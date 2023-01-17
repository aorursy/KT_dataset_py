import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set_style('darkgrid')

import os
print(os.listdir("../input"))
# 查看 input 下数据有哪些
raw_train = pd.read_csv('../input/train.csv',index_col='Id')
print('train.shape', raw_train.shape)

raw_test = pd.read_csv('../input/test.csv',index_col='Id')
print('test.shape', raw_test.shape)

# 查看train和test数据量，找到目标列 SalePrice
print('target is', set(raw_train.columns) - set(raw_test.columns))
# 我认为训练集和预测集不应该合并
# y_train = raw_train['SalePrice']
# all_data = pd.concat((raw_train.drop('SalePrice', axis=1), raw_test),axis=0,sort=False).reset_index(drop=True)
# print('all_data.shape:', all_data.shape)
# 训练集   (1460, 81)
# 预测数据 (1459, 80)
# 合并    (2919, 80)
# all_data.info()
X_raw_train = raw_train.drop('SalePrice', axis=1)
y_raw_train = raw_train['SalePrice']
def get_missing_ratio(data, transfer_dtypes=True):
    # 获得数据丢失率
    # https://github.com/echo-ray/utils/blob/master/get_missing_ratio.py
    info = {}
    for col in data.columns:
        total_row = data.shape[0]
        unique = data[col].value_counts()
        unique_amount  = len(unique)
        # if (data[col].dtype != 'category'):
        # 将出现很少不同值的数字列 转换为类别列。ps：是否应该作为另一个函数？
        if transfer_dtypes:
            if unique_amount < 50:
                data[col] = data[col].astype('category')
                #data[col] = data[col].astype('str')
        # 如果这一列有超过20个 unique 的值，那只查看前五个
        unique_show = unique.to_dict()
        if unique_amount > 20:
            unique_show = unique.head(5).to_dict()#[:5]
        unique = unique.to_dict()
        total_amount = sum(unique.values())
        missing_row = total_row - total_amount
        missing_ratio = round((missing_row / total_row)*100, 2)
        data_type = data[col].dtype

        info[col] = {
            'colume':col, 'missing_row':missing_row, 'data_type':data_type, 'unique':unique_show,
            'unique_amount':unique_amount, 'missing_ratio':missing_ratio, 
        }
        # print(f"{col:15}|{missing_ratio:>5.2f}%|", unique)
    return pd.DataFrame(info).T.sort_values(by='missing_ratio', ascending=False)

columns_info = get_missing_ratio(X_raw_train)
#.sort_values(by='unique_amount').head(50)
columns_info
# categorical_columns = list(columns_info[columns_info['data_type'] == 'category'].index)
# numeric_columns = list(set(columns_info.index) - set(categorical_columns))
# columns_info['data_type'] != np.number 这样似乎并不对
# 筛选字符型的列
str_data = X_raw_train.select_dtypes(exclude='number')#(include=['category','object'])#
categorical_colums = list(str_data.columns)
print(f'{len(categorical_colums)} categorical_colums: \n', categorical_colums)

# 筛选数值型的列
numeric_data = X_raw_train.select_dtypes(include='number')
numeric_colums = list(numeric_data.columns)
print(f'\n {len(numeric_colums)} numeric_colums: \n', numeric_colums)
# num_data.describe()
print('\n有重复的列吗? ', 'Yes' if ((set(categorical_colums) & set(numeric_colums)) != set()) else 'No')
print('有漏的列吗？  ', 'Yes' if (set(categorical_colums) | set(numeric_colums)) != set(X_raw_train.columns) else 'No')
# 应该只在类别列 查看差异？
# for col in raw_test.columns:
#     if col in ['Id', 'LotArea', 'MasVnrArea', 'BsmtFinSF1']:
#         continue
#     train_unique = set(raw_train[col].unique())
#     test_unique = set(raw_test[col].unique())
#     # 排除数值型特征，也许有更好的办法？
#     if len(test_unique) > 100:
#         continue
#     if train_unique == test_unique:
#         continue
#     else:
#         a = ((test_unique-train_unique) | (train_unique - test_unique))
#         print(col, ':', a)

import sklearn
sklearn.__version__
# 0.20 版后引入了 ColumnTransformer
# http://scikit-learn.org/dev/modules/generated/sklearn.compose.ColumnTransformer.html
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest,SelectPercentile, chi2
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression,BayesianRidge,ElasticNet
from sklearn.decomposition import PCA

categorical_features = categorical_colums
category_trans = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_features = numeric_colums
numeric_trans = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

column_transfer = ColumnTransformer(
    transformers=[
        #('category', category_trans, categorical_features),
        ('numeric', numeric_trans, numeric_features)
    ],
    # remainder=MinMaxScaler()
    # remainder='passthrough' # 保留原始数据？
    # remainder='drop' # 丢掉原始数据？
)
# 测试下 column_transfer 有没有错误
# column_transfer.fit(X_train)

lgb = Pipeline([
    ('preprocessor', column_transfer),
    #('SelectKBest', SelectKBest(chi2, k=3)),
    ('SVD', TruncatedSVD(n_components=15)),  # 如何拿到这15个特征的名字？ 
    #('PCA', SelectPercentile(percentile=0.8)),#PCA(n_components=10)),
    ('LGB', LGBMRegressor()),
    #('reg', LogisticRegression())
])

# -------------- 切分数据 -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_raw_train, y_raw_train,
    random_state=111, #shuffle=False
)

# ------------------训练-预测---------------------------
lgb.fit(X_train, y_train)
y_pred = lgb.predict(X_test)
#print()
print("model score: %.3f" % lgb.score(X_test, y_test))
# 拿到 pipeline 组件的一些数据
# SVD_ = lgb.steps[1][1]
# vars(SVD_)
# vars(lgb.steps[2][1])
param_grid={
    #'SVD__n_components':[2, 6, 9, 12, 15, 20],
    'LGB__num_leaves':[10, 20, 30, 40, 60, 90]
}

grid_search = GridSearchCV(lgb, param_grid, cv=3, return_train_score=False)
grid_search.fit(X_train, y_train)
print("model score: %.3f" % grid_search.score(X_test, y_test))
grid_search.cv_results_

from sklearn import linear_model

'''
column_transfer1 = column_transfer2 = ColumnTransformer(
    [
        ('category', OneHotEncoder(), str_colums),#CountVectorizer(analyzer=lambda x: [x]), 'city'),
        ('num', RobustScaler(), num_colums),#CountVectorizer(), 'title')
    ],
    remainder=MinMaxScaler()
)

column_transfer2 = ColumnTransformer(
    [
        ('category', OneHotEncoder(), str_colums),#CountVectorizer(analyzer=lambda x: [x]), 'city'),
        ('num', RobustScaler(), num_colums),#CountVectorizer(), 'title')
    ],
    remainder=MinMaxScaler()
)
column_transfer.fit_transform(all_data)


numeric_features = ['age', 'fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['embarked', 'sex', 'pclass']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

X = data.drop('survived', axis=1)
y = data['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
'''
