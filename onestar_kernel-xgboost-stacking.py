#load packages， 打印，便于可复现
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)


#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

'''
预测SalePrice的值
'''

# train_df.columns 
# train_df.shape
train_df.describe()
# train_df.info()
# print '%' * 40
# test_df.info()
# train_df.head(10)
# train_df.info()
# print train_df.describe(include = 'all')
# print '%' * 40
# test_df.info()
# # Missing Data的百分比
# total = train_df.isnull().sum().sort_values(ascending=False)
# percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# # print missing_data.head(20)

# total2 = test_df.isnull().sum().sort_values(ascending=False)
# percent2 = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
# missing_data2 = pd.concat([total2, percent2], axis=1, keys=['Total', 'Percent'])
# # print missing_data2.head(20)
train_df.describe(include = 'all')
print ('%' * 40)

# print train_df.loc[:,'MasVnrType']

# train_df['MasVnrType'] = train_df['MasVnrType'].replace(['None'],None )
# train_df.loc[:,'MasVnrType'] =  train_df['MasVnrType'].apply(lambda x: None if x == 'None' else x)
# print type(train_df.loc[1,'MasVnrType'])

limit_missing_values = 0.25
train_limit_missing_values = len(train_df) * limit_missing_values 
print ("Train columns with null values:\n", train_df.columns[train_df.isnull().sum().values > train_limit_missing_values])  # 依列为标准，column
print ('%'*40)

test_limit_missing_values = len(test_df) * limit_missing_values
print ("Test columns with null values:\n", test_df.columns[test_df.isnull().sum().values > test_limit_missing_values])


missing_columns = list(train_df.columns[train_df.isnull().sum() != 0])
print (missing_columns)
# train_df[missing_columns].describe(include = 'all')
print ('%' * 40)
test_missing_columns = list(test_df.columns[test_df.isnull().sum() != 0])
print (test_missing_columns)
train_missing_numerical = list(train_df[missing_columns].dtypes[train_df[missing_columns].dtypes != 'object'].index)
train_missing_category = [i for i in missing_columns if i not in train_missing_numerical]

test_missing_numerical = list(test_df[test_missing_columns].dtypes[test_df[test_missing_columns].dtypes != 'object'].index)
test_missing_category = [i for i in test_missing_columns if i not in test_missing_numerical]

print ("Train missing numerical: ", train_missing_numerical, "\n")
print ("Train missing category: ", train_missing_category,  "\n")
print ("Test missing numerical: ", test_missing_numerical, "\n")
print ("Test missing category: ", test_missing_category, "\n")
# 取众数的特征
train_categories_Mode = ['Electrical']
test_categories_Mode = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']

train_categories_none = [i for i in train_missing_category if i not in train_categories_Mode]
test_categories_none = [i for i in test_missing_category if i not in test_categories_Mode]


# 通过字符串“None”填充 
for category in train_categories_none:
    train_df[category].fillna("None", inplace=True)
    
for category in test_categories_none:
    test_df[category].fillna("None", inplace=True)
    
for category in train_categories_Mode:
    train_df[category].fillna(train_df[category].mode()[0], inplace = True)

for category in test_categories_Mode:
    test_df[category].fillna(test_df[category].mode()[0], inplace = True)
# 取0的数值特征

for col in ('GarageArea', 'GarageCars', 'GarageYrBlt'):
    train_df[col] = train_df[col].fillna(0)
    test_df[col] = test_df[col].fillna(0)
    
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    train_df[col] = train_df[col].fillna(0)
    test_df[col] = test_df[col].fillna(0)
    
# 通过中位数填充
for column in train_missing_numerical:
    train_df[column].fillna(train_df[column].median(), inplace=True)

for column in test_missing_numerical:
    test_df[column].fillna(test_df[column].median(), inplace=True) 
print (train_df.isnull().sum().max())
print (test_df.isnull().sum().max())
# print test_df.loc[test_df.isnull()
train_df['SalePrice'].describe()
sns.distplot(train_df['SalePrice'], color='green')
print ("偏度为 %f " % train_df['SalePrice'].skew())
print ("峰度为 %f"  % train_df['SalePrice'].kurt())
var = 'BsmtFinSF1' # 房屋居住面积

# concat - Series默认行合并； axis = 1，列合并
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1) 
# print dataFrame
data.plot.scatter(x=var, y='SalePrice', ylim = (0, 800000)) # y轴限制
var = 'TotalBsmtSF'  # 房屋地下室的面积
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x = var, y = 'SalePrice', ylim = (0, 800000))
var = 'OverallQual'  # 房屋整体材料和光洁度的评估
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)  # 横坐标类别，纵坐标目标变量
fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'  # 房屋建造年份
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))  # 改变大小
fig = sns.boxplot(x=var, y="SalePrice", data=data)  # 横坐标类别，纵坐标目标变量
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);               # x如果都水平放置，会粘到一起。x倾斜90度，减少占的位置
cols = ['MSSubClass', 'OverallCond', 'BedroomAbvGr', 'YrSold', 'MoSold']

for col in cols:
    train_df[col] = train_df[col].apply(str)
    test_df[col] = test_df[col].apply(str)
corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(corrmat, vmax= .8, square=True);   # vmax 颜色区别，最浅的颜色在0.8
# help(corrmat.nlargest)
# # cols = corrmat.nlargest(k, 'SalePrice').index
# # print cols
# len (train_df[cols].values)  # 数组，1460 * 10 | 转置 10 * 1460
old_features = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']
train_df['TotalSF'] = 0
test_df['TotalSF'] = 0
for i in old_features:
    print (train_df['SalePrice'].corr(train_df[i]))
    train_df['TotalSF'] += train_df[i]
    test_df['TotalSF'] += test_df[i]
train_df['SalePrice'].corr(train_df['TotalSF'])
corrmat = train_df.corr().abs()
#saleprice correlation matrix
k = 36   #number of variables for heatmap，热力图变量数量 

# nlargest - 根据SalePrice列排序，返回前10个跟SalePrice相关性最高的行
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index 

# cm = corrmat.loc[cols,cols] 同以下cm赋值相同
# 训练集中取出目标列的样本，转置，计算10个特征之间的相关性
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
print (len(train_df.columns))
# print len(drop_columns)
print (len(train_df.dtypes[train_df.dtypes == 'object'].index))
# cols
# #scatterplot
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(train_df[cols], size = 2.5)
# plt.show();
# olss = []
# numerical_features = ['GrLivArea']
# for feature in numerical_features:
    
#     # 计算25%分位点
#     Q1 = np.percentile(train_df[feature], 25)
    
#     # 计算75%分位点
#     Q3 = np.percentile(train_df[feature], 75)
#     print Q3
    
#     # 异常阶（1.5倍四分位距）IQR
#     step = 1.5 * (Q3 - Q1)
    
#     print "Feature" + feature + "Outlines"
#     print train_df[ (train_df[feature] <= Q1 - step) | (train_df[feature] >= Q3 + step)][numerical_features]
#     ols = features_train[ (train_df[feature] <= Q1 - step) | (train_df[feature] >= Q3 + step)].index.tolist()
#     olss.append(ols)
    
# olss_new = [ii for i in olss for ii in i]
# # print olss_new

# # 列表方法 .count(i) 统计列表中某个元素出现的次数
# more_than_one = list(set([i for i in olss_new if olss_new.count(i) > 1]))
# more_than_two = list(set([i for i in olss_new if olss_new.count(i) > 2]))
# print len(more_than_one), len(more_than_two)

# ## 移除异常点
# # features_train_new = features_train.drop(features_train.index[olss_new]).reset_index(drop = True)
# # labels_train_new = labels_train.drop(features_train.index[olss_new]).reset_index(drop = True)
var = 'GrLivArea'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# 从散点图中确定去除GrLivArea > 4000 且SalePrice < 200000的点
train_df = train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 200000)].index)
train_df.dtypes[(train_df.dtypes == 'object')].index
var = 'BldgType'  # 房屋整体材料和光洁度的评估
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)  # 横坐标类别，纵坐标目标变量
fig.axis(ymin=0, ymax=800000);
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])
sns.distplot(train_df['SalePrice'], color='blue')

print (train_df['SalePrice'].skew())
train_df['SalePrice'].kurt()
from sklearn.preprocessing import LabelEncoder
# categories = [i for i in train_df.columns if i not in cols]
# process columns, apply LabelEncoder to categorical features

categories = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'BedroomAbvGr']

for c in categories:
    lbl = LabelEncoder() 
    train_df[c] = lbl.fit_transform(list(train_df[c].values))
    test_df[c] = lbl.fit_transform(list(test_df[c].values))


# shape        
print ('Shape all_data: {}'.format(train_df.shape))
print ('Shape all_data: {}'.format(test_df.shape))

numeric_features = list(train_df.dtypes[train_df.dtypes != 'object'].index)
numeric_features.remove('SalePrice')
len(numeric_features)
# box-cox变换
from scipy.special import boxcox1p
# skewed_features = list(skewness)
lam = 0.15
for feature in numeric_features:
    #all_data[feat] += 1
    train_df[feature] = boxcox1p(train_df[feature], lam)
    test_df[feature] = boxcox1p(test_df[feature], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
scaler.fit(train_df[numeric_features])
train_df[numeric_features] = scaler.transform(train_df[numeric_features])
test_df[numeric_features] = scaler.transform(test_df[numeric_features])

# scaler = MinMaxScaler()
# scaler.fit(train_df[numeric_features])
# train_df[numeric_features] = scaler.transform(train_df[numeric_features])
# test_df[numeric_features] = scaler.transform(test_df[numeric_features])
category_columns = train_df.dtypes[train_df.dtypes == 'object'].index
for i in category_columns:
    if len(train_df[i].value_counts().index) <= 2:
        print  ("Train\n" +  i)
        print  (train_df[i].value_counts())
        
    if len(test_df[i].value_counts().index) <= 2:
        print ("Test\n" + i)
        print (test_df[i].value_counts())
drop_columns = 'Utilities'
train_df = train_df.drop(drop_columns, axis=1)
test_df = test_df.drop(drop_columns, axis=1)
features_train = train_df.drop(['SalePrice'], axis=1)
labels_train = train_df['SalePrice']
features_test = test_df
features_train = pd.get_dummies(features_train)
features_test = pd.get_dummies(test_df)

missing_cols = set(features_train.columns) - set(features_test.columns)
for column in missing_cols:
    features_test[column] = 0
    
# 保证测试集columns的顺序同训练集columns相同，特别重要！！！！！！
features_test = features_test[features_train.columns]
print (features_train.columns)
# from sklearn.model_selection import train_test_split

# # 分割features_train 和 labels_train, 测试集大小 = 20%，状态：随机，可复现
# # 顺序：测试特征，训练特征，测试目标，训练目标

# X_train, X_test, y_train, y_test = train_test_split(features_train, labels_train, test_size = 0.2, random_state = 42)

# 输出数量观察

X_train = features_train
y_train = labels_train
print (len(X_train))
# # 导入算法模型和评分标准 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import fbeta_score, make_scorer, r2_score ,mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# def rmsle_cv(model, train, value):
#     kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train.values)
#     rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
#     return rmse.mean()

# # for key_num in range(10,25,3):

# #     cols_model = cols[:key_num]
# #     drop_columns = [i for i in corrmat.columns if i not in cols_model]
# #     print drop_columns , "\n"
# #     X_train_model = X_train.drop(drop_columns, axis=1)

# #         # 初始化,确定随机状态，可复现
# #     reg1 = DecisionTreeRegressor(random_state = 42)
# #     reg2 = LinearRegression()
# #     reg3 = RandomForestRegressor(random_state = 42)
# #     reg4 = XGBRegressor()
# #     reg5 = Lasso(alpha=0.001, random_state= 42)
# #     reg6 = SVR(kernel = 'rbf')
# #     reg7 = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)

# #     # 建立字典，收集学习器的效果
# #     # 学习，收集预测得分
# #     results = {}
# #     for reg in [reg1, reg2, reg3, reg4, reg5, reg6, reg7]:
# #         # 回归器的名称
# #         reg_name = reg.__class__.__name__
# # #         reg.fit(X_train_model, y_train)
# # #         pred_test = reg.predict(X_test_model)
# # #         results[reg_name] = rmse(y_test, pred_test)
# #         results[reg_name] = rmsle_cv(reg, X_train_model, y_train)
# #     print key_num, "---", results
# #     print '\n'
cols = cols[:22] 
drop_columns = [i for i in corrmat.columns if i not in cols]

print (drop_columns , "\n")
X_train = X_train.drop(drop_columns, axis=1)
print (len(X_train.columns))
features_test = features_test.drop(drop_columns, axis=1)
from sklearn.metrics import fbeta_score, make_scorer, r2_score ,mean_squared_error

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
# 模型：RandomForest
# 导入Grid
from sklearn.model_selection import GridSearchCV

# 初始化回归模型
reg = RandomForestRegressor(random_state=42)

# 确定参数列表
parameters = {
    'max_leaf_nodes': range(20, 100 ,10)
}

# 确定评分标准
scorer = 'neg_mean_squared_error'

# 回归模型使用网格搜索
grid_reg = GridSearchCV(reg, parameters, scoring = scorer)

# 训练
grid_reg.fit(X_train, y_train)

# grid_reg.cv_results_

# 获得最佳拟合回归器
best_reg_rf = grid_reg.best_estimator_
# print grid_reg.best_estimator_
print (grid_reg.best_estimator_)
pred_rf = best_reg_rf.predict(X_train)
rmsle(pred_rf, y_train)
# 模型：线性回归
from sklearn.model_selection import GridSearchCV

# 初始化回归模型
reg = Lasso( alpha=0.001, random_state= 42)

# 确定参数列表
parameters = {
    'normalize': [True, False],
    'alpha': [0, 0.0001, 0.0005, 0.001]
}

# 确定评分标准
scorer = 'neg_mean_squared_error'

# 回归模型使用网格搜索
grid_reg = GridSearchCV(reg, parameters, scoring = scorer)

# 训练
grid_reg.fit(X_train, y_train)

# 获得最佳拟合回归器
best_reg_lasso = grid_reg.best_estimator_
print (grid_reg.best_estimator_)
pred_lasso = best_reg_lasso.predict(X_train)
rmsle(pred_lasso, y_train)
reg = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state = 42)

parameters = {
    'alpha':[0, 0.0001, 0.0005, 0.001],
    'l1_ratio':np.arange(0, 1, 0.1)
}

scorer = 'neg_mean_squared_error'

grid_reg = GridSearchCV(reg, parameters, scoring= scorer)

grid_reg.fit(X_train, y_train)

best_elasticNet_reg =  grid_reg.best_estimator_
print (grid_reg.best_estimator_)
pred_elasticNet_reg = best_elasticNet_reg.predict(X_train)
rmsle(pred_elasticNet_reg, y_train)
from sklearn.svm import SVR

reg = SVR(kernel = 'rbf')

parameters = {
    'C': np.arange(1.1,2,0.1),
    'gamma':np.arange(0,0.1,0.01)
}

scorer = 'neg_mean_squared_error'

grid_reg = GridSearchCV(reg, parameters, scoring = scorer)

grid_reg.fit(X_train, y_train)

best_reg_SVR = grid_reg.best_estimator_
print (grid_reg.best_estimator_)
pred_lasso = best_reg_lasso.predict(X_train)
rmsle(pred_lasso, y_train)
# # 确定最佳决策树的数量, xgboost中的cv函数来确定最佳的决策树数量, 提高Xgboost调参速度
# # 后边有 early_stopping_rounds 个rmse没有下降的就停止
# # 原话：
# # Activates early stopping. CV error needs to decrease at least
# #        every <early_stopping_rounds> round(s) to continue.
# #        Last entry in evaluation history is the one from best iteration.

# import xgboost as xgb
# from xgboost.sklearn import XGBRegressor

# def modelfit(alg, X_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
#     if useTrainCV:
#         xgb_param = alg.get_xgb_params()
#         xgtrain = xgb.DMatrix(X_train.values, label= y_train.values)
#         cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
#             metrics='rmse', early_stopping_rounds = early_stopping_rounds)
#         alg.set_params(n_estimators=cvresult.shape[0])
#         print cvresult

#     #Fit the algorithm on the data
#     alg.fit(X_train, y_train,eval_metric='rmse')

#     #Predict training set:
#     dtrain_predictions = alg.predict(X_train)

#     # 模型训练报告
#     print "\nModel Report"
#     print "RMSE Score (Train): %f" % rmse(y_train, dtrain_predictions)
    
#     # 特征重要性
#     feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)[:10,]
#     feat_imp.plot(kind='bar', title='Feature Importances')
#     plt.ylabel('Feature Importance Score')
# 模型：Xgboost
from sklearn.model_selection import GridSearchCV

best_reg_xgb = XGBRegressor(learning_rate= 0.01, n_estimators = 5000, 
                    max_depth= 4, min_child_weight = 1.5, gamma = 0.1, 
                   subsample = 0.7, colsample_bytree = 0.6, 
                   seed = 27)

# modelfit(reg, X_train, y_train)
best_reg_xgb.fit(X_train, y_train)
pred_y_XGB = best_reg_xgb.predict(X_train)
print (rmsle(pred_y_XGB, y_train))
# rmsle_cv(best_reg_xgb, X_train, y_train)
# param_test1 = {
#     'max_depth': range(3,10,2),
#     'min_child_weight': range(1,6,2)
# }

# scorer = make_scorer(rmse)

# # 负均方误差
# grid_reg1 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.01, n_estimators=1605, 
#                                                 max_depth=5, min_child_weight = 1,gamma = 0,
#                                                 subsample = 0.8, colsample_bytree = 0.8, seed = 27), 
#                          param_grid = param_test1, scoring = 'neg_mean_squared_error')
# grid_reg1.fit(X_train, y_train)
# grid_reg1.grid_scores_, grid_reg1.best_params_, grid_reg1.best_score_

# param_test1 = {
#     'max_depth': [4, 5, 6],
#     'min_child_weight': np.arange(1.0, 4.0, 0.5)
# }

# scorer = make_scorer(rmse)

# # 负均方误差
# grid_reg1 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=115, 
#                                                 max_depth=5, min_child_weight = 1,gamma = 0,
#                                                 subsample = 0.8, colsample_bytree = 0.8, seed = 27), 
#                          param_grid = param_test1, scoring = 'neg_mean_squared_error')
# grid_reg1.fit(X_train, y_train)
# grid_reg1.grid_scores_, grid_reg1.best_params_, grid_reg1.best_score_
# param_test3 = {
#     'gamma' : [i/10.0 for i in range(0,5)]
# }


# grid_reg1 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=115, 
#                                                 max_depth=4, min_child_weight = 1.5,gamma = 0,
#                                                 subsample = 0.8, colsample_bytree = 0.8, seed = 27), 
#                          param_grid = param_test3, scoring = 'neg_mean_squared_error')
# grid_reg1.fit(X_train, y_train)
# grid_reg1.grid_scores_, grid_reg1.best_params_, grid_reg1.best_score_
# param_test4 = {
#     'subsample':np.arange(0.5, 1.0 ,0.05),
#     'colsample_bytree':np.arange(0.5, 1.0, 0.05)
# }


# grid_reg1 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=115, 
#                                                 max_depth=4, min_child_weight = 1.5,gamma = 0,
#                                                 subsample = 0.8, colsample_bytree = 0.8, seed = 27), 
#                          param_grid = param_test4, scoring = 'neg_mean_squared_error')
# grid_reg1.fit(X_train, y_train)
# grid_reg1.grid_scores_, grid_reg1.best_params_, grid_reg1.best_score_
# param_test5 = {
#     "reg_alpha":np.arange(0.0, 1.1, 0.1),
#     "reg_lambda":np.arange(0.0, 1.1, 0.1)
# }

# grid_reg1 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=115, 
#                                                 max_depth=4, min_child_weight = 1.5,gamma = 0,
#                                                 subsample = 0.7, colsample_bytree = 0.6, seed = 27), 
#                          param_grid = param_test5, scoring = 'neg_mean_squared_error')
# grid_reg1.fit(X_train, y_train)
# grid_reg1.grid_scores_, grid_reg1.best_params_, grid_reg1.best_score_
# reg2 = XGBRegressor(learning_rate=0.1, n_estimators= 1000, 
#                          max_depth=4, min_child_weight = 1.5,gamma = 0,
#                         subsample = 0.7, colsample_bytree = 0.6, seed = 27)

# modelfit(reg2, X_train, y_train )
# pred_y_test = reg2.predict(X_test)
# rmse(pred_y_test, y_test)
from mlxtend.regressor import StackingRegressor

# metal_reg = Lasso(alpha= 0.0, random_state=42)
metal_reg = SVR(kernel= 'rbf', C = 20)

stregr = StackingRegressor(regressors = [ best_reg_lasso, best_elasticNet_reg], meta_regressor = metal_reg)

# params = {'meta-lasso__alpha':[0.1, 1.0, 10.0] }

# grid = GridSearchCV(estimator=stregr,
#                     param_grid=params,
#                     cv=5,
#                     refit=True)

# grid.fit(X_train, y_train)
# for params, mean_score, scores in grid.grid_scores_:
#     print("%0.3f +/- %0.2f %r"
#         % (mean_score, scores.std() / 2.0, params))


# rmsle_cv(stregr, X_train, y_train).mean()
stregr.fit(X_train, y_train)
stacked_y_pred  = stregr.predict(X_train)
rmsle(y_train, stacked_y_pred)
# print(__doc__)

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.datasets import load_digits
# from sklearn.model_selection import learning_curve
# from sklearn.model_selection import ShuffleSplit


# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
#     """
#     Generate a simple plot of the test and training learning curve.

#     Parameters
#     ----------
#     estimator : object type that implements the "fit" and "predict" methods
#         An object of that type which is cloned for each validation.

#     title : string
#         Title for the chart.

#     X : array-like, shape (n_samples, n_features)
#         Training vector, where n_samples is the number of samples and
#         n_features is the number of features.

#     y : array-like, shape (n_samples) or (n_samples, n_features), optional
#         Target relative to X for classification or regression;
#         None for unsupervised learning.

#     ylim : tuple, shape (ymin, ymax), optional
#         Defines minimum and maximum yvalues plotted.

#     cv : int, cross-validation generator or an iterable, optional
#         Determines the cross-validation splitting strategy.
#         Possible inputs for cv are:
#           - None, to use the default 3-fold cross-validation,
#           - integer, to specify the number of folds.
#           - An object to be used as a cross-validation generator.
#           - An iterable yielding train/test splits.

#         For integer/None inputs, if ``y`` is binary or multiclass,
#         :class:`StratifiedKFold` used. If the estimator is not a classifier
#         or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

#         Refer :ref:`User Guide <cross_validation>` for the various
#         cross-validators that can be used here.

#     n_jobs : integer, optional
#         Number of jobs to run in parallel (default 1).
#     """
#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()

#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")

#     plt.legend(loc="best")
#     return plt
# title = "Learning Curves (Xgboost)"
# # Cross validation with 100 iterations to get smoother mean test and train
# # score curves, each time with 20% data randomly selected as a validation set.
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

# estimator = XGBRegressor()
# plot_learning_curve(estimator, title, features_train, labels_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

# plt.show()
pred = np.expm1(stregr.predict(features_test) * 0.5 + best_reg_xgb.predict(features_test) * 0.5)
pred
# 增加索引，列名，构建DataFrame，符合输出数据格式

test_df = pd.read_csv('../input/test.csv')
predict_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': pred})

# DataFrame设定index
predict_df = predict_df.set_index('Id')

# 重命名DataFrame的列，列名 = 字典{原：替换后}
# predict_df.rename(columns = {predict_df.columns[0]: 'Id'}, inplace=True)

predict_df
predict_df.to_csv('Submission.csv')