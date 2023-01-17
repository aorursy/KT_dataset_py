# 导入相关库

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
# 读取数据

X_full = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')
# 移除缺失值的目标列

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)



# 分离预测目标

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



# 划分训练集，测试集

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8,

                                                                test_size=0.2, random_state=0)





# 选取分类变量

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and

                    X_train_full[cname].dtype == 'object']





# 选取数值变量

numerical_cols = [cname for cname in X_train_full.columns if

                  X_train_full[cname].dtype in ['int64', 'float64']]



# 仅保留选取变量

my_cols = categorical_cols+numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
X_train.head()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# 数据预处理

numerical_transformer=SimpleImputer(strategy='constant')



# 分类数据预处理(缺失值，独热编码)

categorical_transformer=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),

                                       ('onehot',OneHotEncoder(handle_unknown='ignore')),])



# 数值变量与分类变量处理

preprocessor=ColumnTransformer(transformers=[

    ('num',numerical_transformer,numerical_cols),

    ('cat',categorical_transformer,categorical_cols)

])



# 定义模型

from xgboost import XGBRegressor

model=XGBRegressor(n_estimators=900,learing_rate=0.1)

# 打包管道预处理及建模代码

clf=Pipeline(steps=[('preprocessor',preprocessor),

                   ('model',model)])



# 训练模型

clf.fit(X_train,y_train)



# 预测

preds=clf.predict(X_valid)



# MAE

print("MAE:",mean_absolute_error(y_valid,preds))
# 预处理测试模型，拟合模型

preds_test=clf.predict(X_test)



#  输出

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)