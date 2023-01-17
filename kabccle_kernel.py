# 环境准备

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# 0 读取训练数据, 并从中提取要预测的salePrice.

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")


# 查看数据

train.head(5)
test.head(5)
# 1.1 了解数据集整体情况.，

train.describe()

test.describe()
train.info()
test.info()
# 1.2 了解train与test的大小.

# train数据集1460条数据, 81个特征.

# test数据集1459条数据,80个特征.

print("train-------------------shape")

print(train.shape)

print("test--------------------shape")

print(test.shape)
# 1.3 保存要预测值

y = train['SalePrice']

y
# 2 清洗数据

# 缺失对null值的处理

# 设置处理数据的集合为df_train与df_tes
# 定义清洗数据集,清洗完后为df...

df_train = train.copy()

df_test  = test.copy()
# 统计空值

def df_statistics_null(dfset):

    total = dfset.isnull().sum().sort_values(ascending=False)

    percent = (dfset.isnull().sum()/dfset.shape[0]).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data
# 统计缺失情况：train集

missing_data_train = df_statistics_null(df_train)

missing_data_train.head(20)
# 统计缺失情况：test集

missing_data_test = df_statistics_null(df_test)

missing_data_test.head(20)
# columns where NaN values have meaning e.g. no pool etc.

cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',

               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',

               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',

               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',

               'MSZoning', 'Utilities']



# replace 'NaN' with 'None' in these columns

for col in cols_fillna:

    df_train[col].fillna('None',inplace=True)

    df_test[col].fillna('None',inplace=True)

    

# fillna with mean for the remaining columns: LotFrontage, GarageYrBlt, MasVnrArea

# 为什么这两个变量不进行填none

df_train.fillna(df_train.mean(), inplace=True)

df_test.fillna(df_test.mean(), inplace=True)
a = df_statistics_null(df_test)

a.head(20)
# 3 挖掘数据 处理完后的数据集为df_XX_ml

# 数据的处理方式分为numerical与categories的处理

# 3.1 numerical各个feature之间相关性

corrmat = df_train.corr()

f,ax = plt.subplots(figsize=(20,15))# 调节画布的大小,现在的是20*15

sns.heatmap(corrmat, vmax=.8, annot=True)
# 进一步统计出相关性>0.5的features

corrmat = df_train.corr()

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]

plt.figure(figsize=(10,10))

g = sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# 根据上面的热力相关图, 可以得出: 

# 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF',

# 'GrLivArea',   'FullBath',  'TotRmsAbvGrd', 'GarageCars',  'GarageArea',

#  这10个features与salePrice有很强的相关性, 初步选出这10features

numerical_features = top_corr_features[:-1]

numerical_features
train_num = df_train[numerical_features]

train_num.head(5)
test_num = df_test[numerical_features]
# 3.2 categories的features处理

categorical_features = df_train.select_dtypes(include=['object']).columns

categorical_features
# 将字符型数据集转换成数值型

train_cat = df_train[categorical_features]

test_cat  = df_test[categorical_features]

mix       = pd.concat([test_cat,train_cat])

all_cat   = pd.get_dummies(mix)
train_cat_ecode = all_cat[-1460:]

test_cat_ecode  = all_cat[:1459]
df_train_ml = pd.concat([train_cat_ecode,train_num],axis=1)#axis=1,为列合并

df_test_ml  = pd.concat([test_cat_ecode,test_num],axis=1)
# 4.0创建模型, 用筛选出的features训练模型.

from sklearn.linear_model import LinearRegression

lReg = LinearRegression()

lReg.fit(df_train_ml,y)
pred = lReg.predict(df_test_ml)

pred
# 选择feature

# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

# df_train = train[cols]

# df_train.head(5)
# 分离出要预测的变量

# y = df_train.SalePrice

# y.head(5)
# 切分dateset in train

# from sklearn.model_selection import cross_val_score, train_test_split

# X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size = 0.3, random_state = 0)

# print("X_train : " + str(X_train.shape))

# print("X_test : " + str(X_test.shape))

# print("y_train : " + str(y_train.shape))

# print("y_test : " + str(y_test.shape))
# # 对features进行正则化处理

# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()

# df_train_ml_sc = sc.fit_transform(x_train)

# df_test_ml_sc = sc.transform(x_test)
# df_train_ml_sc = pd.DataFrame(df_train_ml_sc)

# df_test_ml_sc = pd.DataFrame(df_test_ml_sc)
# # model, 创建模型

# from sklearn.linear_model import LinearRegression

# LR = LinearRegression()

# LR.fit(df_train_ml_sc,y)
# accuracy = LR.score(X_test, y_test)

# accuracy
# pred = LR.predict(df_test_ml_sc)

# # pred = pd.DataFrame(pred)

# # id = test['Id']

# # result = pd.concat([id,pred],axis=1)

# # print(type(pred))

# # print(type(id))

# # result.head(5)
# 提交答案

pred.astype(int)

submission = pd.DataFrame()

submission['Id'] = test.Id

submission['SalePrice'] = pred

submission.to_csv('output.csv', index=False)
submission.head(5)