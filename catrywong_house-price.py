#基本模块

import numpy as np

import pandas as pd

#画图模块

import matplotlib.pyplot as plt

from sklearn import preprocessing

#模型训练前把数据分组用的train_test_split, 用到基础模型xgboost,

from sklearn.model_selection import train_test_split

import xgboost as xgb

from xgboost import plot_importance

#忽略wainings

import warnings

warnings.filterwarnings('ignore')

#清理内存

import gc

#图表显示设置

%matplotlib inline

#打印数据表格的文件

import os

os.listdir('../input')

import math
os.listdir('../input')
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_test.head()
df_all = pd.concat([df_train.loc[ : , 'MSSubClass':'SaleCondition'],

                   df_test.loc[ : , 'MSSubClass':'SaleCondition']])

df_all = df_all.reset_index(drop = True)



print(df_train.shape, df_test.shape, df_all.shape)
#用df_all表

total = df_all.isnull().sum().sort_values(ascending = False)

percent = (100 * df_all.isnull().sum() / df_all.isnull().count()).sort_values(ascending = False).round(2)

missing_table = pd.concat([total, percent], axis = 1)

missing_table = missing_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

print(missing_table.head(20))
feat_num = df_all.dtypes[df_all.dtypes != 'object'].index

feat_object = df_all.dtypes[df_all.dtypes == 'object'].index



print('feat_num :', feat_num)

print(feat_num.shape)



print('-----------------------------------------------------------')



print('feat_object :', feat_object)

print(feat_object.shape)
fig, axes = plt.subplots(3, 3, figsize=(20, 20))

plt.subplots_adjust(wspace = 0.2, hspace = 0.2)

tem = df_train[feat_num].fillna(-100)

tem['SalePrice'] = df_train['SalePrice']

for a in range(3):

    for b in range(3):

        k = a * 3 + b

        df_is_null = tem[tem.iloc[ : , k] == -100]#用-100表示空值

        df_no_null = tem[tem.iloc[ : , k] != -100]

        axes[a, b].scatter(df_no_null.iloc[ : , k], df_no_null['SalePrice'])

        axes[a, b].scatter(df_is_null.iloc[ : , k], df_is_null['SalePrice'], c = 'r', s = 40)

        axes[a, b].set_xlabel(feat_num[k])

plt.show()
feat_discrete = ['MSSubClass','OverallQual', 'OverallCond' ]

feat_continus = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1']

#填补空缺值（用df_all）

df_all['LotFrontage'].fillna(df_all['LotFrontage'].mean(), inplace = True)

df_all['MasVnrArea'].fillna(0, inplace = True)

df_all['BsmtFinSF1'].fillna(0, inplace = True)

#查看是有还有空值

print(df_all[feat_num[ : 9]].isnull().sum())



drop_row = [ ]

drop_row += df_train['LotFrontage'][df_train['LotFrontage'] > 250].index.tolist()

drop_row += df_train['BsmtFinSF1'][df_train['BsmtFinSF1'] > 5000].index.tolist()



fig, axes = plt.subplots(3, 3, figsize=(20, 20))

plt.subplots_adjust(wspace = 0.2, hspace = 0.2)

tem = df_train[feat_num].fillna(-100)

tem['SalePrice'] = df_train['SalePrice']

for a in range(3):

    for b in range(3):

        k = a * 3 + b + 9

        df_is_null = tem[tem.iloc[ : , k] == -100]

        df_no_null = tem[tem.iloc[ : , k] != -100]

        axes[a, b].scatter(df_no_null.iloc[ : , k], df_no_null['SalePrice'])

        axes[a, b].scatter(df_is_null.iloc[ : , k], df_is_null['SalePrice'], c = 'r', s = 40)

        axes[a, b].set_xlabel(feat_num[k])

plt.show()

#填补空缺值（用df_all）

feat_discrete += ['BsmtFullBath', 'BsmtHalfBath','LowQualFinSF']

feat_continus += ['BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea']

#填补空值（用df_all）

tem = ['BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

for i in tem:

    df_all[i].fillna(0, inplace = True)

#查看是有还有空值

print(df_all[feat_num[9 : 18]].isnull().sum())



drop_row += df_train['BsmtFinSF2'][df_train['BsmtFinSF2'] > 1400].index.tolist()

drop_row += df_train['GrLivArea'][(df_train['GrLivArea'] > 4000) & (df_train['SalePrice'] < 300000)].index.tolist()

fig, axes = plt.subplots(3,3, figsize=(20, 20))

plt.subplots_adjust(wspace = 0.2, hspace = 0.2)

tem = df_train[feat_num].fillna(-100)

tem['SalePrice'] = df_train['SalePrice']

for a in range(3):

    for b in range(3):

        k = a * 3 + b + 18

        df_is_null = tem[tem.iloc[:, k] == -100]

        df_not_null = tem[tem.iloc[:, k] != -100]

        axes[a, b].scatter(df_not_null.iloc[ : , k], df_not_null['SalePrice'])

        axes[a, b].scatter(df_is_null.iloc[ : , k], df_is_null['SalePrice'], c = 'r', s = 40)

        axes[a, b].set_xlabel(feat_num[k])

plt.show()
#填补空缺值（用df_all）

feat_discrete += ['FullBath', 'HalfBath','BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces','GarageYrBlt', 'GarageCars']

feat_continus += ['GarageArea']

#填补空值（用df_all）

df_all['GarageYrBlt'].fillna(0, inplace = True)

df_all['GarageCars'].fillna(0, inplace = True)

df_all['GarageArea'].fillna(0, inplace = True)

#查看是有还有空值

print(df_all[feat_num[18 : 27]].isnull().sum())



drop_row += df_train['KitchenAbvGr'][df_train['KitchenAbvGr'] == 0].index.tolist()

fig, axes = plt.subplots(3,3, figsize=(20, 20))

plt.subplots_adjust(wspace = 0.2, hspace = 0.2)

tem = df_train[feat_num].fillna(-100)

tem['SalePrice'] = df_train['SalePrice']

for a in range(3):

    for b in range(3):

        k = a * 3 + b + 27

        df_is_null = tem[tem.iloc[ : , k] == -100]

        df_not_null = tem[tem.iloc[ : , k] != -100]

        axes[a, b].scatter(df_not_null.iloc[ : , k], df_not_null['SalePrice'])

        axes[a, b].scatter(df_is_null.iloc[ : , k], df_is_null['SalePrice'], c = 'r', s = 40)

        axes[a, b].set_xlabel(feat_num[k])

plt.show()
#填补空缺值（用df_all）

feat_discrete += ['EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal','MoSold', 'YrSold']

feat_continus += ['WoodDeckSF', 'OpenPorchSF']

#查看是有还有空值

print(df_all[feat_num[27:36]].isnull().sum())



drop_row += df_train['MiscVal'][df_train['MiscVal'] > 4800].index.tolist()
#outliers所在的行序

drop_row = list(set(drop_row))

print(drop_row)

#离散型数值列列数

print(len(feat_discrete))

print(len(feat_continus))
#去除outliers

df_all.drop(drop_row, inplace = True)
print(df_train.shape, df_test.shape, df_all.shape)
for b in feat_object:

    le = preprocessing.LabelEncoder()

    df_all[b].fillna('nan', inplace = True)

    le.fit(df_all[b])

    df_all[b] = le.transform(df_all[b])
#验证df_all的空值已经完全填好

df_all.isnull().any().sum()
type(feat_object)
#合并文本列及离散类数值列，用feat_one_hot表示

a = feat_object.tolist()

feat_one_hot = a + feat_discrete

print(len(feat_one_hot))
df_allX = pd.get_dummies(df_all, columns = feat_one_hot )
df_allX.head()



del df_all

gc.collect()
df_train.shape
df_train.loc[drop_row ,]
df_train.drop(drop_row, axis = 0, inplace = True)
df_train_fin = df_allX.iloc[ : df_train.shape[0], : ]

df_test_fin = df_allX.iloc[df_train.shape[0] : , ]

#加上SalePrice(去除outliers行后)



df_train_fin.loc[ : , 'SalePrice'] = df_train['SalePrice'].values



print(df_train_fin.shape,df_test_fin.shape)



# del df_allX, df_train

# gc.collect()
X = df_train_fin.drop(['SalePrice'], axis= 1).values

y = df_train_fin['SalePrice'].values



del df_train_fin

gc.collect()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 1)



#用xgb.XGBRegressor训练3个不同参数的模型，生成3份预测结果后求均值（ans表示测试组的预测答案，xxx表示训练组的预测答案）

model1 = xgb.XGBRegressor(max_depth=3, learning_rate=0.3, n_estimators=350, silent=True, subsample = 0.8, objective='reg:gamma')

model1.fit(X_train, y_train)

ans1 = model1.predict(X_test)

xxx1 = model1.predict(X_train)



model2 = xgb.XGBRegressor(max_depth=4, learning_rate=0.1, n_estimators=400, silent=True, subsample = 0.8, objective='reg:gamma')

model2.fit(X_train, y_train)

ans2 = model2.predict(X_test)

xxx2 = model2.predict(X_train)



model3 = xgb.XGBRegressor(max_depth=4, learning_rate=0.2, n_estimators=400, silent=True, subsample = 0.8, objective='reg:gamma')

model3.fit(X_train, y_train)

ans3 = model3.predict(X_test)

xxx3 = model3.predict(X_train)



#求测试组和训练组的预测答案均值

ans = (ans1 + ans2 + ans3) / 3

xxx = (xxx1 + xxx2 + xxx3) / 3



#查看效果

a = []

for i in range(len(ans)):

    a.append((np.log(y_test[i]) - np.log(ans[i])) * (np.log(y_test[i]) - np.log(ans[i])))

rmse = math.sqrt(sum(a)/len(ans))



b = []

for i in range(len(xxx)):

    b.append((np.log(y_train[i]) - np.log(xxx[i])) * (np.log(y_train[i]) - np.log(xxx[i])))

rmse_xxx = math.sqrt(sum(b)/len(xxx))



#用t最终test生成3次答案

ans_a = model1.predict(df_test_fin.values)

ans_b = model2.predict(df_test_fin.values)

ans_c = model3.predict(df_test_fin.values)

#求均值

ans = (ans_a + ans_b + ans_c) / 3



#生成csv

df_test['SalePrice'] = ans

df_new = df_test.loc[ : , ['Id','SalePrice']]

df_new.to_csv ("002.csv" , encoding = "utf-8", index=0)
