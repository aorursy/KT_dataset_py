# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
import phik
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

path = os.path.join(dirname,'train.csv')
df = pd.read_csv(path, delimiter = ",", index_col = 'Id')
missing = df.isnull().sum()
        
missing = missing[missing>0]
missing.sort_values(inplace=True)
missing.plot.bar()
        
# Show in percent
perc_miss = missing/len(df)
print(perc_miss[perc_miss > 0].sort_values(ascending=False))
perc_miss = df.isnull().mean()
index = perc_miss[perc_miss > 0.8].index
df = df.drop(index, axis = 1)
df
# Correlation between the attribute variables and the target variable
corr = df.phik_matrix()
corr_SP = corr['SalePrice']
corr_SP = corr_SP[corr_SP < 0.2]
corr_SP.plot.bar(figsize=(10,5))
# Removing variables with low correlation in relation to the target variable
index = corr_SP.index
df = df.drop(index, axis = 1)
df
# Removing correlation between attribute variables
# Choose the variable with the highest correlation with the target variable

corr = df.phik_matrix()
corr_sp = corr['SalePrice']
tp = []
rem = []
for c in corr.columns:
    index = corr.loc[corr.loc[:,c] > 0.8].index.tolist()
    for i in index:
        if c != i:
            if corr_sp[c] > corr_sp[i]:
                aux = i
            else:
                aux = c
            if aux not in rem:
                rem = rem + [aux]
                        
            tp = tp + [[c,i,corr.loc[i,c]]]
        
df = df.drop(rem, axis = 1)
df
missing = df.isnull().sum()
        
missing = missing[missing>0]
missing.sort_values(inplace=True)
missing.plot.bar()
missing
# segregate categorical variables from numeric
s = df[missing.index].dtypes == "object"
cat = s[s].index
num = s[~s].index
print("CAT:", cat)
print("NUM:", num)
# MasVnrType and MasVnrArea
# Missing means absence of the characteristic
print(df[['MasVnrType','MasVnrArea']][df.MasVnrType.isnull()])

df['MasVnrType'] = df['MasVnrType'].fillna('NONE')
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
# BsmtCond, BsmtFinType1, BsmtExposure
# Missing means No Basement
# GarageType, GarageFinish, GarageQual
# Missing means No Garagem
# FireplaceQu - Missing means No FirePlace
col = ['BsmtCond', 'BsmtFinType1', 'BsmtExposure', 'GarageType', 'GarageFinish', 'GarageQual', 'FireplaceQu']
df[col] = df[col].fillna('NONE')
# Electrical
print(df[['Electrical', 'SalePrice']].groupby(['Electrical'], as_index = True).count())
# Fill with most frequently
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].value_counts().idxmax())
# LotFrontage
aux = df[['LotFrontage', 'SalePrice']]
aux['miss'] = np.where(aux.LotFrontage.isnull(), 'SIM', 'NAO')

sns.distplot(aux.SalePrice[aux.miss == "NAO"], label = 'NAO' )
sns.distplot(aux.SalePrice[aux.miss == "SIM"], label = 'SIM' )
plt.legend(prop={'size': 12})
plt.title('SalePrice by miss or not LotFrontage')
corr = df.phik_matrix()
corrLF = corr['LotFrontage'].sort_values(ascending=False)
corrLF
# Fill with Linear Regression
# I will use as independent variable TotalBsmtSF, Neighborhood, PoolArea and MSSubClass.
df_LF = df[['TotalBsmtSF', 'Neighborhood', 'PoolArea', 'MSSubClass', 'LotFrontage']]
df_LF = pd.get_dummies(df_LF, columns=['Neighborhood'])
cond = (df_LF['LotFrontage'] > 250)
df_LF = df_LF[~cond]

df_LF_target = df_LF[['LotFrontage']][~df_LF['LotFrontage'].isnull()]

df_LF_test = df_LF[df_LF['LotFrontage'].isnull()]
df_LF_test = df_LF_test.drop(['LotFrontage'], axis=1)

df_LF_train = df_LF[~df_LF['LotFrontage'].isnull()]
df_LF_train = df_LF_train.drop(['LotFrontage'], axis=1)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(df_LF_train,df_LF_target)
print('Score:', reg.score(df_LF_train, df_LF_target))

plt.scatter(df_LF_target, reg.predict(df_LF_train))
pred = pd.DataFrame(data = reg.predict(df_LF_test), columns = ['LotFrontage1'])
pred.index = df_LF_test.index
df = pd.concat([df,pred], axis = 1)
df.LotFrontage = df.LotFrontage.fillna(df.LotFrontage1)
df = df.drop(['LotFrontage1'], axis = 1)
df_num = df.select_dtypes(exclude=['object']).copy()
df_num = df_num.drop(['SalePrice'], axis =1)
fig = plt.figure(figsize=(20,18))
for i in range(len(df_num.columns)):
    fig.add_subplot(5, 4, i+1)
    sns.scatterplot(df_num.iloc[:, i],df.SalePrice)
plt.tight_layout()
# Looking at the charts above, I will remove the outliers
# LotFrontage > 250
df = df[df.LotFrontage < 250]
# LotArea > 100000
df = df[df.LotArea < 100000]
# MasVnrArea > 1300
df = df[df.MasVnrArea < 1300]
# TotalBsmtSF > 4000
df = df[df.TotalBsmtSF < 4000]
# WoodDeckSF > 800
df = df[df.WoodDeckSF < 800]
# OpenPorchSF > 480
df = df[df.OpenPorchSF < 480]
# reading test data
path = os.path.join(dirname,'test.csv')
test = pd.read_csv(path, delimiter = ",", index_col = 'Id')

# keep the same columns
col = df.columns.drop(['SalePrice'])
test = test[col]
test.shape
# Missing data
missing = test.isnull().sum()
        
missing = missing[missing>0]
missing.sort_values(inplace=True)
missing.plot.bar()
missing
# We will fill in the same way as in the training data
col = ['MasVnrType', 'BsmtCond', 'BsmtFinType1', 'BsmtExposure', 'GarageType', 'GarageFinish', 'GarageQual', 'FireplaceQu']
test[col] = test[col].fillna('NONE')
test['MasVnrArea'] = test['MasVnrArea'].fillna(0)
# Fill TotalBsmtSF with the median - TotalBsmtSF, BsmtUnfSF, BsmtFullBath and GarageCars
col = ['TotalBsmtSF', 'BsmtUnfSF', 'BsmtFullBath', 'GarageCars']
for i in col:
    test[i] = test[i].fillna(test[i].median())
# Fill with more frequency value - Exterior2nd and SaleType
col = ['Exterior2nd', 'SaleType']
for i in col:
    test[i] = test[i].fillna(test[i].value_counts().idxmax())
# LotFrontage
df_LF = test[['TotalBsmtSF', 'Neighborhood', 'PoolArea', 'MSSubClass', 'LotFrontage']]
df_LF = df_LF[~df_LF.isnull()]
df_LF = pd.get_dummies(df_LF, columns=['Neighborhood'])
print(df_LF.shape)

df_LF_target = df_LF[['LotFrontage']][~df_LF['LotFrontage'].isnull()]
print(df_LF_target.shape)

df_LF_test = df_LF[df_LF['LotFrontage'].isnull()]
df_LF_test = df_LF_test.drop(['LotFrontage'], axis=1)
print(df_LF_test.shape)

df_LF_train = df_LF[~df_LF['LotFrontage'].isnull()]
df_LF_train = df_LF_train.drop(['LotFrontage'], axis=1)
print(df_LF_train.shape)

reg = LinearRegression()
reg.fit(df_LF_train,df_LF_target)
print('Score:', reg.score(df_LF_train, df_LF_target))

plt.scatter(df_LF_target, reg.predict(df_LF_train))

pred = pd.DataFrame(data = reg.predict(df_LF_test), columns = ['LotFrontage1'])
pred.index = df_LF_test.index
test = pd.concat([test,pred], axis = 1)
test.LotFrontage = test.LotFrontage.fillna(test.LotFrontage1)
test = test.drop(['LotFrontage1'], axis = 1)
test.LotFrontage.isnull().sum()
train = df.drop(['SalePrice'], axis = 1)
target = df['SalePrice']
train['type'] = 'train' 
test['type'] = 'test'
fulldata = train.append(test)
train.shape, test.shape, target.shape, fulldata.shape
s = fulldata.dtypes == "object"
cat = s[s].index
cat = cat.drop("type")

OHE = OneHotEncoder(handle_unknown='ignore', sparse=False)
OHE.fit(fulldata[cat])
data_OHE = OHE.transform(fulldata[cat])
data_OHE = pd.DataFrame(data = data_OHE)
data_OHE.index = fulldata.index
fulldata = pd.concat([fulldata,data_OHE], axis = 1)
fulldata = fulldata.drop(cat, axis = 1)

train = fulldata[fulldata['type'] == "train"]
test = fulldata[fulldata['type'] == "test"]

train = train.drop(['type'], axis = 1)
test = test.drop(['type'], axis = 1)
print(train.shape, target.shape, test.shape)
clf = GradientBoostingRegressor(n_estimators = 100, learning_rate=0.1, min_samples_leaf=8, subsample=0.8)
clf.fit(train,target)

feature_importances = pd.DataFrame(clf.feature_importances_,index = train.columns, columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances[feature_importances.importance == 0.0])
index = feature_importances[feature_importances.importance == 0.0].index
train = train.drop(index, axis = 1)
test = test.drop(index, axis = 1)
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [200, 500],
              'min_samples_split': [ 20, 30, 50],
              'max_depth': [30, 50 ],
              'max_features': [0.5]
             }
clf = GradientBoostingRegressor(learning_rate=0.1, subsample=0.8)
GS = GridSearchCV(estimator = clf, param_grid = param_grid, cv=6)
GS.fit(train, target)
print('Best estimator:', GS.best_estimator_)
print('Best param:', GS.best_params_)
print('Score:', GS.best_score_)
clf = GradientBoostingRegressor(learning_rate=0.1, 
                                 subsample=0.8, 
                                 n_estimators = 200,
                                 min_samples_split =  20,
                                 max_depth = 30,
                                 max_features = 0.5)
clf.fit(train, target)
saleprice = clf.predict(test)

pred = pd.DataFrame(np.around(saleprice), columns = ['SalePrice'], index = test.index).astype('int64')
# Salva o arquivo no formato csv
pred.to_csv('csv_to_submit.csv', index = False)