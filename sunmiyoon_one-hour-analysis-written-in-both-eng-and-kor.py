import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(2)
# train.dtypes
train.describe()
import matplotlib.pyplot as plt
import seaborn as sns
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(train.corr(), vmax=.8, square=True);
def strongerRelationSalePrice(f1, f2):
    f1Corr = train.corr().loc[f1,'SalePrice']
    f2Corr = train.corr().loc[f2,'SalePrice']
#     print(f1Corr, f2Corr)
    return (f1, f2) if (f1Corr >= f2Corr) else (f2, f1) 
print('{} has stronger relation with SalePrice than {}'.format(strongerRelationSalePrice('YearBuilt', 'GarageYrBlt')[0], strongerRelationSalePrice('YearBuilt', 'GarageYrBlt')[1]))
print('{} has stronger relation with SalePrice than {}'.format(strongerRelationSalePrice('GrLivArea', 'TotRmsAbvGrd')[0], strongerRelationSalePrice('GrLivArea', 'TotRmsAbvGrd')[1]))
print('{} has stronger relation with SalePrice than {}'.format(strongerRelationSalePrice('GarageCars', 'GarageArea')[0], strongerRelationSalePrice('GarageCars', 'GarageArea')[1]))
# train = train.drop(['GarageYrBlt', 'TotRmsAbvGrd', 'GarageArea'], axis=1)
# test = test.drop(['GarageYrBlt', 'TotRmsAbvGrd', 'GarageArea'], axis=1)
%matplotlib inline
import matplotlib.pyplot as plt

# I think this graph is more elegant than pandas.hist()
# train['SalePrice'].hist(bins=100)
sns.distplot(train['SalePrice'])
import matplotlib.pyplot as plt

fig, axes = plt.subplots(14, 6, figsize=(15, 45), sharey=True)
for col, a in zip(train.columns, axes.flatten()):
    if col == 'SalePrice':   
        a.set_title(col)
        a.scatter(df['SalePrice'], df['SalePrice'])
    else:
        df = train[['SalePrice', col]].dropna()
        a.set_title(col)
        a.scatter(df[col], df['SalePrice'])
df = pd.concat([train, test]).drop('SalePrice', axis=1)

yTrain = train.SalePrice.values
nTrain = len(train)
testId = test['Id']

print('Training data size is {}'.format(train.shape))
print('Test data size is {}'.format(test.shape))
print('Total data size is: {}'.format(df.shape))
dfNullPct = df.isnull().sum() * 100 / len(df)
missings = pd.DataFrame(dfNullPct.drop(dfNullPct[dfNullPct == 0].index).sort_values(ascending=False), columns=['Missing Pct'])
missings.head(5)
# features which contains null value
missings.index
fillNanNone = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', \
               'GarageQual', 'GarageFinish', 'GarageCond', 'GarageType', \
               'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', \
               'MasVnrType', 'GarageYrBlt', 'MSSubClass', 'Functional']

fillNanZero = ['GarageArea', 'GarageCars', \
              'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', \
              'MasVnrArea']

for col in list(missings.index):
    if col == 'Functional':
        df[col] = df[col].fillna('Typ')
    elif col in fillNanNone:
        df[col] = df[col].fillna('None')
    elif col in fillNanZero:
        df[col] = df[col].fillna(0)
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
df = df.drop(['Utilities'], axis=1)
for col in ['Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'MSZoning']:
    df[col] = df[col].fillna(df[col].mode()[0])
    print('Most frequently shown in feature \'{}\' is \'{}\''.format(col, df[col].mode()[0]))
dfNullPct = df.isnull().sum() * 100 / len(df)
pd.DataFrame(dfNullPct.drop(dfNullPct[dfNullPct == 0].index).sort_values(ascending=False), columns=['Missing Pct'])
# Year and month sold have to be transformed into categoriacal feature
# 연도, 달을 나타내는 변수는 반드시 카테고리 변수로 바꿔주어야 합니다. 어떤 데이터를 보더라도 주의해야 하는 사항입니다.
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', \
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', \
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', \
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir']

for col in cols:
    print('{}: {}'.format(col, df[col].unique()))
df = df.replace({'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, \
                 'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, \
                 'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, \
                 'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, \
                 'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, \
                 'ExterQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, \
                 'ExterCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, \
                 'HeatingQC': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, \
                 'PoolQC': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, \
                 'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, \
                 'BsmtFinType1': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}, \
                 'BsmtFinType2': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}, \
                 'Functional': {'Sel': 6, 'Sev': 5, 'Maj2': 4, 'Maj1': 3, 'Mod': 2, 'Min1': 1, 'Min2': 1, 'Typ': 0}, \
                 'BsmtExposure': {'Gd': 3, 'Av': 2, 'Mn': 1, 'No': 0, 'None': 0}, \
                 'Fence': {'GdPrv': 2, 'GdWo': 2, 'MnPrv': 1, 'MnWw': 1, 'None': 0}, \
                 'GarageFinish': {'Fin': 3, 'Unf': 2, 'RFn': 1, 'None': 0}, \
                 'LandSlope': {'Gtl': 2, 'Mod': 1, 'Sev': 0}, \
                 'LotShape': {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0}, \
                 'PavedDrive': {'Y': 2, 'P': 1, 'N': 0}, \
                 'Street': {'Pave': 1, 'Grvl': 0}, \
                 'Alley': {'Pave': 2, 'Grvl': 1, 'None': 0}, \
                 'CentralAir': {'Y': 1, 'N': 0}})
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df = pd.get_dummies(df)
print(df.shape)
train = df[:nTrain]
test = df[nTrain:]
import xgboost as xgb
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
kfold = KFold(5, shuffle=True, random_state=42)
robScaler = RobustScaler()
trainRobustScaled = robScaler.fit_transform(train)
testRobustScaled = robScaler.fit_transform(test)
ridg = Ridge(alpha=0.0001, random_state=None)
scores = np.sqrt(-cross_val_score(ridg, trainRobustScaled, yTrain, cv=kfold, scoring='neg_mean_squared_error'))
print('{:.4f}'.format(scores.mean()))
elastNet = ElasticNet()
scores = np.sqrt(-cross_val_score(elastNet, trainRobustScaled, yTrain, cv=kfold, scoring='neg_mean_squared_error'))
print('{:.4f}'.format(scores.mean()))
ridg.fit(trainRobustScaled, yTrain)
elastNet.fit(trainRobustScaled, yTrain)
ridgPredict = pd.DataFrame(ridg.predict(testRobustScaled))
elastNetPredict = pd.DataFrame(elastNet.predict(testRobustScaled))
sub = ridgPredict * 0.3 + elastNetPredict * 0.7
sub['Id'] = testId
sub.columns = ['SalePrice', 'Id']
sub.to_csv('./submission.csv', index=False)
