
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


import os
print(os.listdir("../input"))


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.groupby('Neighborhood').Id.count().sort_values().plot(kind='barh', figsize=(8,8), color = '#e0bb28',edgecolor = 'k', alpha = 0.7)
plt.title('Neighborhoods popularity', fontsize = 20, ha = 'left', x=0)
plt.yticks(fontsize = 15)
plt.xticks(fontsize = 15)
num_features = train.select_dtypes(include=['int64','float64']).columns.drop(['Id','SalePrice'])

cat_features = train.select_dtypes(include=['object']).columns
num_features
f = pd.melt(train, value_vars=num_features)
g = sns.FacetGrid(f, col='variable', col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.distplot, 'value');
train.isnull().sum()[train.isnull().sum()!=0].sort_values(ascending = False)
train[cat_features].head()
train['Alley'].replace({'Grvl':1, 'Pave':2}, inplace = True)
train['Street'].replace({'Grvl':1, 'Pave':2}, inplace = True)
test['Alley'].replace({'Grvl':1, 'Pave':2}, inplace = True)
test['Street'].replace({'Grvl':1, 'Pave':2}, inplace = True)
cat_dict = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
for i in [18, 19, 21, 22, 27, 30, 32, 35, 36, 38]: 
    train[cat_features[i]].replace(cat_dict,inplace = True)
    test[cat_features[i]].replace(cat_dict,inplace = True)
train['MasVnrType'].replace({'BrkCmn':1,'None':1, 'Stone':3,'BrkFace': 2}, inplace = True)
test['MasVnrType'].replace({'BrkCmn':1,'None':1, 'Stone':3,'BrkFace': 2}, inplace = True)

train['LotConfig'].replace({'Inside':1,'FR2':1,'Corner':1,'CulDSac':2,'FR3':2}, inplace = True)
test['LotConfig'].replace({'Inside':1,'FR2':1,'Corner':1,'CulDSac':2,'FR3':2}, inplace = True)

train['LandSlope'] = train['LandSlope'].replace({'Sev':1, 'Mod':1, 'Gtl':2})
test['LandSlope'] = test['LandSlope'].replace({'Sev':1, 'Mod':1, 'Gtl':2})

train['LotShape'] = train['LotShape'].replace({'Reg':2,'IR3':1,'IR2':1,'IR1':1})
test['LotShape'] = test['LotShape'].replace({'Reg':2,'IR3':1,'IR2':1,'IR1':1})

train['MSZoning'].replace({'FV':1,'RL':2,'RH':3,'RM':4,'C (all)':5,'A (agr)':6,'I (all)':7}, inplace = True)
test['MSZoning'].replace({'FV':1,'RL':2,'RH':3,'RM':4,'C (all)':5,'A (agr)':6,'I (all)':7}, inplace = True)

train['Utilities'].replace({'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4}, inplace=True)
test['Utilities'].replace({'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4}, inplace=True)

train['BsmtExposure'].replace({'No':1, 'Mn':2, 'Av':3, 'Gd':4}, inplace=True)
test['BsmtExposure'].replace({'No':1, 'Mn':2, 'Av':3, 'Gd':4}, inplace=True)

train['BsmtFinType1'].replace({'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}, inplace=True)
test['BsmtFinType1'].replace({'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}, inplace=True)

train['LandContour'].replace({'Low':1, 'HLS':2, 'Bnk':3, 'Lvl':4}, inplace=True)
test['LandContour'].replace({'Low':1, 'HLS':2, 'Bnk':3, 'Lvl':4}, inplace=True)

train['BsmtFinType2'].replace({'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}, inplace=True)
test['BsmtFinType2'].replace({'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}, inplace=True)

train['Functional'].replace({'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8}, inplace=True)
test['Functional'].replace({'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8}, inplace=True)

train['GarageFinish'].replace({'Unf':1, 'RFn':2, 'Fin':3}, inplace=True)
test['GarageFinish'].replace({'Unf':1, 'RFn':2, 'Fin':3}, inplace=True)

train['PavedDrive'].replace({'N':1, 'P':2, 'Y':3}, inplace=True)
test['PavedDrive'].replace({'N':1, 'P':2, 'Y':3}, inplace=True)

train['CentralAir'].replace({'Y':1, 'N':0}, inplace = True)
test['CentralAir'].replace({'Y':1, 'N':0}, inplace = True)
corr = train[num_features.append(pd.Index(['SalePrice']))].corr()
fig = plt.figure(figsize=(16,15))
ax = fig.add_subplot(111)

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.index.values, cmap="PiYG");
sns.set(rc={'figure.figsize':(15,15)})
sns.heatmap(train.isnull(),xticklabels=False,yticklabels=False,cmap='viridis')
train.isnull().sum()[train.isnull().sum()!=0].sort_values(ascending=False)
train.columns
col_with_null = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageYrBlt', 'GarageType', 'GarageFinish', 'GarageQual',
'GarageCond', 'BsmtFinType2','BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'MasVnrArea', 'MasVnrType', 'Electrical']

train[col_with_null]

corr = train[['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageYrBlt', 'GarageType', 'GarageFinish', 'GarageQual',
'GarageCond', 'BsmtFinType2','BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'MasVnrArea', 'MasVnrType', 'Electrical']+['SalePrice']].corr()
corr['SalePrice'].sort_values(ascending=False)

train.drop(['PoolQC','MiscFeature','Fence'],axis=1,inplace=True)
test.drop(['PoolQC','MiscFeature','Fence'],axis=1,inplace=True)
train.isnull().sum()[train.isnull().sum()!=0].sort_values(ascending=False)
train.fillna(train.mean(), inplace=True)
test.fillna(train.mean(), inplace=True)
train.isnull().sum()[train.isnull().sum()!=0].sort_values(ascending=False)
train['GarageType'].fillna('No', inplace=True)
train['Electrical'].fillna('No', inplace=True)
test['GarageType'].fillna('No', inplace=True)
test['Electrical'].fillna('No', inplace=True)
train.isnull().sum()[train.isnull().sum() != 0]
test.isnull().sum()[test.isnull().sum() != 0]
test[['Exterior1st','Exterior2nd','SaleType']].head()
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
test.fillna('No', inplace=True)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.columns
X = train.drop('SalePrice',axis=1)
y = train['SalePrice']
print(len(X.columns))
print(len(test.columns))
categ = X.select_dtypes(include=[object])
print(len(categ.columns))
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
onh = onehotencoder.fit(categ)

train_dum = pd.get_dummies(X,drop_first=True)
test_dum = pd.get_dummies(test, drop_first=True)
train_dum
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_dum, y, test_size=0.3, random_state=42)
train,test_eval = X_train.align(X_test, join='outer', axis=1, fill_value=0)
train,test = train.align(test_dum, join='outer', axis=1, fill_value=0)
test_eval,test = test_eval.align(test, join='outer', axis=1, fill_value=0)
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
eval_set = [(test_eval, y_test)]
model = XGBRegressor(n_estimators=1000)
model.fit(train, y_train, early_stopping_rounds=5, eval_metric="error", eval_set = eval_set, verbose=False)
predictions = model.predict(test)
len(train.columns)
len(test_eval.columns)
len(test.columns)
predictions_df = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})
# you could use any filename. We choose submission here
predictions_df.to_csv('sample_submission.csv', index=False)
