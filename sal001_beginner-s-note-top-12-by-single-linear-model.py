import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train_size = len(df)
features = pd.concat([df.drop('SalePrice',axis=1),df_test],sort=False).reset_index(drop=True)

target = df['SalePrice'].copy()
df.head()
print('number of numeric columns: ' + str(len(df.select_dtypes(['int64','float64']).columns)))

print('number of categorical columns: ' + str(len(df.select_dtypes(['object']).columns)))
#correlation to SalePrice

plt.figure(figsize=(12,6))

df.corr()['SalePrice'].drop('SalePrice').sort_values().plot(kind='bar')
sns.jointplot(x='OverallQual', y='SalePrice', data=df)



sns.jointplot(x='GrLivArea', y='SalePrice', data=df)
#23 catergorical columns with missing values

missing_str = features.select_dtypes('object').isnull().sum().sort_values(ascending=False)

missing_str[missing_str>0]
fill_none = ['MiscFeature','Alley','Fence','FireplaceQu','GarageType','GarageCond', 'GarageQual','GarageFinish','BsmtCond','BsmtFinType1', 'BsmtFinType2', 'BsmtExposure', 'BsmtQual','MasVnrType']

features[fill_none] = features[fill_none].fillna('None',inplace=False)
sns.scatterplot(x='PoolArea',y='SalePrice',data=df.fillna('Missing'),hue='PoolQC')
features['PoolQC'].fillna('N',inplace=True)

features['PoolQC'].replace({'Ex':'Y','Fa':'Y','Gd':'Y'},inplace=True)

features.drop('PoolArea',axis=1,inplace=True)
otr_cate = ['MSZoning', 'Utilities', 'Functional', 'Electrical', 'SaleType', 'Exterior2nd', 'Exterior1st', 'KitchenQual']

for c in otr_cate:

    features[c] = features[c].fillna(features[c].mode()[0],inplace=False)
#11 catergorical columns with missing values

missing_num = features.select_dtypes(['int64','float64']).isnull().sum().sort_values(ascending=False)

missing_num[missing_num>0]
df.plot(x='LotFrontage',y='LotArea',kind='scatter')
features['LotFrontage'].fillna(features['LotFrontage'].median(),inplace=True)
features[features['GarageCond']=='None']['GarageYrBlt'].isnull().sum()
features['GarageYrBlt'].fillna(0,inplace=True)
fill_zero = ['GarageArea','GarageCars','MasVnrArea','BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']

features[fill_zero] = features[fill_zero].fillna(0)
# focusing on first row, and found some point does not follow the trend

sns.pairplot(df[['SalePrice','GrLivArea','1stFlrSF','TotalBsmtSF','LotFrontage','LotArea']].dropna())
drop_index = [] # list to contain all rows should be excluded

drop_index.append(df[(df['GrLivArea']>4500) & (df['SalePrice']<400000)].loc[:train_size].index)

drop_index.append(df[(df['1stFlrSF']>4000) & (df['SalePrice']<400000)].loc[:train_size].index)

drop_index.append(df[(df['TotalBsmtSF']>6000) & (df['SalePrice']<400000)].loc[:train_size].index)

drop_index.append(df[df['LotArea']>100000].index)

drop_index.append(df[df['LotFrontage']>300].index)



drop_index = list(map(list,drop_index)) # IndexObject -> python list

tmp = []

for sublist in drop_index:

    for item in sublist:

        tmp.append(item)

drop_index = list(set(tmp))  # merge into single list and take set() to remove the duplicated



features.drop(drop_index,inplace=True)

target.drop(drop_index,inplace=True)
features['MSSubClass'] = features['MSSubClass'].astype('object')
features['Utilities'].value_counts()
features.drop('Utilities',axis=1,inplace=True)
features.drop('Id',axis=1,inplace=True)
df.plot(x='LowQualFinSF',y='SalePrice',kind='scatter')
features['LowQualFinSF'] = features['LowQualFinSF'].apply(lambda x: 'Y' if x>0 else 'N')
#before take square root

sns.jointplot(x=df['GrLivArea'],y=df['SalePrice'].apply(np.log1p),data=df,kind='reg')
#after take square root

sns.jointplot(x=df['GrLivArea']**0.5,y=df['SalePrice'].apply(np.log1p),data=df,kind='reg')
features['GrLivArea'] = features['GrLivArea']**0.5
#predictor pairwise correlation check

corr_matrix = features.corr()

colinearity = {}

for column in corr_matrix.columns:

    index = corr_matrix[corr_matrix[column]>0.6].index

    for indice in index:

        if not column == indice:

            if not indice+' '+column in colinearity.keys():

                colinearity[column+' '+indice]=corr_matrix.loc[indice,column]

colinearity
high_collinerarity = ['GarageArea','TotRmsAbvGrd','1stFlrSF','2ndFlrSF','BsmtFinSF1','BsmtFinSF2','FullBath']

features = features.drop(high_collinerarity,axis=1)
categorical_cols = list(features.select_dtypes('object').columns)

dummies = pd.get_dummies(features[categorical_cols],drop_first=True)

features = pd.concat([features.drop(categorical_cols,axis=1),dummies],axis=1)
X_train = features.loc[0:train_size-1]

y_train = target

X_test = features.loc[train_size:]
X_train.shape
from sklearn.linear_model import RidgeCV

reg = RidgeCV()

reg.fit(X_train,np.log1p(y_train))

#reg.fit(X_train,y_train)
from sklearn.model_selection import cross_val_score

log_rms = np.sqrt(-np.mean(cross_val_score(reg, X_train,np.log1p(y_train), cv=5,scoring='neg_mean_squared_error')))

#log_rms = np.sqrt(-np.mean(cross_val_score(reg, X_train,y_train, cv=5,scoring='neg_mean_squared_log_error')))

print(f'RMLS : {log_rms}')
pred = np.expm1(reg.predict(X_train))

#pred = reg.predict(X_train)
comparison = pd.DataFrame({'prediction':pred.reshape(pred.shape[0],),'actual':y_train,'error':pred.reshape(pred.shape[0],)-y_train})

sns.distplot(comparison['error'])
comparison.plot(x='prediction',y='error',kind='scatter')
comparison.plot(x='prediction',y='actual',kind='scatter')
# output result for submission

pred = np.expm1(reg.predict(X_test))

pred = pd.DataFrame(pred.reshape(1459, ))

output = pd.concat([df_test['Id'],pred],axis=1).rename(columns={0:'SalePrice'})

output.to_csv('submission.csv',index=False)
steps = ['Control','Outliers','Log_y','Modification','Non-linearity','Collinearity']

rmls = [0.15494899171649665,0.14154336306738766,0.11543537404168344,0.11479554199015254,0.11394995795225088,0.11326714995779572]

scores = pd.DataFrame({'steps':steps,'rmls':rmls})



scores.plot(x='steps',y='rmls',marker='o',rot=90)

plt.xlabel('Preprocessing Actions',fontsize=14)

plt.ylabel('RMLS',fontsize=14)

plt.title('RMLS Changing wrt to Preprocessing Actions',fontsize=16)