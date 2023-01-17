import warnings

warnings.filterwarnings('ignore')
import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats



%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(3)
print('Number of rows in training set: ',train.shape[0])

print('Number of columns in training set: ', train.shape[1])
test.head(3)
print('Number of rows in test dataset: ', test.shape[0])

print('Number of columns in test dataset: ', test.shape[1])
df = pd.concat([train.drop('SalePrice', axis = 1),test], axis = 0)
df.head(3)
print('Number of rows in dataset: ', df.shape[0])

print('Number of columns in dataset: ', df.shape[1])
df.drop('Id', axis = 1).describe().T                   #T = transpose of the dataset
print('No. of categorical attributes: ', df.select_dtypes(exclude = ['int64','float64']).columns.size)
print('No. of numerical attributes: ', df.select_dtypes(exclude = ['object']).columns.size)
plt.figure(figsize=(20,6))

sns.heatmap(df.select_dtypes(exclude=['object']).isnull(), yticklabels=False, cbar = False, cmap = 'viridis')

plt.title('Null Values present in Numerical Attributes',fontsize=18)

plt.show()



plt.figure(figsize=(20,6))

sns.heatmap(df.select_dtypes(exclude=['int64','float64']).isnull(), yticklabels=False, cbar = False, cmap = 'viridis')

plt.title('Null Values present in Categorical Attributes',fontsize=18)

plt.show()
null_val = df.isnull().sum()/len(df)*100

null_val.sort_values(ascending = False, inplace = True)

null_val = pd.DataFrame(null_val, columns = ['missing %'])

null_val = null_val[null_val['missing %'] > 0]



sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

sns.barplot(x = null_val.index, y = null_val['missing %'], palette='Set1')

plt.xticks(rotation = 90)

plt.show()
sns.set_style('whitegrid')

df.hist(bins = 30, figsize = (20,15), color = 'darkgreen')

plt.show()

plt.tight_layout()
plt.figure(figsize=(30,20))

sns.heatmap(df.corr(), annot = True,cmap='GnBu')

plt.title('Heatmap of all Features',fontsize=18)

plt.show()
sns.set_style('whitegrid')

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols])

plt.show()
plt.figure(figsize=(10,6))

sns.scatterplot(x='1stFlrSF',y='SalePrice', data = train,color = 'orange')

plt.title('SalePrice vs. 1stFlrSF')

plt.show()
plt.figure(figsize=(10,6))

sns.scatterplot(x='GrLivArea',y='SalePrice', data = train,color = 'limegreen')

plt.title('SalePrice vs. OverallQual')

plt.show()
plt.figure(figsize=(10,6))

sns.scatterplot(x='TotalBsmtSF',y='SalePrice', data = train,color = 'royalblue')

plt.title('SalePrice vs. TotalBsmtSF')

plt.show()
plt.figure(figsize=(10,6))

sns.scatterplot(x='GarageArea',y='SalePrice', data = train,color = 'royalblue')

plt.title('SalePrice vs. GarageArea')

plt.show()
sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

sns.boxplot(x='OverallQual', y='SalePrice', data = train,palette='magma')

plt.show()
plt.figure(figsize=(5,6))

sns.boxplot(x='Street', y='SalePrice', data = train,palette='magma')

plt.title('SalePrice vs. Street')

plt.show()
plt.figure(figsize=(20,12))

sns.boxplot(x='YearBuilt', y='SalePrice', data = train)

plt.xticks(rotation = 90)

plt.title('SalePrice vs. YearBuilt', fontsize=15)

plt.show()
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

df['LotFrontage'] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
#GarageType, GarageFinish, GarageQual and GarageCond these are replacing with None

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    df[col] = df[col].fillna('None')
#GarageYrBlt, GarageArea and GarageCars these are replacing with zero

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:

    df[col] = df[col].fillna(int(0))
#BsmtFinType2, BsmtExposure, BsmtFinType1, BsmtCond, BsmtQual these are replacing with None

for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):

    df[col] = df[col].fillna('None')
#MasVnrArea : replace with zero

df['MasVnrArea'] = df['MasVnrArea'].fillna(int(0))
#MasVnrType : replace with None

df['MasVnrType'] = df['MasVnrType'].fillna('None')
#There is put mode value 

df['Electrical'] = df['Electrical'].fillna(df['Electrical']).mode()[0]
#There is no need of Utilities

df = df.drop(['Utilities'], axis=1)
df['PoolQC'] = df['PoolQC'].fillna('None')
df['MiscFeature'].fillna('None', inplace = True)
df['Alley'].fillna('None', inplace = True)
df['Fence'].fillna('None', inplace = True)
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
df['KitchenQual'].fillna(df['KitchenQual'].mode()[0], inplace = True)

df['BsmtFullBath'].fillna(0, inplace = True)
df['FullBath'].fillna(df['FullBath'].mode()[0],inplace = True)
for col in ['SaleType','KitchenQual','Exterior2nd','Exterior1st','Electrical']:

    df[col].fillna(df[col].mode()[0],inplace=True)
df['MSZoning'].fillna(df['MSZoning'].mode()[0],inplace=True)
df['Functional'].fillna(df['Functional'].mode()[0],inplace=True)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    df[col].fillna(0,inplace=True)
#Checking there is any null value or not

plt.figure(figsize=(15, 4))

sns.heatmap(df.isnull(),yticklabels=False)

plt.show()
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',

        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature', 

        'SaleType', 'SaleCondition', 'Electrical', 'Heating')
from sklearn.preprocessing import LabelEncoder

for c in cols:

    lbl = LabelEncoder()

    lbl.fit(list(df[c].values))

    df[c] = lbl.transform(list(df[c].values))
train_data = df.iloc[:1460,:]

test_data = df.iloc[1460:,:]
train_data.shape
test_data.shape
X = train_data

y = train['SalePrice']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.21, random_state = 7)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X_train.drop('Id',axis = 1), y_train)
lin_reg.score(X_test.drop('Id',axis = 1),y_test)
prediction = lin_reg.predict(test_data.drop('Id',axis = 1))
#Train the model

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=1000)
model.fit(X_train.drop('Id',axis = 1), y_train)
model.score(X_test.drop('Id',axis = 1),y_test)
from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)
GBR.fit(X_train.drop('Id',axis = 1), y_train)
GBR.score(X_test.drop('Id',axis = 1),y_test)
GBR.fit(X.drop('Id',axis = 1),y)
predictions = GBR.predict(test_data.drop('Id',axis = 1))
submission = pd.DataFrame({'Id':test_data['Id'],'SalePrice':predictions})
submission.to_csv('housepricesub.csv',index=False)