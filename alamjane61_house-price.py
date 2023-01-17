import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 





import os

print(os.listdir("../input"))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(train.head())

print('**'* 50)

print(test.head())
print(train.info())

print('**'* 50)

print(test.info())
n_train = train.shape[0]

n_test = test.shape[0]

y = train['SalePrice'].values

data = pd.concat((train, test)).reset_index(drop=True)

data.drop(['SalePrice'], axis=1, inplace=True)
data.head()
sns.lmplot(x='YearBuilt',y='SalePrice',data=train)
plt.figure(figsize=(16,8))

sns.boxplot(x='GarageCars',y='SalePrice',data=train)

plt.show()
plt.figure(figsize=(16,8))

sns.barplot(x='GarageArea',y = 'SalePrice',data=train, estimator=np.mean)

plt.show()
plt.figure(figsize=(16,8))

sns.barplot(x='FullBath',y = 'SalePrice',data=train)

plt.show()
sns.lmplot(x='1stFlrSF',y='SalePrice',data=train)
data = data[['LotArea','Street', 'Neighborhood','Condition1', 'Condition2','BldgType','HouseStyle','OverallCond', 'Heating','CentralAir','Electrical','1stFlrSF','2ndFlrSF','BsmtHalfBath','FullBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','GarageCars','GarageArea','PoolArea']]
data.info()
data['BsmtHalfBath'] = data['BsmtHalfBath'].fillna(data['BsmtHalfBath'].mean())

data['Electrical'] = data['Electrical'].fillna('SBrkr')

data['GarageCars'] = data['GarageCars'].fillna(data['GarageCars'].mean())

data['GarageArea'] = data['GarageArea'].fillna(data['GarageArea'].mean())
# Categorical boolean mask

categorical_feature_mask = data.dtypes==object

# filter categorical columns using mask and turn it into alist

categorical_cols = data.columns[categorical_feature_mask].tolist()
data = pd.get_dummies(data, columns=categorical_cols)
data.info()
data.shape
train =data[:n_train]

test = data[n_train:]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=101)


y_train= y_train.reshape(-1,1)

y_test= y_test.reshape(-1,1)



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)

y_train = sc_X.fit_transform(y_train)

y_test = sc_y.fit_transform(y_test)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)

print(lm)
print(lm.intercept_)
print(lm.coef_)
predictions = lm.predict(X_test)

predictions= predictions.reshape(-1,1)
plt.figure(figsize=(15,8))

plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()