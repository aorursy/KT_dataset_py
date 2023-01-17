import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.info()
sns.distplot(train['SalePrice'])
plt.figure(figsize=(12,9))
sns.heatmap(train.corr(), cmap='coolwarm')

plt.scatter(train['GrLivArea'], train['SalePrice'])
sns.scatterplot(x="SalePrice", y="OverallQual" , data=train)
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#dealing with missing data
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max() #just checking that there's no missing data missing
train.head()
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);

data = train[['OverallQual','YearBuilt','FullBath','TotRmsAbvGrd',
             'GarageCars','GarageArea','PoolArea']]
y=train["SalePrice"]
X=data
X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm)
print(lm.intercept_)
print(lm.coef_)
predictions=lm.predict(X_test)


data_test = test[['OverallQual','YearBuilt','FullBath','TotRmsAbvGrd',
             'GarageCars','GarageArea','PoolArea']]
data_test.head()
data_test.isnull().sum()
data_test.info()

data_test.fillna(0, inplace=True)
data_test.isnull().sum()
scaler=StandardScaler()
X=scaler.fit_transform(data_test)

predictions1 = lm.predict(X)
output = pd.DataFrame({'Id':test['Id'],'SalePrice':predictions1})


output.to_csv('submission.csv', index=False)
