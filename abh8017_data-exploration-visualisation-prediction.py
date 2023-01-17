# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test =pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train.describe(include='all')
train.info()
test.head()
test.describe(include='all')
test.info()
test.info()
import matplotlib.pyplot as plt

%matplotlib inline

train.hist(figsize=(20,30))
import seaborn as sns

sns.boxplot(x = "YrSold", y= "SalePrice",data=train)
sns.distplot(train['SalePrice'],norm_hist=True);

#sns.pairplot(data)
import seaborn as sns

sns.boxplot(x = "MSZoning", y= "SalePrice",data=train)
sns.distplot(train['SalePrice'])

sns.distplot(train['YearBuilt'])
sns.distplot(train['LotArea'],color="blue", label="LotArea")

corr =train.corr()
corr
corr[corr['SalePrice']>0.3].index
train = train[['LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',

       'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',

       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',

       'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'SalePrice']]

test=test[['LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',

       'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',

       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',

       'GarageArea', 'WoodDeckSF', 'OpenPorchSF']]
print(train.info())

print("#"*100)

print(test.info())
threshold = 0.7

#Dropping columns with missing value rate higher than threshold

train = train[train.columns[train.isnull().mean() < threshold]]

#Dropping rows with missing value rate higher than threshold

train = train.loc[train.isnull().mean(axis=1) < threshold]



#Filling missing values with medians of the columns

train = train.fillna(train.median())
threshold = 0.7

#Dropping columns with missing value rate higher than threshold

test = test[test.columns[test.isnull().mean() < threshold]]

#Dropping rows with missing value rate higher than threshold

test = test.loc[test.isnull().mean(axis=1) < threshold]



#Filling missing values with medians of the columns

test = test.fillna(test.median())
train.info()
test.describe()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('SalePrice', axis=1), train['SalePrice'], test_size=0.3, random_state=101)
y_train= y_train.values.reshape(-1,1)

y_test= y_test.values.reshape(-1,1)



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)

y_train = sc_X.fit_transform(y_train)

y_test = sc_y.fit_transform(y_test)
print(X_train)
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators = 100, random_state = 0)

rfr.fit(X_train, y_train)
rfr_pred= rfr.predict(test)

rfr_pred = rfr_pred.reshape(-1,1)
rfr_pred.shape
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error
print('MAE:', mean_absolute_error(y_test, rfr_pred))

print('MSE:', mean_squared_error(y_test, rfr_pred))

print('RMSE:', np.sqrt(mean_squared_error(y_test, rfr_pred)))
plt.figure(figsize=(16,8))

plt.plot(y_test,label ='Test')

plt.plot(rfr_pred, label = 'predict')

plt.show()
a = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
a
test_id = a['Id']

a = pd.DataFrame(test_id, columns=['Id'])
test = sc_X.fit_transform(test)
test.shape
a
test.shape
X_test.shape
rfr_pred= rfr.predict(test)

rfr_pred = rfr_pred.reshape(-1,1)
rfr_pred.shape
test_prediction_rfr =sc_y.inverse_transform(rfr_pred)
test_prediction_rfr = pd.DataFrame(test_prediction_rfr, columns=['SalePrice'])
test_prediction_rfr
result = pd.concat([a,test_prediction_rfr], axis=1)
result
result.to_csv('submission.csv',index=False)