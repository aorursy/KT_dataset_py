import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



USAhousing = pd.concat([ train, test ])
USAhousing.head()
USAhousing.describe()
USAhousing.info()

# sns.heatmap(combined.isnull(),yticklabels=False,cbar=False,cmap='viridis')
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):

    print(USAhousing.isnull().sum())
plt.figure(figsize=(20, 20))



sns.heatmap(USAhousing.isnull(),

            square=False,

            annot=False,

            yticklabels=False,

            cbar=False,

            cmap='viridis'            

           )



plt.title('Features with missing values');
# sns.heatmap(USAhousing.corr())



corr=USAhousing.corr()



plt.figure(figsize=(20, 20))



sns.heatmap(corr, 

            vmax=.8, 

            linewidths=0.01,

            square=True,

            annot=True,

            cmap='YlGnBu',

            linecolor="white")



plt.title('Correlation between features');
# Clean Alley column

USAhousing.fillna(value={

    'Alley': 'NA', 

    'Fence': 'NA', 

    'FireplaceQu':'NA', 

    'MiscFeature':'NA',

    'PoolQC': 'NA',

    'BsmtCond': 'NA',

    'BsmtExposure': 'NA',

    'BsmtFinType1': 'NA',

    'BsmtFinType2': 'NA',

    'BsmtFinSF1': 0,

    'BsmtFinSF2': 0,

    'BsmtFullBath': 0,

    'BsmtHalfBath': 0,

    'BsmtUnfSF': 0,

    'TotalBsmtSF': 0,

    'BsmtQual': 'NA',

    'GarageCond': 'NA',

    'GarageFinish': 'NA',

    'GarageQual': 'NA',

    'GarageType': 'NA',

    'GarageYrBlt': USAhousing['YearBuilt'],

    'MasVnrArea': 0,

    'MasVnrType': 'None',

    'Electrical': 'SBrkr',

    'Functional': 'Typ',

    'GarageArea': 0,

    'GarageCars': 0,

    'MSZoning': 'RL',

    'Utilities': 'AllPub',

    'Exterior1st': 'VinylSd',

    'Exterior2nd': 'VinylSd',

    'KitchenQual': 'TA',

    'SaleType': 'WD'    

}, inplace=True)



# drop lotfrontage

USAhousing.drop('LotFrontage', axis=1, inplace=True)
objColumns = [col for col in list(USAhousing.columns) if USAhousing[col].dtypes == object]



dummies = pd.get_dummies(USAhousing, columns=objColumns, drop_first=True)



USAhousing = USAhousing.drop(objColumns, axis=1)



result = pd.concat([USAhousing, dummies], axis=1)



with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    print(result[:5])
train = USAhousing[USAhousing['SalePrice'].notnull()]



test = USAhousing[USAhousing['SalePrice'].isnull()]

test = test.drop('SalePrice', axis=1)
X = train.drop('SalePrice', axis=1)

y = train['SalePrice']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
sns.distplot((y_test - predictions))
from sklearn import metrics
print('MAE:  ', metrics.mean_absolute_error(y_test, predictions))

print( 'MSE:  ', metrics.mean_squared_error(y_test, predictions))

print( 'RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestRegressor(n_estimators=100)

rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)
plt.scatter(y_test, rfc_pred)

sns.distplot((y_test - rfc_pred))
print( 'MAE:  ', metrics.mean_absolute_error(y_test, rfc_pred))

print( 'MSE:  ', metrics.mean_squared_error(y_test, rfc_pred))

print( 'RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, rfc_pred)))
# Check for any remaining missing values

print("Remaining NaN?", np.any(np.isnan(test)) )

#np.all(np.isfinite(test))
#set ids as PassengerId and predict survival 

ids = test.Id

predictions = rfc.predict(test)



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'Id' : ids, 'SalePrice': predictions })

output.to_csv('submission.csv', index=False)