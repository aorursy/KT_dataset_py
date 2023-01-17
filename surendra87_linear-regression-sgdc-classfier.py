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
# Load require libraries 
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline
trainData = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
testData = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
trainData.head()
testData.head()
trainData.info()
# Check the missing values
trainData.isna().sum().sort_values(ascending=False)[:20]
testData.isna().sum().sort_values(ascending=False)[:35]
sns.countplot(x="SalePrice", hue='HouseStyle', data=trainData, palette="Set3")
#####  Remove the clumns where missing data is in hingh numbers
trainData = trainData.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1)
testData = testData.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1)
trainData.isna().sum().sort_values(ascending=False)[:20]
missing_data_feature = ['GarageType', 'GarageYrBlt','GarageFinish','GarageCond','GarageQual','BsmtExposure','BsmtFinType2','BsmtFinType1',
'BsmtCond', 'BsmtQual', 'MasVnrType', 'MasVnrArea', 'Electrical']
trainData[missing_data_feature].dtypes
from sklearn.impute import SimpleImputer
float_impute = SimpleImputer(missing_values=np.nan, strategy='mean')

trainData[['GarageYrBlt', 'MasVnrArea']] = float_impute.fit_transform(trainData[['GarageYrBlt', 'MasVnrArea']])
testData[['GarageYrBlt', 'MasVnrArea']] = float_impute.fit_transform(testData[['GarageYrBlt', 'MasVnrArea']])
string_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
string_var = ['GarageType', 'GarageFinish','GarageCond','GarageQual','BsmtExposure','BsmtFinType2','BsmtFinType1',
'BsmtCond', 'BsmtQual', 'MasVnrType', 'Electrical']
trainData[string_var] = string_imputer.fit_transform(trainData[string_var])
testData[string_var] = string_imputer.fit_transform(testData[string_var])
trainData.isnull().sum().sort_values(ascending=False)[:10]
testData.isnull().sum().sort_values(ascending=False)[:20]
missing_data_feature = ['MSZoning', 'BsmtFullBath','Utilities','BsmtHalfBath', 'Functional', 'TotalBsmtSF', 'GarageArea', 
'BsmtFinSF2', 'BsmtUnfSF', 'SaleType', 'Exterior2nd', 'Exterior1st', 'KitchenQual', 'GarageCars', 'BsmtFinSF1']
testData[missing_data_feature].dtypes
# Float Values
float_columns = ['BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF', 'GarageArea', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageCars', 'BsmtFinSF1']
testData[float_columns] = float_impute.fit_transform(testData[float_columns])

# String Values
string_var = ['MSZoning', 'Utilities', 'Functional', 'SaleType', 'Exterior2nd', 'Exterior1st', 'KitchenQual']
testData[string_var] = string_imputer.fit_transform(testData[string_var])
testData.isnull().sum().sort_values(ascending=False)[:10]
plt.figure(figsize=(23,12))
sns.swarmplot(trainData['Neighborhood'], trainData['SalePrice'], hue=trainData['MSSubClass'], palette='winter')
plt.title("Swarm Plot of Neighborhood VS SalePrice with variations based on MSSubClass")
plt.xlabel("Neighborhood")
plt.ylabel("SalePrice")
plt.show()
plt.figure(figsize=(15,8))
sns.regplot(trainData['LotArea'], trainData['SalePrice'])
plt.title("Regression plot or Living area VS SalesPrice")
plt.xlabel("Living Area")
plt.ylabel("SalePrice")
plt.show()
from sklearn.preprocessing import LabelEncoder

encode_columns = ['Street', 'LotShape', 'LandContour', 'Utilities', 'LandSlope', 'BldgType', 'HouseStyle', 'ExterQual', 
                  'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 
                  'CentralAir', 'KitchenQual', 'Functional', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive'
                  ]
#Train Data Set
train_le = {}
for col in encode_columns:
    train_le[col] = LabelEncoder()
    trainData[col] = train_le[col].fit_transform(trainData[col])
    
#Test Data Set 
test_le = {}
for col in encode_columns:
    test_le[col] = LabelEncoder()
    testData[col] = test_le[col].fit_transform(testData[col])
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
# Drop object columns
trainData = trainData.select_dtypes(exclude='object')
testData = testData.select_dtypes(exclude='object')
X = trainData.drop(['Id', 'SalePrice', 'YrSold'], axis=1)
y = trainData['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
regression = LinearRegression()
regression.fit(X_train, y_train)
print(regression.intercept_)
print(regression.coef_)
#Predict Value
y_pred = regression.predict(X_test)
predict = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predict['error'] = predict['Actual'] - predict['Predicted']
predict.head()
predict = predict.head(25)
predict.plot(kind='bar', figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='Blue')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='Blue')
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
plt.figure(figsize=(15,8))
plt.scatter(y_test, y_pred, color='gray', marker='+')
#plt.plot(y_test, y_pred, color='Blue', linewidth=1)
plt.show()
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
from sklearn import linear_model
SGDClf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3, penalty='elasticnet')
SGDClf.fit(X_train, y_train)
y_pred = SGDClf.predict(X_test)
SGDCpredict = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
SGDCpredict['error'] = SGDCpredict['Actual'] - SGDCpredict['Predicted']
SGDCpredict.head()
SGDCpredict = SGDCpredict.head(25)
SGDCpredict.plot(kind='bar', figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='Blue')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='Blue')
plt.show()
X_testData = testData.drop(['Id', 'YrSold'], axis=1)
testpred = regression.predict(X_testData)
testPredData = pd.DataFrame({'ID': testData['Id'], 'SalePrice': testpred})
testPredData.head()
