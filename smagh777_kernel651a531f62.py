# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test_path='../input/house-prices-advanced-regression-techniques/test.csv'

train_path='../input/house-prices-advanced-regression-techniques/train.csv'

test = pd.read_csv(test_path)

train = pd.read_csv(train_path)
train.shape

#test.shape

#train=train.drop(columns="LotFrontage")
test.shape
print(train.isnull().sum())
#handling the missing data with NA values(more than 80%)

train=train.drop(["Alley","PoolQC","Fence","MiscFeature"],axis=1)



train.shape
#handling the missing data with NA values(less than 80%)



from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

#imputer = imputer.fit(train[:,])

imputer=imputer.fit(train[['LotFrontage','FireplaceQu','GarageYrBlt','GarageQual','GarageCond','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','MasVnrArea']])

train[['LotFrontage','FireplaceQu','GarageYrBlt','GarageQual','GarageCond','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','MasVnrArea']]=imputer.transform(train[['LotFrontage','FireplaceQu','GarageYrBlt','GarageQual','GarageCond','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','MasVnrArea']])







#checking once more for na values

missing_val_count_by_column = (train.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
#filtering the categorical columns from the data set

categorical_feature_mask = train.dtypes==object
#transfering categorical columns into a list

categorical_cols = train.columns[categorical_feature_mask].tolist()

train[categorical_cols].head(10)
#Encoding the categorical data

from sklearn.preprocessing import LabelEncoder

labelencoder_train=LabelEncoder()

train[categorical_cols] = train[categorical_cols].apply(lambda col: labelencoder_train.fit_transform(col))
X=train.iloc[:,0:76]

y=train.iloc[:,76]
# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=1000,random_state = 1)

regressor.fit(X,y)
# Fitting SVR to the dataset

#from sklearn.svm import SVR

#regressor = SVR(kernel = 'rbf')

#regressor.fit(X, y)
 #Fitting Decision Tree Regression to the dataset

#from sklearn.tree import DecisionTreeRegressor

#regressor = DecisionTreeRegressor(random_state = 0)

#regressor.fit(X, y)
 #Fitting Simple Linear Regression to the Training set

#from sklearn.linear_model import LinearRegression

#regressor = LinearRegression()

#regressor.fit(X, y)
#ElasticNet Regression

from sklearn.linear_model import ElasticNet

from sklearn.datasets import make_regression

X, y = make_regression(n_features=76, random_state=0)

regressor = ElasticNet(random_state=0)

regressor.fit(X, y)
#Data processing for test data



missing_val_count_by_column = (test.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
#handling the missing data with NA values(more than 80%)

test=test.drop(["Alley","PoolQC","Fence","MiscFeature"],axis=1)
test.shape
#handling the missing data with NA values(less than 80%)



from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

#imputer = imputer.fit(train[:,])

imputer=imputer.fit(test[['GarageArea','Exterior2nd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','KitchenQual','KitchenQual','Functional','GarageCars','SaleType','MSZoning','LotFrontage','FireplaceQu','GarageYrBlt','GarageQual','GarageCond','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','MasVnrArea','Utilities','Exterior1st']])

test[['GarageArea','Exterior2nd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','KitchenQual','KitchenQual','Functional','GarageCars','SaleType','MSZoning','LotFrontage','FireplaceQu','GarageYrBlt','GarageQual','GarageCond','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','MasVnrArea','Utilities','Exterior1st']]=imputer.transform(test[['GarageArea','Exterior2nd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','KitchenQual','KitchenQual','Functional','GarageCars','SaleType','MSZoning','LotFrontage','FireplaceQu','GarageYrBlt','GarageQual','GarageCond','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','MasVnrArea','Utilities','Exterior1st']])







categorical_feature_mask = test.dtypes==object
#transfering categorical columns into a list

categorical_cols = test.columns[categorical_feature_mask].tolist()

test[categorical_cols].head(10)
#Encoding the categorical data

from sklearn.preprocessing import LabelEncoder

labelencoder_test=LabelEncoder()

test[categorical_cols] = test[categorical_cols].apply(lambda col: labelencoder_test.fit_transform(col))
#predicting new result



train_predictions=regressor.predict(X)

train_predictions
from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(train_predictions,y)

mae
test_predictions=regressor.predict(test)
test_predictions
output = pd.DataFrame({'Id': test.Id,

                       'SalePrice': test_predictions})

output.to_csv('submission_RF_v_3.csv', index=False)
#submission['Id']=test['Id'].to_numpy()
#submission.head(20)
#submission['SalePrice']=test_predictions
#submission.to_csv(r'Users\baljindersmagh\Desktop\FinalSubmission.csv')
#print(submission)