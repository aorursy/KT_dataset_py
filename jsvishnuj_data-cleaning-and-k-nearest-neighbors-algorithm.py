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
# importing the training data

train_dataset = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
# checking the information about the dataset including the data types and numerical counts

train_dataset.info()
# making the Id as index feature

train_dataset = train_dataset.set_index('Id')
# checking for missing values

missing_dataframe = pd.concat([train_dataset.isnull().sum()], axis = 1)

print(missing_dataframe[missing_dataframe[0]>0])
train_dataset = train_dataset.drop(['LotFrontage', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)
# checking for missing values

missing_dataframe = pd.concat([train_dataset.isnull().sum()], axis = 1)

print(missing_dataframe[missing_dataframe[0]>0])
# now we can remove the missing samples from the dataset using dropna()

train_dataset = train_dataset.dropna()
# checking for missing values

missing_dataframe = pd.concat([train_dataset.isnull().sum()], axis = 1)

print(missing_dataframe[missing_dataframe[0]>0])
numeric_columns = train_dataset.describe().columns

nonnumeric_columns = [col for col in train_dataset.columns if col not in train_dataset.describe().columns]
print("numeric columns count: ", len(numeric_columns))

print("non numeric columns count: ", len(nonnumeric_columns))
#import label encoder

from sklearn.preprocessing import LabelEncoder
def encoding(dataframe_feature):

    if(dataframe_feature.dtype == 'object'):

        return LabelEncoder().fit_transform(dataframe_feature)

    else:

        return dataframe_feature
train_dataset = train_dataset.apply(encoding)
numeric_columns = train_dataset.describe().columns

nonnumeric_columns = [col for col in train_dataset.columns if col not in train_dataset.describe().columns]

print("numeric columns count: ", len(numeric_columns))

print("non numeric columns count: ", len(nonnumeric_columns))
# importing seaborn library for data visualization

import seaborn as sns

import matplotlib.pyplot as plt
train_dataset.head()
sns.heatmap(train_dataset.corr())
correlation_matrix = train_dataset.corr()

essential_features = correlation_matrix.index[abs(correlation_matrix['SalePrice']) > 0.6]

plt.figure(figsize = (8, 8))

sns.heatmap(train_dataset[essential_features].corr(), cbar = False, annot = True, square = True)
sns.pairplot(train_dataset[essential_features])
train_dataset = train_dataset[(train_dataset["SalePrice"] < 500000) &

              (train_dataset["GrLivArea"] < 3000) &

              (train_dataset["TotalBsmtSF"] < 2300) &

              (train_dataset["1stFlrSF"] < 2200) & 

              (train_dataset["GarageArea"] < 1200)]
train_dataset.shape
sns.pairplot(train_dataset[essential_features])
# making the input and target features

X = train_dataset.drop(['SalePrice'], axis = 1)

y = train_dataset['SalePrice'].values
# Splitting the training and testing data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# importing metrics for accuracy calculation

from sklearn.metrics import mean_squared_error
# importing KNN regression

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()

# fitting the model with the training dataset

knn.fit(X_train, y_train)

# predicting the values

predicted_value = knn.predict(X_test)

# calculating the accuracy

rmse_before_cleaning = np.sqrt(mean_squared_error(predicted_value, y_test))

print(rmse_before_cleaning)
sns.distplot(train_dataset['SalePrice'], bins = 10)
from sklearn.model_selection import GridSearchCV

parameters = {

            'n_neighbors' : [1,2,3,4,5,6,7,8,9,10],

            'algorithm' : ['ball_tree', 'brute']

             }

grid_search_cv = GridSearchCV(KNeighborsRegressor(), parameters)

grid_search_cv.fit(X_train, y_train)
grid_search_cv.best_params_
knn = KNeighborsRegressor(n_neighbors = 4)

# fitting the model with the training dataset

knn.fit(X_train, y_train)

# predicting the values

predicted_value = knn.predict(X_test)

# calculating the accuracy

rmse_after_cleaning = np.sqrt(mean_squared_error(predicted_value, y_test))

print(rmse_after_cleaning)
print('RMSE:',rmse_after_cleaning)
# importing test data

test_dataset = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_dataset.head()
test_dataset = test_dataset.set_index(['Id'])
test_dataset.info()
test_dataset.isnull().sum()
test_dataset = test_dataset.drop(['LotFrontage', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)
test_dataset = test_dataset.dropna()
# converting the test data into numerical and then normalizing the data.

def encoding(dataframe_feature):

    if(dataframe_feature.dtype == 'object'):

        return LabelEncoder().fit_transform(dataframe_feature)

    else:

        return dataframe_feature

test_dataset = test_dataset.apply(encoding)
print(train_dataset.shape)

print(test_dataset.shape)
SalePrice = knn.predict(test_dataset)
submission_dataset = pd.DataFrame()

submission_dataset['Id'] = test_dataset.index

submission_dataset['SalePrice'] = SalePrice
submission_dataset.head()

# submission_dataset.to_csv("KNN_Housing_Regression.csv", index=False)
# importing training and testing datasets again.

train_dataset = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_dataset1 = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_dataset = test_dataset1.copy()
train_dataset = train_dataset.set_index(['Id'])

test_dataset = test_dataset.set_index(['Id'])
train_numerical_columns = train_dataset.describe().columns

train_dataset = train_dataset[train_numerical_columns]

train_dataset.head()
test_numerical_columns = test_dataset.describe().columns

test_dataset = test_dataset[test_numerical_columns]

test_dataset.head()
# importing the simple imputer missing values. 

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = 'mean')
imputed_data = imputer.fit_transform(train_dataset)

train_dataset = pd.DataFrame(data = imputed_data, columns = train_dataset.columns)

train_dataset.isnull().sum()
imputed_data = imputer.fit_transform(test_dataset)

test_dataset = pd.DataFrame(data = imputed_data, columns = test_dataset.columns)

test_dataset.isnull().sum()
X = train_dataset.drop(['SalePrice'], axis = 1)

y = train_dataset['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 22)
parameters = {

            'n_neighbors' : [1,2,3,4,5,6,7,8,9,10],

            'algorithm' : ['ball_tree', 'brute']

             }

grid_search_cv = GridSearchCV(KNeighborsRegressor(), parameters)

grid_search_cv.fit(X_train, y_train)

grid_search_cv.best_params_
knn = KNeighborsRegressor(n_neighbors = 6)

# fitting the model with the training dataset

knn.fit(X_train, y_train)

# predicting the values

predicted_value = knn.predict(X_test)

# calculating the accuracy

rmse = np.sqrt(mean_squared_error(predicted_value, y_test))

print(rmse)
print(train_dataset.shape)

print(test_dataset.shape)
SalePrice = knn.predict(test_dataset)
submission_dataset = pd.DataFrame({'Id' : test_dataset1['Id'], 'SalePrice': SalePrice})
submission_dataset.to_csv("KNN_Housing_Regression.csv", index=False)