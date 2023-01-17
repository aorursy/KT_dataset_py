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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import Imputer

from sklearn.base import TransformerMixin

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor
# Importing the dataset

dataset = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', na_values = ['NULL', 'NaN', 'nan', 'null'], keep_default_na = False)

dataset = dataset.drop([1379], axis = 0) 
column_list = []

for col in dataset.columns:

    column_list.append(col)
column_int = ['Id','MSSubClass',	'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',

               'YearBuilt',	'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF', 

               'TotalBsmtSF', '1stFlrSF',	'2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',	

               'BsmtHalfBath', 'FullBath',	'HalfBath', 'BedroomAbvGr',	'KitchenAbvGr', 'TotRmsAbvGrd',	

               'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 

               'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 

               'YrSold', 'SalePrice']
# Training Data Preprocessing

for index, row in dataset.iterrows():

    if dataset.loc[index,'LotFrontage'] == "NA":

        dataset.loc[index,'LotFrontage'] = np.nan

        

for index, row in dataset.iterrows():

    if dataset.loc[index,'MasVnrArea'] == "NA":

        dataset.loc[index,'MasVnrArea'] = int(0)

    elif dataset.loc[index,'GarageYrBlt'] == "NA":

        dataset.loc[index,'GarageYrBlt'] = int(0)

    elif dataset.loc[index,'MasVnrType'] == "NA":

        dataset.loc[index,'MasVnrType'] = "None"



for col in ['LotFrontage']:

    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

    imputer.fit(dataset[[col]])

    dataset[col]=imputer.transform(dataset[[col]]).ravel()



for col in column_list:

    if col in column_int:

        for index, row in dataset.iterrows():

            dataset.loc[index,col] = int(dataset.loc[index,col])

    else:

        le = LabelEncoder()

        dataset[col] = le.fit_transform(dataset[col])
# Assigning dependent and independent classes

X = dataset.iloc[:, :-1].values

Y = dataset.iloc[:, -1].values
# Creating regression model

regressor = RandomForestRegressor(n_estimators = 300)

regressor.fit(X,Y)
# Importing test dataset

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', na_values = ['NULL', 'NaN', 'nan', 'null'], keep_default_na = False)
column_list_test = []

for col in test.columns:

    column_list_test.append(col)

    

col_test = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 

            'GarageYrBlt','GarageCars', 'GarageArea']

col_imp = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']

col_imp_int = ['LotFrontage', 'BsmtFullBath', 'BsmtHalfBath']
#categorical imputer

class DataFrameImputer(TransformerMixin):



    def __init__(self):

        """Impute missing values.



        Columns of dtype object are imputed with the most frequent value 

        in column.



        Columns of other types are imputed with mean of column.



        """

    def fit(self, X, y=None):



        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],

            index=X.columns)



        return self



    def transform(self, X, y=None):

        return X.fillna(self.fill)
# Testing data preprocessing

for col in col_imp_int:

    for index, row in test.iterrows():

        if test.loc[index,col] == "NA":

            test.loc[index,col] = np.nan

            

for col in col_imp:

    for index, row in test.iterrows():

        if test.loc[index,col] == "NA":

            test.loc[index,col] = np.nan

                        

test_imp = pd.DataFrame()



for col in col_imp:

    test_imp[col] = test[col]



for col in col_imp:

    test_imp = DataFrameImputer().fit_transform(test_imp)

    

for col in col_imp:

    test[col] = test_imp[col]



for index, row in test.iterrows():

    for col in col_test:

        if test.loc[index,col] == "NA":

            test.loc[index,col] = int(0)

    if test.loc[index,'MasVnrType'] == "NA":

            test.loc[index,'MasVnrType'] = "None"

        

from sklearn.preprocessing import Imputer

for col in col_imp_int:

    if col == 'LotFrontage':

        imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)  

        imputer.fit(test[[col]])

        test[col]=imputer.transform(test[[col]]).ravel()

    else:

        imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)  

        imputer.fit(test[[col]])

        test[col]=imputer.transform(test[[col]]).ravel()



from sklearn.preprocessing import LabelEncoder

for col in column_list_test:

    if col in column_int:

        for index, row in test.iterrows():

            test.loc[index,col] = int(test.loc[index,col])

    else:

        le = LabelEncoder()

        test[col] = le.fit_transform(test[col])
# Assigning independent classes

X_test = test.values
# Prediction

Y_pred = regressor.predict(X_test)
# Creating CSV file

subm = pd.DataFrame()

subm['Id'] = X_test[:,0]

subm['SalePrice'] = Y_pred



subm.to_csv('Submission.csv', index = False)