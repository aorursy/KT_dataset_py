# Tutorials etc. used

# Data descriptions: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

# RMSLE function: https://www.kaggle.com/marknagelberg/rmsle-function



# Import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

import math
# Import Root Mean Squared Logarithmic Error (RMSLE) function

# Created by Alexandre Gazagnes

# https://www.kaggle.com/marknagelberg/rmsle-function

def rmsle(y_test, y_pred) : 

    assert len(y_test) == len(y_pred)

    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))
def clean_categorical_data(dataset):

    #print("Substitute missing data:")

    # Part 1: Features where we will substitute missing data with "None"/"NA"

    # TODO check wether to use None or NA

    catfeats_fillnaNone = [

        'PoolQC', # Pool quality (Can be Ex, Gd, TA, Fa, NA)

        'MiscFeature', # Miscellaneous feature not covered in other categories (Can be Elev, Gar2, Othr, Shed, TenC, NA)

        'Alley', # Type of alley access to property (Can be Grvl, Pave, NA )

        'Fence', # Fence quality 

        'FireplaceQu', # Fireplace quality

        'GarageType', # Garage location

        'GarageFinish', # Interior finish of the garage

        'GarageQual', # Garage quality

        'GarageCond', # Garage condition

        'BsmtFinType2', # Quality of second finished area (if present)

        'BsmtExposure', # Walkout or garden level basement walls

        'BsmtFinType1', # Quality of basement finished area 

        'BsmtCond', # General condition of the basement

        'BsmtQual'# Height of the basement

         ]



    dataset.loc[:,catfeats_fillnaNone] = dataset[catfeats_fillnaNone].fillna('None')



    # Part 2: Features where we don't know how to substitute missing data

    missing_features = [

        'LotFrontage', # Linear feet of street connected to property

        'MasVnrArea', # Masonry veneer area in square feet (brick facade)

        'MasVnrType', #Masonry veneer type (brick facade)

        'Electrical' #Electrical system

    ]



    # Solution: Average or drop

    #print("Features where we don't know how to substitute missing data:")

    #print(missing_features)

    

    return dataset
def select_features(train_test_combined):

    #print("Feature selection")

    # TODO we should engineer away all missing ones

   # missing = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'GarageArea', 'GarageCars', 'GarageYrBlt', 'LotFrontage', 'MasVnrArea', 'TotalBsmtSF']

   # missing = ['']

    features = train_test_combined.select_dtypes(include=[np.number]) #drop categoricals  #only numericals # TODO also include others

    features = features.dropna(axis=0) # TODO do we need this?

    features = features.columns

    features = features.drop(['SalePrice','Id'])

   # features = features.drop(missing)

    

    return features
def calculate(submission, features):

    original_y = train.SalePrice

    original_X = train[features]



    # Prepare train_X, val_X, train_y, val_y

    if(submission == True):

        train_X = original_X

        train_y =  original_y

        val_X = test[features]

    else: # Train-Test-Split

        train_X, val_X, train_y, val_y = train_test_split(original_X, original_y, random_state = 0)



    # Define model, Fit & Predict

    model = RandomForestRegressor(random_state=1,n_estimators=100)

    model.fit(train_X, train_y)

    y_prediction = model.predict(val_X)



    # Output submission file OR local metrics

    if(submission == True):

        submission = pd.DataFrame()

        submission['Id'] = test.Id

        submission['SalePrice'] = y_prediction

        submission.to_csv('submission_test.csv', index=False)

    else: 

        metric_mae = mean_absolute_error(val_y, y_prediction)

        metric_rmse = rmsle(val_y, y_prediction)

        print("\n\n\n--------------------------------")

        print("Validation MAE for Random Forest Model: {}$".format(int(metric_mae)))

        print("Validation RMSQ for Random Forest Model: {}".format(metric_rmse))
def count_missing_numerical(dataset):

    data = dataset.select_dtypes(include=[np.number])

    null_cols = data.columns[data.isnull().any(axis=0)]

    X_null = data[null_cols].isnull().sum()

    X_null = X_null.sort_values(ascending=False)

    print(X_null)
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import Imputer



def clean_numerical_data(dataset):

    '''

    print("\n\nBEFORE CLEANING: \n")

    count_missing_numerical(dataset)

    print(dataset['LotFrontage'].describe())

    '''



    cols_with_missing = ['MasVnrArea', # 8+15 - Masonry veneer area in square feet

                             'LotFrontage', # 259 - Linear feet of street connected to property

 #                            'GarageYrBlt', # 81 - 

                             'LotFrontage',

                             'BsmtHalfBath',

                             'BsmtFullBath',

                             'GarageArea',

                             'GarageCars',

                             'TotalBsmtSF',

                             'BsmtUnfSF',

                             'BsmtFinSF2',

                             'BsmtFinSF1'

                            ]

    

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    imp.fit(dataset[cols_with_missing])

    dataset.loc[:, cols_with_missing]= imp.transform(dataset[cols_with_missing])



    # For some features, we can do better than the Imputer. We can substitute missing data with other meaningful data

    # GarageYrBlt: Year garage was built: We will use the year the house was built

    dataset.loc[:,'GarageYrBlt'] = dataset['GarageYrBlt'].fillna(dataset.YearBuilt)

    

    '''

    print("\n\nAFTER CLEANING: \n")

    count_missing_numerical(dataset)

    print(dataset['LotFrontage'].describe())

    '''

    

    return dataset
# Toggle between submission file generation and local train-test-split

submission = True



# Import files

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train_test_combined = pd.concat([train, test], sort=False)



train = clean_categorical_data(train)

test = clean_categorical_data(test)

#print("Clean train set \n")

train = clean_numerical_data(train)

#print("\n-----------------\n")

#print("Clean test set \n")

test = clean_numerical_data(test)

features = select_features(train_test_combined)

calculate(submission, features)