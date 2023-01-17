#Import the necessary libraries

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression

from sklearn.svm import LinearSVR



# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

file_path = '../input/train.csv'

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



data = pd.read_csv(file_path, index_col='Id')

# Create target object and call it y
#Check whether the dataset contains columns with high missing value

data.info()
#Take the list of columns with more than 300 missing value

cols_with_missing = [col for col in data

                     if data[col].isnull().sum() > 300]

cols_with_missing
#Drop columns with high missing value

data.drop(cols_with_missing, axis=1)
#Visualize the dataset to check are there lots of outliers in it

data.hist(bins = 50, figsize = (20, 15))

plt.show()
#Data normalization with feature clipping technique to deal with outliers

data['1stFlrSF'] = np.clip(data['1stFlrSF'], a_min = None, a_max = 2250) 

data['2ndFlrSF'] = np.clip(data['2ndFlrSF'], a_min = None, a_max = 1250) 

data['GarageArea'] = np.clip(data['GarageArea'], a_min = None, a_max = 1000) 

data['BsmtFinSF1'] = np.clip(data['BsmtFinSF1'], a_min = None, a_max = 2000) 

data['GrLivArea'] = np.clip(data['GrLivArea'], a_min = 500, a_max = 3250)

data['LotFrontage'] = np.clip(data['LotFrontage'], a_min = None, a_max = 150) 

data['MasVnrArea'] = np.clip(data['MasVnrArea'], a_min = None, a_max = 500) 

data['TotalBsmtSF'] = np.clip(data['TotalBsmtSF'], a_min = None, a_max = 2500)

data['OpenPorchSF'] = np.clip(data['OpenPorchSF'], a_min = None, a_max = 175)
#Column with outliers

outlier_cols = ['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'GarageArea', 'GrLivArea', 'LotFrontage', 'MasVnrArea', 'OpenPorchSF', 'TotalBsmtSF']

#Visualize the result of data normalization

data[outlier_cols].hist(bins = 50, figsize = (20, 15))

plt.show()
#sSelect the feature and label column

y = data.SalePrice

X = data.drop(['SalePrice'], axis=1)



# Split into validation and training data

train_X_full, valid_X_full, train_y, valid_y = train_test_split(X, y, train_size=0.85, test_size=0.15, random_state=1)



#create categorical columns

obj_cols = [cols for cols in train_X_full.columns if train_X_full[cols].dtype == 'object']

#create numerical columns

num_cols = [cols for cols in train_X_full.columns if train_X_full[cols].dtype in ['int64', 'float64']]



my_cols = obj_cols + num_cols

train_X = train_X_full[my_cols].copy()

valid_X = valid_X_full[my_cols].copy()



#Impute missing values in numerical columns with mean value

num_transform = SimpleImputer(strategy='mean')



#Impute missing values in categorical columns with most frequent value

#Change the categorical column to numerical column with One-hot encoder

obj_transform = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),

                              ('OH_encode', OneHotEncoder(handle_unknown='ignore'))])



#Applies transformers to columns in the dataset

preprocessor = ColumnTransformer(transformers=[('num', num_transform, num_cols),

                                          ('obj', obj_transform, obj_cols)])



#Pipeline to Connect the data preprocessing and modeling progress

def train_test_model(model):

    my_model = Pipeline(steps=[('preprocessor', preprocessor),

                        ('model', model)])



    my_model.fit(train_X, train_y)

    prediction = my_model.predict(valid_X)

    score = mean_absolute_error(prediction, valid_y)

    print(score)
train_test_model(RandomForestRegressor(n_estimators = 100))
train_test_model(XGBRegressor(learning_rate=0.02,

                     n_estimators=750,

                     random_state=1))
train_test_model(LinearRegression())
train_test_model(LinearSVR())