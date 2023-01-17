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



pd.set_option('display.max_columns', None)
data = pd.read_csv('../input/home-data-for-ml-course/train.csv',index_col='Id')

test = pd.read_csv('../input/home-data-for-ml-course/test.csv',index_col='Id')
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] +data['2ndFlrSF']

data['Total_Bathrooms'] = data['FullBath'] + (0.5* data['HalfBath']) + data['BsmtFullBath'] + (0.5* data['BsmtHalfBath'])

data['Total_sqrt_footage'] = data['BsmtFinSF1'] +data['BsmtFinSF2'] + data['1stFlrSF']+data['2ndFlrSF']

data['Total_porch_SF'] = data['OpenPorchSF'] + data['3SsnPorch'] +data['EnclosedPorch'] +  data['ScreenPorch'] + data['WoodDeckSF']



#new features 

data['haspool'] = data['PoolArea'].apply(lambda x:1 if x>0 else 0)

data['has2ndFloor'] = data['2ndFlrSF'].apply(lambda x:1 if x>0 else 0)

data['hasgarage'] = data['GarageArea'].apply(lambda x:1 if x>0 else 0)

data['hasbsmt'] = data['TotalBsmtSF'].apply(lambda x:1 if x>0 else 0)

data['hasfireplace'] = data['Fireplaces'].apply(lambda x:1 if x>0 else 0)
import datetime

now = datetime.datetime.now()

building_age = now.year - data['YearBuilt']

data['building_age'] = now.year - data['YearBuilt']



#data.info()
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder
numrical_missing = SimpleImputer(strategy='constant')

categorical_missing = SimpleImputer(strategy='most_frequent')

onehot = OneHotEncoder(handle_unknown='ignore')

label_encoder = LabelEncoder()
data.head()

#before using IMputer split train and test set

y = data.SalePrice

X = data.drop('SalePrice', axis=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1,test_size=0.2)
#numerical column

numerical_column = [col for col in X.columns if X[col].dtypes in ['int64','float64']]

#categorical column has lees then 10 unique values

categirical_column_less_10 = [col for col in X.columns if 

                             X[col].dtype == 'object' and

                             X[col].nunique() < 10]



#categorical values has more then 10 unique values

categirical_column_more_10 = [col for col in X.columns if 

                             X[col].dtype == 'object' and

                             X[col].nunique() > 10]



all_cat = categirical_column_less_10+ categirical_column_more_10
train_X
train_X_Imputer_num = pd.DataFrame(numrical_missing.fit_transform(train_X[numerical_column]), columns=train_X[numerical_column].columns)

val_X_Imputer_num = pd.DataFrame(numrical_missing.fit_transform(val_X[numerical_column]), columns=val_X[numerical_column].columns)



train_X_Imputer_cat = pd.DataFrame(categorical_missing.fit_transform(train_X[all_cat]), columns=train_X[all_cat].columns)

val_X_Imputer_cat = pd.DataFrame(categorical_missing.fit_transform(val_X[all_cat]), columns=val_X[all_cat].columns)



train_X_Imputer_cat_label =train_X_Imputer_cat.copy()

val_X_Imputer_cat_label =val_X_Imputer_cat.copy()



for col in all_cat:

    train_X_Imputer_cat_label[col] = label_encoder.fit_transform(train_X_Imputer_cat[col])

    val_X_Imputer_cat_label[col] = label_encoder.fit_transform(val_X_Imputer_cat[col])
train_X_Imputer_num.join(train_X_Imputer_cat_label)

val_X_Imputer_num.join(val_X_Imputer_cat_label)
X_train = train_X_Imputer_num.copy()

X_valid = val_X_Imputer_num.copy()
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=37)



X_new = selector.fit_transform(train_X_Imputer_num, train_y)



# Get back the kept features as a DataFrame with dropped columns as all 0s

selected_features = pd.DataFrame(selector.inverse_transform(X_new),

                                 index= train_X_Imputer_num.index,

                                 columns=train_X_Imputer_num.columns)

dropped_columns = selected_features.columns[selected_features.var() == 0]

print(dropped_columns)

X_train_final = X_train.drop(dropped_columns, axis=1)

X_valid_final = X_valid.drop(dropped_columns, axis=1)

model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(X_train_final, train_y)

preds = model.predict(X_valid_final)

score = mean_absolute_error(val_y, preds)

print('MAE: ', score)
my_cols =['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'TotalSF', 'Total_Bathrooms', 'Total_sqrt_footage', 'Total_porch_SF', 'haspool', 'has2ndFloor', 'hasgarage', 'hasbsmt', 'hasfireplace', 'Neighborhood_count', 'Exterior1st_count', 'Exterior2nd_count', 'building_age', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
result = set(my_cols) - set(dropped_columns)

print(result)