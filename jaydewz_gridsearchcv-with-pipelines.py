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
from pathlib import Path

from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import GridSearchCV
train_path = '../input/home-data-for-ml-course/train.csv'

test_path = '../input/home-data-for-ml-course/test.csv'

train = pd.read_csv(train_path,index_col='Id')

test = pd.read_csv(test_path,index_col='Id')

# Drop N/A values for sale price

train.dropna(axis=0,subset=['SalePrice'],inplace=True)

y = train.SalePrice

X = train.drop(['SalePrice'],axis=1)
X.isnull().sum().sort_values(ascending=False).head(20)
# Pool Data was very few and far between. 

#Converted the quality of the pool to whether the house had a pool or not

X['Pool'] = X.PoolQC.notnull().astype('int')

X = X.drop(['PoolQC'],axis=1)

test['Pool'] = test.PoolQC.notnull().astype('int')

test = test.drop(['PoolQC'],axis=1)
fence_map = {'MnWw':1,'GdWo':2,'MnPrv':3,'GdPrv':4} 

X['Fence'] = X.Fence.map(fence_map)

X.Fence.fillna(value=0,inplace=True)

test['Fence'] = test.Fence.map(fence_map)

test.Fence.fillna(value=0,inplace=True)



# Central Air

airMap = {'Y':1,'N':0}

X['CentralAir'] = X.CentralAir.map(airMap)

X.CentralAir.fillna(value=0,inplace=True)

test['CentralAir'] = test.CentralAir.map(airMap)

test.CentralAir.fillna(value=0,inplace=True)



# Garage Year Built fill NA rows

X['GarageYrBlt'].fillna(X['YearBuilt'],inplace=True)

X.GarageType.fillna(value='NoGarage',inplace=True)

test['GarageYrBlt'].fillna(test['YearBuilt'],inplace=True)

test.GarageType.fillna(value='NoGarage',inplace=True)



# Masonry Veneer

X['MasVnrType'].fillna(value='None',inplace=True)

X['MasVnrArea'].fillna(value=0,inplace=True)

test['MasVnrType'].fillna(value='None',inplace=True)

test['MasVnrArea'].fillna(value=0,inplace=True)



# Lot Frontage Assume square lot and take the square root of lot area if no value given

X.LotFrontage.fillna(X['LotArea']**0.5,inplace=True)

test.LotFrontage.fillna(test['LotArea']**0.5,inplace=True)



# Drop Misc Feature

X.drop(columns=['MiscFeature'],inplace=True)

test.drop(columns=['MiscFeature'],inplace=True)



# Sale Type

X['SaleType'].fillna(value='Other',inplace=True)

test['SaleType'].fillna(value='Other',inplace=True)


# Get X columns that are categorical/numerical

category_list = [cname for cname in X.columns if X[cname].dtype == 'object']

num_list = [cname for cname in X.columns if X[cname].dtype != 'object']
conditionSet1 = [0,'Po','Fa','TA','Gd','Ex']

conditionSet2 = [0, 'Unf','RFn','Fin']

conditionSet3 = [0,'N','P','Y'] #Paved Driveway

conditionSet4 = [0, 'Mix','FuseP','FuseF','FuseA','SBrkr']

conditionSet5 = [0,'Grvl', 'Pave'] #Access

conditionSet6 = [0, 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']

conditionSet7 = [0, 'Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']

conditionSet8 = [0,'ELO','NoSeWa','NoSewr','AllPub']

conditionSet9 = ['TwnhsI','TwnhsE','Duplx','2FmCon','1Fam'] # House Style

conditionSet10 = [0,'No','Mn','Av','Gd'] #BsmtExposure

conditions = [conditionSet1, conditionSet2, conditionSet3, conditionSet4, conditionSet5, conditionSet6, conditionSet7, conditionSet8, conditionSet9, conditionSet10]
# Create a mapping system for columns that have responses fulfilling one of the conditions

def cond_map(df,column,condition):

    weightings = [weight for weight in range(len(condition))]

    condition_map = dict(zip(condition,weightings))

    df[column].fillna(value=0,inplace=True)

    return df[column].map(condition_map)



for column in category_list:

    for condition in conditions:

        if set(X[column].fillna(value=0).values).issubset(condition):

            X[column] = cond_map(X,column,condition)

            test[column] = cond_map(test,column,condition)

        else:

            continue
# Reload X columns that are categorical/numerical

category_list = [cname for cname in X.columns if X[cname].dtype == 'object']

num_list = [cname for cname in X.columns if X[cname].dtype != 'object']

yearList = ['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']



# Want to fill missing values with zero except columns that are years

for column in np.setdiff1d(num_list,yearList):

    X[column].fillna(value=0,inplace=True)

    test[column].fillna(value=0,inplace=True)
# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])

# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, num_list),

        ('cat', categorical_transformer, category_list)

    ])



# Bundle preprocessing and modeling code in a pipeline

pipeline = Pipeline(steps=[('preprocessor',preprocessor),('xgbrg', XGBRegressor())])
# param_grid = {'xgbrg__learning_rate':[0.05,0.075,0.1],

# 'xgbrg__n_estimators':[300,450,600],

# 'xgbrg__max_depth':[2,4,6]}



# """

# Result:

# 0.8907309348826631

# {'xgbrg__learning_rate': 0.05, 'xgbrg__max_depth': 2, 'xgbrg__n_estimators': 600}

# """



# search = GridSearchCV(pipeline, param_grid,n_jobs=-1,cv=5)



# # Preprocessing of training data, fit model

# #my_pipeline.fit(X_train,y_train)

# search.fit(X,y)

# print(search.best_score_)

# print(search.best_params_)

my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),('xgbrg', XGBRegressor(learning_rate=0.05,max_depth=2,n_estimators=600))])

my_pipeline.fit(X,y)

test_pred = my_pipeline.predict(test)
output = pd.DataFrame({'Id': test.index,'SalePrice': test_pred})

output.to_csv('home_prices_submission.csv', index=False)