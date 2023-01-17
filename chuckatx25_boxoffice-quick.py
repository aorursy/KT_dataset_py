# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
movie_train_path = '../input/train.csv'

movie_test_path = '../input/test.csv'

train = pd.read_csv(movie_train_path)

test = pd.read_csv(movie_test_path)

test_data = pd.read_csv(movie_test_path)
train.head()
test.head()
train.describe()
train.isna().sum()/len(train)
test.isna().sum()/len(test)
# Revise wrong information



# Revising some wrong information



train.loc[train['id'] == 16,'revenue'] = 192864          # Skinning

train.loc[train['id'] == 90,'budget'] = 30000000         # Sommersby          

train.loc[train['id'] == 118,'budget'] = 60000000        # Wild Hogs

train.loc[train['id'] == 149,'budget'] = 18000000        # Beethoven

train.loc[train['id'] == 313,'revenue'] = 12000000       # The Cookout 

train.loc[train['id'] == 451,'revenue'] = 12000000       # Chasing Liberty

train.loc[train['id'] == 464,'budget'] = 20000000        # Parenthood

train.loc[train['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II

train.loc[train['id'] == 513,'budget'] = 930000          # From Prada to Nada

train.loc[train['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol

train.loc[train['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip

train.loc[train['id'] == 850,'budget'] = 90000000        # Modern Times

train.loc[train['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman

train.loc[train['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   

train.loc[train['id'] == 1359,'budget'] = 10000000       # Stir Crazy 

train.loc[train['id'] == 1542,'budget'] = 1              # All at Once

train.loc[train['id'] == 1542,'budget'] = 15800000       # Crocodile Dundee II

train.loc[train['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp

train.loc[train['id'] == 1714,'budget'] = 46000000       # The Recruit

train.loc[train['id'] == 1721,'budget'] = 17500000       # Cocoon

train.loc[train['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed

train.loc[train['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget

train.loc[train['id'] == 2491,'revenue'] = 6800000       # Never Talk to Strangers

train.loc[train['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus

train.loc[train['id'] == 2612,'budget'] = 15000000       # Field of Dreams

train.loc[train['id'] == 2696,'budget'] = 10000000       # Nurse 3-D

train.loc[train['id'] == 2801,'budget'] = 10000000       # Fracture

test.loc[test['id'] == 3889,'budget'] = 15000000       # Colossal

test.loc[test['id'] == 6733,'budget'] = 5000000        # The Big Sick

test.loc[test['id'] == 3197,'budget'] = 8000000        # High-Rise

test.loc[test['id'] == 6683,'budget'] = 50000000       # The Pink Panther 2

test.loc[test['id'] == 5704,'budget'] = 4300000        # French Connection II

test.loc[test['id'] == 6109,'budget'] = 281756         # Dogtooth

test.loc[test['id'] == 7242,'budget'] = 10000000       # Addams Family Values

test.loc[test['id'] == 7021,'budget'] = 17540562       #  Two Is a Family

test.loc[test['id'] == 5591,'budget'] = 4000000        # The Orphanage

test.loc[test['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee
def date_features(df):

    df['release_date'] = pd.to_datetime(df['release_date'])

    df['release_year'] = df['release_date'].dt.year

    df['release_month'] = df['release_date'].dt.month

    df['release_day'] = df['release_date'].dt.day

    df['release_quarter'] = df['release_date'].dt.quarter

    df.drop(columns=['release_date'], inplace=True)

    return df



train=date_features(train)

test=date_features(test)



train['release_year'].head(10)
train.columns
def drop_columns(df):

    df.drop(labels=['id', 'belongs_to_collection', 'genres', 'homepage',

       'imdb_id', 'original_language', 'original_title', 'overview',

        'poster_path', 'production_companies',

       'production_countries', 'spoken_languages',

       'tagline', 'title', 'Keywords', 'cast', 'crew'],axis=1,inplace=True)

    return df
train=drop_columns(train)

test=drop_columns(test)
from sklearn.model_selection import train_test_split

df_train,df_valid = train_test_split(train,test_size=0.2)
tr,va=df_train.copy(),df_valid.copy()

X_train,X_valid,y_train,y_valid = tr.drop(['revenue'],axis=1),va.drop(['revenue'],axis=1), tr['revenue'],va['revenue']
cat_col=[col for col in X_train.columns if X_train[col].dtype=="O"]

num_col=[col for col in X_train.columns if X_train[col].dtype in ['int64','int32','float32', 'float64']]
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
numerical_transformer = SimpleImputer(strategy='mean')

categorical_transformer= Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')), 

                                       ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('num',numerical_transformer,num_col),

                                              ('cat',categorical_transformer, cat_col)])
from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor

gb = XGBRegressor(n_estimators=100,learning_rate = 0.05)

rf = RandomForestRegressor(n_estimators=100,min_samples_split=5,min_samples_leaf=3,max_features=4)

cat = CatBoostRegressor(iterations=100,learning_rate=0.05,

                                 depth=3,

                                 eval_metric='RMSE',

                                 colsample_bylevel=0.8,

                                 random_seed = 1,

                                 bagging_temperature = 0.2,

                                 metric_period = None,

                                 early_stopping_rounds=200

                                )
xgb_pl = Pipeline(steps=[('preprocessor', preprocessor),('xgb',gb)])

rf_pl = Pipeline(steps=[('preprocessor', preprocessor),('randomforest',rf)])

cat_pl = Pipeline(steps=[('preprocessor', preprocessor),('catBoost',cat)])

xgb_pl.fit(X_train,y_train)

rf_pl.fit(X_train,y_train)

cat_pl.fit(X_train,y_train)
xgb_pred = xgb_pl.predict(X_valid)

rf_pred = rf_pl.predict(X_valid)

cat_pred = cat_pl.predict(X_valid)
from sklearn.metrics import mean_absolute_error

mae_rf = mean_absolute_error(rf_pred,y_valid)

mae_xgb =  mean_absolute_error(xgb_pred,y_valid)

mae_cat= mean_absolute_error(cat_pred,y_valid)

print('MAE RF: ', mae_rf, 'MAE XGB: ', mae_xgb,'MAE CAT: ', mae_cat )
np.mean(np.abs((rf_pred - y_valid) / rf_pred)) * 100, np.mean(np.abs((xgb_pred - y_valid) / xgb_pred)) * 100,np.mean(np.abs((cat_pred - y_valid) / cat_pred)) * 100
pred= xgb_pl.predict(test)
submission = pd.DataFrame({'id':test_data['id'],

                          'revenue':pred})
submission.to_csv('submission.csv',index=False)