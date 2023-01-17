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

movie_data = pd.read_csv(movie_train_path)

test_data = pd.read_csv(movie_test_path)
movie_data.head()
movie_data.drop(['homepage','imdb_id','original_title','overview','poster_path','crew'], axis= 1 , inplace = True)
movie_data[['release_month','release_day','release_year']]=df['release_date'].str.split('/',expand=True).replace(np.nan, 0).astype(int)

movie_data['release_year'] = df['release_year']

movie_data.loc[ (df['release_year'] <= 18) & (df['release_year'] < 100), "release_year"] += 2000

movie_data.loc[ (df['release_year'] > 18)  & (df['release_year'] < 100), "release_year"] += 1900



movie_data.drop('release_date',axis= 1 , inplace = True)
movie_data.head()
movie_data['belongs_to_collection'].apply(lambda x: 1 if x != {} else 0)
categorical_cols = [cname for cname in movie_data.columns if movie_data[cname].dtype=='object']



numerical_cols = [cname for cname in movie_data.columns if movie_data[cname].dtype in ['int64', 'float64']]



my_cols = categorical_cols + numerical_cols
categorical_cols
X = movie_data[my_cols]

y = movie_data.revenue

from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid = train_test_split(X,y,random_state=1)
X.isnull().sum()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
numerical_transformer = SimpleImputer(strategy='mean')

categorical_transformer= Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')), 

                                       ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('num',numerical_transformer,numerical_cols),

                                              ('cat',categorical_transformer, categorical_cols)])
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=100,learning_rate = 0.05)
from sklearn.metrics import mean_absolute_error

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])

my_pipeline.fit(X_train,y_train)

prediction1 = my_pipeline.predict(X_valid)

MAE = mean_absolute_error(prediction1,y_valid)

print('MAE: ', MAE)
my_pipeline.fit(X,y)

prediction=my_pipeline.predict(test_data)
submission = pd.DataFrame({'id':test_data['id'],

                          'revenue':prediction})
submission.to_csv('submission.csv',index=False)