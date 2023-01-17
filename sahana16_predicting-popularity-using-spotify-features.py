# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.plotting import scatter_matrix





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
filename='/kaggle/input/top50spotify2019/top50.csv'

spotify_data=pd.read_csv(filename,encoding='ISO-8859-1')

spotify_data.head()
#Understand the data

#spotify_data.info()

#for columns in spotify_data.columns:

#    print(spotify_data[columns].value_counts())
new_spotify_data=spotify_data.drop(['Track.Name','Unnamed: 0'],axis=1,inplace=False)

print(new_spotify_data.info())
#for columns in spotify_data.columns:

#    print(columns,spotify_data[columns].value_counts())

new_spotify_data.describe()

numericalattributes=new_spotify_data.columns[2:]

print(new_spotify_data.columns)

scatter_matrix(new_spotify_data,figsize=(15,20))

print(new_spotify_data.corr())

from sklearn.model_selection import train_test_split

print(new_spotify_data.columns)

popula_array=new_spotify_data['Popularity']

new_spotify_data.drop(['Popularity'],axis=1,inplace=True)

X_train, X_test, y_train, y_test = train_test_split(new_spotify_data,popula_array ,test_size=0.2, random_state=42)
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline,FeatureUnion

from sklearn.preprocessing import StandardScaler,LabelBinarizer,LabelEncoder

class DataFrameSelector(BaseEstimator, TransformerMixin): 

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names 

    def fit(self, X, y=None):

        return self

    def transform(self, X):

            return X[self.attribute_names].values

        

from sklearn.base import BaseEstimator, TransformerMixin

class new_LabelBina(BaseEstimator, TransformerMixin): 

    def __init__(self):

        self.encoder=LabelBinarizer()

    def fit(self, X, y=None):

        self.encoder.fit(X)

        return self

    def transform(self, X):

            return self.encoder.transform(X)

class new_Labelencoder(BaseEstimator, TransformerMixin): 

    def __init__(self):

        self.encoder=LabelEncoder()

    def fit(self, X, y=None):

        self.encoder.fit(X)

        return self

    def transform(self, X):

            return self.encoder.transform(X).reshape(-1,1)           






num_pipeline=Pipeline([('selector',DataFrameSelector(numericalattributes[:-1])),

    ('std_scalar',StandardScaler())])



artist_pipeline = Pipeline([

('selector', DataFrameSelector(['Artist.Name'])), ('label_binarizer', new_LabelBina()),

])

genre_pipeline = Pipeline([

('selector', DataFrameSelector(['Genre'])), ('label_binarizer2',new_LabelBina()),

])

full_pipeline = FeatureUnion(transformer_list=

                             [ ("num_pipeline", num_pipeline), ("artist_pipeline", artist_pipeline),

                                              ("genre_pipeline", genre_pipeline)])

spotify_preparared=full_pipeline.fit_transform(X_train)

#Training Model

from sklearn.linear_model import LinearRegression

linear=LinearRegression()

scale_output=StandardScaler()

new_y=scale_output.fit_transform(np.array(y_train).reshape(-1,1))

linear.fit(spotify_preparared,new_y)

test_data=full_pipeline.transform(X_test)

predictions=linear.predict(test_data)

from sklearn.metrics import mean_squared_error

new_y=scale_output.transform(np.array(y_test).reshape(-1,1))



lin_mse = np.sqrt(mean_squared_error(new_y,predictions))

print(lin_mse)





    