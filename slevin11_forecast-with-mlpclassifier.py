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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline
Hotel=pd.read_csv("/kaggle/input/las-vegas-tripadvisor-reviews/lasvegas_tripadvisor.csv",sep=',')
sns.countplot(Hotel.Score)

y=Hotel.Score

X=Hotel.drop('Score',axis=1)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



numerical_cols = list(X.columns[X.dtypes == 'int64'])

categorical_cols= list(X.columns[X.dtypes == 'object'])
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler



# Preprocessing for numerical data

numerical_transformer = Pipeline([('Imputer',SimpleImputer(strategy='constant')),('Scale',StandardScaler())])



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

    ])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1) 
from sklearn.metrics import mean_absolute_error

from sklearn import metrics



# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', MLPClassifier())])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_test)



# Evaluate the model

score = mean_absolute_error(y_test, preds)

print('MAE:', score)

print('Accuracy: ',metrics.accuracy_score(y_test,preds))
pd.crosstab(y_test,preds)