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

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error

data_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv') 

data_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv') 
import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score

# Read the data

X = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

X_test_full =pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice              

X.drop(['SalePrice'], axis=1, inplace=True)



# Break off validation set from training data





# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and 

                        X[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_copy= X[my_cols].copy()



X_test = X_test_full[my_cols].copy()



numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numeric_cols ),

        ('cat', categorical_transformer, low_cardinality_cols)

    ])

#def get_score(n_estimators):

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', XGBRegressor(n_estimators=1500, learning_rate=0.03) )

                             ])

my_pipeline.fit(X_copy, y)

        #return scores.mean()

#results ={}

#for i in range(1,9):  #别忘了冒号！！！！！

    #results[250*i]=get_score(250*i)###记住  ！！！！# Your code here  字典的创建和索引！！！！！

#import matplotlib.pyplot as plt

#%matplotlib inline



#plt.plot(results.keys(), results.values())

#plt.show()

preds_test =my_pipeline.predict(X_test)

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)


