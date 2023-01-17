# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')
X_full.info()
X_full.drop(['Alley', 'PoolQC', 'MiscFeature', 'Fence'], axis=1)

X_test_full.drop(['Alley', 'PoolQC', 'MiscFeature', 'Fence'], axis=1)
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full['SalePrice']

X_full.drop(['SalePrice'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split



X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)



categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype=='object']

numerical_cols=[cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



numerical_transformer = SimpleImputer(strategy='constant')



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers = [

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ]

)
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score



def get_score(n_estimators):

    """Return the average MAE over 3 CV folds of random forest model.

    

    Keyword argument:

    n_estimators -- the number of trees in the forest

    """

    # Replace this body with your own code

    my_pipeline = Pipeline(steps=[

        ('preprocessor', preprocessor),

        ('model', GradientBoostingRegressor(n_estimators=n_estimators, random_state=0))

    ])

    

    scores = -1 * cross_val_score(my_pipeline, X_train, y_train,

                              cv=3,

                              scoring='neg_mean_absolute_error')

    

    return scores.mean()
results = {}

for i in range(1, 9):

    results[100*i] = get_score(100*i)
%matplotlib inline



plt.plot(results.keys(), results.values())

plt.show()
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score



def get_score(learning_rate):

    """Return the average MAE over 3 CV folds of random forest model.

    

    Keyword argument:

    n_estimators -- the number of trees in the forest

    """

    # Replace this body with your own code

    my_pipeline = Pipeline(steps=[

        ('preprocessor', preprocessor),

        ('model', GradientBoostingRegressor(n_estimators=400, learning_rate=learning_rate, random_state=0))

    ])

    

    scores = -1 * cross_val_score(my_pipeline, X_train, y_train,

                              cv=3,

                              scoring='neg_mean_absolute_error')

    

    return scores.mean()
results = {}

for i in range(1, 9):

    results[0.1*i] = get_score(0.1*i)
%matplotlib inline



plt.plot(results.keys(), results.values())

plt.show()
my_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', GradientBoostingRegressor(n_estimators=400, random_state=0))

])



scores = -1 * cross_val_score(my_pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')

print('MAE scores:\n', scores)

print('Average MAE score: ' + str(scores.mean()))
from sklearn.metrics import mean_absolute_error



my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_valid)

score = mean_absolute_error(y_valid, preds)



print('MAE: ' + str(score))
preds_test = my_pipeline.predict(X_test)

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)