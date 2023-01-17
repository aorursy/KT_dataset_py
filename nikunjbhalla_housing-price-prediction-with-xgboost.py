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

from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
X_train.head()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from xgboost import XGBRegressor



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
from sklearn.model_selection import cross_val_score

def get_score(n_estimators):

    """Return the average MAE over 3 CV folds of random forest model.

    

    Keyword argument:

    n_estimators -- the number of trees in the forest

    """

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', XGBRegressor(random_state=0, learning_rate = 0.05, n_estimators = n_estimators))

                             ])

    scores = -1 * cross_val_score(my_pipeline, X_train, y_train,

                              cv=3,

                              scoring='neg_mean_absolute_error')

    #print(n_estimators, ' : ',scores.mean())

    return scores.mean()

results = dict()

for est in range(500,600,10):

    results[est] = get_score(est)
best_estimator = min(results, key=results.get)

print('Best estimator is', best_estimator, 'with MAE =', results[min(results, key=results.get)])
import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(results.keys(), results.values())

plt.show()
# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', XGBRegressor(random_state=0, learning_rate = 0.05, n_estimators = best_estimator))

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_valid)



from sklearn.metrics import mean_absolute_error

# Evaluate the model

score = mean_absolute_error(y_valid, preds)

print('MAE:', score)
preds_test = my_pipeline.predict(X_test)



output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)