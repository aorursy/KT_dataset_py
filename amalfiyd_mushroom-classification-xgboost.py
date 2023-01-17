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
# Importing additional modules

import xgboost as xgb

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder, OrdinalEncoder

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import roc_auc_score
# Creating dataframe selector class

from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names=attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values
# Import data to memory

filename = r'/kaggle/input/mushroom-classification/mushrooms.csv'

df = pd.read_csv(filename)
df.describe()
df.info()
df.head()
for col in df.columns:

    print("Columns : ", col, " Nunique : ", df[col].nunique())
# Creating pipeline for each feature type

class_col = ['class']

binary_cols = ['bruises','gill-attachment','gill-spacing','gill-size','stalk-shape']

ordinal_cols = ['gill-spacing','gill-size','ring-number','population']

nominal_cols  = [col for col in df.columns if col not in class_col + binary_cols + ordinal_cols]
# Creating X and y for dataset

df_X = df[binary_cols + ordinal_cols + nominal_cols]

df_y = df[class_col]
# Pipeline 01 Binary and Ordinal

pipeline_bin_ord = Pipeline([

    ('selector', DataFrameSelector(binary_cols + ordinal_cols)),

    ('ord', OrdinalEncoder())

])

    

# Pipeline 02 Nominal

pipeline_nom = Pipeline([

    ('selector', DataFrameSelector(nominal_cols)),

    ('nominal', OneHotEncoder(sparse=False))

])



# Pipeline 04 Class

pipeline_bin = Pipeline([

    ('selector',DataFrameSelector(class_col)),

    ('ord', OrdinalEncoder())

])
# Full pipeline

df_bin_ord = pipeline_bin_ord.fit_transform(df)

df_nom = pipeline_nom.fit_transform(df)

df_target = pipeline_bin.fit_transform(df)



X = np.c_[df_bin_ord, df_nom]

y = df_target.ravel()
# Create training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



print(X_train.shape)

print(X_test.shape)
# Model development - defining parameters

clf = xgb.XGBClassifier()

parameters = {

    'learning_rate':[0.1],

    'n_estimators': [500],

    'max_depth': [5],

    'min_child_weight': [1],

    'gamma': [0.1],

    'subsample': [0.8],

    'colsample_bytree': [0.8],

    'objective': ['binary:logistic'],

    'seed': [42]

}



grid = GridSearchCV(clf,

                    parameters, 

                    n_jobs=-1,

                    scoring="roc_auc",

                    cv=3,

                    verbose=1)
# Model development - fitting training set

grid.fit(X_train, y_train)
# Print roc_auc

grid.best_score_
# Get best model

best_clf = grid.best_estimator_
# Predict test set

y_pred = best_clf.predict(X_test)



# Evaluate prediction

roc_auc = roc_auc_score(y_pred, y_test)

print("AUC score : ", roc_auc)