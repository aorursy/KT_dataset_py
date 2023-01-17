# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

import numpy as np



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load the data

train_df = pd.read_csv('../input/train.csv', header=0)

test_df = pd.read_csv('../input/test.csv', header=0)



# We'll impute missing values using the median for numeric columns and the most

# common value for string columns.

# This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948

from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],

            index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.fill)



feature_columns_to_use = ['Pclass','Sex','Age','Fare','Parch']

nonnumeric_columns = ['Sex']



# Join the features from train and test together before imputing missing values,

# in case their distribution is slightly different

big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])

big_X_imputed = DataFrameImputer().fit_transform(big_X)



le = LabelEncoder()

for feature in nonnumeric_columns:

    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

    

print("Train DF Size: {:,}".format(train_df.shape[0]))

print("Test DF Size: {:,}".format(test_df.shape[0]))

print("Train Row 1: {}".format(big_X.as_matrix()[0]))

print("Train Row 1 Imputed: {}".format(big_X_imputed.as_matrix()[0]))
# Prepare the inputs for the model

# leave out last 100 samples for second model

train_X = big_X_imputed[0:train_df.shape[0]-100].as_matrix()

test_X = big_X_imputed[train_df.shape[0]::].as_matrix()

train_y = train_df['Survived'][0:-100]



# You can experiment with many other options here, using the same .fit() and .predict()

# methods; see http://scikit-learn.org

# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost

xgb1 = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)

predictions1 = xgb1.predict(test_X)

print("Done Training 1st model")

submission1 = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],

                            'Survived': predictions1 })



train_X2 = big_X_imputed[0:train_df.shape[0]].as_matrix()

test_X2 = big_X_imputed[train_df.shape[0]::].as_matrix()

train_y2 = train_df['Survived']

xgb2 = xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.10, ).fit(train_X, train_y, xgb_model=xgb1.get_booster())

xgb3 = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05, ).fit(train_X, train_y, xgb_model=xgb1.get_booster())

predictions2 = xgb2.predict(test_X)

predictions3 = xgb3.predict(test_X)

print("Done Training 2nd and 3rd models")

submission2 = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],

                             'Survived1': predictions1, 'Survived2': predictions2

                           , 'Survived3': predictions3})

print(submission2[0:20])
booster1 = xgb1.get_booster()

booster2 = xgb2.get_booster()

booster3 = xgb3.get_booster()

print(dir(booster1))
from xgboost import plot_importance

plot_importance(booster1) # first model

plot_importance(booster2) # deeper model

plot_importance(booster3) # same as first with more training
booster1.save_model('xgb1')

booster2.save_model('xgb2')

booster3.save_model('xgb3')

print(check_output(["ls", "-lh"]).decode("utf8"))