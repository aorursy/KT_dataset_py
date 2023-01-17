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
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder
train_df = pd.read_csv('../input/titanic/train.csv', header=0)

test_df = pd.read_csv('../input/titanic/test.csv', header=0)
# impute missing values using the median for numeric columns 

# and the most common value for string columns



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
# joing the features from train and test together before imputing missing values,



big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])

big_X_imputed = DataFrameImputer().fit_transform(big_X)
# XGBoost doesn't (yet) handle categorical features automatically,

# so we need to change them to columns of integer values



le = LabelEncoder()

for feature in nonnumeric_columns:

    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])
# Prepare the inputs for the model

train_X = big_X_imputed[0:train_df.shape[0]].values

test_X = big_X_imputed[train_df.shape[0]::].values

train_y = train_df['Survived']
# You can experiment with many other options here, using the same .fit() and .predict()

# this example uses the current build of XGBoost



gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)

predictions = gbm.predict(test_X)
submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],

                          'Survived':predictions})

submission.to_csv('submission.csv', index=False)