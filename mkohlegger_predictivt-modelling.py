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
data = pd.read_csv("/kaggle/input/direct-marketing/dm-1.csv")

data.drop("Cust_Id", axis=1, inplace=True)

data.head(4)
from sklearn.model_selection import train_test_split
in_features = ['Age', 'Gender', 'OwnHome', 'Married', 'Location', 'Salary', 'Children', 'History', 'Catalogs']

out_features = ['AmountSpent']



X, y = data[in_features], data[out_features]

X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from sklearn.pipeline import Pipeline, FeatureUnion
class ColumnSelector:

    

    """This is a sklearn conform Transformer that allows you to select columns in a pipeline"""

    

    def __init__(self, select_numeric=True):

        assert type(select_numeric) == bool, "select_numeric needs to be boolean"

        self.sn = select_numeric

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        from pandas.core.frame import DataFrame

        assert type(X) == DataFrame, "X input needs to be pandas DataFrame"

        if self.sn:

            return X.select_dtypes(include="number")

        elif not self.sn:

            return X.select_dtypes(exclude="number")
class CategorialImputer:

    

    def __init__(self):

        self.col_most_frequent = {}

    

    def fit(self, X, y=None):

        from pandas.core.frame import DataFrame

        assert type(X) == DataFrame, "X needs to be pandas DataFrame"

        

        for column in X.columns:

            

            # store most frequent value in self

            v = X[column].mode()[0]

            self.col_most_frequent[column] = v

            

            # now we also need to impute the fitting data to pass on to the next step

            X[column].fillna(v)

        

        return self

    

    def transform(self, X, y=None):

        from pandas.core.frame import DataFrame

        assert type(X) == DataFrame, "X needs to be pandas DataFrame"

        

        for colum in X.columns:

            X[colum].fillna(self.col_most_frequent[column])

            

        return X
X_pipeline = FeatureUnion(transformer_list=[

    ("numeric pipeline", Pipeline(steps=[

        ("select numeric", ColumnSelector(select_numeric=True)),

        ("impute", SimpleImputer(strategy="median")),

        ("scale", MinMaxScaler())

    ])),

    ("non numeric pipeline", Pipeline(steps=[

        ("select numeric", ColumnSelector(select_numeric=False)),

        ("impute", DataFrameImputer()),

        ("encode", OneHotEncoder())

    ]))

])



y_pipeline = Pipeline(steps=[

    ("scale", MinMaxScaler())

])
X_pipeline.fit(X_train)

X_train_pp = X_pipeline.transform(X_train)

X_test_pp = X_pipeline.transform(X_test)



y_pipeline.fit(y_train)

y_train_pp = y_pipeline.transform(y_train)

y_test_pp = y_pipeline.transform(y_test)



y_train_pp = y_train_pp.ravel()

y_test_pp = y_test_pp.ravel()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train_pp, y_train_pp)
from matplotlib import pyplot as plt
prediction = y_pipeline.inverse_transform(rf.predict(X_test_pp).reshape(-1, 1))



plt.figure(figsize=(20,10))

plt.scatter(x=y_test, y=prediction, label="predicted")

plt.plot([0,5000], [0,5000], "r-")
print(f"R² score is {r2_score(y_test, prediction)}")

print(f"Mean abs error is {mean_absolute_error(y_test, prediction)}")

print(f"Mean error is {(mean_squared_error(y_test, prediction))**0.5}")