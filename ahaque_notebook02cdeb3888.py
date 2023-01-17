# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv", index_col="id")

df_train.head()
df_test = pd.read_csv("../input/test.csv", index_col="id")

df_test.head()
X = df_train[df_train.columns[:-1]]

y = df_train["loss"]
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
class MultiLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.encoders = defaultdict(LabelEncoder)

        

    def fit(self, X):

        X.apply(lambda x: self.encoders[x.name].fit(x))

        return self

        

    def transform(self, X):

        return X.apply(lambda x: self.encoders[x.name].transform(x))

        

    def fit_transform(self, X):

        return X.apply(lambda x: self.encoders[x.name].fit_transform(x))

        

    def inverse_transform(self, X):

        return X.apply(lambda x: self.encoders[x.name].inverse_transform(x))
class Transform_df(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.mlencode = MultiLabelEncoder()

    

    def fit(self, X):

        self.cat_cols = list(filter(lambda x: "cat" in x, X.columns.values))

        self.cont_cols = list(filter(lambda x: "cont" in x, X.columns.values))

        # self.mlencode.fit(X[self.cat_cols])

        return self

     

    def transform(self, X):

        #return pd.concat([self.mlencode.transform(X[self.cat_cols]), 

        return pd.concat([X[self.cat_cols].applymap(lambda x: ord(x[0])), 

                          X[self.cont_cols]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)
feat_trans = Transform_df()

X_train = feat_trans.fit_transform(X_train)

X_test = feat_trans.transform(X_test)

X_val = feat_trans.transform(df_test)
clf = RandomForestRegressor(200)

clf.fit(X_train, y_train)
mean_absolute_error(y_test, clf.predict(X_test))
y_p = clf.predict(X_val)
output = pd.DataFrame({'loss': y_p}, index=df_test.index)

output
output.to_csv("submission_01.csv")