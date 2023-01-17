# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
train.info()
train.describe(include="all")
# import matplotlib.pyplot as plt

# train.hist(bins=50, figsize=(20,15))

# plt.show()
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('std_scaler', StandardScaler()),

    ])
train_num = train.select_dtypes(include=[np.number])

train_num_tr = num_pipeline.fit_transform(train_num)

train_num_tr
from sklearn.base import BaseEstimator, TransformerMixin



class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                       index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
from sklearn.preprocessing import OneHotEncoder



cat_pipeline = Pipeline([

    ('imputer', MostFrequentImputer()),

    ('cat_encoder', OneHotEncoder(sparse=False))

])



train_cat = train.select_dtypes(include=[np.object])

train_cat_tr = cat_pipeline.fit_transform(train_cat)

train_cat_tr
from sklearn.compose import ColumnTransformer



num_attribs = list(train_num)

cat_attribs = list(train_cat)



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs)

#         ("cat", cat_pipeline, cat_attribs),

    ])



train_prepared = train.drop(['Id','SalePrice'], axis=1)

train_prepared = full_pipeline.fit_transform(train)

train_prepared
test_prepared = test.drop('Id', axis=1)

test_prepared = full_pipeline.transform(test_prepared)
y_train = train['SalePrice']
from sklearn.model_selection import cross_val_score



def crossValScore(model, x, y):

    scores = cross_val_score(model, x, y, cv=10)

    return scores.mean()
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(train_prepared, y_train)

print(crossValScore(lin_reg, train_prepared, y_train))
ids = test['Id']

predictions = lin_reg.predict(test_prepared)

output = pd.DataFrame({ 'Id' : ids, 'SalePrice': predictions })

output.to_csv('submission.csv', index=False)