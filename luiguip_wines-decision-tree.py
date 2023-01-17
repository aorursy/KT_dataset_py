# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
winemag = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv', index_col=0)
winemag.head()
winemag.info()
winemag.isna().sum()
len(winemag['region_1'].unique())
X = pd.get_dummies(winemag[['country', 'price', 'variety']])

y = winemag['points']

X.columns
X.isna().sum()
X['price'].describe()
X['price'] = X['price'].fillna(X['price'].mean()).round()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
d_tree_classifier = DecisionTreeClassifier(random_state=1)

d_tree_classifier.fit(X_train, y_train)

predict_tree_default = d_tree_classifier.predict(X_val)
mean_absolute_error(y_val, predict_tree_default)
def get_mae(max_leaf_nodes):

    d_tree_classifier = DecisionTreeClassifier(random_state=1, max_leaf_nodes=max_leaf_nodes)

    d_tree_classifier.fit(X_train, y_train)

    predict_tree_default = d_tree_classifier.predict(X_val)

    return mean_absolute_error(y_val, predict_tree_default)
[get_mae(x) for x in [5, 50, 500, 5000]]
d_tree_classifier_test = DecisionTreeClassifier(random_state=1, max_leaf_nodes=500)

d_tree_classifier_test.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

predict_tree_test = d_tree_classifier_test.predict(X_test)

mean_absolute_error(y_test, predict_tree_test)
random_forest_classifier_test = RandomForestClassifier(random_state=1, n_estimators=10)

random_forest_classifier_test.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

predict_random_forest_test = random_forest_classifier_test.predict(X_test)

mean_absolute_error(y_test, predict_random_forest_test)
from xgboost import XGBClassifier

xgb_classifier_test = XGBClassifier(random_state=1, n_estimators=10)

xgb_classifier_test.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

predict_xgb_test = xgb_classifier_test.predict(X_test)

mean_absolute_error(y_test, predict_xgb_test)