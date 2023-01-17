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

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
df = pd.read_csv("/kaggle/input/winetesttask/WINE.csv")
df.head()
df.fillna(-9999, inplace=True)
df.isnull().sum()
df.head()
one_hot_encoded_training_predictors = pd.get_dummies(df)
one_hot_encoded_training_predictors.drop("Index",axis=1, inplace=True)

X = one_hot_encoded_training_predictors.drop("Target", axis=1)

y = one_hot_encoded_training_predictors["Target"]
fig, axes = plt.subplots(len(one_hot_encoded_training_predictors.columns)//2, 2, figsize=(12, 48))



i = 0

for triaxis in axes:

    for axis in triaxis:

        one_hot_encoded_training_predictors.hist(column = one_hot_encoded_training_predictors.columns[i], bins = 100, ax=axis)

        i = i+1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
y_train = np.array(y_train).reshape(-1,1)
clf = RandomForestClassifier(n_estimators=1000, max_depth=16, verbose=1)

clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
accuracy_score(y_test, predicted)
X_train.head()
X_train_alco = X_train.drop([ "E", "Di", "Density", "Nitrogen", "Sugar", "pH", "V"], axis = 1)

X_test_alco = X_test.drop(["E", "Di", "Density", "Nitrogen", "Sugar", "pH", "V", ], axis = 1)

clf_2 = RandomForestClassifier(n_estimators=100, max_depth=20, verbose=1)
clf_2.fit(X_train_alco, y_train)
predicted = clf_2.predict(X_test_alco)
accuracy_score(y_test, predicted)
from sklearn.model_selection import cross_val_score
iters = list(range(50, 250, 50))

for i in iters:

    clf_cv = RandomForestClassifier(n_estimators=i)

    rfc_eval = cross_val_score(estimator=clf_cv

                               , X=X.drop([ "E", "Di", "Density"

                                           , "Nitrogen", "Sugar", "pH", "V"], axis = 1), y=y, cv=10)

    print(rfc_eval.mean(), '.Estimators:', i)
from sklearn.ensemble import AdaBoostClassifier
ada_boost = AdaBoostClassifier(n_estimators=100, random_state=0)

ada_boost.fit(X_train_alco, y_train)
predicted = ada_boost.predict(X_test_alco)
accuracy_score(y_test, predicted)
from sklearn.ensemble import GradientBoostingClassifier
X_train_gmb = X_train.drop([ "E", "Di", "Density", "Nitrogen", "Sugar", "pH", "V"], axis=1)

X_test_gmb = X_test.drop([ "E", "Di", "Density", "Nitrogen", "Sugar", "pH", "V"], axis=1)
y_train.reshape((1,-1))[0]
iters = list(range(50, 250, 50))

for i in iters:

    bmg_cv = GradientBoostingClassifier(n_estimators=i, learning_rate=.1,

                                     max_depth=10, random_state=0)

    gbm_eval = cross_val_score(estimator=bmg_cv, X=X_train_gmb, y=y_train.reshape((1,-1))[0], cv=10)

    print(gbm_eval.mean(), '.Estimators:', i)
gbm_clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=.1,

                                    max_depth=7, random_state=0).fit(X_train_gmb, y_train)
predicted = gbm_clf.predict(X_test_gmb)
accuracy_score(y_test, predicted)
X_test_gmb.describe()
df2 = pd.read_csv("/kaggle/input/winetesttask/WINE.csv")
df2 = pd.get_dummies(df2)
df2.drop("Index", axis=1, inplace=True)
df2 = df2.dropna()
df2.isnull().sum()
y = df2["Target"]
X = df2.drop("Target", axis = 1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.33, random_state=0)
clf = RandomForestClassifier(n_estimators=1000, max_depth=16, verbose=1)

clf.fit(X_train2, y_train2)

predicted = clf.predict(X_test2)
accuracy_score(y_test2, predicted)
# X_train2_dp_ft = X_train2.drop([ "E", "Di", "Density", "Nitrogen", "Sugar", "pH", "V"], axis=1)

# X_test2_dp_ft = X_test2.drop([ "E", "Di", "Density", "Nitrogen", "Sugar", "pH", "V"], axis=1)

X_train2_dp_ft = X_train2

X_test2_dp_ft = X_test2
clf.fit(X_train2_dp_ft, y_train2)

predicted = clf.predict(X_test2_dp_ft)

accuracy_score(y_test2, predicted)
X_train2_dp_ft.shape
X_train2_dp_ft.columns
numeric_cols = ['Alcohol', 'A', 'B', 'C', 'D', 'H', 'S', 'U']
X_train2_dp_ft[numeric_cols].info()
print(X_train2_dp_ft.shape, y_train2.shape)
rem_outliers = X_train2_dp_ft
rem_outliers.head()
rem_outliers["Target"] = y_train2
rem_outliers.head()
X_train2_dp_out = X_train2_dp_ft#[np.abs(X_train2_dp_ft.D-X_train2_dp_ft.D.mean()) <= (3*X_train2_dp_ft.D.std())]

# X_train2_dp_out = X_train2_dp_out[np.abs(X_train2_dp_out.Alcohol-X_train2_dp_out.Alcohol.mean()) <= (3*X_train2_dp_out.Alcohol.std())]

X_train2_dp_out = X_train2_dp_out[np.abs(X_train2_dp_out.Density-X_train2_dp_out.Density.mean()) <= (3*X_train2_dp_out.Density.std())]

# X_train2_dp_out = X_train2_dp_out[np.abs(X_train2_dp_out.H-X_train2_dp_out.H.mean()) <= (3*X_train2_dp_out.H.std())]

# X_train2_dp_out = X_train2_dp_out[np.abs(X_train2_dp_out.U-X_train2_dp_out.U.mean()) <= (3*X_train2_dp_out.U.std())]



# X_train2_dp_out = X_train2_dp_out[np.abs(X_train2_dp_out.S-X_train2_dp_out.S.mean()) <= (3*X_train2_dp_out.S.std())]

X_train2_dp_out = X_train2_dp_out[np.abs(X_train2_dp_out.A-X_train2_dp_out.A.mean()) <= (3*X_train2_dp_out.A.std())]

#X_train2_dp_out = X_train2_dp_out[np.abs(X_train2_dp_out.Sugar-X_train2_dp_out.Sugar.mean()) <= (3*X_train2_dp_out.Sugar.std())]

#X_train2_dp_out = X_train2_dp_out[np.abs(X_train2_dp_out.pH-X_train2_dp_out.pH.mean()) <= (3*X_train2_dp_out.pH.std())]

# X_train2_dp_out = X_train2_dp_out[np.abs(X_train2_dp_out.Nitrogen-X_train2_dp_out.Nitrogen.mean()) <= (3*X_train2_dp_out.Nitrogen.std())]
y_train2_dp_out = X_train2_dp_out["Target"]

X_train2_dp_out = X_train2_dp_out.drop("Target", axis = 1)
X_train2_dp_ft.shape
X_train2_dp_out.shape
clf = RandomForestClassifier(n_estimators=1000, verbose=1)

clf.fit(X_train2_dp_out, y_train2_dp_out)

predicted = clf.predict(X_test2)

accuracy_score(y_test2, predicted)
X_train2_dp_out.describe()
from sklearn.naive_bayes import GaussianNB
#X_train2_dp_out = X_train2_dp_out.drop("Density", axis = 1)

#X_test2_bayes = X_train2_dp_out.drop("Density", axis = 1)
nb_clf = GaussianNB()

nb_clf.fit(X_train2_dp_out, y_train2_dp_out)

predicted = nb_clf.predict(X_test2)

accuracy_score(y_test2, predicted)
from sklearn.naive_bayes import MultinomialNB
X_train2_bayes = X_train2.drop("Density", axis = 1)

X_test2_bayes = X_test2.drop("Density", axis = 1)
nb_clf = GaussianNB()

nb_clf.fit(X_train2, y_train2)

predicted = nb_clf.predict(X_test2)

accuracy_score(y_test2, predicted)