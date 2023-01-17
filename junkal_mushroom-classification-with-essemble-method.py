import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

import time
data = pd.read_csv('../input/mushrooms.csv', index_col=False)

data.head(5)
print(data.shape)
data.describe()
encoder = LabelEncoder()



for col in data.columns:

    data[col] = encoder.fit_transform(data[col])

 

data.head()
print(data.groupby('class').size())
Y = data['class'].values

X = data.drop('class', axis=1).values



X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.30, random_state=21)
# ensembles

ensembles = []

ensembles.append(('AB', AdaBoostClassifier()))

ensembles.append(('GBM', GradientBoostingClassifier()))

ensembles.append(('RF', RandomForestClassifier()))

ensembles.append(('ET', ExtraTreesClassifier()))
import warnings



results = []

names = []

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    for name, model in ensembles:

        kfold = KFold(n_splits=10, random_state=21)

        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

        results.append(cv_results)

        names.append(name)

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)
# prepare the model

model = RandomForestClassifier(random_state=21, n_estimators=100) 

model.fit(X_train, Y_train)
predictions = model.predict(X_test)

print("Accuracy score %f" % accuracy_score(Y_test, predictions))

print(classification_report(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))