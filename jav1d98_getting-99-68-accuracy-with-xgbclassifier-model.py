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
df = pd.read_csv('../input/nasa.csv')

df.head()
from xgboost import XGBClassifier

from matplotlib import pyplot

from xgboost import plot_importance

from numpy import sort

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectFromModel
df['avg_dia'] = df[['Est Dia in KM(min)', 'Est Dia in KM(max)']].mean(axis=1)
X = df[['Absolute Magnitude','avg_dia', 'Relative Velocity km per hr','Miss Dist.(kilometers)','Orbit Uncertainity',

        'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant','Epoch Osculation','Eccentricity','Semi Major Axis',

        'Inclination','Asc Node Longitude','Orbital Period','Perihelion Distance','Perihelion Arg',

        'Aphelion Dist','Perihelion Time','Mean Anomaly','Mean Motion']]

X.head()
y = df['Hazardous'].astype(int)

y.head()
model = XGBClassifier()

model.fit(X, y)

# plot feature importance

plot_importance(model)

pyplot.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)



model = XGBClassifier()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



thresholds = sort(model.feature_importances_)

for thresh in thresholds:



	selection = SelectFromModel(model, threshold=thresh, prefit=True)

	select_X_train = selection.transform(X_train)



	selection_model = XGBClassifier()

	selection_model.fit(select_X_train, y_train)



	select_X_test = selection.transform(X_test)

	y_pred = selection_model.predict(select_X_test)

	predictions = [round(value) for value in y_pred]

	accuracy = accuracy_score(y_test, predictions)

	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
X = df[['Absolute Magnitude','Minimum Orbit Intersection']]

y = df['Hazardous'].astype(int)

X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

model = XGBClassifier()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))