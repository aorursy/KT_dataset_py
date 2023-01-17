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
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('../input/StudentsPerformance.csv')
df.head()
df.info()
scores = df.iloc[:,-3:]
scores.head()
x = df['math score']
y = df['reading score']
z = df['writing score']

fig = plt.figure(figsize=(12,10))
ax = plt.axes(projection='3d')

ax.scatter(x, y, z,)
sns.pairplot(df,hue='gender')
sns.pairplot(df,hue='lunch')
sns.countplot(df['race/ethnicity'])
from sklearn.model_selection import train_test_split
X = scores
y = df['gender']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)
forest.fit(X_train, y_train)
pred = forest.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print (classification_report(y_test, pred), '\n', confusion_matrix(y_test,pred))
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_pred = dtc.predict(X_test)
print (classification_report(y_test, dtc_pred), '\n', confusion_matrix(y_test,dtc_pred))
y_lunch = df['lunch']
X_train, X_test, y_train, y_test = train_test_split(X, y_lunch, test_size=0.33, random_state=42)
forest.fit(X_train, y_train)
lunch_pred = forest.predict(X_test)
print (classification_report(y_test, lunch_pred), '\n', confusion_matrix(y_test,lunch_pred))
dtc.fit(X_train,y_train)
lunch_dtc_pred = dtc.predict(X_test)
print (classification_report(y_test, lunch_dtc_pred), '\n', confusion_matrix(y_test,lunch_dtc_pred))
y_race = df['race/ethnicity']
X_train, X_test, y_train, y_test = train_test_split(X, y_race, test_size=0.33, random_state=42)
forest.fit(X_train,y_train)
race_pred = forest.predict(X_test)
print (classification_report(y_test, race_pred), '\n', confusion_matrix(y_test,race_pred))
