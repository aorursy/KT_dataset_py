import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualization tools

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling as pp

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/iris/Iris.csv")
df.sample(5)
df.info()
report = pp.ProfileReport(df)



report.to_file("report.html")



report
import missingno as msno

msno.matrix(df)

plt.show()
f,ax = plt.subplots(figsize=(15, 10))

sns.heatmap(df.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
X=df.iloc[:, 1:5]

X.head()
df.Species.unique()
from sklearn.preprocessing import LabelEncoder

y=df.iloc[:,-1]



encoder = LabelEncoder()

y = encoder.fit_transform(y)

print(y)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics



classifier = KNeighborsClassifier(n_neighbors = 10)

classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)
print('Accuracy Score:')

print(metrics.accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(cm, annot=True, linewidths=0.5,linecolor="red", fmt= '.0f',ax=ax)

plt.show()

plt.savefig('ConfusionMatrix.png')