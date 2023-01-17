import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

df.head()
print ((df["Class"].value_counts()[1]/df["Class"].value_counts()[0] )* 100 , "% of true fraud detected")
df["Time"].plot(figsize = (40,10))
# fraud

df["Amount"][df["Class"] == 1].plot(figsize = (40,10))

plt.show()
df["Amount"][df["Class"] == 1].hist(figsize=(10,10))

plt.show()
# normal

df["Amount"][df["Class"] == 0].plot(figsize = (40,10))
df["Amount"][df["Class"] == 0].hist(figsize=(40,10), bins = 50)

plt.show()
df.isnull().values.sum()
sns.distplot(df.loc[df['Class'] == 0]["Time"], hist=True)

sns.distplot(df.loc[df['Class'] == 1]["Time"], hist=True)
sns.boxplot(x="Class", y="Amount", data=df)
sns.scatterplot(x="Amount", y="V1",hue="Class",data=df)
X = df.drop(["Class", "Time"], axis = 1)

y = df["Class"]
from sklearn.ensemble import IsolationForest

from sklearn.metrics import classification_report,accuracy_score

import warnings  

warnings.filterwarnings('ignore')
fraction = df["Class"].value_counts()[1]/df["Class"].value_counts()[0]
algorithm = IsolationForest(behaviour='new',contamination=fraction,random_state=42)

y_pred = algorithm.fit(X).predict(X)

scores_prediction = algorithm.decision_function(X)
(y_pred.min(), y_pred.max())
y_pred[y_pred == 1] = 0

y_pred[y_pred == -1] = 1

n_errors = (y_pred != y).sum()
# Run Classification Metrics

print("error: {}".format(n_errors))

print("Accuracy Score :")

print(accuracy_score(y,y_pred))

print("Classification Report :")

print( classification_report(y, y_pred))



    
# Importing KNN module from PyOD

!pip install Pyod
from pyod.models.knn import KNN



# Train kNN detector

clf = KNN(contamination=fraction, n_neighbors=3)

clf.fit(X)
# Get the prediction labels of the training data

y_train_pred = clf.labels_ 

# Outlier scores

y_train_scores = clf.decision_scores_
plt.hist(y_train_pred)
from pyod.utils import evaluate_print



# Evaluate on the training data

evaluate_print('KNN', y, y_train_scores)
(y_train_pred.shape, y.shape)
from sklearn.metrics import classification_report

print( classification_report(y, y_train_pred))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify = y)
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)

from sklearn.metrics import classification_report

print( classification_report(y_test, y_pred))