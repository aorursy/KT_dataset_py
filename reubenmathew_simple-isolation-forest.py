import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv("../input/creditcard.csv")
print(data.head())

print(data.describe())

print(data.shape)
sns.countplot(data['Class'])

plt.show()
data = data.sample(frac = 0.50, random_state = 1)

print(data.shape)
correlation_matrix = data.corr()

fig = plt.figure(figsize = (10,10))

sns.heatmap(correlation_matrix,square = True)

plt.show()
data = data.drop(['V10','V12','V14','V17'],1)
fraud = data[data['Class']==1]

valid = data[data['Class']==0]



percent_fraud = len(fraud)/len(valid)

print(percent_fraud)
columns = data.columns.tolist()



target = "Class"



columns = [c for c in columns if c not in [target]]

print(columns)



X = data[columns]

Y = data[target]



from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = .5)

print(Y_train.shape,Y_test.shape)



print(X.shape,Y.shape)

from sklearn.metrics import classification_report, accuracy_score

from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings("ignore")

clf = IsolationForest(max_samples=len(X_train))

clf.fit(X_train)

y_pred = clf.predict(X_test)

y_pred[y_pred==1] = 0

y_pred[y_pred==-1] = 1
print(accuracy_score(y_pred,Y_test))
clf = IsolationForest(contamination = percent_fraud, max_samples = len(X_train))

clf.fit(X_train)

y_pred = clf.predict(X_test)

y_pred[y_pred==1] = 0

y_pred[y_pred==-1] = 1
print(accuracy_score(y_pred,Y_test))
from sklearn.metrics import classification_report

print(classification_report(y_pred,Y_test))