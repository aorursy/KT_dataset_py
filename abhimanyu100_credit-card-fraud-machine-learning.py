# load the dataset from this link

!wget https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv
# Import all necessary library

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sklearn

import seaborn as sns

import scipy

import sys

df = pd.read_csv('https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv')
print(df.shape) # shape (i.e, number of row and column in dataset)

print(df.columns)

# print(df.head())
#Check the overall important value of the dataset

print(df.describe())
#previous shape

print(df.shape)

print(df.head())
class_count = pd.value_counts(df['Class'])

class_count.plot(kind = 'bar')

plt.xlabel("Class")

plt.ylabel("Total case/Frequency")

plt.title("Histogram of Fruad case")
#10 percent of the data

df = df.sample(frac = 0.1, random_state = 1)

print(df.shape)
df.hist(figsize = (20, 10))
#Calculate number of fraud cases in dataset (0 class represent fraud)

fraud = df[df['Class'] == 1]

Valid = df[df['Class'] == 0]



outlier_fraction = len(fraud) / float(len(Valid))

print(outlier_fraction)

print("Fraud Cases:  {}".format(len(fraud)))

print("Valid Cases: {}".format(len(Valid)))







cor = df.corr()

fig = plt.figure(figsize = (12, 9))



#heatmap in python

sns.heatmap(cor, vmax = 0.8, square = True)

plt.show()
print(df.columns)
columns = df.columns.tolist()



columns = [c for c in columns if c not in ["Class"]]



target = "Class"



X = df[columns]

y = df[target]



print(X.shape)

print(y.shape)
from sklearn.metrics import classification_report, accuracy_score

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor

# randomm seed

state = 1



# dictionary of classifier

# create outlier detection for fraud case problem

classifiers = {

    "Isolation Forest": IsolationForest(max_samples=len(X), contamination=outlier_fraction, random_state = state),

    "Locak Outlier factor": LocalOutlierFactor(

    n_neighbors = 20,

    contamination = outlier_fraction)

}
# fitting the model

n_outliers = len(fraud)

print(n_outliers)
for i, (clf_name, clf) in enumerate(classifiers.items()):

  

    if clf_name == 'Local Outlier factor':

        y_pred = clf.fit_predict(X)

        scores_pred = clf.negative_outlier_factor

    

    else:

        clf.fit(X)

        scores_pred = clf.decision_function(X)

        y_pred = clf.predict(X)

      

# reshape prediction value 0 for valid, 1 for fraud

    y_pred[y_pred == 1] = 0

    y_pred[y_pred == -1] = 1

    

    n_errors = (y_pred != y).sum()

    

    print('{}: {}'.format(clf_name, n_errors))

    print(accuracy_score(y, y_pred))

    print(classification_report(y, y_pred))