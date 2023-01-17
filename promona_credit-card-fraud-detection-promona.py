# importing libraries 

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy

import sklearn 
# Load data 

data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
# Show dataset columns

print(data.columns)
# Show data shape

print(data.shape)
# Show data information 

print(data.describe())
# Scale down data size 

data = data.sample(frac=0.1 , random_state=1)

print(data.shape)
# Plot histogram of each feature

data.hist(figsize = (20,20))

plt.show()
# Determine outlier fraction 

fraud = data[data['Class']==1]

valid = data[data['Class']==0]



outlier_fraction = len(fraud)/len(valid)

print(outlier_fraction)

print("fraud classes : {}".format(len(fraud)))

print("valid classes : {}".format(len(valid)))

# Correlation 

correlation_matrix = data.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(correlation_matrix,vmax=0.8,square=True)

plt.show()
columns = data.columns.tolist()

columns= [c for c in columns if c not in ['Class']]

target = 'Class'

X = data[columns]

Y = data[target]

print(X.shape)
from sklearn.metrics import classification_report , accuracy_score

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor



state =1 

classifiers = {

    'Isolation Forest' : IsolationForest(max_samples = len(X),

                                        contamination = outlier_fraction,

                                        random_state = state ),

    'Local Outlier Factor' : LocalOutlierFactor(n_neighbors=20, contamination = outlier_fraction)

}
n_outliers =len(fraud)

for i,(c_name,c_object) in enumerate(classifiers.items()):

    if c_name == 'Local Outlier Factor':

        y_pred = c_object.fit_predict(X)

        score_pred = c_object.negative_outlier_factor_

    else:

        c_object.fit(X)

        score_pred = c_object.decision_function(X)

        y_pred = c_object.predict(X)

    y_pred[y_pred==1] = 0 

    y_pred[y_pred == -1] = 1

    

    n_errors = (y_pred != Y).sum()

    print('Class Name : {} , Errors : {}'.format(c_name,n_errors))

    print(accuracy_score(Y,y_pred))

    print(classification_report(Y,y_pred))
