# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/creditcard.csv")
data.head()
data.columns
data.shape
data["Class"].value_counts()
data.describe()
data=data.sample(frac=0.1,random_state=1)

data.shape
data.hist(figsize=(20,20))

plt.show()
fraud=data[data["Class"]==1]

valid=data[data["Class"]==0]

outlier_fraction=len(fraud)/len(valid)

outlier_fraction
correlation=data.corr()

fig=plt.figure(figsize=(12,9))

sns.heatmap(correlation,square=True)

plt.show()
#get all columns from data

columns=data.columns.tolist()

#remove class

columns=[c for c in columns if c not in ["Class"]]

target="Class"

X=data[columns]

Y=data[target]

print(X.shape)

print(Y.shape)
from sklearn.metrics import  classification_report,accuracy_score

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor

#random state

state=1

#define outlier detection methods

classifiers={"Isolation Forest":IsolationForest(max_samples=(len(X)),contamination=outlier_fraction,

                                                           random_state= state),

            "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20,contamination=outlier_fraction)}

#fit the model

n_outliers=len(fraud)

for i,(clf_name,clf) in enumerate(classifiers.items()):

    #fit data and tag outlier

    if clf_name=="Local Outlier Factor":

        y_pred=clf.fit_predict(X)

        scores_pred=clf.negative_outlier_factor_

    else:

        clf.fit(X)

        scores_pred=clf.decision_function(X)

        y_pred=clf.predict(X)

    #reshape prediction values to 0 for valid,1 for fraud

    y_pred[y_pred==1]=0

    y_pred[y_pred== -1]=1

    n_errors=(y_pred!=Y).sum()

    #run classification metrices

    print("{}:{}".format(clf_name,n_errors))

    print(accuracy_score(Y,y_pred))

    print(classification_report(Y,y_pred))

    

        

        