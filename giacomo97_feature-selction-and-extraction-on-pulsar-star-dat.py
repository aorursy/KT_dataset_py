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



# Any results you write to the current directory are saved as output.
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

import matplotlib.pyplot as plt



def evaluate(y_test, y_pred):

    return {

        "accuracy" : accuracy_score(y_test, y_pred),

        "precision" : precision_score(y_test, y_pred),

        "recall" : recall_score(y_test, y_pred),

        "AUC" : roc_auc_score(y_test, y_pred),

        "f1-measure" :  f1_score(y_test, y_pred, average='micro')

    }

def metricsKeeper(listOfMetrics,newMetrics,i):

    for key,value in newMetrics.items():

        if key not in listOfMetrics.keys():

            listOfMetrics[key]=[]

        listOfMetrics[key].insert(i,value)

    return listOfMetrics



def plotMetrics(metrics):

    plt.figure()

    for key, value in metrics.items():

        plt.plot(range(1,len(value)+1),value,label=key)

        plt.legend()

    plt.show()
data=pd.read_csv("/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv")

data.sample(frac=1,random_state=69)

data.head()
X=data[data.columns.difference(['target_class'])].to_numpy()

y=data["target_class"].to_numpy()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.svm import SVC



clf = SVC(gamma='auto')

clf.fit(X_train, y_train)

predictions=clf.predict(X_test)

evaluate(y_test,predictions)
from sklearn.feature_selection import SelectKBest

listOfMetrics={}

for i in range(1,9):

    sel = SelectKBest(k=i)

    sel.fit(X_train,y_train)

    X_train_transformed=sel.transform(X_train)

    X_test_transformed=sel.transform(X_test)



    clf = SVC(gamma='auto')

    clf.fit(X_train_transformed, y_train)

    predictions=clf.predict(X_test_transformed)

    newMetrics=evaluate(y_test,predictions)

    metricsKeeper(listOfMetrics,newMetrics,i)

plotMetrics(listOfMetrics)
from sklearn.feature_selection import SelectPercentile



listOfMetrics={}

for i in range(1,9):

    sel = SelectPercentile(percentile=i*10)

    sel.fit(X_train,y_train)

    X_train_transformed=sel.transform(X_train)

    X_test_transformed=sel.transform(X_test)



    clf = SVC(gamma='auto')

    clf.fit(X_train_transformed, y_train)

    predictions=clf.predict(X_test_transformed)

    newMetrics=evaluate(y_test,predictions)

    metricsKeeper(listOfMetrics,newMetrics,i)

plotMetrics(listOfMetrics)
from sklearn.decomposition import PCA



listOfMetrics={}



for i in range(1,9):

    sel = PCA(n_components=i)

    sel.fit(X_train)

    X_train_transformed=sel.transform(X_train)

    X_test_transformed=sel.transform(X_test)



    clf = SVC(gamma='auto')

    clf.fit(X_train_transformed, y_train)

    predictions=clf.predict(X_test_transformed)

    newMetrics=evaluate(y_test,predictions)

    metricsKeeper(listOfMetrics,newMetrics,i)

plotMetrics(listOfMetrics)