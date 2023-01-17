# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

import scipy
data=pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.head()
data.columns
data.shape
data.describe()
data.hist(figsize=(20,20))

plt.show()
Valid= data[data['Class']==0]

Fraud= data[data['Class']==1]



print("Fraud cases {}".format(len(Fraud)))

print("Valid cases {}".format(len(Valid)))
outlier_fraction= len(Fraud)/ float(len(Valid))

print(outlier_fraction)
plt.figure(figsize=(18,12))

sns.heatmap(data.corr(),annot=True,fmt='.0%')
columns=data.columns.tolist()



#filter the columns that we don't want

columns= [c for c in columns if c not in ['Class']] #Class column will be removed



Target= "Class"

X= data[columns]

Y=data[Target]



print(X.shape)

print(Y.shape)
from sklearn.metrics import classification_report,accuracy_score

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor

from sklearn.svm import OneClassSVM



State=1



Classifiers={

    "Isolation Forest": IsolationForest(max_samples=len(X),

                                       contamination=outlier_fraction,

                                        random_state=State),

    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20,

                                            contamination=outlier_fraction)

                                               

                                

    }
n_outliers = len(Fraud)

for i, (clf_name,clf) in enumerate(Classifiers.items()):

    #Fit the data and tag outliers

    if clf_name == "Local Outlier Factor":

        y_pred = clf.fit_predict(X)

        scores_prediction = clf.negative_outlier_factor_

    else:    

        clf.fit(X)

        scores_prediction = clf.decision_function(X)

        y_pred = clf.predict(X)

    #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions

    y_pred[y_pred == 1] = 0

    y_pred[y_pred == -1] = 1

    n_errors = (y_pred != Y).sum()

    # Run Classification Metrics

    print("{}: {}".format(clf_name,n_errors))

    print("Accuracy Score :")

    print(accuracy_score(Y,y_pred))

    print("Classification Report :")

    print(classification_report(Y,y_pred))
