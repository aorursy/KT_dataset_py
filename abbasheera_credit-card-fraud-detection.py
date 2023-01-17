# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import sys

#import matplotlib as plt

import matplotlib.pyplot as plt

import seaborn as sns

import scipy

import sklearn





print('Python: {}'.format(sys.version))

print('Numpy: {}'.format(np.__version__))

print('Pandas: {}'.format(pd.__version__))

#print('MatplotLib: {}'.format(plt.__version__))

print('seaborn: {}'.format(sns.__version__))

print('sklearn: {}'.format(sklearn.__version__))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/creditcard.csv")

data.columns
data.shape
data.describe()
data= data.sample(frac=0.1,random_state=1)

data.shape
data.hist(figsize = (40,30))

plt.show
fraud =data[data["Class"]==1]

valid =data[data["Class"]==0]



outlier_fraction = len(fraud) / float(len(valid))

print(outlier_fraction)



print("Fraud Case: {}".format(len(fraud)))

print("Valid Case: {}".format(len(valid)))
#Correlation Matrix

corrmat=data.corr().abs()

plt.figure(figsize=(12,9))

sns.heatmap(corrmat,vmax=0.8,square=True)

plt.show()
data.corr()["Class"].sort_values()
#get all the columns from the dataframe

columns = data.columns.tolist()



#filter the column to remove data dnt want

columns =[c for c in columns if c not in ["Class"]]



#Store the variable we will be predicting 

target ="Class"



X=data[columns]

Y=data[target]



X.shape
Y.shape
from sklearn.metrics import classification_report,accuracy_score

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor
# define a random state

state = 1



# define the outlier detection methods

classifiers = {

        "Isolation Forest": IsolationForest(max_samples=len(X),

                                                   contamination= outlier_fraction,

                                                   random_state = state),

        "Local Outlier Factor ":LocalOutlierFactor(n_neighbors =20,novelty=True,

                                                          contamination=outlier_fraction)

    }
#fit the model

n_outlier=len(fraud)



for i ,(clf_name,clf ) in enumerate(classifiers.items()):

    #fit the data and tag outliers

    if clf_name == "Local Outlier Factor" :

        y_pred = clf.fit_predict(X)

        scores_pred =clf.negative_outlier_factor_

    else:

        clf.fit(X)

        scores_pred = clf.decision_function(X)

        y_pred=clf.predict(X)

    #Reshape the prediction values to 0 for valid , 1 for fraud

    y_pred[y_pred == 1] =0

    y_pred[y_pred == -1] = 1

    

    n_errors = (y_pred != Y).sum()

    

    #Run Classification Mertices

    print('{}: {}'.format(clf_name,n_errors))

    print('accuracy_score: {}'.format(accuracy_score(Y,y_pred)))

    print('Classification Report: {}'.format(classification_report(Y,y_pred)))