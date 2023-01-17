# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import sklearn

import scipy

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import classification_report,accuracy_score

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor

from sklearn.svm import OneClassSVM

from pylab import rcParams

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/creditcardfraud/creditcard.csv",sep=',')

data.head()
rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42

LABELS = ['Normal', 'Fraud']
data.info()
#check for null values

data.isnull().any()
count_classes = pd.value_counts(data['Class'], sort = True)



count_classes.plot(kind = 'bar', rot=0)



plt.title("Transaction Class Distribution")



plt.xticks(range(2), LABELS)



plt.xlabel("Class")



plt.ylabel("Frequency")
## Get the Fraud and the normal dataset 



fraud = data[data['Class']==1]



normal = data[data['Class']==0]
print(fraud.shape,normal.shape)
## We need to analyze more amount of information from the transaction data

#How different are the amount of money used in different transaction classes?

fraud.Amount.describe()
normal.Amount.describe()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

f.suptitle('Amount per transaction by class')

bins = 50

ax1.hist(fraud.Amount, bins = bins)

ax1.set_title('Fraud')

ax2.hist(normal.Amount, bins = bins)

ax2.set_title('Normal')

plt.xlabel('Amount ($)')

plt.ylabel('Number of Transactions')

plt.xlim((0, 20000))

plt.yscale('log')

plt.show();

# We Will check Do fraudulent transactions occur more often during certain time frame ? Let us find out with a visual representation.



f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

f.suptitle('Time of transaction vs Amount by class')

ax1.scatter(fraud.Time, fraud.Amount)

ax1.set_title('Fraud')

ax2.scatter(normal.Time, normal.Amount)

ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')

plt.ylabel('Amount')

plt.show()
## Take some sample of the data



data1= data.sample(frac = 0.1,random_state=1)



data1.shape
data.shape
#Determine the number of fraud and valid transactions in the dataset



Fraud = data1[data1['Class']==1]



Valid = data1[data1['Class']==0]



outlier_fraction = len(Fraud)/float(len(Valid))
## Correlation

import seaborn as sns

#get correlations of each features in dataset

corrmat = data1.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#Create independent and Dependent Features

columns = data1.columns.tolist()

# Filter the columns to remove data we do not want 

columns = [c for c in columns if c not in ["Class"]]

# Store the variable we are predicting 

target = "Class"

# Define a random state 

state = np.random.RandomState(42)

X = data1[columns]

Y = data1[target]

X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))

# Print the shapes of X & Y

print(X.shape)

print(Y.shape)
##Define the outlier detection methods



classifiers = {

    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 

                                       contamination=outlier_fraction,random_state=state, verbose=0),

    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 

                                              leaf_size=30, metric='minkowski',

                                              p=2, metric_params=None, contamination=outlier_fraction),

    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 

                                         max_iter=-1, random_state=state)

   

}
type(classifiers)
n_outliers = len(fraud)

for i, (clf_name,clf) in enumerate(classifiers.items()):

    #Fit the data and tag outliers

    if clf_name == "Local Outlier Factor":

        y_pred = clf.fit_predict(X)

        scores_prediction = clf.negative_outlier_factor_

    elif clf_name == "Support Vector Machine":

        clf.fit(X)

        y_pred = clf.predict(X)

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