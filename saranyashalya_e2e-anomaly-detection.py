import numpy as np

import pandas as pd

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor

from sklearn.svm import OneClassSVM

from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.graph_objs as go

import chart_studio.plotly as py

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode, iplot



init_notebook_mode(connected = True)
data = pd.read_csv("../input/creditcardfraud/creditcard.csv")
data.columns
data.head()
data.describe()
data.shape
data.isnull().values.any()
LABELS =['Normal','Fraud']

count_classes = pd.value_counts(data['Class'], sort = True)

count_classes.plot(kind ='bar',rot =0)

plt.title('Transaction Class Distribution')

plt.xticks(range(2),LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")
Fraud = data[data['Class']==1]

Normal = data[data['Class']==0]
Fraud.shape, Normal.shape
Fraud.Amount.describe()
Normal.Amount.describe()
f,(ax1,ax2) = plt.subplots(2,1, sharex = True)

f.suptitle('Amount per transaction by class')

bins = 50

ax1.hist(Fraud.Amount, bins = bins)

ax1.set_title("Fraud Amount distribution")

ax2.hist(Normal.Amount, bins = bins)

ax2.set_title("Normal Amount distribution")

plt.xlabel("Amount($)")

plt.ylabel("Number of transactions")

plt.xlim(0,20000)

plt.yscale('log')

plt.show()
f,(ax1,ax2) = plt.subplots(2,1, sharex = True)

f.suptitle("Distribution of Amount vs time")

ax1.scatter(Fraud.Time, Fraud.Amount)

ax1.set_title("Fraud - Time vs Amount")

ax2.scatter(Normal.Time, Normal.Amount)

ax2.set_title("Normal - Time vs Amount")

plt.xlabel("Time")

plt.ylabel("Amount")

plt.show()
trace = go.Scatter(

            x = Fraud.Time,

            y = Fraud.Amount,

            mode = 'markers')

data =[trace]
iplot({'data':data})
data = pd.read_csv("../input/creditcardfraud/creditcard.csv")

data1 = data.sample(frac = 0.1, random_state=1)

data1.shape
data1.hist(figsize =(20,20))

plt.show()
Fraud = data1[data1['Class']==1]

Valid = data1[data1['Class']==0]

outlier_fraction = len(Fraud)/float(len(Valid))
outlier_fraction
len(Fraud), len(Valid)
import seaborn as sns

correlation_matrix = data1.corr()

fig = plt.figure(figsize =(15,8))

sns.heatmap(correlation_matrix, vmax =0.8, square = True)

plt.show()
X = data1.drop(['Class'],axis =1)

Y= data1.Class
import scipy

state = np.random.RandomState(42)

X_outliers = state.uniform(low=0 , high=1, size =(X.shape[0], X.shape[1]))

X.shape, Y.shape
#Model prediction

classifiers = {

    'Isolation Forest': IsolationForest(n_estimators = 100, max_samples = len(X),contamination = outlier_fraction,random_state = state, verbose =0), #Contamination - The amount of contamination of the data set, i.e. the proportion of outliers in the data set

    'Local Outlier Factor' : LocalOutlierFactor(n_neighbors = 20, algorithm ='auto',leaf_size =30, metric = 'minkowski',p=2, metric_params = None, contamination=outlier_fraction),

    'Support Vector Machine' : OneClassSVM(kernel ='rbf',degree=3, gamma =0.1, nu =0.05, max_iter =-1,random_state = state)

}
n_outliers = len(Fraud)

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