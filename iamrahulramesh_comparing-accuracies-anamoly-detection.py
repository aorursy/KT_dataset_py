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
import sklearn

import scipy

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import classification_report,accuracy_score

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor

from sklearn.svm import OneClassSVM



from pylab import rcParams

rcParams['figure.figsize'] =14,8

RANDOM_SEED = 36

LABELS =['Normal','Fraud']

data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

data.head()
data.info()
data.isnull().values.any()
cp = sns.countplot(x ='Class',data = data)

plt.xticks(np.arange(2),LABELS)

plt.xlabel('Class')

plt.ylabel('Count')

plt.title('Normal vs Fraud Transaction Count')

plt.legend(loc ='best')

plt.show()
normal =data[data['Class'] ==0]

fraud =data[data['Class'] ==1]
print(normal.shape,fraud.shape)
fraud.Amount.describe()
normal.Amount.describe()
f, (ax1,ax2) = plt.subplots(2,1, sharex = True)

f.suptitle('Amount of transaction by class')

bins = 50

ax1.hist(fraud.Amount,bins = bins)

ax1.set_title('Fraud')

ax2.hist(normal.Amount,bins = bins)

ax2.set_title('Normal')

plt.xlabel('Amount in $')

plt.ylabel('Number of Transactions')

plt.xlim((0,20000))

plt.yscale('log') # for viewing the data well in histogram
f, (ax1,ax2) = plt.subplots(2,1,sharex = True)

f.suptitle('Time of transaction vs amount by class')

ax1.scatter(fraud.Time,fraud.Amount)

ax1.set_title('Fraud')

ax1.set_xlabel('Time ( in seconds)')

ax1.set_ylabel('Amount')

ax2.scatter(normal.Time,normal.Amount)

ax2.set_title('Normal')

ax2.set_xlabel('Time ( in seconds)')

ax2.set_ylabel('Amount')
data1 = data.sample(frac = 0.1, random_state = 1)

data1.shape
data.shape
normal1 = data1[data1['Class'] ==0]

fraud1 = data1[data1['Class'] ==1]



print(normal1.shape,fraud1.shape)
fraud_fraction = len(fraud1)/float(len(normal1))

print(fraud_fraction)
print('Number of Fraud cases : {}'.format(len(fraud1)))

print('Number of Normal cases: {}'.format(len(normal1)))

print('Fraud percentage : {} %'.format(round(fraud_fraction*100,3)))
core = data1.corr()

top_features = core.index

plt.figure(figsize =(20,20))



h = sns.heatmap(data[top_features].corr(),annot = True, cmap = 'coolwarm')
columns =data1.columns.tolist()

columns = [c for c in columns if c not in ['Class']]



target = 'Class'



state = np.random.RandomState(36)



X = data1[columns]

y = data1[target]



print(X.shape)

print(y.shape)

n_outliers = len(fraud)

n_outliers
#Using Isolation Forest

IF =IsolationForest(n_estimators=100, max_samples=len(X), 

                                       contamination=fraud_fraction,random_state=state, verbose=0) 

clf = IF.fit(X)

scores_prediction = clf.decision_function(X)

y_pred = clf.predict(X)

y_pred[y_pred == 1] = 0

y_pred[y_pred == -1] = 1

n_errors = (y_pred != y).sum()



print("Isolation Forest: {}".format(n_errors))

print("Accuracy Score :")

print(accuracy_score(y,y_pred))

print("Classification Report :")

print(classification_report(y,y_pred))
#Using Local Outlier Factor algorithm



LOF = LocalOutlierFactor(n_neighbors=20, algorithm='auto', 

                                              leaf_size=30, metric='minkowski',

                                              p=2, metric_params=None, contamination=fraud_fraction)

y_pred =LOF.fit_predict(X)





y_pred[y_pred == 1] = 0

y_pred[y_pred == -1] = 1

n_errors = (y_pred != y).sum()



print("LocalOutlierFactor: {}".format(n_errors))

print("Accuracy Score :")

print(accuracy_score(y,y_pred))

print("Classification Report :")

print(classification_report(y,y_pred))
#Using Support Vector machine

from sklearn.ensemble import BaggingClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC



n_estimators = 10



ORC = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'))



clf2 = ORC.fit(X,y)



y_pred = clf2.predict(X)



y_pred[y_pred == 1] = 0

y_pred[y_pred == -1] = 1

n_errors = (y_pred != y).sum()



print("OneVsRestClassifier: {}".format(n_errors))

print("Accuracy Score :")

print(accuracy_score(y,y_pred))

print("Classification Report :")

print(classification_report(y,y_pred))