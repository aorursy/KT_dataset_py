# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv",sep = ',')

data.head()
data.info()
data.isnull().values.any()
count_classes = pd.value_counts(data['Class'], sort = True)
count_classes.plot(kind ='bar', rot = 0)
plt.title("Transaction class distribution")
plt.xticks(range(2))
plt.xlabel("Class")
plt.ylabel("Frequency")

fraud = data[data['Class']==1]
normal = data[data['Class']==0]
print(fraud.shape)
print(normal.shape)
fraud.Amount.describe()
normal.Amount.describe()
f, (ax1,ax2) = plt.subplots(2,1,sharex = True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount($)')
plt.ylabel('Number of transactions')
plt.xlim(0,20000)
plt.yscale('log')
plt.show();


f, (ax1,ax2) = plt.subplots(2,1,sharex = True)
f.suptitle('Time Vs Amount')
ax1.scatter(fraud.Time,fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time,normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.show();

data1 = data.sample(frac = 0.1, random_state=1)
data1.shape

data.shape
Fraud = data1[data1['Class']==1]
Valid = data1[data1['Class']==0]
outlier_fraction= len(Fraud)/float(len(Valid))
print(outlier_fraction)
print("Fraud Cases : {}".format(len(Fraud)))
print("Valid Cases : {}".format(len(Valid)))
import seaborn as sns
corrmat = data1.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20,20))
g = sns.heatmap(data[top_corr_features].corr(),annot = True, cmap="RdYlGn")
columns = data1.columns.tolist()
columns = [c for c in columns if c not in ['Class']]
target = 'Class'
state = np.random.RandomState(42)
X = data1[columns]
Y = data1[target]
X_outliers = state.uniform(low =0, high =1, size=(X.shape[0],X.shape[1]))
print(X.shape)
print(Y.shape)
classifiers = {
    "Isolation Forest": IsolationForest(n_estimators=100,max_samples=len(X), contamination = outlier_fraction,random_state=state,verbose=0),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, algorithm='auto',leaf_size=30,metric ='minkowski', p=2, metric_params=None, contamination = outlier_fraction),
    "Support Vector Machine": OneClassSVM(kernel ='rbf', degree=3 , gamma=0.1,nu=0.05,max_iter=-1)
    
}
n_outliers=len(Fraud)
for i,(clf_name,clf) in enumerate(classifiers.items()):
    if clf_name == 'Local Outlier Factor':
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    n_errors = (y_pred != Y).sum()
    print("{}:{}".format(clf_name,n_errors))
    print("Accuracy Score:")
    print(accuracy_score(Y,y_pred))
    print("Classification Report:")
    print(classification_report(Y,y_pred))
