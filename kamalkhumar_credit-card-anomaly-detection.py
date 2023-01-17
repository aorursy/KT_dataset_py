# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df.sample(10)
df.shape
#Description of DataFrame
df.describe(include = "all")
#See if missing values are present
df.isna().sum()
#How many types of values are present
print(pd.value_counts(df['Class']))
LABELS = ["Normal", "Fraud"]
#Display plot of value counts
class_counts = pd.value_counts(df['Class'], sort = True)
class_counts.plot(kind = 'bar', rot=0, color=['blue','red']) #Added colors as a list 
plt.title('Counts of Fraud/Normal')
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Count")
fraud = df[df["Class"]==1]

normal = df[df["Class"]==0]
print(fraud.shape, normal.shape)
#Use describe to look at how much the range of values are in the entire DataFrames
#fraud.describe()
#For specific column add column name like so
fraud.Amount.describe()
normal.Amount.describe()
#Subplots can be used to add multiple plots in the same window
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.suptitle("Amount per class", fontsize='large')
bins = 50 #Best view possible

#Specify which axis where you are going to plot
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraudulent amount')
#ax1.set_ylabel('Number of Transactions') Line shortened

ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal amount')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
#plt.xlim((0, 20000))
plt.yscale('log') # Used to better view the second plot
plt.show()
df_sample = df.sample(frac=0.1, random_state=1)

X = df_sample[df.columns.tolist()]
y = df_sample["Class"]

outlier_fraction = len(fraud)/ float(len(normal))
state = np.random.RandomState(42)
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=outlier_fraction, random_state=state, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                         max_iter=-1) #Random state not in OneClassSVM
   
}
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
    n_errors = (y_pred != y).sum()
    # Run Classification Metrics
    print("{}: {}".format(clf_name,n_errors))
    print("Accuracy Score :")
    print(accuracy_score(y,y_pred))
    print("Classification Report :")
    print(classification_report(y,y_pred))