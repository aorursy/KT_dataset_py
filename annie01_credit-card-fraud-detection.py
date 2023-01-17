# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
data.head()
data.describe()
data.columns
data.info()
data.shape
#check for missing values
data.isnull().sum().sum()
import seaborn as sns
print(data['Class'].value_counts())
sns.set_style("darkgrid")
sns.countplot(data['Class']);
fraud_data = data[data['Class']==1]
genuine_data = data[data['Class']==0]
fraud_data.Amount.describe()
genuine_data.Amount.describe()
import matplotlib.pyplot as plt
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
f.suptitle('Amount per transaction by class')

ax1.hist(fraud_data.Amount, 100)
ax1.set_title('Fraud')
ax2.hist(genuine_data.Amount, 100)
ax2.set_title('Genuine')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();
#scaling the amount column using a standard scaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
data["Amount"]=scaler.fit_transform(np.array(data["Amount"]).reshape(-1,1))
# We Will check Do fraudulent transactions occur more often during certain time frame ? Let us find out with a visual representation.

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud_data.Time, fraud_data.Amount)
ax1.set_title('Fraud')
ax2.scatter(genuine_data.Time, genuine_data.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()
#splitting data for training and testing
import sklearn
from sklearn.model_selection import train_test_split
x = data.drop("Class", axis=1)
y = data['Class']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 100)
from sklearn.svm import LinearSVC
clf=LinearSVC()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
from sklearn.metrics import confusion_matrix
import mlxtend 

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("F1 score:",metrics.f1_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics

#splitting data for training and testing
x = data.drop("Class", axis=1)
y = data['Class']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 100)

#Classifying data using LinearSVC
clf=LinearSVC()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

# Performance Evaluation
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#SMOTE
from collections import Counter
from imblearn.over_sampling import SMOTE

x_resampled, y_resampled = SMOTE().fit_sample(x_train, y_train)
print(sorted(Counter(y_resampled).items()))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
clf=LogisticRegression(solver='saga')
clf.fit(x_resampled,y_resampled)
y_pred=clf.predict(x_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("F1 Score:",metrics.f1_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
#plot_confusion_matrix(clf, x_test, y_test) 
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, y_pred),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()
from sklearn.svm import LinearSVC
clf=LinearSVC(max_iter = 190) 
clf.fit(x_resampled,y_resampled)
y_pred=clf.predict(x_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("F1 Score:",metrics.f1_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
#plot_confusion_matrix(clf, x_test, y_test) 
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, y_pred),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()
clf=RandomForestClassifier()

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_resampled,y_resampled)
y_pred=clf.predict(x_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("F1 Score:",metrics.f1_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, y_pred),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()
# Separate input features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) # 80% training and 20% test
clf=RandomForestClassifier()

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("F1 Score:",metrics.f1_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, y_pred),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()