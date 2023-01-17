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
#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
data_sets = load_digits()
data_sets
X,y = data_sets.data,data_sets.target
for class_name,class_count in zip(data_sets.target_names, np.bincount(data_sets.target)):
    print(class_name,class_count)
y_binary_imbalanced = y.copy()
y_binary_imbalanced[y_binary_imbalanced != 1] = 0

print('Original labels:\t', y[1:30])
print('New binary labels:\t', y_binary_imbalanced[1:30])
np.bincount(y_binary_imbalanced)
from sklearn.svm import SVC
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y_binary_imbalanced,
                                                 random_state=0)
svm = SVC(kernel='rbf',C=1)
svm.fit(X_train,y_train)
svm.score(X_test,y_test)
from sklearn.dummy import DummyClassifier
# since negative class 0 is most frequent.
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train,y_train)
dummy.score(X_test,y_test)
dummy_prediction = dummy.predict(X_test)
dummy_prediction
svm = SVC(kernel='linear',C=1)
svm.fit(X_train,y_train)
svm.score(X_test,y_test)
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
#negative class 0 is most frequent
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train,y_train)
dummy_predicted = dummy.predict(X_test)
confusion = confusion_matrix(y_test,dummy_predicted)
plot_confusion_matrix(dummy,X_test,y_test,cmap=plt.cm.Blues)

print('Most frequent class (dummy classifier):\n',confusion)

# produces random predictions w/ same class proportion as training set
dummy = DummyClassifier(strategy='stratified')
dummy.fit(X_train,y_train)
dummy_predicted = dummy.predict(X_test)
confusion = confusion_matrix(y_test,dummy_predicted)
plot_confusion_matrix(dummy,X_test,y_test,cmap=plt.cm.Blues)

print('Random class proportional Prediction (dummy classifier):\n',confusion)
svm = SVC(kernel='linear',C=1)
svm.fit(X_train,y_train)
svm_predict = svm.predict(X_test)
confusion = confusion_matrix(y_test,svm_predict)
plot_confusion_matrix(svm,X_test,y_test)

print('Support vector machine classifier:\n',confusion)
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train,y_train)
lg_pred = lg.predict(X_test)
confusion = confusion_matrix(y_test,lg_pred)
plot_confusion_matrix(lg,X_test,y_test,cmap=plt.cm.Blues)
print('Logistic Regression Classifier:\n',confusion)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=2)
dt.fit(X_train,y_train)
dt_pred = dt.predict(X_test)
confusion = confusion_matrix(y_test,dt_pred)
plot_confusion_matrix(dt,X_test,y_test)

print('Decision Tree Classifier:\n',confusion)
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
# Accuracy = TP + TN / (TP + TN + FP + FN)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate
# F1 = 2 * Precision * Recall / (Precision + Recall) 

print('Accuracy: {:.2f}'.format(accuracy_score(y_test,dt_pred)))
print('Precision: {:.2f}'.format(precision_score(y_test,dt_pred)))
print('recall: {:.2f}'.format(recall_score(y_test,dt_pred)))
print('f1: {:.2f}'.format(f1_score(y_test,dt_pred)))
from sklearn.metrics import classification_report
print('Random class proportional dummy\n:',
     classification_report(y_test,dummy_predicted,target_names=['Not 1','1']))
print('Svm\n:',
     classification_report(y_test,svm_predict,target_names=['Not 1','1']))
print('Logistic Regression\n:',
     classification_report(y_test,lg_pred,target_names=['Not 1','1']))
print('Decision Tree\n:',
     classification_report(y_test,dt_pred, target_names=['Not 1','1']))