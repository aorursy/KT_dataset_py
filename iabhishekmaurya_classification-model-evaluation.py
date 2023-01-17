# Importing necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold

sns.set()
# Reading the Dataset

clas_data = pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')
def data_info(data):

    print('\t\t Data Info:')

    print(clas_data.info())

    print('\n\n\t\t Data Head:')

    print(clas_data.head())

    print('\n\n\t\t Data Describe:')

    print(clas_data.describe())

    print('\n\nData Shape: ',clas_data.shape)

    print('\n\n\t\t Null Values')

    print(clas_data.isna().sum())
data_info(clas_data)
var = 'target'

sns.countplot(clas_data[var])
var = 'sex'

sns.countplot(clas_data[var])
var = 'age'

f, ax = plt.subplots(figsize=(15,8))

sns.distplot(clas_data[var])

plt.xlim([0,80])
var = 'chol'

f, ax = plt.subplots(figsize=(15,8))

sns.distplot(clas_data[var])

plt.xlim([0,600])
var = 'trestbps'

f, ax = plt.subplots(figsize=(15,8))

sns.distplot(clas_data[var])

plt.xlim([0,250])
plt.figure(figsize=(18,18))

sns.heatmap(clas_data.corr(),annot=True,cmap='RdYlGn')



plt.show()
clas_data.columns
X = clas_data.iloc[:,:-1]

y = clas_data.iloc[:,-1]

print("\n\n\t\tIndependent features of Dataset: ")

print(X.head())

print("\n\n\t\tDependent features of Dataset: ")

print(y.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 25)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
from sklearn import metrics

print("Classification Model Accuracy is: ",metrics.accuracy_score(y_test, y_pred))
! pip install -q scikit-plot
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test,y_pred)
from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import plot_precision_recall_curve

average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))

disp = plot_precision_recall_curve(log_reg, X_test, y_test)

disp.ax_.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
from sklearn.metrics import f1_score

print("Macro F1 Score: ",f1_score(y_test, y_pred, average='macro'))

print("Micro F1 Score: ",f1_score(y_test, y_pred, average='micro'))

print("Weighted F1 Score: ",f1_score(y_test, y_pred, average='weighted'))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



plt.plot(fpr, tpr)

plt.title('ROC curve for Heart Attack classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
from sklearn.metrics import classification_report

target_names = ['class 0', 'class 1']

print(classification_report(y_test, y_pred, target_names=target_names))