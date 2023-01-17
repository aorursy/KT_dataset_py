import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report, accuracy_score

from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import confusion_matrix

from sklearn import preprocessing

from sklearn.metrics import log_loss

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

import xgboost as xg_boost

from xgboost import plot_tree

from xgboost import XGBClassifier

from pylab import rcParams

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVC

import statsmodels.api as sm

from sklearn import linear_model

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
data_train = pd.read_csv('../input/forest_data.csv')

data_test = pd.read_csv('../input/forest_data_teste.csv')
data_train.head()
data_test.head()
x_train = data_train.drop(['Label'], 1)

y_train = pd.Series(LabelEncoder().fit_transform(data_train['Label']))



x_test = data_test.drop(['Label'], 1)

y_test = pd.Series(LabelEncoder().fit_transform(data_test['Label']))
data_train.isna().sum()
print("Numero de linhas e colunas no dataset : ",data_train.shape)
print("Numero de linhas e colunas no dataset : ",data_test.shape)
counts = data_train['Label'].value_counts()

print(counts)
X=np.array(data_train.drop(['Label'], 1))

y=np.array(data_train.Label)
classifier = RandomForestClassifier(n_estimators=200, random_state=0)

classifier.fit(x_train, y_train)
probRF=classifier.predict_proba(x_test)

print('Score:', classifier.score(x_test,y_test))

print('LogLoss: ', log_loss(y_test,probRF))
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

from sklearn import metrics

## Confusion Matrix

cnf_matrix = metrics.confusion_matrix(y_test, classifier.predict(x_test))

class_names=[0, 1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Predicted label')

plt.xlabel('Atual label')
model = xg_boost.XGBClassifier()

model.fit(x_train, y_train)
probXGB=model.predict_proba(x_test)

#probXGB = probXGB[:, 1] #Segunda coluna até o final dela;

print('Score:', model.score(x_test,y_test))

print('LogLoss: ',log_loss(y_test,probXGB))
## Confusion Matrix

cnf_matrix = metrics.confusion_matrix(y_test, model.predict(x_test))

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Predicted label')

plt.xlabel('Atual label')
gb=GradientBoostingClassifier(n_estimators=50)

gb.fit(x_train,y_train)
probGB=gb.predict_proba(x_test)

#probGB = probGB[:, 1]

print('Score:', gb.score(x_test,y_test))

print('LogLoss: ',log_loss(y_test,probGB))
## Confusion Matrix

cnf_matrix = metrics.confusion_matrix(y_test, gb.predict(x_test))

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Predicted label')

plt.xlabel('Atual label')
naive=GaussianNB()

naive.fit(x_train,y_train)
probNV=naive.predict_proba(x_test)

print('Score:',naive.score(x_test,y_test))

print('LogLoss: ',log_loss(y_test,probNV))
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

from sklearn import metrics

## Confusion Matrix

cnf_matrix = metrics.confusion_matrix(y_test, naive.predict(x_test))

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
knn=KNeighborsClassifier (n_neighbors=3)

knn.fit(x_train,y_train)
probKNN=knn.predict_proba(x_test)

print('Score:',knn.score(x_test,y_test))

print('LogLoss: ',log_loss(y_test,probKNN))
## Confusion Matrix

cnf_matrix = metrics.confusion_matrix(y_test, knn.predict(x_test))

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Predicted label')

plt.xlabel('Atual label')
logisticRegr = LogisticRegression()

result = logisticRegr.fit(x_train, y_train)
probLR=result.predict_proba(x_test)

print('Score:',result.score(x_test,y_test))

print('LogLoss: ',log_loss(y_test,probLR))
## Confusion Matrix

cnf_matrix = metrics.confusion_matrix(y_test, result.predict(x_test))

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')