import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression #LogisticRegression

from sklearn.ensemble import RandomForestClassifier #RandomForest

from sklearn.tree import DecisionTreeClassifier #DecisionTree

from sklearn.naive_bayes import GaussianNB #Naive Bayes

import xgboost as xgb #XGBoost



from sklearn import tree

import graphviz
test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv', delimiter=',')

test.head()
sample_submission = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv', delimiter=',')

sample_submission.head()
train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv', delimiter=',')

train.head()
test.info()
test.describe()
train.info()
train.describe()
train.shape, test.shape, sample_submission.shape
new_train = train.copy()

new_train = new_train.iloc[:,:2]

new_train_without_target = train.copy()

new_train_without_target.drop(["ID_code", "target"], inplace=True, axis=1)

new_train['var_mean'] = new_train_without_target.mean(axis=1)

new_train.head()
colors = ['palegreen','salmon']

plt.figure(figsize=(7,7))

plt.pie(train["target"].value_counts(), explode=(0, 0.25), labels= ["0", "1"], startangle=225, autopct='%1.1f%%', colors=colors)

plt.axis('equal')

plt.show()
sns.countplot(train['target'], palette='Set2')
f,ax=plt.subplots(1,2,figsize=(15,7))

new_train[new_train['target']==0].var_mean.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='gray')

ax[0].set_title('target= 0')

x1=list(range(3,12,1))

ax[0].set_xticks(x1)

new_train[new_train['target']==1].var_mean.plot.hist(ax=ax[1],color='purple',bins=20,edgecolor='black')

ax[1].set_title('target= 1')

x2=list(range(3,12,1))

ax[1].set_xticks(x2)

plt.show()
sns.distplot(new_train['var_mean'])
X = train.iloc[:,2:]

y = train.loc[:, train.columns == 'target']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 1,stratify =y)
logreg = LogisticRegression().fit(X_train, y_train)

y_pred_logreg = logreg.predict(X_test)
print("Confusion matrix:\n", confusion_matrix(y_test,y_pred_logreg))

print("\n\nLogistic Regression model accuracy(in %): ", accuracy_score(y_test, y_pred_logreg)*100)

print("\n\nClassification report:\n", classification_report(y_test, y_pred_logreg))
gnb = GaussianNB().fit(X_train, y_train)

y_pred_gnb = gnb.predict(X_test)
print("Confusion matrix:\n", confusion_matrix(y_test,y_pred_gnb))

print("\n\nNaive Bayes model accuracy(in %): ", accuracy_score(y_test, y_pred_gnb)*100)

print("\n\nClassification report:\n", classification_report(y_test, y_pred_gnb))
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(X_train, y_train)

features = [c for c in train.columns if c not in ['ID_code', 'target']]

tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=features)

graphviz.Source(tree_graph)
y_pred_tree = tree_model.predict(X_test)
print("Confusion matrix:\n", confusion_matrix(y_test,y_pred_tree))

print("\n\nDecision Tree model accuracy(in %): ", accuracy_score(y_test, y_pred_tree)*100)

print("\n\nClassification report:\n", classification_report(y_test, y_pred_tree))
rf = RandomForestClassifier(n_estimators=100,

                                       bootstrap = True,

                                       criterion='entropy').fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Confusion matrix:\n", confusion_matrix(y_test,y_pred_rf))

print("\n\nRandom Forest model accuracy(in %): ", accuracy_score(y_test, y_pred_rf)*100)

print("\n\nClassification report:\n", classification_report(y_test, y_pred_rf))
xgb_model = xgb.XGBClassifier().fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
print("Confusion matrix:\n", confusion_matrix(y_test,y_pred_xgb))

print("\n\nXGBoost model accuracy(in %): ", accuracy_score(y_test, y_pred_xgb)*100)

print("\n\nClassification report:\n", classification_report(y_test, y_pred_rf))
roc_auc_score(y_test, y_pred_logreg)