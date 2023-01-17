import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl



from sklearn.metrics import silhouette_score, silhouette_samples

import sklearn.metrics

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from sklearn.mixture import GaussianMixture

import seaborn as sns

import itertools

import scipy



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train.head()
train.info()
test.head()
sns.heatmap(train.isnull(),yticklabels=False, cbar=False,cmap='viridis')
sns.countplot(x='Survived', data=train, hue='Sex')
train['Parch'].value_counts()
sns.countplot(x='Survived', data=train, hue='Pclass')
sns.distplot(train['Age'].dropna(), kde=False, bins=30)
sns.countplot(x='SibSp', data=train)
train['Fare'].hist(bins=40)
def impute_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        if Pclass==1: 

            return 37

        elif Pclass==2:

            return 29

        else:

            return 24

    else:

        return Age

        
train['Age']=train[['Age', 'Pclass']].apply(impute_age,axis=1)
train['Age']=np.log10(train['Age'])
train['Fare']=np.log10(train['Fare']+10)
sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')
train.drop(['Cabin'],axis=1,inplace=True)
train.dropna(inplace=True)
sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')
dums=pd.get_dummies(train[['Sex', 'Embarked']], drop_first=True)
train=pd.concat([train,dums], axis=1)
train.head()
train.drop(['Sex', 'Embarked','Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
train.head()
f = plt.figure(figsize=(19, 15))

plt.matshow(train.corr(), fignum=f.number)

plt.xticks(range(train.shape[1]), train.columns, fontsize=14, rotation=45)

plt.yticks(range(train.shape[1]), train.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
X=train.drop(['Survived'], axis=1)

Y=train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3,random_state=176)
from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier(random_state=42, criterion="entropy",

                             min_samples_split=10, min_samples_leaf=10, max_depth=3, max_leaf_nodes=5)

clf.fit(X_train, Y_train)



y_pred_dt = clf.predict(X_test)
feature_names = X.columns

class_names = [str(x) for x in clf.classes_]
print(clf.tree_.node_count)

print(clf.tree_.impurity)

print(clf.tree_.children_left)

print(clf.tree_.threshold)
from sklearn.metrics import confusion_matrix



confusion_matrix(Y_test, y_pred_dt)
from sklearn.metrics import classification_report


print(classification_report(Y_test, y_pred_dt, target_names=class_names))
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss, roc_curve, auc



print("Accuracy = {:.2f}".format(accuracy_score(Y_test, y_pred_dt)))

print("Kappa = {:.2f}".format(cohen_kappa_score(Y_test, y_pred_dt)))

print("F1 Score = {:.2f}".format(f1_score(Y_test, y_pred_dt)))

print("Log Loss = {:.2f}".format(log_loss(Y_test, y_pred_dt)))
def plot_roc(clf, X_test, y_test, name, ax, show_thresholds=True):

    y_pred_rf = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, thr = roc_curve(y_test, y_pred_rf)



    ax.plot([0, 1], [0, 1], 'k--');

    ax.plot(fpr, tpr, label='{}, AUC={:.2f}'.format(name, auc(fpr, tpr)));

    ax.scatter(fpr, tpr);



    if show_thresholds:

        for i, th in enumerate(thr):

            ax.text(x=fpr[i], y=tpr[i], s="{:.2f}".format(th), fontsize=14, 

                     horizontalalignment='left', verticalalignment='top', color='black',

                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1', alpha=0.1));

        

    ax.set_xlabel('False positive rate', fontsize=18);

    ax.set_ylabel('True positive rate', fontsize=18);

    ax.tick_params(axis='both', which='major', labelsize=18);

    ax.grid(True);

    ax.set_title('ROC Curve', fontsize=18)
plt.style.use('default');

figure = plt.figure(figsize=(10, 6));    

ax = plt.subplot(1, 1, 1);

plot_roc(clf, X_test, Y_test, "Decision Tree", ax)

plt.legend(loc='lower right', fontsize=18);

plt.tight_layout();
test['Age']=test[['Age', 'Pclass']].apply(impute_age,axis=1)
test['Age']=np.log10(test['Age'])
test['Fare']=np.log10(test['Fare']+10)
sns.heatmap(test.isnull(),yticklabels=False, cbar=False, cmap='viridis')
test.drop(['Cabin'],axis=1,inplace=True)
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
sns.heatmap(test.isnull(),yticklabels=False, cbar=False, cmap='viridis')
dumst=pd.get_dummies(test[['Sex', 'Embarked']], drop_first=True)
test=pd.concat([test,dumst], axis=1)
test.drop(['Sex', 'Embarked','Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
test.head()
predictions1=clf.predict(test)
predictions1
dataset1 = pd.DataFrame(predictions1)
dataset1.to_csv("DT1.csv")