# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set_style('darkgrid')



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



#from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

#from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

#from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split, validation_curve, cross_val_predict

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, roc_auc_score, make_scorer

from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA

import time



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
submission = pd.read_csv("../input/santander-customer-transaction-prediction/sample_submission.csv")

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
submission.head(10)
train.head(10)
test.head(10)
train.describe()
train.info()
test.info()
train.columns
test.columns
train.isna().sum()
test.isna().sum()
columns = train.columns

percent_missing = train.isnull().sum() * 100 / len(train)

missing_value_data = pd.DataFrame({'column_name': columns,

                                 'percent_missing': percent_missing})

missing_value_data.sort_values('percent_missing')
columnss = test.columns

percent_missing = test.isnull().sum() * 100 / len(test)

missing_value_data = pd.DataFrame({'column_name': columnss,

                                 'percent_missing': percent_missing})

missing_value_data.sort_values('percent_missing')
sns.heatmap(train.corr())

train.corr()
train_correlations = train.drop(["target"], axis=1).corr()

train_correlations = train_correlations.values.flatten()

train_correlations = train_correlations[train_correlations != 1]



test_correlations = test.corr()

test_correlations = test_correlations.values.flatten()

test_correlations = test_correlations[test_correlations != 1]



plt.figure(figsize=(20,5))

sns.distplot(train_correlations, color="Red", label="train")

sns.distplot(test_correlations, color="Yellow", label="test")

plt.xlabel("Correlation values found in train (except 1)")

plt.ylabel("Density")

plt.title("Are there correlations between features?"); 

plt.legend();
features = train.columns.values[2:202]

unique_max_train = []

unique_max_test = []

for feature in features:

    values = train[feature].value_counts()

    unique_max_train.append([feature, values.max(), values.idxmax()])

    values = test[feature].value_counts()

    unique_max_test.append([feature, values.max(), values.idxmax()])
np.transpose((pd.DataFrame(unique_max_train, columns=['Feature', 'Max duplicates', 'Value'])).\

            sort_values(by = 'Max duplicates', ascending=False).head(15))
train.shape, test.shape, submission.shape
train.target.dtype
train.target = train.target.astype(np.int32)

train.target.dtype
f,ax=plt.subplots(1,2, figsize=(12,4))

train.target.value_counts().plot.pie(explode=[0,0.12],autopct='%1.3f%%',ax=ax[0])

sns.countplot(train['target'])

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ax = sns.distplot(train[train['target']==0].var_10, bins = 30, ax = axes[0], kde = False)

ax.set_title('0')

ax = sns.distplot(train[train['target']==1].var_10, bins = 30, ax = axes[1], kde = False)

ax.set_title('1')
f,ax = plt.subplots(1,2,figsize=(18,10))

zero = train.loc[train["target"] == 0]

one = train.loc[train["target"] == 1]

k = sns.kdeplot(zero.var_10,zero.var_100,shade=True,shade_lowest=True,cmap="magma",ax=ax[0])

k = sns.kdeplot(one.var_10,one.var_100,shade=True,shade_lowest=True,cmap="plasma_r",ax=ax[1])

ax[0].set_xlabel("var_10", fontsize=18)

ax[1].set_xlabel("var_10", fontsize=18)

ax[0].set_ylabel("var_100", fontsize=18)

ax[1].set_ylabel("var_100", fontsize=18)

ax[0].set_title("Zero", fontsize=18)

ax[1].set_title("One", fontsize=18)

plt.show()
f,ax = plt.subplots(1,2,figsize=(12,12))

n = sns.scatterplot(train["target"],train["var_20"],hue=train["target"],s=100,ax=ax[0])

n = sns.barplot(train["target"],train["var_20"],ax=ax[1])

plt.show()
plt.figure(figsize=(18,18))

n = sns.pairplot(train[["var_20","var_40","var_60","var_80","var_100","var_150","target"]],hue="target",palette="husl")

plt.show()
plt.figure(figsize=(18,18))

n = sns.pairplot(train[["var_10","var_30","var_50","var_70","var_90","var_180","target"]],hue="target",palette="husl")

plt.show()
X, y = train.iloc[:,2:], train.iloc[:,1]
scale = MinMaxScaler()

scale.fit(X)

features = pd.DataFrame(scale.transform(X))
X
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# 1-Nearest Neighbor Classifier

#NeNeClassifier = KNeighborsClassifier(n_neighbors=3)

#-------------------------------------------------------------------

# 2-Logistic Regression

LRClassifer = LogisticRegression()

#-------------------------------------------------------------------

# 3-SVM Sigmoid Classifier

#SVMClassifer = SVC(kernel='sigmoid')

#-------------------------------------------------------------------

# 4-Naive Bayes Classifier

NBClassifer = GaussianNB()

#-------------------------------------------------------------------

# 5-Decision Tree Classifier

DTClassifer = DecisionTreeClassifier(min_impurity_split=2, min_samples_leaf=9, random_state=25)

#-------------------------------------------------------------------

# 6-Random Forest Classifier

#RFClassifer = RandomForestClassifier(n_estimators=300, random_state=10)

#-------------------------------------------------------------------

# 6-XGBoost Classifier

XGBClassifer = XGBClassifier(n_estimators=100)
#NeNeClassifier.fit(X_train, y_train)

LRClassifer.fit(X_train, y_train)

#SVMClassifer.fit(X_train, y_train)
NBClassifer.fit(X_train, y_train)
DTClassifer.fit(X_train, y_train)
#RFClassifer.fit(X_train, y_train)
XGBClassifer.fit(X_train, y_train)
#y_preds = NeNeClassifier.predict(X_test)

y_preds1 = LRClassifer.predict(X_test)

#y_preds2 = SVMClassifer.predict(X_test)
y_preds3 = NBClassifer.predict(X_test)
y_preds4 = DTClassifer.predict(X_test)
#y_preds5 = RFClassifer.predict(X_test)
y_preds6 = XGBClassifer.predict(X_test)
print(y_preds1[:10],'\n',y_test[:10])

print("*******************************************************")

print(y_preds3[:10],'\n',y_test[:10])

print("*******************************************************")

print(y_preds4[:10],'\n',y_test[:10])

print("*******************************************************")

print(y_preds6[:10],'\n',y_test[:10])
print("Accuracy of Logistic Regression",accuracy_score(y_test, y_preds1))

print("Accuracy of Naive Bayes",accuracy_score(y_test, y_preds3))

print("Accuracy of Decision Tree",accuracy_score(y_test, y_preds4))

print("Accuracy of XGBoost",accuracy_score(y_test, y_preds6))
print("\nLogistic Regression\n",classification_report(y_test, y_preds1))

print("\nNaive Bayes\n",classification_report(y_test, y_preds3))

print("\nDecision Tree\n",classification_report(y_test, y_preds4))

print("\nXGBoost\n",classification_report(y_test, y_preds6))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, LRClassifer.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, LRClassifer.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
nb_roc_auc = roc_auc_score(y_test, NBClassifer.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, NBClassifer.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Naive Bayes (area = %0.2f)' % nb_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('')

plt.legend(loc="lower right")

plt.savefig('NB_ROC')

plt.show()
dt_roc_auc = roc_auc_score(y_test, DTClassifer.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, DTClassifer.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Decision Tree (area = %0.2f)' % dt_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('')

plt.legend(loc="lower right")

plt.savefig('DT_ROC')

plt.show()
xgboost_roc_auc = roc_auc_score(y_test, XGBClassifer.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, XGBClassifer.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='XGBoost (area = %0.2f)' % xgboost_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('')

plt.legend(loc="lower right")

plt.savefig('XG_ROC')

plt.show()