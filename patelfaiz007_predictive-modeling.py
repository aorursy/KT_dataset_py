# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv("../input/train_data.csv", header = None, delimiter = " *, *", engine = 'python')
data.head(5)
data.shape
data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
                'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
data.tail(5)
data.info()
data.describe()
data.isnull().sum()
data = data.replace(to_replace = ["?"], value = np.nan)
data.isnull().sum()
data.workclass.unique()
data.occupation.unique()
data.native_country.unique()
data_copy = pd.DataFrame.copy(data)
data_copy.describe(include = 'all')
for value in ['workclass', 'occupation', 'native_country']:
    data_copy[value].fillna(data_copy[value].mode()[0], inplace = True)
data_copy.isnull().sum()
data_copy.dtypes
colname = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income']
from sklearn import preprocessing

le = {}

for x in colname:
    le[x] = preprocessing.LabelEncoder()

for x in colname:
    data_copy[x] = le[x].fit_transform(data_copy[x])
data_copy.tail(5)
data_copy.dtypes
X = data_copy.values[:, :-1]
y = data_copy.values[:, -1]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)
X = scaler.transform(X)
print(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
# Adjusting the Threshold
y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)
y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value > 0.45:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
confusion_matrix(y_test, y_pred_class)
for a in np.arange(0,1,0.01):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", cfm[1,0]," , type 1 error:", cfm[0,1])
from sklearn import metrics
fpr, tpr, z = metrics.roc_curve(y_test, y_pred_class)
auc = metrics.auc(fpr, tpr)
print(auc)
print(fpr)
print(tpr)
plt.title("Receiver Operating Characteristic")
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = "lower right")
plt.plot([0,1], [0, 1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
fpr, tpr, z = metrics.roc_curve(y_test, y_pred_prob[:, 1:])
auc = metrics.auc(fpr, tpr)
print(auc)
plt.title("Receiver Operating Characteristic")
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = "lower right")
plt.plot([0,1], [0, 1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
#Feature Selection using RFE
colname = data_copy.columns[:]
from sklearn.feature_selection import RFE
rfe = RFE(classifier, 7)
model_rfe = rfe.fit(X_train, y_train)
print("Num Features: ", model_rfe.n_features_)
print("Selected Features: ")
print(list(zip(colname, model_rfe.support_)))
print("Feature Ranking: ", model_rfe.ranking_)
y_pred = model_rfe.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
# Feature Selection using Univariate Selection
X = data_copy.values[:,:-1]
y = data_copy.values[:, -1]
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
test = SelectKBest(score_func=chi2, k=13)
fit1 = test.fit(X, y)

print(fit1.scores_)
print(list(zip(colname,fit1.get_support())))
X = fit1.transform(X)

print(X)
scaler = StandardScaler()

scaler.fit(X)
X = scaler.transform(X)
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

classifier = LogisticRegression()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
