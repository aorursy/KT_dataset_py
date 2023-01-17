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

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
sns.set(rc={'figure.figsize':(8,5)})
train = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
train.head()
#Time for some Exploratory data analysis.
sns.countplot(data = train, x=train['Response'])
fig, a=plt.subplots(2,2, figsize=(12, 8))
sns.distplot(train['Age'], kde=False, ax=a[0, 0])
sns.countplot(data=train, x=train['Gender'], ax=a[0, 1])
sns.countplot(data=train, x=train['Gender'], hue=train['Response'], ax=a[1, 0])
sns.countplot(data=train, x=train['Gender'], hue=train['Previously_Insured'], ax=a[1, 1])
fig, a=plt.subplots(2,2, figsize=(12, 8))
sns.countplot(data=train, x=train['Vehicle_Damage'], hue=train['Response'], ax=a[0, 0])
sns.countplot(data=train, x=train['Vehicle_Damage'], hue=train['Gender'], ax=a[0, 1])
sns.countplot(data=train, x=train['Vehicle_Age'], hue=train['Response'], ax=a[1, 0])
sns.countplot(data=train, x=train['Response'], ax=a[1, 1])
sns.set(rc={'figure.figsize':(12,10)})
sns.heatmap(train.corr(), annot=True)
X = train.drop(['Response', 'id', 'Driving_License'], axis = 1)
y = train['Response']
cat = []
num = []


def feature(df):
    
    
    for i in df.columns:
        if df[i].dtype == 'object':
            cat.append(i)
        else:
            num.append(i)

mms = MinMaxScaler()
def scale(num):
    
    num_new = pd.DataFrame(mms.fit_transform(num))
    num_new.columns = num.columns
    return num_new
def ohe(cat):
    cat_new = pd.DataFrame(pd.get_dummies(cat, drop_first = True))
    return cat_new

feature(X)
X_new = pd.concat([scale(X[num]), ohe(X[cat])], axis=1)
X_new
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.20, random_state = 100)

print(y_train.value_counts())
print(y_test.value_counts())
smote = SMOTE()

X_sm, y_sm = smote.fit_sample(X_train, y_train)

print(X_sm.shape)
print(y_sm.value_counts())
def mod(any_model):
    
    model = any_model
    
    model.fit(X_sm, y_sm)
    
    y_pred = model.predict(X_test)
    
    print(roc_auc_score(y_test, y_pred)) 
    print(classification_report(y_pred, y_test))
    print(confusion_matrix(y_pred, y_test))
import re

regex = re.compile(r"\[|\]|<", re.IGNORECASE)

X_sm.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_sm.columns.values]
X_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_test.columns.values]

model = XGBClassifier(objective="binary:logistic", random_state=42)

model.fit(X_sm, y_sm)

y_pred = model.predict(X_test)

print(roc_auc_score(y_test, y_pred))
mod(LogisticRegression())
mod(RandomForestClassifier(n_estimators=200))
mod(KNeighborsClassifier(n_neighbors=20, weights='uniform', n_jobs=-1))
from sklearn.naive_bayes import GaussianNB

mod(GaussianNB())