# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn import tree

from sklearn import svm

from sklearn import ensemble

from sklearn import neighbors

from sklearn import linear_model

from sklearn import metrics

from sklearn import preprocessing

from sklearn import preprocessing

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.datasets import make_blobs

import seaborn as sns
# load the dataset

df= pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

print("The dataset has %d rows and %d columns." % df.shape)
percent_missing = df.isnull().sum() * 100 / len(df)

percent_missing
df.Attrition.replace(to_replace = dict(Yes = 1, No = 0), inplace = True)
df['Attrition'].value_counts(normalize=True)
df.head()
df.describe()
fig, (axis1) = plt.subplots(1,1,figsize=(15,10))

sns.countplot(df['Age'], hue=df['Attrition'])

fig, (axis1) = plt.subplots(1,1,figsize=(15,10))

sns.countplot(df['DailyRate'], hue=df['Attrition'])

fig, (axis1) = plt.subplots(1,1,figsize=(15,10))

sns.countplot(df['DistanceFromHome'], hue=df['Attrition'])

fig, (axis1) = plt.subplots(1,1,figsize=(15,10))

sns.countplot(df['Education'], hue=df['Attrition'])

fig, (axis1) = plt.subplots(1,1,figsize=(15,10))

sns.countplot(df['EnvironmentSatisfaction'], hue=df['Attrition'])

fig, (axis1) = plt.subplots(1,1,figsize=(15,10))

sns.countplot(df['HourlyRate'], hue=df['Attrition'])

fig, (axis1) = plt.subplots(1,1,figsize=(15,10))

sns.countplot(df['Age'], hue=df['Attrition'])
print("------  Data Types  ----- \n",df.dtypes)
import seaborn as sns

corr = df.corr()

plt.figure(figsize=(15,10))

sns.heatmap(corr,cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},cmap='YlGnBu')
#Lets drop few variables which doesnt look helpful

df = df.drop(['EmployeeCount','EmployeeNumber'], axis=1)
dataset =  df.drop(['OverTime','MaritalStatus','JobRole','Gender','Department','EducationField','BusinessTravel','Over18'], axis=1)

BusinessTravel = pd.get_dummies(df.BusinessTravel).iloc[:,1:]

Department = pd.get_dummies(df.Department).iloc[:,1:]

OverTime = pd.get_dummies(df.OverTime).iloc[:,1:]

MaritalStatus = pd.get_dummies(df.MaritalStatus).iloc[:,1:]

JobRole = pd.get_dummies(df.JobRole).iloc[:,1:]

Gender = pd.get_dummies(df.Gender).iloc[:,1:]

EducationField = pd.get_dummies(df.EducationField).iloc[:,1:]

Over18 = pd.get_dummies(df.Over18).iloc[:,1:]

dataset = pd.concat([dataset,Over18,BusinessTravel,Department,OverTime,MaritalStatus,JobRole,Gender,], axis=1)

dataset
print("------  Data Types  ----- \n",dataset.dtypes)
X =  dataset.drop(['Attrition'], axis=1)

y = dataset['Attrition']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10,random_state=1)

from sklearn.ensemble import RandomForestClassifier
classifier =  RandomForestClassifier(n_estimators = 400,random_state = 42)

classifier.fit(X_train, y_train)  

predictions = classifier.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_test,predictions ))  

print(accuracy_score(y_test, predictions ))
fig, (axis1) = plt.subplots(1,1,figsize=(15,10))

feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 70% training and 30% test
import xgboost as xgb

from sklearn.model_selection import KFold, cross_val_score, train_test_split

xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.08, objective= 'binary:logistic',n_jobs=-1).fit(X_train, y_train)

print('Accuracy of XGB classifier on training set: {:.2f}'

       .format(xgb_model.score(X_train, y_train)))

print('Accuracy of XGB classifier on test set: {:.2f}'

       .format(xgb_model.score(X_test[X_train.columns], y_test)))
y_pred = xgb_model.predict(X_test)

print(classification_report(y_test, y_pred))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)  # 80% training and 20% test
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()

gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split



classifier.fit(X_train, y_train)



rf_predict_probabilities = classifier.predict_proba(X_test)[:,1]

gb_predict_probabilities = gb.predict_proba(X_test)[:,1]

y_predict_xgb = xgb_model.predict_proba(X_test)[:,1]





gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_predict_probabilities)

gb_roc_auc = auc(gb_fpr, gb_tpr)



rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_predict_probabilities)

rf_roc_auc = auc(rf_fpr, rf_tpr)



xgb_fpr, xgb_tpr, _ = roc_curve(y_test, y_predict_xgb)

xgb_roc_auc = auc(xgb_fpr, xgb_tpr)





plt.figure()

plt.plot(gb_fpr, gb_tpr, color='darkorange',

         lw=2, label='random Forrest (area = %0.2f)' % gb_roc_auc)

plt.plot(rf_fpr, rf_tpr, color='darkgreen',

         lw=2, label='Gradient Boosting (area = %0.2f)' % rf_roc_auc)

plt.plot(xgb_fpr, xgb_tpr, color='black',

         lw=2, label='XGBoost (area = %0.2f)' % xgb_roc_auc)



plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
pred_for_submission = xgb_model.predict(X_test).astype(int)

pred_for_submission
xgb_probs = xgb_model.predict_proba(X_test)

xgb_probs