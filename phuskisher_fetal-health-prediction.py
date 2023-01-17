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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('/kaggle/input/fetal-health-classification/fetal_health.csv')

df.head()
# shape of datset

print(f'rows : {df.shape[0]}')

print(f'columns : {df.shape[1]}')
# info of dataset

df.info()
# nan values

df.isna().sum()
# hence no nan values
# statistical info of dataset

df.describe().T
# correlation matrix for the dataset

plt.figure(figsize=(15,15))

sns.heatmap(df.corr(), annot=True)
sns.countplot(df['fetal_health'])
# 1: Normal, 2; suspect, 3: pathologic
# boxplots

fig, axes=plt.subplots(7,3, figsize=(20,25))

for i,j in enumerate(df.columns):

  ax=axes[int(i/3), i%3]

  sns.boxplot(df[j], ax=ax)

y=df['fetal_health']

X=df.drop('fetal_health', axis=1)
# distribution plot of dataset

fig, axes=plt.subplots(11,2, figsize=(20,20))

for i,j in enumerate(df.columns):

  ax=axes[int(i/2), i%2]

  sns.distplot(df[j], ax=ax)
# scaling the dataset

from sklearn.preprocessing import StandardScaler

scalar=StandardScaler()

X=scalar.fit_transform(X)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=42)
# shape after spliting

print('shape after spliting')

print('*'*30)

print(f'X_test : {X_test.shape}')

print(f'X_train : {X_train.shape}')

print(f'y_test : {y_test.shape}')

print(f'y_train : {y_train.shape}')
# logistic regression

lor=LogisticRegression()

lor.fit(X_train,y_train)

pred_lor=lor.predict(X_test)

score_lor=cross_val_score(lor,X,y,cv=5)

print(score_lor)
score_lor.mean()
# classification report

from sklearn.metrics import classification_report, confusion_matrix

report_lor=classification_report(y_test,pred_lor)

print(report_lor)
# 88% accuracy for logistic regression
# confusion matrix

cm=confusion_matrix(y_test,pred_lor)

sns.heatmap(cm,annot=True)

plt.xlabel('Predicted Value')

plt.ylabel('Actual Values')
# random forest

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

rf.fit(X_train,y_train)

pred_rf=rf.predict(X_test)

score_rf=cross_val_score(rf,X,y,cv=5)

print(score_rf)
score_rf.mean()
# classification report

from sklearn.metrics import classification_report, confusion_matrix

report_rf=classification_report(y_test,pred_rf)

print(report_rf)



# confusion matrix

cm=confusion_matrix(y_test,pred_rf)

sns.heatmap(cm,annot=True)

plt.xlabel('Predicted Value')

plt.ylabel('Actual Values')
# 94% accuracy with random forest classification
# decision tree

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(X_train,y_train)

pred_dt=dt.predict(X_test)

score_dt=cross_val_score(dt,X,y,cv=5)

print(score_dt)
score_dt.mean()
# classification report

from sklearn.metrics import classification_report, confusion_matrix

report_dt=classification_report(y_test,pred_dt)

print(report_dt)



# confusion matrix

cm=confusion_matrix(y_test,pred_dt)

sns.heatmap(cm,annot=True)

plt.xlabel('Predicted Value')

plt.ylabel('Actual Values')
# 94% accuracy with decision tree classification
# xgb classifier

from xgboost import XGBClassifier

xgb=XGBClassifier()

xgb.fit(X_train,y_train)

pred_xgb=xgb.predict(X_test)

score_xgb=cross_val_score(xgb,X,y,cv=5)

print(score_xgb)
score_xgb.mean()
# classification report

from sklearn.metrics import classification_report, confusion_matrix

report_xgb=classification_report(y_test,pred_xgb)

print(report_xgb)



# confusion matrix

cm=confusion_matrix(y_test,pred_xgb)

sns.heatmap(cm,annot=True)

plt.xlabel('Predicted Value')

plt.ylabel('Actual Values')
# 95% accuracy with XGB classifier
# hence XGB classfier with 95% accuracy