# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
data
for i in data.columns:
    if len(data[i].value_counts()) <= 2:
        print(i)
    else:
        pass
sns.countplot(data['anaemia'], hue=data['DEATH_EVENT'])
print(">>Percentage of People with no anaemia that died, out of total number of people who had no Anaemia: ", (len(data[(data['anaemia'] == 0) & (data['DEATH_EVENT'] == 1)])/len(data[data['anaemia'] == 0]))*100)
print(">>People with no Anaemia: ", len(data[data['anaemia']==0]))
print(">>Percentage of People with Anaemia that died, out of total number people who had Anaemia: ", (len(data[(data['anaemia'] == 1) & (data['DEATH_EVENT'] == 1)])/len(data[data['anaemia'] == 1]))*100)
print(">>People with Anaemia: ", len(data[data['anaemia']==1]))
sns.countplot(data['diabetes'], hue=data['DEATH_EVENT'])
print(">>Percentage of People with no diabetes that died, out of total number of people who actually had no Diebetes: ", (len(data[(data['diabetes'] == 0) & (data['DEATH_EVENT'] == 1)])/len(data[data['diabetes'] == 0]))*100)
print(">>People with no Diabetes: ", len(data[data['diabetes']==0]))
print(">>Percentage of People with diabetes that died, out of total no people who had diabetes: ", (len(data[(data['diabetes'] == 1) & (data['DEATH_EVENT'] == 1)])/len(data[data['diabetes'] == 1]))*100)
print(">>People with Diabetes: ", len(data[data['diabetes']==1]))
sns.countplot(data['high_blood_pressure'], hue=data['DEATH_EVENT'])
print(">>Percentage of People with no high_blood_pressure that died, out of total number of people who had no high_blood_pressure: ", (len(data[(data['high_blood_pressure'] == 0) & (data['DEATH_EVENT'] == 1)])/len(data[data['high_blood_pressure'] == 0]))*100)
print(">>People with no high_blood_pressure: ", len(data[data['high_blood_pressure']==0]))
print(">>Percentage of People with high_blood_pressure that died, out of total number people who had high_blood_pressure: ", (len(data[(data['high_blood_pressure'] == 1) & (data['DEATH_EVENT'] == 1)])/len(data[data['high_blood_pressure'] == 1]))*100)
print(">>People with high_blood_pressure: ", len(data[data['high_blood_pressure']==1]))
sns.countplot(data['sex'], hue=data['DEATH_EVENT'])
print(">>Percentage of Female that died, out of total no of Females: ", (len(data[(data['sex'] == 0) & (data['DEATH_EVENT'] == 1)])/len(data[data['sex'] == 0]))*100)
print(">>Total Females: ", len(data[data['sex']==0]))
print(">>Percentage of Male, out of total no Male: ", (len(data[(data['sex'] == 1) & (data['DEATH_EVENT'] == 1)])/len(data[data['sex'] == 1]))*100)
print(">>Total Males: ", len(data[data['sex']==1]))
sns.countplot(data['smoking'], hue=data['DEATH_EVENT'])
print(">>Percentage of People who didn't smoke that died, out of total no of people who actually didn't smoke: ", (len(data[(data['smoking'] == 0) & (data['DEATH_EVENT'] == 1)])/len(data[data['smoking'] == 0]))*100)
print(">>People with no smoking: ", len(data[data['smoking']==0]))
print(">>Percentage of People with smoking that died, out of total no people who had been smoking: ", (len(data[(data['smoking'] == 1) & (data['DEATH_EVENT'] == 1)])/len(data[data['smoking'] == 1]))*100)
print(">>People with smoking: ", len(data[data['smoking']==1]))
sns.scatterplot(x=data['creatinine_phosphokinase'], y=data['age'], hue=data['DEATH_EVENT'])
sns.distplot(data['creatinine_phosphokinase'])
sns.scatterplot(x=data['platelets'], y=data['age'], hue=data['DEATH_EVENT'])
sns.distplot(data['platelets'])
sns.scatterplot(x=data['serum_creatinine'], y=data['age'], hue=data['DEATH_EVENT'])
sns.distplot(data['serum_creatinine'])
sns.scatterplot(x=data['ejection_fraction'], y=data['age'], hue=data['DEATH_EVENT'])
sns.distplot(data['ejection_fraction'])
sns.scatterplot(x=data['time'], y=data['time'], hue=data['DEATH_EVENT'])
sns.distplot(data['time'])
from sklearn.model_selection import train_test_split

X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
logr = LogisticRegression()
logr.fit(X_train, y_train)
y_predlr = logr.predict(X_test)

print(confusion_matrix(y_test, y_predlr))
print(classification_report(y_test, y_predlr))
print(accuracy_score(y_test, y_predlr))
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
y_pred_ada = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred_ada))
print(classification_report(y_test, y_pred_ada))
print(accuracy_score(y_test, y_pred_ada))
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_predrf = rf.predict(X_test)

print(confusion_matrix(y_test, y_predrf))
print(classification_report(y_test, y_predrf))
print(accuracy_score(y_test, y_predrf))
grb = GradientBoostingClassifier()
grb.fit(X_train, y_train)
y_predgrb = grb.predict(X_test)

print(confusion_matrix(y_test, y_predgrb))
print(classification_report(y_test, y_predgrb))
print(accuracy_score(y_test, y_predgrb))
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)
y_pred_knn = model.predict(X_test)

print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
print(accuracy_score(y_test, y_pred_knn))

result = pd.DataFrame({'Model': ['LogisticRegression','AdaBoostClassifier','RandomForestClassifier','GradientBoostingClassifier', 'KNeighborsClassifier'],
                       'Score': [accuracy_score(y_test, y_predlr), accuracy_score(y_test, y_pred_ada), accuracy_score(y_test, y_predrf), accuracy_score(y_test, y_predgrb), accuracy_score(y_test, y_pred_knn)]})
result