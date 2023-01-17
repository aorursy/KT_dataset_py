# make sure we have the latest seaborb package
!pip install seaborn --upgrade
# should be version 11
import seaborn as sns
sns.__version__
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/employee-attrition/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.info()
# get the dtype and unique values for each column of the data frame
for feat in df.columns:
    print(feat)
    print(df[feat].dtype)
    print(df[feat].unique())
    print('#'*30)
df.drop(columns=['EmployeeCount', 'Over18', 'EmployeeNumber', 'StandardHours'], inplace=True)
list_ratio = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
              'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole',
              'YearsSinceLastPromotion', 'YearsWithCurrManager']
list_binary = ['Gender', 'OverTime']
list_cat = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
list_ord = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'PerformanceRating',
            'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']
len(list_ratio)
# make a grid space 
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(18,12))
# hspace lets us see the names of each feature
fig.subplots_adjust(hspace=0.5)
# gives the plot a title
fig.suptitle('Numeric features against the target')
# for loop to populate each subplot with a chart
for feat, ax in zip(list_ratio, axes.flatten()):
    sns.histplot(data=df, x=feat, hue='Attrition', ax=ax)
list_binary
fig, axes = plt.subplots(1, 2, figsize=(12,4))
fig.suptitle('Binary features against the target')
for feat, ax in zip(list_binary, axes.flatten()):
    sns.countplot(data=df, x=feat, hue='Attrition', ax=ax)
len(list_cat)
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.subplots_adjust(hspace=0.8)
fig.suptitle('categorical varibles against the target')
for feat, ax in zip(list_cat, axes.flatten()):
    sns.countplot(data=df, x=feat, hue='Attrition', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, horizontalalignment='right')
# creat dummy varibales
list_dummy = list(df.select_dtypes('object'))
df = pd.get_dummies(df, columns=list_dummy, drop_first=True)
# import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
X = df.drop(columns='Attrition_Yes').values
y = df['Attrition_Yes']
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# scale data
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
logreg = LogisticRegression()
logreg.fit(X_train_sc, y_train)
logreg_pred = logreg.predict(X_test_sc)

print('accuracy: ', accuracy_score(y_test, logreg_pred))
print(confusion_matrix(y_test, logreg_pred))
print(classification_report(y_test, logreg_pred))
# feature importance
importance = logreg.coef_
for i, v in enumerate(importance.flatten()):
    print('Feature', i, ':', v)
rf = RandomForestClassifier()
rf.fit(X_train_sc, y_train)
rf_pred = rf.predict(X_test_sc)

print('accuracy: ', accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
