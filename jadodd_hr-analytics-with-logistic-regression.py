#importing the usual suspects

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/HR_comma_sep.csv')
df.head()
df.info()
df.describe()
sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap='viridis')
#just to make sure

df[df.isnull()].count()
g = sns.PairGrid(df,hue='left')

g = g.map_diag(plt.hist)

g = g.map_offdiag(plt.scatter)
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(),cmap='viridis',annot=True)
sns.set_style('whitegrid')

sns.distplot(df['satisfaction_level'],bins=20)
plt.figure(figsize=(16,6))

sns.countplot(df['satisfaction_level'],hue=df['left'])

plt.tight_layout()
plt.figure(figsize=(14,6))

sns.countplot(df['last_evaluation'],hue=df['left'])

plt.tight_layout()
plt.figure(figsize=(14,7))

g = sns.barplot(x=df['last_evaluation'],y=df['satisfaction_level'])

plt.tight_layout()
plt.figure(figsize=(14,7))

g = sns.barplot(x=df['last_evaluation'],y=df['satisfaction_level'],hue=df['left'])

plt.tight_layout()
plt.figure(figsize=(14,6))

sns.countplot(df['average_montly_hours'],hue=df['left'])

plt.tight_layout()

plt.legend(loc=1)
plt.figure(figsize=(14,6))

sns.countplot(df['time_spend_company'],hue=df['left'])

plt.tight_layout()
sns.countplot(df['number_project'],hue=df['left'])
plt.figure(figsize=(12,6))

sns.countplot(df['sales'],hue=df['left'])
plt.figure(figsize=(12,6))

sns.countplot(df['promotion_last_5years'],hue=df['left'])

plt.tight_layout()
df[df['promotion_last_5years']==1].count()
df[(df['promotion_last_5years']==1) & (df['left']==1)].count()
#Proportion of those with promotion that left

print((19.0/310)*100)
df[df['promotion_last_5years']==0].count()
df[(df['promotion_last_5years']==0) & (df['left']==1)].count()
#Proportion of those without promotion that left

print((3552.0/14680)*100)
sns.countplot(df['Work_accident'])
sns.countplot(df['Work_accident'],hue=df['left'])
#low salaried workers seems to have a higher rate of leaving

#obvious but worth looking into

sns.countplot(df['salary'],hue=df['left'])
df['salary'].value_counts()
def sal_class(x):

    if x == "low":

        return 1

    elif x == "medium":

        return 2

    elif x == "high":

        return 3
df['sal_class'] = df['salary'].apply(sal_class)
df['sal_class'].value_counts()
df['sales'].value_counts()
def job_class(x):

    if x == "sales":

        return 1

    elif x == "technical":

        return 2

    elif x == "support":

        return 3

    elif x == "IT":

        return 4

    elif x == "product_mng":

        return 5

    elif x == "marketing":

        return 6

    elif x == "RandD":

        return 7

    elif x == "accounting":

        return 8

    elif x == "hr":

        return 9

    elif x == "technical":

        return 10

    elif x == "management":

        return 11
df['job_class'] = df['sales'].apply(job_class)
df['job_class'].value_counts()
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
X = df[['satisfaction_level','last_evaluation','number_project','average_montly_hours',

        'time_spend_company','Work_accident','promotion_last_5years','sal_class']]

y = df['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))
X = df[['satisfaction_level','last_evaluation','number_project','average_montly_hours',

       'promotion_last_5years','sal_class','Work_accident']]

y = df['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))
X = df[['satisfaction_level','number_project','average_montly_hours',

       'sal_class','Work_accident']]

y = df['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))
X = df[['satisfaction_level','number_project',

       'sal_class','Work_accident']]

y = df['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))