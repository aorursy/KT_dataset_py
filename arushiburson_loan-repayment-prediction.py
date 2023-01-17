import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
loans = pd.read_csv("../input/loan_data.csv")

loans.head()
loans.info()
loans.describe()
sns.set_style('darkgrid')

plt.hist(loans['fico'].loc[loans['credit.policy']==1], bins=30, label='Credit.Policy=1')

plt.hist(loans['fico'].loc[loans['credit.policy']==0], bins=30, label='Credit.Policy=0')

plt.legend()

plt.xlabel('FICO')

plt.figure(figsize=(10,6))

loans[loans['not.fully.paid']==1]['fico'].hist(bins=30, alpha=0.5, color='blue', label='not.fully.paid=1')

loans[loans['not.fully.paid']==0]['fico'].hist(bins=30, alpha=0.5, color='green', label='not.fully.paid=0')

plt.legend()

plt.xlabel('FICO')


#creating a countplot to see the counts of purpose of loans by not.fully.paid

plt.figure(figsize=(12,6))

sns.countplot(data=loans, x='purpose', hue='not.fully.paid')
#checking the trend between FICO and the interest rate

plt.figure(figsize=(10,6))

sns.jointplot(x='fico', y='int.rate', data=loans)
#understanding the relationship between credit.policy and not.fully.paid

sns.lmplot(data=loans, x='fico', y='int.rate', hue='credit.policy', col='not.fully.paid', palette='Set2')
loans.head()
#handling categorical variable purpose

purpose_c = pd.get_dummies(loans['purpose'], drop_first=True)

loans_f = pd.concat([loans, purpose_c], axis=1).drop('purpose', axis=1)

loans_f.head()
#checking for null values

sns.heatmap(loans.isnull())
#Splitting the dataset into test and train set

from sklearn.model_selection import train_test_split

y = loans_f['not.fully.paid'] 

X = loans_f.drop('not.fully.paid', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
#using decision tree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

prediction = dtree.predict(X_test)



#checking performance of the model

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))
#using random forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=800)

rfc.fit(X_train, y_train)

predictionRF = rfc.predict(X_test)



#checking performance of the model

print(confusion_matrix(y_test, predictionRF))

print(classification_report(y_test, predictionRF))
