#Importing necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#loading the dataset

loans = pd.read_excel('/kaggle/input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx','Data')
# displaying the dataset

loans.head()
loans.info()
#Looking at various statistics of the data

loans.describe().transpose()



loans[loans['Experience']<0]['Experience'].count()
sns.set_style('whitegrid')

sns.jointplot(data=loans, x='Age', y='Experience')
# Evaluating the age range of those with negtive experience

loans[loans['Experience']<0]['Age'].value_counts()
q=int(np.ceil(loans[loans['Age']<=29]['Experience'].mean()))
# Since there are only three negative values (-3,-2,-1) we only replace this three values

loans['Experience'].replace(to_replace=[-3,-2,-1], value=q, inplace=True)
# Now we can check if the 'Experience' column is positive

loans[loans['Experience']<0]['Experience'].count()
# The columns 'ID' and 'ZIP Code' are not needed for your prediction so we can remove them.

loans.drop(columns=['ID', 'ZIP Code'], axis=1, inplace=True)
#checking the columns again

loans.columns
#checking if any column is zero.

loans.isnull().sum()
loans.isna().sum()
sns.distplot(loans['Income'], bins=30, color='g')

sns.distplot(loans['Age'], bins=30, color='r')
sns.boxplot(data=loans, x='Family', y='Income', hue='Personal Loan')
sns.boxplot(data=loans, x='Education', y='Income', hue='Personal Loan', color='m')
g = sns.PairGrid(data=loans, vars=['Income', 'Mortgage', 'CD Account', 'CCAvg'],

                 hue='Personal Loan', palette='RdBu_r')

g.map(plt.scatter, alpha=0.8)

g.add_legend();
plt.figure(figsize=(14,5))



corr=loans.corr()

sns.heatmap(corr, annot=True, cmap="YlGnBu")
from sklearn.model_selection import train_test_split
X=loans.drop('Personal Loan', axis=1)
y=loans['Personal Loan']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
logistic=LogisticRegression()
logistic.fit(X_train, y_train)
logistic.score(X_test,y_test)
from sklearn.metrics import classification_report, confusion_matrix
reg_pred=logistic.predict(X_test)
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(criterion='entropy')
dtree.fit(X_train, y_train)
dtree_pred=dtree.predict(X_test)
dtree.score(X_test, y_test) 
print(confusion_matrix(dtree_pred, y_test))

print('\n')

print(classification_report(dtree_pred, y_test))
from sklearn.ensemble import RandomForestClassifier
random_forest= RandomForestClassifier(n_estimators=40, criterion='entropy')
random_forest.fit(X_train, y_train)
random_forest.score(X_test, y_test)
rfc_pred= random_forest.predict(X_test)
print(confusion_matrix(rfc_pred, y_test))

print('\n')

print(classification_report(rfc_pred, y_test))
from sklearn.naive_bayes import GaussianNB
naive_bayes= GaussianNB()
naive_bayes.fit(X_train, y_train)
naive_bayes.score(X_test, y_test)
nb_pred=naive_bayes.predict(X_test)
print(confusion_matrix(nb_pred, y_test))

print('\n')

print(classification_report(nb_pred, y_test))
d = {'Classifier': ['Logistic Regression', 'Decision Tree', 'Random Forest Classifier', 'Naive Bayes' ], 

     'Accuracy': [logistic.score(X_test,y_test),

              dtree.score(X_test,y_test),

              random_forest.score(X_test, y_test), 

              naive_bayes.score(X_test,y_test)]}

df = pd.DataFrame(data=d)

print(df)