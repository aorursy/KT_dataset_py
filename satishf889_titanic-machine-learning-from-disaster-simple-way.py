#Data Rendering

import numpy as np

import pandas as pd

import os

from pandas import Series, DataFrame



#visualization

import seaborn as sb

import matplotlib.pyplot  as plt

from pylab import rcParams



#Correlation libraries

import scipy 

from scipy.stats import spearmanr,chi2_contingency



#Machine Learning Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
%matplotlib inline

rcParams['figure.figsize']=20,14

plt.style.use('seaborn-whitegrid')
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_address="/kaggle/input/titanic/train.csv"

test_address="/kaggle/input/titanic/test.csv"



train_DF=pd.read_csv(train_address)

test_DF=pd.read_csv(test_address)



combined_DF=[train_DF,test_DF]
train_DF.head()
test_DF.head()
train_DF.describe()
test_DF.describe()
survial_count=train_DF['Survived'].value_counts()

survive=survial_count[1]/(survial_count[0]+survial_count[1])*100

print(f'Total % people survied were {survive:0.2f}')
train_DF[['Survived','Sex']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived', ascending=False)
train_DF[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)
train_DF[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived', ascending=False)
train_DF[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived', ascending=False)
train_DF[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sb.FacetGrid(train_DF, col='Survived')

g.map(plt.hist, 'Age', bins=20)
train_DF['AgeBand'] = pd.cut(train_DF['Age'], 5)

train_DF[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
train_DF=train_DF.drop(['PassengerId','Name','Ticket','Fare','Cabin','AgeBand'],axis=1)

test_DF=test_DF.drop(['Name','Ticket','Fare','Cabin'],axis=1)

train_DF.head()
train_DF.info()
frequent_embarked=train_DF['Embarked'].mode()[0]

frequent_embarked
#Fill NaN values of Embarked with 'S'

train_DF['Embarked']=train_DF["Embarked"].fillna(frequent_embarked)

#Fill NaN values of Age with 'median()'

train_DF['Age']=train_DF["Age"].fillna(train_DF['Age'].median())

##Fill NaN values of Age



test_DF['Embarked']=test_DF["Embarked"].fillna(frequent_embarked)

test_DF['Age']=test_DF["Age"].fillna(test_DF['Age'].median())
title_mapping = {"male": 0, "female": 1, "C": 1, "Q": 2, "S": 3}

train_DF["Sex"]=train_DF["Sex"].map(title_mapping)

train_DF["Embarked"]=train_DF["Embarked"].map(title_mapping)

train_DF.loc[(train_DF['Age'] <= 16) , 'Age'] = 0

train_DF.loc[(train_DF['Age'] > 16) & (train_DF['Age'] <= 32), 'Age'] = 1

train_DF.loc[(train_DF['Age'] > 32) & (train_DF['Age'] <= 48), 'Age'] = 2

train_DF.loc[(train_DF['Age'] > 48) & (train_DF['Age'] <= 64), 'Age'] = 3

train_DF.loc[ train_DF['Age'] > 64, 'Age'] = 4 

test_DF["Sex"]=test_DF["Sex"].map(title_mapping)

test_DF["Embarked"]=test_DF["Embarked"].map(title_mapping)

test_DF.loc[(test_DF['Age'] <= 16) , 'Age'] = 0

test_DF.loc[(test_DF['Age'] > 16) & (test_DF['Age'] <= 32), 'Age'] = 1

test_DF.loc[(test_DF['Age'] > 32) & (test_DF['Age'] <= 48), 'Age'] = 2

test_DF.loc[(test_DF['Age'] > 48) & (test_DF['Age'] <= 64), 'Age'] = 3

test_DF.loc[ test_DF['Age'] > 64, 'Age'] = 4 

train_DF.head()
train_DF['Parch'].value_counts()
sb.pairplot(train_DF)
pclass=test_DF["Pclass"]

sex=test_DF["Sex"]

age=test_DF["Age"]

sibsp=test_DF["SibSp"]

parch=test_DF["Parch"]

embarked=test_DF["Embarked"]



spearmanr_coefficient,p_value= spearmanr(pclass,sex)

print(f'Spearman Rank correlation coefficient {spearmanr_coefficient:0.3f}')
spearmanr_coefficient,p_value= spearmanr(pclass,age)

print(f'Spearman Rank correlation coefficient {spearmanr_coefficient:0.3f}')
spearmanr_coefficient,p_value= spearmanr(pclass,sibsp)

print(f'Spearman Rank correlation coefficient {spearmanr_coefficient:0.3f}')
spearmanr_coefficient,p_value= spearmanr(pclass,parch)

print(f'Spearman Rank correlation coefficient {spearmanr_coefficient:0.3f}')
spearmanr_coefficient,p_value= spearmanr(pclass,embarked)

print(f'Spearman Rank correlation coefficient {spearmanr_coefficient:0.3f}')
table= pd.crosstab(pclass,sex)

chi2,p,dof,expected= chi2_contingency(table.values)

print(f'Chi-square statistic {chi2:0.3f} p_value{p:0.3f}')
table= pd.crosstab(pclass,age)

chi2,p,dof,expected= chi2_contingency(table.values)

print(f'Chi-square statistic {chi2:0.3f} p_value{p:0.3f}')
table= pd.crosstab(pclass,sibsp)

chi2,p,dof,expected= chi2_contingency(table.values)

print(f'Chi-square statistic {chi2:0.3f} p_value{p:0.3f}')
table= pd.crosstab(pclass,parch)

chi2,p,dof,expected= chi2_contingency(table.values)

print(f'Chi-square statistic {chi2:0.3f} p_value{p:0.3f}')
table= pd.crosstab(pclass,embarked)

chi2,p,dof,expected= chi2_contingency(table.values)

print(f'Chi-square statistic {chi2:0.3f} p_value{p:0.3f}')
train_DF.head()
X_train = train_DF.drop("Survived", axis=1)

Y_train = train_DF["Survived"]

X_test  = test_DF.drop(['PassengerId'],axis=1).copy()
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred_log = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
svc = SVC()

svc.fit(X_train, Y_train)

Y_pred_svc = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred_knn = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred_decision = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred_random = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest',  

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
submission_df=pd.DataFrame({

    "PassengerId": test_DF["PassengerId"],

    "Survived": Y_pred_random

})

submission_df

# submission_df.to_csv('./submission.csv', index=False)