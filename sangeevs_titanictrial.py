import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



tr = pd.read_csv('../input/train.csv')

te = pd.read_csv('../input/test.csv')



 

tr.head()

tr.tail()

tr.info()

tr['Sex'].unique()

tr['Pclass'].unique()

tr['Embarked'].unique()

tr['SibSp'].unique()

tr['Parch'].unique()

tr.sort_values(by='Survived').describe()



tr.shape

tr['Deck']='C'

tr.loc[tr['Cabin'].notnull(), 'Deck'] = tr['Cabin'].str[0]



tr.isnull().sum()

tr.shape	



y= tr[(tr['Pclass']==1) & (tr["Sex"]=="female")]["Embarked"].mode()



tr["Embarked"]=tr["Embarked"].fillna("S")

am= tr["Age"].mean()

ast =tr["Age"].std()

no= tr["Age"].isnull().sum()

ran=np.random.randint(am-ast,am+ast,no)

tr['Age']=tr['Age'].fillna(ran.mean())





tr['Alone'] = 0

tr['Family'] =  tr["Parch"] + tr["SibSp"]

tr['Alone'].loc[tr['Family'] > 0] = 0

tr['Alone'].loc[tr['Family'] == 0] = 1

tr['Title']='Sir'

for i in range(891):

    tr['Title'].loc[i] = tr['Name'].loc[i].split(',').pop(1).split('.').pop(0).strip()

 

tr['Title'] =tr['Title'].replace('Mr',1)

tr['Title'] =tr['Title'].replace('Mrs',2)

tr['Title'] =tr['Title'].replace('Miss',3)

tr['Title'] =tr['Title'].replace('Ms',4)

tr['Title'] =tr['Title'].replace('Master',4)

tr['Title'] =tr['Title'].replace('Don',4)

tr['Title'] =tr['Title'].replace('Rev',1)

tr['Title'] =tr['Title'].replace('Dr',4)

tr['Title'] =tr['Title'].replace('Mme',3)

tr['Title'] =tr['Title'].replace('Major',5)

tr['Title'] =tr['Title'].replace('Lady',2)

tr['Title'] =tr['Title'].replace('Sir',4)

tr['Title'] =tr['Title'].replace('Mlle',2)

tr['Title'] =tr['Title'].replace('Col',4)

tr['Title'] =tr['Title'].replace('Capt',5)

tr['Title'] =tr['Title'].replace('the Countess',3)

tr['Title'] =tr['Title'].replace('Jonkheer',4)

tr['Title'] =tr['Title'].replace('Mlle',2)

tr['Title'] =tr['Title'].replace('Dona',3)



tr['Sex'] =tr['Sex'].replace('male',1)

tr['Sex'] =tr['Sex'].replace('female',2)

tr['Embarked'] =tr['Embarked'].replace('S',1)

tr['Embarked'] =tr['Embarked'].replace('C',2)

tr['Embarked'] =tr['Embarked'].replace('Q',3)



tr['Deck'] =tr['Deck'].replace('A',1)

tr['Deck'] =tr['Deck'].replace('B',2)

tr['Deck'] =tr['Deck'].replace('C',3)

tr['Deck'] =tr['Deck'].replace('D',4)

tr['Deck'] =tr['Deck'].replace('E',5)

tr['Deck'] =tr['Deck'].replace('F',6)

tr['Deck'] =tr['Deck'].replace('G',8)

tr['Deck'] =tr['Deck'].replace('T',7)





tr.loc[ tr['Age']<= 8 , 'Age'] = 0

tr.loc[(tr['Age']> 8)&(tr['Age']<=16), 'Age'] = 1

tr.loc[(tr['Age']>16)&(tr['Age']<=24), 'Age'] = 2

tr.loc[(tr['Age']>24)&(tr['Age']<=32), 'Age'] = 3

tr.loc[(tr['Age']>32)&(tr['Age']<=40), 'Age'] = 4

tr.loc[(tr['Age']>40)&(tr['Age']<=48), 'Age'] = 5

tr.loc[(tr['Age']>48)&(tr['Age']<=56), 'Age'] = 6

tr.loc[(tr['Age']>56)&(tr['Age']<=64), 'Age'] = 7

tr.loc[(tr['Age']>64)&(tr['Age']<=72), 'Age'] = 8

tr.loc[(tr['Age']>72), 'Age'] = 9



tr['Child']=0

tr.loc[(tr['Age']<16),'Child']=1



g = sns.FacetGrid(tr, col='Survived')

g.map(plt.hist, 'Age', bins=20)





sns.barplot(x='Family', y='Survived', data=tr, order=[1,0])

pd.crosstab(tr['Title'], tr['Sex'])

sns.barplot(x='Deck', y='Survived',hue='Pclass', data=tr)

sns.barplot(x='Pclass', y='Survived',hue='Sex', data=tr)

sns.barplot(x='Age', y='Survived',hue='Sex', data=tr)

sns.barplot(x='Sex', y='Survived', data=tr)

sns.barplot(x='Child', y='Survived',hue='Pclass', data=tr)

tr = tr.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

te   = te.drop(['Name','Ticket','Cabin'], axis=1)

sns.countplot(x='Embarked', data=tr)

sns.countplot(x='Survived', hue="Embarked", data=tr, order=[1,0])







xtr = tr.drop("Survived", axis=1)

ytr = tr["Survived"]

xte  = te.drop("PassengerId", axis=1).copy()





from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



print(tr.head(10))

# Logistic regression

logreg = LogisticRegression()

logreg.fit(xtr,ytr)

acc_log = logreg.score(xtr,ytr)

acc_log



# Support vector machines

svc = SVC()

svc.fit(xtr,ytr)

acc_svc = svc.score(xtr,ytr)

acc_svc





# k nearest neighbors - primitive lazy learning technique

knn = KNeighborsClassifier()

knn.fit(xtr,ytr)

acc_knn = knn.score(xtr,ytr)

acc_knn



# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(xtr, ytr)

acc_gaussian = gaussian.score(xtr, ytr)

acc_gaussian





# Perceptron

perceptron = Perceptron()

perceptron.fit(xtr, ytr)

acc_perceptron = perceptron.score(xtr, ytr)

acc_perceptron



# Stochastic Gradient Descent

sgd = SGDClassifier()

sgd.fit(xtr, ytr)

acc_sgd = sgd.score(xtr, ytr)

acc_sgd





# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(xtr, ytr)

acc_decision_tree = decision_tree.score(xtr, ytr)

acc_decision_tree





# Evaluate models

models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_gaussian, acc_perceptron, 

              acc_sgd, acc_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)


