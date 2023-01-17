import pandas as pd

import numpy as np

import re

# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.decomposition import PCA
titanic = pd.read_csv('../input/train.csv')

titanic_test = pd.read_csv('../input/test.csv')

test = titanic_test
titanic.info()
Male_titanic = titanic.loc[titanic.Sex== 'male',:]

Female_titanic = titanic.loc[titanic.Sex== 'female',:]
g0=sns.FacetGrid(Male_titanic, col='Survived')

g0.map(plt.hist, 'Age', bins=20)

g1=sns.FacetGrid(Male_titanic, col='Survived')

g1.map(plt.hist, 'Fare', bins=20)

g2=sns.FacetGrid(Male_titanic, col='Survived', row='Pclass')

g2.map(plt.hist, 'Age', bins=20)

g3=sns.FacetGrid(Male_titanic, col='Survived')

g3.map(plt.hist, 'Age', bins=20)

g4=sns.FacetGrid(Male_titanic, col='Survived', row='Embarked')

g4.map(plt.hist, 'Age', bins=20)
l0=sns.FacetGrid(Female_titanic, col='Survived')

l0.map(plt.hist, 'Age', bins=20)

l1=sns.FacetGrid(Female_titanic, col='Survived')

l1.map(plt.hist, 'Fare', bins=20)

l2=sns.FacetGrid(Female_titanic, col='Survived', row='Pclass')

l2.map(plt.hist, 'Age', bins=20)

l3=sns.FacetGrid(Female_titanic, col='Survived')

l3.map(plt.hist, 'Age', bins=20)

l4=sns.FacetGrid(Female_titanic, col='Survived', row='Embarked')

l4.map(plt.hist, 'Age', bins=20)
titanic.head()
titanic = titanic.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)

titanic_test = titanic_test.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)
total = [titanic,titanic_test]

for dataset in total:

    dataset.loc[dataset['Age']<= 18, 'Age'] = 0

    dataset.loc[(dataset['Age']> 18) & (dataset['Age']<= 32), 'Age'] =1 

    dataset.loc[(dataset['Age']> 32) & (dataset['Age']<=48), 'Age'] = 2

    dataset.loc[(dataset['Age']> 48) & (dataset['Age']<=64), 'Age'] = 3

    dataset.loc[dataset['Age']> 64, 'Age'] = 4
total = [titanic,titanic_test]

titanic.Embarked.value_counts()

Fre_embarked_package = titanic.Embarked.mode()

Fre_age_band = titanic.Age.mode()

for dataset in total:

    dataset['Age']=dataset.Age.fillna(Fre_age_band[0])

    dataset['Embarked']=dataset.Embarked.fillna(Fre_embarked_package[0])
titanic=pd.get_dummies(titanic,columns=['Sex','Embarked'],drop_first=True)

titanic_test=pd.get_dummies(titanic_test,columns=['Sex','Embarked'],drop_first=True)
titanic.info()
print(titanic.corr())
Av_Fare =titanic_test.Fare.mean()

titanic_test['Fare']=titanic_test.Fare.fillna(Av_Fare)
df_1 = titanic.loc[:,['Fare','Pclass']]

df_2 = titanic_test.loc[:,['Fare','Pclass']]

pca =  PCA(n_components=1)

col_1 = pca.fit_transform(df_1)

col_2 = pca.fit_transform(df_2)



titanic['Mod_col_1']=col_1[:,0]

titanic_test['Mod_col_1']=col_2[:,0]



titanic=titanic.drop(['Fare','Pclass'], axis=1)

titanic_test=titanic_test.drop(['Fare','Pclass'], axis=1)
df_3 = titanic.loc[:,['SibSp','Parch']]

df_4 = titanic_test.loc[:,['SibSp','Parch']]

pca =  PCA(n_components=1)

col_3 = pca.fit_transform(df_3)

col_4 = pca.fit_transform(df_4)



titanic['Mod_col_2']=col_3[:,0]

titanic_test['Mod_col_2']=col_4[:,0]



titanic=titanic.drop(['SibSp','Parch'], axis=1)

titanic_test=titanic_test.drop(['SibSp','Parch'], axis=1)
titanic.shape
X_train = titanic.drop('Survived', axis=1)

y_train = titanic['Survived']
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred_1 = logreg.predict(titanic_test)

acc_log = logreg.score(X_train, y_train) * 100

acc_log
svc = SVC()

svc.fit(X_train, y_train)

Y_pred_2 = svc.predict(titanic_test)

acc_svc = svc.score(X_train,y_train) * 100

acc_svc
knn = KNeighborsClassifier(n_neighbors = 10)

knn.fit(X_train, y_train)

Y_pred_3 = knn.predict(titanic_test)

acc_knn = knn.score(X_train, y_train) * 100

acc_knn
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

Y_pred_4 = decision_tree.predict(titanic_test)

acc_decision_tree = decision_tree.score(X_train, y_train) * 100

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=50)

random_forest.fit(X_train,y_train)

Y_pred_5= random_forest.predict(titanic_test)

acc_rf = random_forest.score(X_train,y_train) * 100

acc_rf
test = test.loc[:,['PassengerId']]

test['Survived']=Y_pred_4[:]

test.to_csv('out.csv')