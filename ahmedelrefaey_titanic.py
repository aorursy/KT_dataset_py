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
df_train=pd.read_csv('/kaggle/input/titanic/train.csv')
df_test=pd.read_csv('/kaggle/input/titanic/test.csv')

df_train.head()
# getting information of null values
df_train.info()

# summary ststistics
df_train.describe()

import matplotlib.pyplot as plt
import seaborn as sns

df_train.corr()
#plt.hist(df_train.Age)

sns.boxplot(x='Age',hue='Survived',data=df_train)
plt.show()
sns.boxplot('Pclass','Age',data=df_train)
plt.show()
pclass1Age=(df_train.loc[df_train['Pclass']==1,'Age'].median())
pclass2Age=(df_train.loc[df_train['Pclass']==2,'Age'].median())
pclass3Age=(df_train.loc[df_train['Pclass']==3,'Age'].median())

df_train.loc[df_train['Pclass']==1,'Age']=df_train.loc[df_train['Pclass']==1,'Age'].fillna(pclass1Age)
df_train.loc[df_train['Pclass']==2,'Age']=df_train.loc[df_train['Pclass']==2,'Age'].fillna(pclass2Age)
df_train.loc[df_train['Pclass']==3,'Age']=df_train.loc[df_train['Pclass']==3,'Age'].fillna(pclass3Age)
df_train.Embarked.fillna(df_train.Embarked.value_counts().index[0],inplace=True)
df_train.info()
df_train['Title'] = df_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(df_train['Title'], df_train['Sex'])
print(df_train.Title.unique())
toBeReplaced=[ 'Don' ,'Rev' ,'Dr', 'Mme', 'Ms', 'Major', 'Lady','Sir' ,'Mlle' ,'Col' ,'Capt', 'Countess' ,'Jonkheer']
df_train.Title=df_train.Title.replace(toBeReplaced,'other')
df_train.Title.unique()
df_train.drop(['Name','PassengerId','Ticket','Cabin'],axis=1,inplace=True)

df_train['NumberOfFamily']=df_train.SibSp + df_train.Parch 
df_train.drop(['SibSp','Parch'],axis=1,inplace=True)
df_train.head()
df_train=pd.get_dummies(df_train, prefix=['Title','Embarked','Sex'])
df_train.head()
sns.boxplot('Fare',data=df_train)
plt.show()
df_train['newage']=pd.cut(df_train['Age'],bins=4,labels=[0,1,2,3])
df_train['newfare']=pd.cut(df_train['Fare'],bins=4,labels=[0,1,2,3])
df_train.drop(['Age','Fare'],axis=1,inplace=True)
df_train.head()
df_test.info()
pclass1Age=(df_test.loc[df_test['Pclass']==1,'Age'].median())
pclass2Age=(df_test.loc[df_test['Pclass']==2,'Age'].median())
pclass3Age=(df_test.loc[df_test['Pclass']==3,'Age'].median())

df_test.loc[df_test['Pclass']==1,'Age']=df_test.loc[df_test['Pclass']==1,'Age'].fillna(pclass1Age)
df_test.loc[df_test['Pclass']==2,'Age']=df_test.loc[df_test['Pclass']==2,'Age'].fillna(pclass2Age)
df_test.loc[df_test['Pclass']==3,'Age']=df_test.loc[df_test['Pclass']==3,'Age'].fillna(pclass3Age)


df_test.Fare.fillna(df_test.Fare.median(),inplace=True)


df_test['Title'] = df_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(df_test['Title'], df_test['Sex'])

print(df_test.Title.unique())
toBeReplaced=[ 'Dona','Don' ,'Rev' ,'Dr', 'Mme', 'Ms', 'Major', 'Lady','Sir' ,'Mlle' ,'Col' ,'Capt', 'Countess' ,'Jonkheer']
df_test.Title=df_test.Title.replace(toBeReplaced,'other')
df_test.Title.unique()

df_test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)


df_test['NumberOfFamily']=df_test.SibSp + df_test.Parch 
df_test.drop(['SibSp','Parch'],axis=1,inplace=True)
df_test.head()


df_test=pd.get_dummies(df_test, prefix=['Title','Embarked','Sex'])
df_test.head()


df_test['newage']=pd.cut(df_test['Age'],bins=4,labels=[0,1,2,3])
df_test['newfare']=pd.cut(df_test['Fare'],bins=4,labels=[0,1,2,3])
df_test.drop(['Age','Fare'],axis=1,inplace=True)
df_test.head()
df_train.shape,df_test.shape
Y_train=df_train.Survived

X_train=df_train.drop('Survived',axis=1)

X_test=df_test.drop('PassengerId',axis=1)
X_train.shape, Y_train.shape, X_test.shape
print(X_train.columns)
print(X_test.columns)
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# Random Forest

random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)
Y_pred_forest = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

logreg = LogisticRegression(max_iter=100,tol=0.01)
logreg.fit(X_train, Y_train)
Y_predLOG = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
submission101 = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred_forest
    })
submission101.to_csv('submission101.csv', index=False)
X_train = df_train.drop("Survived", axis=1)
Y_train = df_train["Survived"]
X_test  = df_test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
# Import MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
 
# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
X_train.shape  , Y_train.shape ,X_test.shape
X_train.columns , X_test.columns
X_train.drop('Name',axis=1,inplace=True)
# Logistic Regression

logreg = LogisticRegression(max_iter=100,tol=0.01)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_forest = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


import matplotlib.pyplot as plt
plt.barh(models['Model'],models['Score'])
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)
