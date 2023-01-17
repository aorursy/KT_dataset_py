import numpy as np # linear algebra
import pandas as pd # data processing

#import our data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#data samples
print("Shape of our train dataset ", train.shape)
print("Shape of our test dataset ", test.shape)
train.sample(5)


test.sample(5)
print(train.columns.values)
print(train.isnull().sum())
print('_'*50)
print(train.info())
#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
train[['Pclass','Survived']].groupby(['Pclass']).mean()
pd.crosstab(train.Pclass,train.Survived,margins=True).style.background_gradient(cmap='summer_r')
train[['Sex','Survived']].groupby(['Sex']).mean()
sns.countplot('Sex',hue='Survived',data=train)

plt.hist(x = [train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()
sns.barplot(x="SibSp", y="Survived", data=train)
plt.hist(x = [train[train['Survived']==1]['SibSp'], train[train['Survived']==0]['SibSp']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('SibSp Size Histogram by Survival')
plt.xlabel('SibSp Size (#)')
plt.ylabel('# of Passengers')
plt.legend()

train[['Parch','Survived']].groupby(['Parch']).mean()
sns.barplot(x="Parch", y="Survived", data=train)
plt.hist(x = [train[train['Survived']==1]['Parch'], train[train['Survived']==0]['Parch']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Parch Size Histogram by Survival')
plt.xlabel('Parch Size (#)')
plt.ylabel('# of Passengers')
plt.legend()
plt.subplot(131)
sns.countplot('Embarked',hue='Survived',data=train)
plt.subplot(132)
sns.countplot('Embarked',hue='Pclass',data=train)
plt.subplot(133)
sns.barplot(x="Embarked", y="Survived", data=train)
train = train.drop(['Cabin','Ticket','Fare','Name' ], axis = 1)
test = test.drop(['Cabin','Ticket','Fare','Name' ], axis = 1)
print(test.isnull().sum())
print(train.isnull().sum())
#filling age
train['Age'].fillna(train['Age'].median(), inplace = True)
test['Age'].fillna(test['Age'].median(), inplace = True)
##filling embarked
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace = True)
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()

embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()
from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.20, random_state = 0)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)
# Support Vector Machines
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)
#Decision Tree
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)
# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)
# KNN or k-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)
# Gradient Boosting Classifier
gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)

idpas= test['PassengerId']
predictions = gbk.predict(test.drop('PassengerId', axis=1))


output = pd.DataFrame({ 'PassengerId' : idpas, 'Survived': predictions })
output.to_csv('submission.csv', index=False)