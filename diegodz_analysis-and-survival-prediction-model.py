import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

%matplotlib inline

sns.set_style('darkgrid')

colors = ["windows blue","amber", "greyish", "faded green", "dusty purple"]

sns.set_palette(sns.xkcd_palette(colors))

sns.set_palette('hls')

sns.set_context("notebook", 1.5)

alpha = 0.7
test  = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')
train.info()
test.info()
train.head()
test.head()
plt.figure(figsize=(18,5))

plt.subplot(121)

sns.heatmap(train.isnull(), yticklabels=False, cbar=False)

plt.subplot(122)

sns.heatmap(test.isnull(), yticklabels=False, cbar=False)

plt.show()
plt.figure(figsize=(18,7))

plt.subplot(121)

sns.boxplot(x='Pclass', y='Age', data=train)

plt.title('Training set')

plt.subplot(122)

sns.boxplot(x='Pclass', y='Age', data=test)

plt.title('Test set')

plt.show()
# Compute the average age per class using both training and testing datasets. 

Age_Pclass1 = 0.5*(train[train['Pclass']==1]['Age'].mean() + test[test['Pclass']==1]['Age'].mean())

Age_Pclass2 = 0.5*(train[train['Pclass']==2]['Age'].mean() + test[test['Pclass']==2]['Age'].mean())

Age_Pclass3 = 0.5*(train[train['Pclass']==3]['Age'].mean() + test[test['Pclass']==3]['Age'].mean())



print(Age_Pclass1, Age_Pclass2, Age_Pclass3)
# Compute the average fare per class using both training and testing datasets. 

Fare_Pclass1 = 0.5*(train[train['Pclass']==1]['Fare'].mean() + test[test['Pclass']==1]['Fare'].mean())

Fare_Pclass2 = 0.5*(train[train['Pclass']==2]['Fare'].mean() + test[test['Pclass']==2]['Fare'].mean())

Fare_Pclass3 = 0.5*(train[train['Pclass']==3]['Fare'].mean() + test[test['Pclass']==3]['Fare'].mean())



print(Fare_Pclass1, Fare_Pclass2, Fare_Pclass3)
def input_age(cols):

    

    Pclass = cols[0]

    Age = cols[1]



    if pd.isnull(Age):

        if Pclass==1:

            return Age_Pclass1      

        if Pclass==2:

            return Age_Pclass2

        else:

            return Age_Pclass3

    else:

        return Age
def input_fare(cols):

    

    Pclass = cols[0]

    Fare = cols[1]



    if pd.isnull(Fare):

        if Pclass==1:

            return Fare_Pclass1      

        if Pclass==2:

            return Fare_Pclass2

        else:

            return Fare_Pclass3

    else:

        return Fare
# Replace missing values

train['Age'] = train[['Pclass','Age']].apply(input_age, axis=1)

test['Age'] = test[['Pclass','Age']].apply(input_age, axis=1)



train['Fare'] = train[['Pclass','Fare']].apply(input_age, axis=1)

test['Fare'] = test[['Pclass','Fare']].apply(input_age, axis=1)
plt.figure(figsize=(18,5))

plt.subplot(121)

sns.heatmap(train.isnull(), yticklabels=False, cbar=False)

plt.subplot(122)

sns.heatmap(test.isnull(), yticklabels=False, cbar=False)

plt.show()
train.drop('Cabin', axis=1, inplace=True)

test.drop('Cabin', axis=1, inplace=True)

#test.dropna(inplace=True)
plt.figure(figsize=(18,5))

plt.subplot(121)

sns.heatmap(train.isnull(), yticklabels=False, cbar=False)

plt.subplot(122)

sns.heatmap(test.isnull(), yticklabels=False, cbar=False)

plt.show()
train.drop(['Name','Ticket'], inplace=True, axis=1)

test.drop(['Name', 'Ticket'], inplace=True, axis=1)
train.head()
test.head()
sns.pairplot(train)
plt.figure(figsize=(14,12))

sns.heatmap(train.corr(), annot=True, square=True)
plt.figure(figsize=(20,15))

plt.subplot(331)

sns.countplot(x='Survived', data=train, color='grey', alpha=alpha)

plt.subplot(332)

sns.countplot(x='Sex', data=train, color='grey', alpha=alpha)

plt.subplot(333)

sns.countplot(x='Pclass', data=train, color='grey', alpha=alpha)

plt.subplot(334)

sns.countplot(x='Embarked', data=train, color='grey', alpha=alpha)

plt.subplot(335)

sns.countplot(x='SibSp', data=train, color='grey', alpha=alpha)

plt.subplot(336)

sns.countplot(x='Parch', data=train, color='grey', alpha=alpha)

plt.subplot(337)

sns.distplot(train['Age'], color='grey', kde=False, bins=20)

plt.subplot(338)

sns.distplot(train['Fare'], color='grey', kde=False, bins=30)



plt.tight_layout()
plt.figure(figsize=(20,5))

plt.subplot(121)

sns.countplot(x='Sex', data=train, hue='Survived')

plt.title('Survival by sex')

plt.subplot(122)

sns.countplot(x='Pclass', data=train, hue='Survived')

plt.title('Survival by class')

plt.show()



sns.catplot(x='Sex', hue='Survived', col='Pclass', data=train, kind="count")

plt.show()
plt.figure(figsize=(20,5))

plt.subplot(121)

sns.countplot(x='SibSp', data=train, hue='Survived')

plt.title('Survival by number of siblings')

plt.legend(['Died', 'Survived'], loc=1)

plt.subplot(122)

sns.countplot(x='Parch', data=train, hue='Survived')

plt.title('Survival by number of children')

plt.legend(['Died', 'Survived'], loc=1)

plt.show()
plt.figure(figsize=(20,5))

sns.catplot(x='SibSp', hue='Survived', col='Pclass', data=train, kind="count")

plt.show()
plt.figure(figsize=(20,5))

sns.catplot(x='SibSp', hue='Survived', col='Sex', data=train, kind="count")

plt.show()
train_1class = train[train['Pclass']==1]

train_2class = train[train['Pclass']==2]

train_3class = train[train['Pclass']==3]
g=sns.catplot(x='SibSp', hue='Survived', col='Sex', data=train_1class, kind='count')

g.fig.suptitle('1st class', fontsize=25)

plt.show()

g=sns.catplot(x='SibSp', hue='Survived', col='Sex', data=train_2class, kind='count')

g.fig.suptitle('2nd class', fontsize=25)

plt.show()

g=sns.catplot(x='SibSp', hue='Survived', col='Sex', data=train_3class, kind='count', col_order=['female', 'male'])

g.fig.suptitle('3th class', fontsize=25)

plt.show()
g=sns.catplot(x='Parch', hue='Survived', col='Sex', data=train_1class, kind='count')

g.fig.suptitle('1st class', fontsize=25)

plt.show()

g=sns.catplot(x='Parch', hue='Survived', col='Sex', data=train_2class, kind='count')

g.fig.suptitle('2nd class', fontsize=25)

plt.show()

g=sns.catplot(x='Parch', hue='Survived', col='Sex', data=train_3class, kind='count', col_order=['female', 'male'])

g.fig.suptitle('3th class', fontsize=25)

plt.show()
plt.figure(figsize=(20,5))

plt.subplot(121)

sns.countplot(x='Embarked', data=train, hue='Survived')

plt.subplot(122)

sns.countplot(x='Embarked', hue='Pclass', data=train, palette='Set2')

plt.show()
sex_train = pd.get_dummies(train['Sex'],drop_first=True)

sex_test  = pd.get_dummies(test['Sex'],drop_first=True)



#embarked_train = pd.get_dummies(train['Embarked'],drop_first=True)

#embarked_test  = pd.get_dummies(test['Embarked'],drop_first=True)



train.drop(['Sex', 'Embarked'],axis=1,inplace=True)

test.drop(['Sex', 'Embarked'],axis=1,inplace=True)



train = pd.concat([train,sex_train],axis=1)

test  = pd.concat([test,sex_test],axis=1)
train.head()
test.head()
X_train = train.drop(['Survived', 'PassengerId'], axis=1)

y_train = train['Survived']

X_test = test.drop(['PassengerId'], axis=1)
X_train0, X_test0, y_train0, y_test0 = train_test_split(X_train, y_train, test_size=0.3, random_state=101)
logmodel = LogisticRegression()

logmodel.fit(X_train0,y_train0)

predictions = logmodel.predict(X_test0)

acc_lr = round(accuracy_score(y_test0, predictions) * 100, 2)

print('Accuracy (LR): {}'.format(acc_lr))
coefficients = pd.DataFrame(logmodel.coef_[0], X_train0.columns)

coefficients.columns = ['Coefficient']

coefficients
error_rate=[]

for i in range(1,40):

    KNN = KNeighborsClassifier(n_neighbors=i)

    KNN.fit(X_train0,y_train0)

    pred_i = KNN.predict(X_test0)

    error_rate.append(np.mean(pred_i != y_test0))
plt.figure(figsize=(10,5))

plt.plot(range(1,40), error_rate, linestyle='--', color='blue', marker='o',markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=11)

knn.fit(X_train0, y_train0)

predictions = knn.predict(X_test0)

acc_knn = round(accuracy_score(y_test0, predictions) * 100, 2)

print('Accuracy (KNN): {}'.format(acc_knn))
dt = DecisionTreeClassifier()

dt.fit(X_train0, y_train0)

predictions = dt.predict(X_test0)

acc_dt = round(accuracy_score(y_test0, predictions) * 100, 2)

print('Accuracy (Decision Tree): {}'.format(acc_dt))
error_rate=[]

for i in range(1,150):

    RF = RandomForestClassifier(n_estimators=i)

    RF.fit(X_train0,y_train0)

    pred_i = RF.predict(X_test0)

    error_rate.append(np.mean(pred_i != y_test0))
plt.figure(figsize=(10,5))

plt.plot(range(1,150), error_rate, linestyle='--', color='blue', marker='o',markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. n_estimators')

plt.xlabel('n_estimators')

plt.ylabel('Error Rate')
rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train0, y_train0)

predictions = rfc.predict(X_test0)

acc_rfc = round(accuracy_score(y_test0, predictions) * 100, 2)

print('Accuracy (Random Forest): {}'.format(acc_rfc))
param_grid = {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001,0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose=3)

grid.fit(X_train0, y_train0)
grid.best_estimator_
predictions = grid.predict(X_test0)
acc_svc = round(accuracy_score(y_test0, predictions) * 100, 2)

print('Accuracy (SVM): {}'.format(acc_svc))
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'SVM'],

    'Accuracy': [acc_lr, acc_knn, acc_dt, acc_rfc, acc_svc]})

models
rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

predictions = rfc.predict(X_test)
predictions_df = pd.DataFrame(predictions, test['PassengerId'])

predictions_df.columns = ['Survived']

predictions_df.to_csv("predictions_rfc.csv")