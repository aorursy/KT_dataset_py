# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

np.seed=2
train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

print(train_df.columns.values)
print(test_df.columns.values)

train_df.head()
train_df.tail()

train_df.info()
print('_'*40)
test_df.info()

train_df.describe()
train_df.describe(include=['O'])

train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[['Parch','Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()



#train_avg_age=train_df["Age"].mean()
#train_std_age=train_df['Age'].std()
#train_count_nan=train_df["Age"].isnull().sum()
#
#rand1=np.random.randint(train_avg_age-train_std_age,train_avg_age+train_std_age,size=train_count_nan)
#train_df["Age"][np.isnan(train_df["Age"])] = rand1
#train_df['Age']=train_df['Age'].astype(int)
#
#train_df.head()
#
#test_avg_age=test_df["Age"].mean()
#test_std_age=test_df['Age'].std()
#test_count_nan=test_df["Age"].isnull().sum()
#
#rand2=np.random.randint(test_avg_age-test_std_age,test_avg_age+test_std_age,size=test_count_nan)
#test_df["Age"][np.isnan(test_df["Age"])] = rand2
#test_df['Age']=test_df['Age'].astype(int)
#
#test_df.head()
#
#combine = [train_df, test_df]

for dataset in combine:
    avg_age=dataset["Age"].mean()
    std_age=dataset['Age'].std()
    count_nan=dataset["Age"].isnull().sum()
    rand1=np.random.randint(avg_age-std_age,avg_age+std_age,size=count_nan)
    dataset["Age"][np.isnan(dataset["Age"])] = rand1
    dataset['Age']=dataset['Age'].astype(int)


for dataset in combine:
    dataset['Family'] = dataset['SibSp'] + dataset['Parch']
    dataset['Family'].loc[dataset['Family'] > 0 ] = 1
    dataset['Family'].loc[dataset['Family'] == 0 ] = 0


train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
combine = [train_df, test_df]

train_df.head()


freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

# Logistic Regression

#logreg = LogisticRegression(C=0.7, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, max_iter=1000000, multi_class='ovr', n_jobs=1,
#          penalty='l2', random_state=None, solver='saga', tol=0.0000001,
#          verbose=0, warm_start=False)
#logreg.fit(X_train, Y_train)
#Y_pred = logreg.predict(X_test)
#logreg.score(X_train, Y_train)
pg = {'C': [0.01, 0.1, 1, 10] }
clf = GridSearchCV(cv=5,
             estimator=LogisticRegression(C=1, intercept_scaling=1, max_iter=10000,   
               dual=False, verbose=0, solver='liblinear', fit_intercept=True, penalty='l2', tol=0.000001),param_grid=pg)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
accuracy=clf.score(X_train, Y_train)
print(accuracy)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

print('The Prediction:')
print(submission)                            
submission.to_csv('titanic.csv', index=False)

Y_train_pred=clf.predict(X_train)
confusion_matrix(Y_train, Y_train_pred)
#clf.best_estimator_
