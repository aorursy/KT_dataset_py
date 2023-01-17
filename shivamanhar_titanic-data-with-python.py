import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
path = '../input/'

df = pd.read_csv(path+'train.csv')

test = pd.read_csv(path+'test.csv')
df.head(2)
df.columns = df.columns.str.lower()
df.head(2)
df.dtypes
df.describe(include=['O'])
df = df.drop(['passengerid'], axis=1)
df.head(2)
df[['pclass', 'survived']].groupby(['pclass'], as_index=False).mean().sort_values(by='survived', ascending=False)
df[['sex', 'survived']].groupby(['sex'], as_index=False).mean()
df[['sibsp', 'survived']].groupby(['sibsp']).mean().sort_values(by='survived', ascending=False)
df[['parch', 'survived']].groupby(['parch']).mean().sort_values(by='survived', ascending=False)
g = sns.FacetGrid(df, col='survived', height=4.4)

g.map(plt.hist, 'age', bins=20)

plt.show()
test.head(2)
df = df.drop(['ticket', 'cabin'], axis=1)
df.head(2)
test.columns = test.columns.str.lower()
test.head(2)
test = test.drop(['ticket','cabin','passengerid'], axis=1)
df['title'] = df.name.str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(df['title'],df['sex'])
test['title'] = test.name.str.extract('([A-Za-z]+)\.', expand=False)
df['title'] = df['title'].replace(['Lady','Countess','Capt', 'Col', 'Don','Dr','Major','Rev','Sir','Jonkheer','Don'], 'Rare')
df['title'] = df['title'].replace('Mlle', 'Miss')

df['title'] = df['title'].replace('Ms', 'Miss')

df['title'] = df['title'].replace('Mme', 'Mrs')
test['title'] = test['title'].replace(['Lady','Countess','Capt', 'Col', 'Don','Dr','Major','Rev','Sir','Jonkheer','Don'], 'Rare')
test['title'] = test['title'].replace('Mlle', 'Miss')

test['title'] = test['title'].replace('Ms', 'Miss')

test['title'] = test['title'].replace('Mme', 'Mrs')
df[['title', 'survived']].groupby(['title']).mean().sort_values(by='survived', ascending=False)
sns.countplot(df['title'])

plt.show()
title_map = {'Mr':1,'Mrs':2, 'Miss':3, 'Master':4, 'Rare':5}
df['title'] = df['title'].map(title_map)

df['title'] = df['title'].fillna(0)
test['title'] = test['title'].map(title_map)

test['title'] = test['title'].fillna(0)
df = df.drop(['name'], axis=1)

test = test.drop(['name'], axis=1)

combine = [df, test]
for dataset in combine:

    dataset['sex'] = dataset['sex'].map({'male':1, 'female':2}).astype(int)
df.head(2)
guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['sex'] == i) & (dataset['pclass'] == j+1)]['age'].dropna()

            

            age_guess = guess_df.median()

            print(age_guess)

            guess_ages[i,j] = (age_guess/0.5 + 0.5 ) * 0.5
guess_ages[1] = guess_ages[1].astype(int)
guess_ages[1].round()
df.head()
data1 = df.loc[:,['age','fare']]
data1.plot(subplots=True)

plt.show()
df['age'] = df['age'].interpolate()

test['age'] = test['age'].interpolate()
df['ageband'] = pd.cut(df['age'], 5)
df[['ageband', 'survived']].groupby(['ageband'], as_index=False).mean().sort_values(by='ageband', ascending=True)
for dataset in combine:

    dataset.loc[dataset['age'] <= 16, 'age'] = 0

    dataset.loc[(dataset['age'] > 16) & (dataset['age'] <= 32), 'age'] = 1

    dataset.loc[(dataset['age'] >32) & (dataset['age'] <= 48), 'age'] = 2

    dataset.loc[(dataset['age'] > 48) &(dataset['age'] <= 64), 'age'] = 3

    dataset.loc[dataset['age'] > 64,'age'] =4
df = df.drop(['ageband'], axis=1)

combine= [df, test]

df.head(2)
for dataset  in combine:

    dataset['familysize'] = dataset['sibsp']+dataset['parch']+1

    

df[['familysize', 'survived']].groupby(['familysize'], as_index=False).mean().sort_values(by='survived', ascending=False)
for dataset in combine:

    dataset['isalone'] =0

    dataset.loc[dataset['familysize'] ==1, 'isalone'] = 1

    

df[['isalone', 'survived']].groupby(['isalone'], as_index=False).mean()
df = df.drop(['parch', 'sibsp', 'familysize'], axis=1)

test = test.drop(['parch', 'sibsp', 'familysize'], axis=1)



combine= [df, test]



df.head(2)
for dataset in combine:

    dataset['age_class'] = dataset.age * dataset.pclass



df.loc[:, ['age_class','age', 'pclass']].head(5)
freq_port = df.embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    dataset['embarked'] = dataset['embarked'].fillna(freq_port)
df[['embarked','survived']].groupby(['embarked'], as_index=False).mean().sort_values(by='survived', ascending=False)
for dataset in combine:

    dataset['embarked'] = dataset['embarked'].interpolate()

    dataset['embarked'] = dataset['embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

    

df.head(3)
test['fare'].fillna(df['fare'].dropna().median(), inplace=True)

df.head()
df['fareband'] = pd.qcut(df['fare'], 4)

df[['fareband','survived']].groupby(['fareband'], as_index=False).mean().sort_values(by='fareband', ascending=True)

for dataset in combine:

    dataset.loc[(dataset['fare'] <= 7.91), 'fare'] = 0

    dataset.loc[(dataset['fare'] > 7.91) &(dataset['fare'] <=14.454), 'fare'] = 1

    dataset.loc[(dataset['fare'] > 14.454) &(dataset['fare'] <=31), 'fare'] = 2

    dataset.loc[(dataset['fare'] > 31), 'fare'] =3

    

    

df = df.drop(['fareband'], axis=1)

combine = [df, test]



df.head(5)
test.head(2)
X_train = df.drop('survived', axis=1)

Y_train = df['survived']



X_test = test.copy()

X_train.shape, Y_train.shape, X_test.shape
#assert df['age'].notnull().all()
X_train.dtypes
assert df['fare'].notnull().all()
#X_test.isnull().sum()
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train)*100,2)

acc_log
# Support vector machines

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train)*100, 2)

acc_svc
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train)*100, 2)

acc_gaussian
# Perceptron

perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train)*100, 2)

acc_perceptron
# Linear SVC

linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train)*100, 2)

acc_linear_svc
# Stochatic Gradient Descent

sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train)*100, 2)

acc_sgd
# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train)*100, 2)

acc_decision_tree
# Random Forest

random_forest = RandomForestClassifier(n_estimators =100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train)*100, 2)

acc_random_forest
submission = pd.DataFrame({

        "Index": X_test.index,

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)
submission_csv = pd.read_csv('submission.csv')

submission_csv