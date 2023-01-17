# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#train set

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')



#test set

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
#train_df

print(train_df.shape)
train_df.head()
#test_df

print(test_df.shape)



test_df.head()
train_df.describe()
train_df.describe(include=['O'])
train_df.columns
train_df.info()
train_df['Survived'].value_counts()
df_train_test = pd.concat([train_df, test_df], axis=0, join='outer', ignore_index=False, keys=None,

          levels=None, names=None, verify_integrity=False, copy=True) 

print(df_train_test.shape)



df_train_test.head(10)
df_train_test.isnull().sum()
df_train_test.drop('Cabin',axis=1,inplace=True)
#imputing NaNs



df_train_test['Age'] = df_train_test['Age'].fillna(df_train_test['Age'].median())
df_train_test[df_train_test['Embarked'].isnull()]
print(df_train_test.Embarked.value_counts(dropna=False))
# Update Missing Embarked values with C since C has more values.

df_train_test.Embarked.fillna(df_train_test.Embarked.mode()[0],inplace=True)
print(df_train_test.Embarked.value_counts(dropna=False))
df_train_test.isnull().sum()
df_train_test[df_train_test['Fare'].isnull()]
df_train_test.Fare.fillna(df_train_test.Fare.median(),inplace=True)
df_train_test[df_train_test.Survived.isnull() == True].shape
df_test = df_train_test[df_train_test.Survived.isnull() == True]



df_test = df_test.drop('Survived', axis = 1)



df_test.columns
df_test.shape
df_train = df_train_test.dropna(axis = 0)
df_train.shape
## PClass vs Survived

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
##Sex vs Survived

df_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
## Sibsp vs Survived

df_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
## Parch vs Survived

df_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(df_train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# grid = sns.FacetGrid(train_df, col='Embarked')

grid = sns.FacetGrid(df_train, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(df_train, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
# since 'sex' is a binary column, each value can be represented as such

# convert the 'sex' col into a binary indicator column



sex_train = pd.get_dummies(df_train['Sex'], drop_first = True) 

# drop_first = True to take away redundant info

# The first column was a perfect predicotr of the second column



# The same could be done to the 'embark' col

embark_train = pd.get_dummies(df_train['Embarked'], drop_first = True)

# Removing one column could remove the 'perfect predictor' aspect
# Combine the indicator columns with the original dataset and then remove the original columns that were adjusted

df_train_adj = pd.concat([df_train, sex_train, embark_train], axis = 1)

df_train_adj.head()
df_train_adj.drop('Sex', axis = 1, inplace = True)

df_train_adj.drop('Embarked', axis = 1, inplace = True)

df_train_adj.drop('Ticket', axis = 1, inplace = True)





df_train_adj.head()
corr_matrix=df_train_adj.corr()

corr_matrix["Survived"].sort_values(ascending=False)
sex_test = pd.get_dummies(df_test['Sex'], drop_first = True) 

embark_test = pd.get_dummies(df_test['Embarked'], drop_first = True)
df_test_adj = pd.concat([df_test, sex_test, embark_test], axis = 1)

df_test_adj.head()
df_test_adj.shape
df_test_adj.drop('Sex', axis = 1, inplace = True)

df_test_adj.drop('Embarked', axis = 1, inplace = True)

df_test_adj.drop('Ticket', axis = 1, inplace = True)

df_test_adj.head()
df_train_adj.hist(figsize = (30, 35), bins = 50, xlabelsize = 8, ylabelsize = 8, color='orange');
combine = [df_train_adj, df_test_adj]

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(df_train_adj['Title'], df_train_adj['male'])
combine = [df_train_adj, df_test_adj]



for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

df_train_adj[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



df_train_adj.head()
df_train_adj = df_train_adj.drop(['Name','PassengerId'], axis=1)

df_test_adj = df_test_adj.drop(['Name'], axis=1)

combine = [df_train_adj, df_test_adj]

df_train_adj.shape, df_test_adj.shape
df_train_adj.head()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

grid = sns.FacetGrid(df_train_adj, row='Pclass', col='male', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['male'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.male == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



df_train_adj.head()
df_train_adj['AgeBand'] = pd.cut(df_train_adj['Age'], 5)

df_train_adj[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
# Mapping Fare

combine = [df_train_adj,df_test_adj]

    

for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

df_train_adj.head()
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

df_train_adj.head()
df_train_adj["Family_size"] = df_train_adj["SibSp"] + df_train_adj["Parch"]+1

df_train_adj["IsAlone"] = 1

df_train_adj['IsAlone'].loc[df_train_adj['Family_size'] > 1] = 0
df_test_adj["Family_size"] = df_test_adj["SibSp"] + df_test_adj["Parch"]+1

df_test_adj["IsAlone"] = 1

df_test_adj['IsAlone'].loc[df_test_adj['Family_size'] > 1] = 0
df_train_adj = df_train_adj.drop(['SibSp', 'Parch'], axis=1)

df_test_adj = df_test_adj.drop(['SibSp', 'Parch'], axis=1)
df_train_adj = df_train_adj.drop(['Family_size'], axis=1)

df_test_adj = df_test_adj.drop(['Family_size'], axis=1)
df_test_adj.columns
df_train_adj = df_train_adj[['Survived','Pclass','male','Age', 'Fare',   'Q', 'S','IsAlone', 'Title']]

df_test_adj = df_test_adj[['PassengerId','Pclass','male','Age', 'Fare',   'Q', 'S','IsAlone', 'Title']]



train = df_train_adj.values

test = df_test_adj.values


from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression



classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

	AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



log_cols = ["Classifier", "Accuracy"]

log 	 = pd.DataFrame(columns=log_cols)



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



X = train[0::, 1::]

y = train[0::, 0]



acc_dict = {}



for train_index, test_index in sss.split(X, y):

	X_train, X_test = X[train_index], X[test_index]

	y_train, y_test = y[train_index], y[test_index]

	

	for clf in classifiers:

		name = clf.__class__.__name__

		clf.fit(X_train, y_train)

		train_predictions = clf.predict(X_test)

		acc = accuracy_score(y_test, train_predictions)

		if name in acc_dict:

			acc_dict[name] += acc

		else:

			acc_dict[name] = acc



for clf in acc_dict:

	acc_dict[clf] = acc_dict[clf] / 10.0

	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

	log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
from sklearn.model_selection import train_test_split



predictors = df_train_adj.drop(['Survived'], axis=1)

target = df_train_adj["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_val)

acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gaussian)
# Logistic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_val)

acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_logreg)
# Support Vector Machines

from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_val)

acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_svc)
# Linear SVC

from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_val)

acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_linear_svc)
# Perceptron

from sklearn.linear_model import Perceptron



perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred = perceptron.predict(x_val)

acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_perceptron)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier



decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred = decisiontree.predict(x_val)

acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_decisiontree)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_val)

acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_randomforest)
# KNN or k-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_val)

acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_knn)
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_val)

acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_sgd)
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_val)

acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],

    'Score': [acc_svc, acc_knn, acc_logreg, 

              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,

              acc_sgd, acc_gbk]})

models.sort_values(by='Score', ascending=False)
#set ids as PassengerId and predict survival 

ids = df_test_adj['PassengerId']

predictions = gbk.predict(df_test_adj.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)