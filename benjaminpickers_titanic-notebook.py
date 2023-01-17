# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Data analysis and wrangling

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd



#visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

combine = [train_df, test_df]
#What are the features of data

print(train_df.columns.values)
#preview the data

train_df.head()
train_df.tail()
#What are the data types of the various features

train_df.info()

print('_'*40)

test_df.info()
train_df.describe()

#review survived rate using percentiles=[.61, .62] knowing our problem description mentions 38% survival rate

#Review Parch distribution using percentiles=[.75, .8]

#SibSp distribution [.68, .69]

#Age and fare [.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]
train_df.describe(include=['O'])
#Pivot class against survived, we see that upper class passengers had a higher survival rate

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#We can confirm that females had a higher survival rate than males

train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#SibSp and Parch have zero correlation for certain values. It may be best to derive a feature or set of features

#from these individual features

train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Correlating numerical features

#Observations

#-Infants (age<=4) had high survival rate

#-Oldest passenger (age=80) survived

#-Large number of 15-25 year olds did not survive

#-Most passengers are in 15-35 age range



#Decisions

#-We should consider Age in our model training

#-Complete the Age feature for null values

#-We should band age groups



g= sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
#Correlating numerical and ordinal features

#Observations

#-Pclass=3 had the most passengers however most did not survive

#-Infant passengers in Plass=2 and Pclass=3 mostly survived

#-Most passengers in Pclass=1 survived

#-Pclass varies in terms of Age



#Decision

#-Consider Pclass for model training



grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=0.5, bins=20)

grid.add_legend();
#Correlating categorical features 

#Observations

#-Female passengers had much better survival rate than males

#-Exception in Embarked=C where males had higher survival rate, this could be a correlation between PClass and embarked

#And in turn PClass and survived, not necessarily direct correlation between Embarked and Survived

#-Males had better survival rate in Pclass=3 when compared to Pclass=2 for c and q ports

#-Ports of embarkation have varying survival rates for Pclass=3 and among male passengers



#Decision

#-Add sex feature to model training

#-Complete and add Embarked feature to model training



grid = sns.FacetGrid(train_df, row="Embarked", size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
#Correlating catergorical and numerical features

#Observations

#-Higher fare paying passengers had better survival

#-Port of embarkation correlates with survival rate



#Decision

#-Consider banding Fare feature



grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=0.5, ci=None)

grid.add_legend()
#Correcting by dropping features

#Based on our assumptions and decisions, we drop the Cabin and Ticket features



print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
#Creating a new feature extracting from existing

#We want to analyze if Name feature can be engineered to extract titles and test correlation between title and survival

#Before we dropping name and passenger id features



#Observations

#-Most titles band age groups acurately

#-Survival among Titles Age bands varies slightly

#-Certain titles mostly survived (Mme, Lady, Sir) or did not (Don, Rev, Jonkheer)



#Decision

#-We decide to retain the new Title feature for model training



for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
#We can replace many titles with a more common name or classify them as Rare

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#We can convert the categorical titles to ordinal



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master":4, "Rare":5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()
#now we can safely drop the Name feature and PassengerId from our datasets

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name', 'PassengerId'], axis=1)

combine = [train_df, test_df]

print(train_df.shape, test_df.shape)
#Converting Categorical features

#We'll convert features containing strings into numerical features, this is required by most model algorithms

#First lets convert Sex feature to a new feature called gender where female=1 and male=0



for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)

train_df.head()
#Completing a numerical continuos feature

#Now we should start estimating and completing features with missing values or null values

#We will first do this with the Age feature



grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=0.5, bins=20)

grid.add_legend()
#Let us start by preparing an empty array to contain guessed Age values based on Pclass x Gender



guess_ages = np.zeros((2,3))

guess_ages
#now we iterate over Sex(0,1) and Pclass(1,2,3) to calculate guessed values of Age for the six combinations



for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            

            age_guess = guess_df.median()

            

            #convert random age float to nearest 0.5 age

            guess_ages[i, j] = int(age_guess/0.5 + 0.5) * 0.5

    

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_ages[i, j]

    

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()
#Lets create Age bands and determine correlations with survived

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
#Lets us replace Age with ordinals based on these bands

for dataset in combine:

    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()
#We can now remove the Age band feature

train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
#Create new featue combining existing features

#We can create a new feature for FamilySize which combines Parch and SibSp

#This will lets us drop Parch and SibSp from our datasets



for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#We create another feature called IsAlone



for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
#Let us drop Parch, SibSp and FamilySize in favour of IsAlone



train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]



train_df.head()
#We can alos create an artifical feature combining Pclass and Age



for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
#Embarked feature takes S, Q, c values based onport of embarkation

#Our training data has two missing values, we simply fill these with the most common occurence



freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#We can now convert the Embarked feature by creating a numeric port feature



for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S':0, 'C':1, 'Q':2}).astype(int)

    

train_df.head()
#We can now complete the Fare feature for single missing vakues in test data using mode 

# to get the value that occurs most frequently



test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
#We can now create FareBand

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
#Convert the Fare feature to ordinal values based on FareBand



for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[ (dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[ (dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]



train_df.head(10)
test_df.head(10)
#Model, Predict and Solve

#Since we are doing supervised learning plus classification and regression we narrow our models down to

#-Logistic Regression

#-KNN or k-nearest neighbour

#-Support Vector Machines

#-Naive Bayes classifier

#-Decision Tree

#-Random Forest

#-Perceptron

#-Artifical neural network

#-RVM or relevance vector machine



X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test = test_df.copy()

print(X_train.shape, Y_train.shape, X_test.shape)

#Logistic Regression is a useful model to run early in the workflow

#It measures the relationship between catergorical dependent variable and one or more independent variables

#by estimating probabilities using a logistic function, which is cumulative logistic distribution



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

print(acc_log)
#We can use Logistic Regression to validate our assumptions and decisions for feature creating and completing goals

#This can be done by calculating the coefficent of the features in the decision function

#Positve coeffiecents increase the log odds, and thus increase the probability

#Negative coefiecents decrease the log odds, and thus decrease the probability



coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
#Support Vector Machines are supervised learing models with associated learning algorithms that analyze

#data used for classification and regression analysis.



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

print(acc_svc)
#KNN is a non-parametric method 

#A sample is classified by a majority vote of its neighbour, with the sample being assigned to the class

#most common among its k nearest neighbours (k is a postive integer, typically small)



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

print(acc_knn)
#Gaussian Naive Bayes classifiers are a family of classifiers based on applying bayes theorem with strong

#naive independence assumptions between features. Bayes classifiers are highly scalable , requiring a number

#of parameters linear in the number of variables in a learning problem



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

print(acc_gaussian)

#Perceptron 

perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

print(acc_perceptron)
#Linear SVC

linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

print(acc_linear_svc)
#Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

print(acc_sgd)
#Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

print(acc_decision_tree)
#Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(acc_random_forest)
#Model Evaluation

#We can rank our models to chose the best one

#While both Decision tree and random forest score the same, we chose random forest as they correct

#for the decision trees' habit of overfitting



models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Perceptron',

             'Stochastic Gradient Decent', 'Linear SVC', 'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, acc_perceptron, acc_sgd,

             acc_linear_svc, acc_decision_tree]})



models.sort_values(by='Score', ascending=False)
test_df = pd.read_csv('../input/titanic/test.csv')

submission = pd.DataFrame({

    "PassengerId": test_df["PassengerId"],

    "Survived": Y_pred

})



submission.to_csv('Submission.csv', index=False)