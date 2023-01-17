# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



import os

print(os.listdir("../input"))



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

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
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
train_df.head() # Return the first rows of the DataFrame (preview the data)
train_df.columns.values # Which features are available in the dataset? (Columns of the DataFrame)
train_df.info() # Get info about each features in the training set

print('-'*40)

test_df.info()
print(train_df.describe()) # Get statistical information about each NUMERICAL features

print('-'*40)

print(train_df.describe(include=['O']))
train_df[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by=['Survived'], ascending=False)
train_df[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by=['Survived'], ascending=False)
train_df[['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by=['Survived'], ascending=False)
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
# Remove features which are not important

print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
# Add new feature 'Title'

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_df['Title'], train_df['Sex'])
# Change the feature's domain for the rare title

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',

                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir',

                                                 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# Map the Title feature domain to numerical values.

# When not provided, set the title to 0, meaning no title

# Careful, if you run this block twice, all the title will be 0

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()
# Now that we have the 'Title' feature, we can remove the 'Name' feature

# as well as the 'PassengerId' feature in the training set



train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]



train_df.shape, test_df.shape
# Convert all the categorical features to numerical features

# (Why not keeping track of all the dictionary mapping ?)



for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

    

train_df.head()
# Estimate, complete the features with missing values (check the tuto to see the different techniques)

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))



for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



train_df.head()
# Create a new feature 'AgeBand' which are age intervals



train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand']).mean().sort_values(by='Survived', ascending=True)
# Convert the age value by an ordinal corresponding to its AgeBand

for dataset in combine:    

    #.loc[ logical condition, column to modify]

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()
# We can remove 'AgeBand' feature

# Wouldn't it be intersting to keep track of it, to know what value of age map to which age range ? 



train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
# Add new feature which takes into account SibSp and Parch: 'FamilySize'



for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

train_df[['FamilySize', 'Survived']].groupby(['FamilySize']).mean().sort_values(by='Survived', ascending=False)
# Create another feature: IsAlone

# Isn't it redundant with FamilySize ?



for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset.FamilySize == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
# Drop the feature FamilySize, SibSp and Parch since we have IsAlone feature



train_df = train_df.drop(['FamilySize', 'SibSp', 'Parch'], axis=1)

test_df = test_df.drop(['FamilySize', 'SibSp', 'Parch'], axis=1)

combine = [train_df, test_df]

train_df.head()
# Create a feature which combine 'Age' and 'Pclass'



for dataset in combine:

    #print(dataset.Age * dataset.Pclass)

    dataset['Age*Pclass'] = dataset.Age * dataset.Pclass

    

# loc[ row condition, column condition]

train_df.loc[:, ['Age*Pclass', 'Age']].head()
# 'Embarked' has missing values, complete them with the most frequent value



# mode() returns the most frequent value

freq_port = train_df.Embarked.dropna().mode()[0]

freq_port



for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby('Embarked').mean().sort_values(by='Survived', ascending=False)
# Convert the categorical feature 'Embarked' with a numerical value



for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    

train_df.head()
# Fill the N/A value of Fare in the test set by the most frequent one. (actually the median)

# Why only in the test set ? 



test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()

# Create a FareBand to classify the Fare(real value) into ranges

# Only in the train set



# qcut: bins based on sample quantiles

# cut: bins based on values

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand']).mean().sort_values(by='Survived', ascending=False)

# Convert the Fare feature to ordinal values based on FareBand



for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[ (dataset['Fare'] <= 14.454) & (dataset['Fare'] > 7.91), 'Fare'] = 1

    dataset.loc[ (dataset['Fare'] <= 31) & (dataset['Fare'] > 14.454), 'Fare'] = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

train_df = train_df.drop(['FareBand'], axis = 1)

combine = [train_df, test_df]



train_df.head(10)
test_df.head(10)
# Now the data is ready, we are ready to train a model.

# Split the training set into the input X and output y



X_train = train_df.drop('Survived', axis=1)

Y_train = train_df['Survived']



X_test = test_df.drop('PassengerId', axis=1).copy() # copy() useless ?

X_train.shape, Y_train.shape, X_test.shape
# Train a simple model: Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



# Now make the prediction of the test set

predictions = logreg.predict(X_test)



# Compute the accuracy of the model on the training set (#correct predictions/#total predictions)

# LINEAR HYOPTHESIS

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
# We can use Logistic regression to verify our initial assumptions and create new features



coeff_df = pd.DataFrame(train_df.columns.delete(0)) # Create new dataframe containing the features as entries

coeff_df.columns = ['Features']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0]) # coef_ returns the Theta coefficient ([0] only to unpack the innerarray)



# Theta represent the correlation between the feature and the output

# (It assumes features scaling + features take on positive values only ?)

coeff_df.sort_values(by='Correlation', ascending = False)
# Do the same thing with SVM instead of Logistic Regression



svc = SVC(kernel='rbf') # Gaussian Kernel

svc.fit(X_train, Y_train)



predictions = svc.predict(X_test)



# Compute the accuracy of SVM (with kernel) on the TRAINING SET

# NON LINEAR HYOPTHESIS

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
# Do the same thing with K-Nearest Neighbors 



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)



# Make the prediction on the test set

predictions = knn.predict(X_test)



# Compute the accuracy of KNN on the TRAINING SET

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Do the same thing with Gaussian Naive Bayes

# This learning classier relies on Bayes's theorem AND

# strong independance between the features (could explain why the results are bad)



gaussian = GaussianNB() # I do not like the variable name, but I stick to the tutorial

gaussian.fit(X_train, Y_train)



# Make the prediction on the test set

predictions = gaussian.predict(X_test)



# Compute the accuracy of Gaussian Naive Bayes classifier

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian



# The perceptron is an algorithm for supervised learning of binary classifiers.

# It is a type of linear classifier.

# The algorithm allows for online learning, in that it processes elements in the training set one at a time.



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)



# Make the prediction on the test set

predictions = perceptron.predict(X_test)



# Compute the accuracy of perceptron learning model

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Do the same thing with SVM without kernel

# NON LINEAR HYPOTHESIS



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)



# Make the prediction on the test set

predictions = linear_svc.predict(X_test)



# Compute the accuracy of SVM without kernel

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
# Do the same thing with linear classifier with Stochastic Gradient Descent

# By default SVM without kernel



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)



# Make the predictions on the test set

predictions = sgd.predict(X_test)



# Compute the accuracy of SVM with SGD

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
# Do the same thing with decision tree. The branches represent the features

# The leafs reprent the output

# Decision trees where the target variable can take continuous values

# are called regression trees.



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)



# Make the predictions on the test set

predictions = decision_tree.predict(X_test)



# Compute the accuracy of decision tree model

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Do the same thing with Random Forests (multitude of decision tree)

# Ouput the class that is the 'mode', 'most frequent' among the individual trees



random_forest = RandomForestClassifier(n_estimators=100) # number of individual tree = 100

random_forest.fit(X_train, Y_train)



# Make the predictions on the test set

predictions = random_forest.predict(X_test)



# Compute the accuracy of Random Forest

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
# Rank all the learning model based on the accuracy metric

# Submit the one with the highest score



models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})



models.sort_values(by='Score', ascending=False)
# Submit



submission = pd.DataFrame({'PassengerId': test_df['PassengerId'],

                           'Survived': predictions})



submission.to_csv('submission.csv', index=False)