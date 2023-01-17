# step 1 is to import some useful libraries for data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df,test_df]
train_df.head()
train_df.info()

# we can see that there are 12 columns in the training dataset. 
    # Numerical columns: 
        # Continous: Age, Fare
        # Discrete: SibSp, Parch
    # Categorical columns: 
        # Survived, Sex, Embarked, 
        # Ordinal: Pclass
# check out column names
print(train_df.columns.values)
train_df.tail()
train_df.isnull().any()
test_df.isnull().any()
# see data info

train_df.info()
print('_'*40)
test_df.info()
train_df.describe()
train_df.describe(include=['O'])
# how Pclass associated with survival?
 # look slike higher class had higher chance of survival rate. 

train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', 
                                                                                      ascending = False)
# How gender associate with survival?
# female has 74% of survival rate vs. male only has 18%
train_df[['Sex','Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', 
                                                                                  ascending = False)
# how sibsp associate with survival?
# SibSp = 1 or 2 had the highest survival rate
train_df[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',
                                                                                    ascending = False)
# how Parent and Child associate with survival?

train_df[['Parch','Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by='Survived',
                                                                                      ascending = False)
# import visulization libaries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
g = sns.FacetGrid(train_df, col = 'Survived')
g = g.map(plt.hist, 'Age', bins = 20)
grid = sns.FacetGrid(train_df, col = 'Pclass', row = 'Survived')
grid = grid.map(plt.hist, 'Age', alpha = 0.5, bins = 20)
grid.add_legend()
grid = sns.FacetGrid(train_df,col='Pclass', hue = 'Survived')
grid.map(plt.hist, 'Age', alpha = 0.5)
grid.add_legend()
train_df.head()
grid = sns.FacetGrid(train_df, col='Embarked')
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
grid = sns.FacetGrid(train_df, row = 'Embarked', col = 'Survived')
grid.map(sns.barplot, 'Sex', 'Fare', alpha = 0.5, ci=None)
grid.add_legend()
train_df = train_df.drop(['Ticket', 'Cabin'], axis = 1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis = 1)
combine = [train_df, test_df]
train_df.head()
for dataset in combine: 
    dataset['Title']= dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

pd.crosstab(train_df['Title'], train_df['Sex'])
# We can replace many titles with a more common name or classify them as Rare.

for dataset in combine: 
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Dr', 'Lady','Capt', 'Col',
                                                'Don','Jonkheer', 'Major', 'Rev', 'Sir'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values('Survived', 
                                                                                     ascending = False)
                                       
train_df.head()
grid =sns.FacetGrid(train_df, row = 'Survived', col = 'Title')
grid.map(plt.hist, 'Age')
# since title is categorical, we would like to convert it to numerical
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}

for dataset in combine: 
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title']=dataset['Title'].fillna(0)

train_df.head()
test_df.head()
# then we can drop name now
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape
sex_mapping = {'female': 1, 'male': 0}
for dataset in combine: 
    dataset['Sex'] = dataset['Sex'].map(sex_mapping).astype(int)
train_df.head()
grid = sns.FacetGrid(train_df, row = 'Pclass', col = 'Sex')
grid.map(plt.hist, 'Age', alpha = 0.5, bins = 20)
grid.add_legend()
guess_ages = np.zeros((2,3))
guess_ages
for dataset in combine: 
    for i in range(0,2):
        for j in range(0,3): 
            # a list of non NA age by sex and pclass
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1) ]['Age'].dropna()
            
            age_guess = guess_df.median()
            
            # convert random age float to nearest 0.5 age
            guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5
            
    # assign the guess_ages back to dataset
    for i in range(0,2): 
        for j in range(0,3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i ) & (dataset.Pclass == j+1), \
                        'Age']= guess_ages[i,j]

    dataset['Age'] =dataset['Age'].astype(int)
    
train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'],5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index = False).mean().sort_values('AgeBand', 
                                                                                           ascending = True)
# Let us replace Age with ordinals based on these age bands
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train_df.head()
test_df.head()
# then we can now remove ageband

train_df = train_df.drop('AgeBand', axis = 1)
combine = [train_df, test_df]
train_df.head()
# Create new feature combining existing features
# We can create a new feature for FamilySize which combines Parch and SibSp. 
# This will enable us to drop Parch and SibSp from our datasets.

for dataset in combine: 
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']+1
    
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean().sort_values('Survived', 
                                                                                                 ascending = False)

train_df.head()
# create another feature called isalone

for dataset in combine: 
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index = False).mean()
train_df.head()
test_df.head()
# now we can drop SibSp, Parch, FamilySize in favor of IsAlone
train_df = train_df.drop(['SibSp', 'Parch', 'FamilySize'], axis = 1)
test_df = test_df.drop(['SibSp', 'Parch', 'FamilySize'], axis = 1)

combine = [train_df, test_df]
train_df.head()
# We can also create an artificial feature combining Pclass and Age.

for dataset in combine: 
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
    
train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head()
# see which one still has missing value
train_df.isnull().any()
# noticed that Embarked still has missing value
train_df.describe(include=['O'])

# we can fill in the missing value with the most common occurance
freq_port = train_df['Embarked'].dropna().mode()[0]
freq_port
for dataset in combine: 
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean()

# Looks like C has the highest survival rate
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
for dataset in combine: 
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping).astype(int)
    
train_df.head()
test_df.describe()
# we can see Fair is missing one value 
# 417 total count vs. test dataset has 418 count
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace = True)
test_df.head()
test_df.describe()

# note that Fare now doesn't have any missing value
# create fareband
train_df['FareBand'] = pd.qcut(train_df['Fare'],q = 4)
train_df[['FareBand','Survived']].groupby(['FareBand'], as_index = False).mean().sort_values('FareBand', ascending = True)
test_df.head()
# convert fareband to oridinal numerical columns
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    
# we can now drop fare band for train_df
train_df = train_df.drop('FareBand',axis =1)

combine = [train_df,test_df]

train_df.head()
test_df.head()
# import all the model library
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.svm import SVC, LinearSVC # SVM
from sklearn.naive_bayes import GaussianNB #Naive Bayes classifier
from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.ensemble import RandomForestClassifier #Random Forrest 
from sklearn.linear_model import Perceptron #Perceptron
from sklearn.linear_model import SGDClassifier 

# set the train and test dataset
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
train_df.head()
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
# check the coefficient of the model
coef_df = pd.DataFrame(train_df.columns.delete(0))
coef_df.columns = ['Feature']
coef_df["Correlation"] = pd.Series(logreg.coef_[0])

coef_df.sort_values(by='Correlation', ascending=False)
# SVC
svc = SVC()
svc.fit(X_train,Y_train)
Y_pred=svc.predict(X_test)
acc_svc = round(svc.score(X_train,Y_train)* 100 ,2)
acc_svc

# we can see that the score is 83.84 > logistic regression score of 80.36
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train)*100,2)
acc_gaussian

# this the worst score so far
# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100,2)
acc_perceptron
# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train)*100,2)
acc_linear_svc
# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train)*100,2)
acc_sgd
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred= decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train)*100,2)
acc_decision_tree
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train)*100,2)
acc_random_forest
models = pd.DataFrame(
    {'Model': ['Logistic Regression','KNN','Support Vector Machines','Naive Bayes','Decision Tree','Random Forrest','Perceptron',
               'Stochastic Gradient Decent','Linear SVC'],
     'Score': [acc_log,acc_knn, acc_svc, acc_gaussian,acc_decision_tree,acc_random_forest,acc_perceptron,acc_sgd, acc_linear_svc]
    }
)

models.sort_values(by='Score',ascending = False)

# we can see the best one are Decision Tree and Random Forrest. Since Randomw Forrest is an ensemble of Decision tree, 
# so it will perform better in the long run
# we will use the Random Forrest model for submission
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)

my_submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

# export the submission file
my_submission.to_csv('submission.csv', index = False)

