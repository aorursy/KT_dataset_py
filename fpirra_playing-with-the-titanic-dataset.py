# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



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
# loading train and test sets with pandas 

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



# For easier access, we concatenate here the two dataframes

# This is not a copy, is a reference

mixed_data = [train_df, test_df]
print(train_df.columns.values)
# DataFrame.head(n=5)

# Returns first n rows

train_df.head(6)
# DataFrame.info(verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None)

# Returns a Concise summary of a DataFrame

train_df.info()

print('_'*40)

test_df.info()
# DataFrame.describe()

# Generates descriptive statistics that summarize the central tendency, 

# dispersion and shape of a dataset’s distribution, excluding NaN values.

train_df.describe()
# Now, we want to get the categorical types

train_df.describe(include=['O']) 
# Percent of male/female of survivals

train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Percent of each class survivals

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Pclass', ascending=True)
# Percent of survivals, linked to the fare

train_df[['Fare', 'Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Fare', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Embarked', bins=10)
grid = sns.FacetGrid(train_df, col='Survived', row='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# Here, we are creating a new col named title

for dataset in mixed_data:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



print (pd.crosstab(train_df['Title'], train_df['Pclass']))

print ('_'*25)

print (pd.crosstab(train_df['Pclass'], train_df['Sex']))

pd.crosstab(train_df['Title'], train_df['Sex'])
def sexifyClass(sex, pclass):

    if sex == 'male':

        return (pclass - 1)    # Returns 0, 1, 2 

    else:

        return (pclass + 2)    # Returns 3, 4, 5

      

for dataset in mixed_data:

    for index, row in dataset.iterrows():

        dataset.loc[index, "sexifiedClasses"] = sexifyClass(dataset.loc[index, "Sex"] , dataset.loc[index, "Pclass"] )

        

print( train_df.head() )
for dataset in mixed_data:

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Lady', 'Countess', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Random')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

    dataset['Sex'] = dataset['Sex'].replace('male', 0)

    dataset['Sex'] = dataset['Sex'].replace('female', 1)

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# Normalizing values

for dataset in mixed_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('Q')

    dataset['Embarked'] = dataset['Embarked'].replace('Q', 1)

    dataset['Embarked'] = dataset['Embarked'].replace('S', 2)

    dataset['Embarked'] = dataset['Embarked'].replace('C', 3)



    dataset['Title'] = dataset['Title'].replace('Master', 1)

    dataset['Title'] = dataset['Title'].replace('Miss', 2)

    dataset['Title'] = dataset['Title'].replace('Mr', 3)

    dataset['Title'] = dataset['Title'].replace('Mrs', 4)

    dataset['Title'] = dataset['Title'].replace('Random', 5)

    

print (test_df.head()) 
# But the Age and Fare, are too many different values... lets band them!

for dataset in mixed_data:

    dataset['FareBand'] = pd.qcut(dataset['Fare'], 3)

    dataset['AgeBand'] = pd.qcut(dataset['Age'], 4)

    

print( train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
# And, after banding that values, let´s get them into integer values

from random import randint



for dataset in mixed_data:

    dataset.loc[ dataset['Fare'] <= 50, 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 50) & (dataset['Fare'] <= 100), 'Fare'] = 2

    dataset.loc[(dataset['Fare'] > 100) & (dataset['Fare'] <= 150), 'Fare']   = 3

    dataset.loc[ dataset['Fare'] > 150, 'Fare'] = 4

    dataset['Fare'] = dataset['Fare'].fillna(randint(1, 4))

    

    dataset.loc[ dataset['Age'] <= 18, 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 25), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 40), 'Age']   = 3

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 55), 'Age']   = 4

    dataset.loc[ dataset['Age'] > 55, 'Age'] = 5    

    dataset['Age'] = dataset['Age'].fillna(randint(1, 5))

    

print (train_df.head(5))

print (test_df.head(5))
# So now, we can safetely delete some columns...

for dataset in mixed_data:

    del dataset['Name']

    del dataset['Ticket']

    del dataset['Cabin']

    del dataset['FareBand']

    del dataset['AgeBand']

    del dataset['Age']

    del dataset['Pclass']

   



# we don´t need the passenger id in the train set.    

del train_df['PassengerId']



print (test_df.head())
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape



# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log



lr_submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

lr_submission.to_csv('log-reg.csv', index=False)
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc

svm_submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

svm_submission.to_csv('svm.csv', index=False)
# Simple, KNN



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn

knn_submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

knn_submission.to_csv('knn.csv', index=False)
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian

gauss_submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

gauss_submission.to_csv('bayes.csv', index=False)
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree

dt_submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

dt_submission.to_csv('dec-tree.csv', index=False)
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest

rndfor_submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

rndfor_submission.to_csv('rnd-forest.csv', index=False)
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Support Vector Machines', 'KNN', 

              'Naive Bayes', 'Decision Tree', 'Random Forest'],

    'Score': [acc_log, acc_svc, acc_knn,  

              acc_gaussian, acc_decision_tree, acc_random_forest]})

models.sort_values(by='Score', ascending=False)
# And what if we put all together...



mix_results = pd.DataFrame({

    'linear_reg':lr_submission['Survived'],

    'svm': svm_submission['Survived'],

    'knn': knn_submission['Survived'],

    'gauss': gauss_submission['Survived'], 

    'dec_tree': dt_submission['Survived'], 

    'random_for': rndfor_submission['Survived']})



mix_results.head(10)
def vote(a,b,c,d,e,f):

    if (a+b+c+d+e+f) > 3:

        return 1

    else:

        return c*d    



for index, row in mix_results.iterrows():

    mix_results.loc[index, "Survived"] = vote(mix_results.loc[index, "dec_tree"],

                                              mix_results.loc[index, "knn"],

                                              mix_results.loc[index, "random_for"],

                                              mix_results.loc[index, "svm"],

                                              mix_results.loc[index, "linear_reg"],

                                              mix_results.loc[index, "gauss"])



gauss_submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": mix_results['Survived'].astype(int)

    })

gauss_submission.to_csv('algorithm-mix.csv', index=False)
