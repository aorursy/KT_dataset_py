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

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import GridSearchCV
#-----------------------------------------------------------

# Step 01: load data using panda

#-----------------------------------------------------------

train_df = pd.read_csv('../input/train.csv')  # train set

test_df  = pd.read_csv('../input/test.csv')   # test  set

combine  = [train_df, test_df]
#-----------------------------------------------------------

# Step 02: Acquire and clean data

#-----------------------------------------------------------

train_df.head(5)
train_df.info()
train_df.describe()
train_df.describe(include=['O'])
 # remove Features: Ticket, Cabin

#train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

#test_df  = test_df.drop(['Ticket', 'Cabin'], axis=1)

#combine  = [train_df, test_df]

for dataset in combine:

    dataset['Cabin'] = dataset['Cabin'].fillna('U')

    dataset['Cabin'] = dataset.Cabin.str.extract('([A-Za-z])', expand=False)

    

for dataset in combine:

    dataset['Cabin'] = dataset['Cabin'].map( {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E':4, 

                                            'F':5, 'G':6, 'T':7, 'U':8} ).astype(int)

    

train_df.head()

    
train_df = train_df.drop(['Ticket'], axis=1)

test_df  = test_df.drop(['Ticket'], axis=1)

combine  = [train_df, test_df]





# survival rate distribtion as a function of Pclass

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# obtain Title from name (Mr, Mrs, Miss etc)

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)





for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Dona'],'Royalty')

    dataset['Title'] = dataset['Title'].replace(['Mme'], 'Mrs')

    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'], 'Miss')

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Major','Rev'], 'Officer')

    dataset['Title'] = dataset['Title'].replace(['Jonkheer', 'Don','Sir'], 'Royalty')

    dataset.loc[(dataset.Sex == 'male')   & (dataset.Title == 'Dr'),'Title'] = 'Mr'

    dataset.loc[(dataset.Sex == 'female') & (dataset.Title == 'Dr'),'Title'] = 'Mrs'



#: count survived rate for different titles

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Covert 'Title' to numbers (Mr->1, Miss->2 ...)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royalty":5, "Officer": 6}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



# Remove 'Name' and 'PassengerId' in training data, and 'Name' in testing data

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]



# if age < 16, set 'Sex' to Child

for dataset in combine:

    dataset.loc[(dataset.Age < 16),'Sex'] = 'Child'

    

# Covert 'Sex' to numbers (female:1, male:2)

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0, 'Child': 2} ).astype(int)



train_df.head()
# Age distribution for different values of Pclass and gender

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', bins=20)

grid.add_legend()
# Guess age values using median values for age across set of Pclass and gender frature combinations

guess_ages = np.zeros((2,3))

for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            

            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5



    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

    

    #convert Age to int

    dataset['Age'] = dataset['Age'].astype(int)



# create Age bands and determine correlations with Survived

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']



train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
# Create family size from 'sibsq + parch + 1'

for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)



#create another feature called IsAlone

for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[(dataset['FamilySize'] == 1), 'IsAlone'] = 1

    dataset.loc[(dataset['FamilySize'] > 4), 'IsAlone'] = 2



train_df[['IsAlone','Survived']].groupby(['IsAlone'], as_index=False).mean()





#drop Parch, SibSp, and FamilySize features in favor of IsAlone

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]

train_df.head()
# Create an artfical feature combinbing PClass and Age.

for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head()
# fill the missing values of Embarked feature with the most common occurance

freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)



for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
# fill the missing values of Fare

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)



# Create FareBand

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)



# Convert the Fare feature to ordinal values based on the FareBand

for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
#------------------------------------------------------------------

# Step 03: Learning model

#------------------------------------------------------------------



X_data = train_df.drop("Survived", axis=1)          # data: Features

Y_data = train_df["Survived"]                       # data: Labels

X_test_kaggle  = test_df.drop("PassengerId", axis=1).copy() # test data (kaggle)

Kfold = 5





#split data into random "train" and "test" set 

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=0)

#X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
# Logistic Regression (sklearn)

logreg  = LogisticRegression()

logreg.fit(X_data, Y_data)

acc_log = cross_val_score(logreg, X_data, Y_data, cv=Kfold)

bcc_log = round(logreg.score(X_test, Y_test) * 100, 5)



Y_pred  = logreg.predict(X_test_kaggle)

coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission_Logistic.csv', index=False)
#Learning curve

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit



logreg  = LogisticRegression(C=1)

logreg.fit(X_data, Y_data)

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)



plt.figure()

plt.title("learning curve")

plt.xlabel("Training examples")

plt.ylabel("Score")

train_sizes=np.linspace(.1, 1.0, 15)

train_sizes, train_scores, test_scores = learning_curve(logreg, X_data, Y_data, cv=cv, train_sizes=train_sizes)



train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std  = np.std(train_scores, axis=1)

test_scores_mean  = np.mean(test_scores, axis=1)

test_scores_std   = np.std(test_scores, axis=1)

plt.grid()

    

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")

plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")

                     

plt.legend(loc="best")
# Learning curve

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit



logreg  = LogisticRegression(C=100)

logreg.fit(X_data, Y_data)

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)



plt.figure()

plt.title("Learning Curve")

plt.xlabel("Training examples")

plt.ylabel("Score")

train_sizes=np.linspace(.1, 1.0, 15)



train_sizes, train_scores, test_scores = learning_curve(logreg, X_data, Y_data, cv=cv, n_jobs=4, train_sizes=train_sizes)



train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std  = np.std(train_scores, axis=1)

test_scores_mean  = np.mean(test_scores, axis=1)

test_scores_std   = np.std(test_scores, axis=1)

plt.grid()

    

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")

plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")

                     

plt.legend(loc="best")
# Support Vector Machines

svc = SVC()

svc.fit(X_data, Y_data)

Y_pred  = svc.predict(X_test_kaggle)

acc_svc = cross_val_score(svc, X_data, Y_data, cv=Kfold)

bcc_svc = round(svc.score(X_test, Y_test) * 100, 5)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission_SVM.csv', index=False)
# KNN

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_data, Y_data)

Y_pred = knn.predict(X_test_kaggle)

acc_knn = cross_val_score(knn, X_data, Y_data, cv=Kfold)

bcc_knn = round(knn.score(X_test, Y_test) * 100, 5)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission_KNN.csv', index=False)
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_data, Y_data)

Y_pred = gaussian.predict(X_test_kaggle)

acc_gaussian = cross_val_score(gaussian, X_data, Y_data, cv=Kfold)

bcc_gaussian = round(gaussian.score(X_test, Y_test) * 100, 5)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission_Gassian_Naive_Bayes.csv', index=False)
# Perceptron

perceptron = Perceptron()

perceptron.fit(X_data, Y_data)

Y_pred = perceptron.predict(X_test_kaggle)

acc_perceptron = cross_val_score(perceptron, X_data, Y_data, cv=Kfold)

bcc_perceptron = round(perceptron.score(X_test, Y_test) * 100, 5)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission_Perception.csv', index=False)
# Linear SVC

linear_svc = LinearSVC()

linear_svc.fit(X_data, Y_data)

Y_pred = linear_svc.predict(X_test_kaggle)

acc_linear_svc = cross_val_score(linear_svc, X_data, Y_data, cv=Kfold)

bcc_linear_svc = round(linear_svc.score(X_test, Y_test) * 100, 5)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission_Linear_SVC.csv', index=False)
# Stochastic Gradient Descent

sgd = SGDClassifier()

sgd.fit(X_data, Y_data)

Y_pred = sgd.predict(X_test_kaggle)

acc_sgd = cross_val_score(sgd, X_data, Y_data, cv=Kfold)

bcc_sgd = round(sgd.score(X_test, Y_test) * 100, 5)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission_stochastic_Gradient_Descent.csv', index=False)
# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_data, Y_data)

Y_pred = decision_tree.predict(X_test_kaggle)

acc_decision_tree = cross_val_score(decision_tree, X_data, Y_data, cv=Kfold)

bcc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 5)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission_Decision_Tree.csv', index=False)
# Random Forest

random_forest = RandomForestClassifier(n_estimators=80, random_state =0, min_samples_leaf = 1)

random_forest.fit(X_data, Y_data)

Y_pred = random_forest.predict(X_test_kaggle)

acc_random_forest = cross_val_score(random_forest, X_data, Y_data, cv=Kfold)

bcc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 5)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission_random_forest.csv', index=False)
from sklearn.metrics import roc_auc_score

ne_options = [1,2,4 ,5, 6, 10]



for ne_size in ne_options :

    random_forest = RandomForestClassifier(n_estimators=80, random_state =0, min_samples_leaf = ne_size)

    random_forest.fit(X_data, Y_data)

    bcc_random_forest2 = round(random_forest.score(X_test, Y_test) * 100, 5)

    print(ne_size,bcc_random_forest2)

   
#ensemble votring

ensemble_voting = VotingClassifier(estimators=[('lg', logreg), ('sv', svc), ('rf', random_forest)], voting='hard')

ensemble_voting.fit(X_train, Y_train)

Y_pred = ensemble_voting.predict(X_test_kaggle)



acc_ensemble_voting = cross_val_score(ensemble_voting, X_data, Y_data, cv=Kfold)

bcc_ensemble_voting = round(ensemble_voting.score(X_test, Y_test) * 100, 5)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission_ensemble_voting.csv', index=False)
models = pd.DataFrame({'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',

                                'Random Forest', 'Naive Bayes', 'Perceptron',

                                'Stochastic Gradient Decent', 'Linear SVC',

                                'Decision Tree', 'ensemble_voting'],'KFoldScore': [acc_svc.mean(), acc_knn.mean(), acc_log.mean(),

                                acc_random_forest.mean(), acc_gaussian.mean(), acc_perceptron.mean(),

                                acc_sgd.mean(), acc_linear_svc.mean(), acc_decision_tree.mean(), acc_ensemble_voting.mean()],

                                'Std': [acc_svc.std(), acc_knn.std(), acc_log.std(),

                                acc_random_forest.std(), acc_gaussian.std(), acc_perceptron.std(),

                                acc_sgd.std(), acc_linear_svc.std(), acc_decision_tree.std(), acc_ensemble_voting.std()]})



models.sort_values(by='KFoldScore', ascending=False)
models = pd.DataFrame({'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',

                                'Random Forest', 'Naive Bayes', 'Perceptron',

                                'Stochastic Gradient Decent', 'Linear SVC',

                                'Decision Tree','ensemble_voting'],

                                'CVScore': [bcc_svc/100.0, bcc_knn/100.0, bcc_log/100.0,

                                bcc_random_forest/100.0, bcc_gaussian/100.0, bcc_perceptron/100.0,

                                bcc_sgd/100.0, bcc_linear_svc/100.0, bcc_decision_tree/100.0, bcc_ensemble_voting/100.0]})



models.sort_values(by='CVScore', ascending=False)