# calling libraries

import pandas as pd

import numpy as np

import random as rnd

import scipy as sp



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



#Learning curve

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import validation_curve
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

    dataset['Cabin'] = dataset['Cabin'].map( {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E':0, 

                                            'F':0, 'G':0, 'T':0, 'U':1} ).astype(int)

    

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

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4



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
train_df.describe()
#------------------------------------------------------------------

# Step 03: Learning model

#------------------------------------------------------------------



X_data = train_df.drop("Survived", axis=1)          # data: Features

Y_data = train_df["Survived"]                       # data: Labels

X_test_kaggle  = test_df.drop("PassengerId", axis=1).copy() # test data (kaggle)



cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# grid search

def grid_search_model(X, Y, model, parameters, cv):

    CV_model = GridSearchCV(estimator=model, param_grid=parameters, cv=cv)

    CV_model.fit(X, Y)

    CV_model.cv_results_

    print("Best Score:", CV_model.best_score_," / Best parameters:", CV_model.best_params_)

    
#validation curve

def validation_curve_model(X, Y, model, param_name, parameters, cv, ylim, log=True):



    train_scores, test_scores = validation_curve(model, X, Y, param_name=param_name, param_range=parameters,cv=cv, scoring="accuracy")

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)



    plt.figure()

    plt.title("Validation curve")

    plt.fill_between(parameters, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(parameters, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")



    if log==True:

        plt.semilogx(parameters, train_scores_mean, 'o-', color="r",label="Training score")

        plt.semilogx(parameters, test_scores_mean, 'o-', color="g",label="Cross-validation score")

    else:

        plt.plot(parameters, train_scores_mean, 'o-', color="r",label="Training score")

        plt.plot(parameters, test_scores_mean, 'o-', color="g",label="Cross-validation score")



    #plt.ylim([0.55, 0.9])

    if ylim is not None:

        plt.ylim(*ylim)



    plt.ylabel('Score')

    plt.xlabel('Parameter C')

    plt.legend(loc="best")

    

    return plt
# Learning curve

def Learning_curve_model(X, Y, model, cv, train_sizes):



    plt.figure()

    plt.title("Learning curve")

    plt.xlabel("Training examples")

    plt.ylabel("Score")





    train_sizes, train_scores, test_scores = learning_curve(model, X, Y, cv=cv, n_jobs=4, train_sizes=train_sizes)



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

    return plt
# lrearning, prediction and printing results

def predict_model(X, Y, model, Xtest, submit_name):

    model.fit(X, Y)

    Y_pred  = model.predict(Xtest)

    score   = cross_val_score(model, X, Y, cv=cv)



    submission = pd.DataFrame({

            "PassengerId": test_df["PassengerId"],

            "Survived": Y_pred

        })

    submission.to_csv(submit_name, index=False)

    

    return score 
search_param = 0   # 1 -- grid search / 0 -- don't search

plot_vc      = 0   # 1--display validation curve/ 0-- don't display

plot_lc      = 0   # 1--display learning curve/ 0 -- don't display
#grid search: Logistic Regression

model = LogisticRegression()

if search_param==1:

    

    param_range = np.logspace(-6, 5, 12)

    param_grid = dict(C=param_range)

    grid_search_model(X_data, Y_data, model, param_grid, cv)
#Validation Curve: Logistic Regression

if plot_vc == 1:

    param_range = np.logspace(-6, 3, 10)

    param_name="C"

    ylim=[0.55, 0.9]

    validation_curve_model(X_data, Y_data, model, "C", param_range, cv, ylim)
#learn curve

logreg  = LogisticRegression(C=1000)



if plot_lc==1:

    train_size=np.linspace(.1, 1.0, 15)

    Learning_curve_model(X_data, Y_data, logreg, cv, train_size)
# Logistic Regression 

acc_log = predict_model(X_data, Y_data, logreg, X_test_kaggle, 'submission_Logistic.csv')
search_param = 0   # 1 -- grid search / 0 -- don't search

plot_vc      = 0   # 1--display validation curve/ 0-- don't display

plot_lc      = 0   # 1--display learning curve/ 0 -- don't display




#grid search: SVM

search_param = 0

if search_param==1:

    param_range = np.linspace(0.5, 5, 9)

    param_grid = dict(C=param_range)



    grid_search_model(X_data, Y_data, SVC(), param_grid, cv)
#Validation Curve: SVC

if plot_vc == 1:

    param_range = np.linspace(0.1, 10, 10)

    param_name="C"

    ylim=[0.78, 0.90]

    validation_curve_model(X_data, Y_data, SVC(), "C", param_range, cv, ylim, log=False)
#learn curve: SVC

svc = SVC(C=1)



if plot_lc == 1:

    train_size=np.linspace(.1, 1.0, 15)

    Learning_curve_model(X_data, Y_data, svc, cv, train_size)
# Support Vector Machines

acc_svc = predict_model(X_data, Y_data, svc, X_test_kaggle, 'submission_SVM.csv')
search_param = 0   # 1 -- grid search / 0 -- don't search

plot_vc      = 0   # 1--display validation curve/ 0-- don't display

plot_lc      = 0   # 1--display learning curve/ 0 -- don't display
#grid search: KNN

if search_param==1:

    param_range = (np.linspace(1, 10, 10)).astype(int)

    param_grid = dict(n_neighbors=param_range)



    grid_search_model(X_data, Y_data, KNeighborsClassifier(), param_grid, cv)
#Validation Curve: KNN

if plot_vc==1:

    param_range = np.linspace(2, 20, 10).astype(int)

    param_name="n_neighbors"

    ylim=[0.75, 0.90]

    validation_curve_model(X_data, Y_data, KNeighborsClassifier(), "n_neighbors", param_range, cv, ylim, log=False)
#learn curve: KNN

knn = KNeighborsClassifier(n_neighbors = 10)



if plot_lc==1:

    train_size=np.linspace(.1, 1.0, 15)

    Learning_curve_model(X_data, Y_data, knn, cv, train_size)
# KNN

acc_knn = predict_model(X_data, Y_data, knn, X_test_kaggle, 'submission_KNN.csv')
# Gaussian Naive Bayes

gaussian = GaussianNB()

acc_gaussian = predict_model(X_data, Y_data, gaussian, X_test_kaggle, 'submission_Gassian_Naive_Bayes.csv')
# Perceptron

perceptron = Perceptron()

acc_perceptron = predict_model(X_data, Y_data, perceptron, X_test_kaggle, 'submission_Perception.csv')
# Linear SVC

linear_svc = LinearSVC()

acc_linear_svc = predict_model(X_data, Y_data, linear_svc, X_test_kaggle, 'submission_Linear_SVC.csv')
# Stochastic Gradient Descent

sgd = SGDClassifier()

acc_sgd = predict_model(X_data, Y_data, sgd, X_test_kaggle, 'submission_stochastic_Gradient_Descent.csv')
# Decision Tree

decision_tree = DecisionTreeClassifier()

acc_decision_tree = predict_model(X_data, Y_data, decision_tree, X_test_kaggle, 'submission_Decision_Tree.csv')
search_param = 0   # 1 -- grid search / 0 -- don't search

plot_vc      = 0   # 1--display validation curve/ 0-- don't display

plot_lc      = 0   # 1--display learning curve/ 0 -- don't display
#grid search: KNN (This step is very slow)

#param_range = (np.linspace(10, 110, 10)).astype(int)

#param_leaf = (np.linspace(1, 2, 2)).astype(int)

#param_grid = {'n_estimators':param_range, 'min_samples_leaf':param_leaf}



#grid_search_model(X_data, Y_data, RandomForestClassifier(), param_grid, cv)
if plot_vc==1:

    param_range = np.linspace(10, 110, 10).astype(int)

    ylim=[0.75, 0.90]

    validation_curve_model(X_data, Y_data, RandomForestClassifier(min_samples_leaf=12), "n_estimators", param_range, cv, ylim, log=False)
if plot_vc==1:

    param_range = np.linspace(1, 21, 10).astype(int)

    ylim=[0.75, 0.90]

    validation_curve_model(X_data, Y_data, RandomForestClassifier(n_estimators=80), "min_samples_leaf", param_range, cv, ylim, log=False)
# Random Forest

random_forest = RandomForestClassifier(n_estimators=80, random_state =0, min_samples_leaf = 12)

acc_random_forest = predict_model(X_data, Y_data, random_forest, X_test_kaggle, 'submission_random_forest.csv')
#ensemble votring

ensemble_voting = VotingClassifier(estimators=[('lg', logreg), ('sv', svc), ('rf', random_forest),('kn',knn)], voting='hard')

acc_ensemble_voting = predict_model(X_data, Y_data, ensemble_voting, X_test_kaggle, 'submission_ensemble_voting.csv')
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