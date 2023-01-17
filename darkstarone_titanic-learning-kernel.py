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



from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV, KFold, train_test_split , StratifiedKFold

from sklearn.feature_selection import RFECV



# Data 

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
def plot_histograms( df , variables , n_rows , n_cols ):

    fig = plt.figure( figsize = ( 16 , 12 ) )

    for i, var_name in enumerate( variables ):

        ax=fig.add_subplot( n_rows , n_cols , i+1 )

        df[ var_name ].hist( bins=10 , ax=ax )

        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")

        ax.set_xticklabels( [] , visible=False )

        ax.set_yticklabels( [] , visible=False )

    fig.tight_layout()  # Improves appearance a bit.

    plt.show()



def plot_distribution( df , var , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )

    facet.map( sns.kdeplot , var , shade= True )

    facet.set( xlim=( 0 , df[ var ].max() ) )

    facet.add_legend()



def plot_categories( df , cat , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , row = row , col = col )

    facet.map( sns.barplot , cat , target )

    facet.add_legend()



def plot_correlation_map( df ):

    corr = df.corr()

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )



def describe_more( df ):

    var = [] ; l = [] ; t = []

    for x in df:

        var.append( x )

        l.append( len( pd.value_counts( df[ x ] ) ) )

        t.append( df[ x ].dtypes )

    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )

    levels.sort_values( by = 'Levels' , inplace = True )

    return levels



def plot_variable_importance( X , y ):

    tree = DecisionTreeClassifier( random_state = 99 )

    tree.fit( X , y )

    plot_model_var_imp( tree , X , y )

    

def plot_model_var_imp( model , X , y ):

    imp = pd.DataFrame( 

        model.feature_importances_  , 

        columns = [ 'Importance' ] , 

        index = X.columns 

    )

    imp = imp.sort_values( [ 'Importance' ] , ascending = True )

    imp[ : 10 ].plot( kind = 'barh' )

    print (model.score( X , y ))
# Checks

train_df.head()

train_df.tail()



train_df.info()

print('_'*40)

test_df.info()



print(train_df.describe())

print(train_df.describe(include=['O']))



plot_correlation_map(train_df)
# Pivot Tables

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Visualizations



plot_categories( train_df , cat = 'Embarked' , target = 'Survived' )

plot_categories( train_df , cat = 'Sex' , target = 'Survived' )

plot_categories( train_df , cat = 'Pclass' , target = 'Survived' )

plot_categories( train_df , cat = 'SibSp' , target = 'Survived' )

plot_categories( train_df , cat = 'Parch' , target = 'Survived' )



# Age vs Survival - Numerical

g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)



# Age vs PClass and Survival - Numerical and Ordinal 

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();



# Embarked and Gender and PClass - Categorical 

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()



# Fare vs Embarked and Survived - Categorical and Numerical 

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
# Wrangle Data 

train_df = train_df.drop(['Ticket'], axis=1)

test_df = test_df.drop(['Ticket'], axis=1)

combine = [train_df, test_df]

# Feature Creation



'''

Numerical mapping can also be done with:

from sklearn import preprocessing

def encode_features(df_train, df_test):

    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']

    df_combined = pd.concat([df_train[features], df_test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test

    

data_train, data_test = encode_features(data_train, data_test)

'''



# Extracts titles from names, and converts them into ordinal groups. 

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]

    

# Make Sex a numerical categorical value.

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    

# Handle Cabin

cabin_mapping = {"U": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E" : 5,"F" : 6,"G":7,"T":8}

for dataset in combine:

    dataset['Cabin'] = dataset.Cabin.str.extract('([A-Z])', expand=False)

    dataset['Cabin'] = dataset['Cabin'].fillna('U')

    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

    

# Fill in age group blanks.

# This is done by getting the median of sex and pclass combinations and using that as a basis 

# For the age, with a 0.5 

guess_ages = np.zeros((2,3))

for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)

    

# Band ages and convert to categorical 



train_df['AgeBand'] = pd.cut(train_df['Age'], 5)



for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

    

train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]



# Create IsAlone feature 

for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



# Combine Age and PClass

for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass

    

# Fill missing embarkation values and convert to int.

freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

# Convert fare to numeric.

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)



for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]
# Model, Predict, Solve

X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()



print(X_train, Y_train, X_test)



# Logistic Regression

# Logistic regression measures the relationship between the categorical dependent variable (feature) 

# and one or more independent variables (features) by estimating probabilities using a logistic function, 

# which is the cumulative logistic distribution.

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)



# Correlations.

coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



print(coeff_df.sort_values(by='Correlation', ascending=False))



# Support Vector Machines.

# Supervised learning models with associated learning algorithms that analyze data used for classification

# and regression analysis. Given a set of training samples, each marked as belonging to 

# one or the other of two categories, an SVM training algorithm builds a model that assigns 

# new test samples to one category or the other, making it a non-probabilistic binary linear classifier. 

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)



# k-Nearest Neighbors

# is a non-parametric method used for classification and regression. 

# A sample is classified by a majority vote of its neighbors, with the sample being assigned to the 

# class most common among its k nearest neighbors (k is a positive integer, typically small). 

# If k = 1, then the object is simply assigned to the class of that single nearest neighbor. 

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)



# Gaussian Naive Bayes

# naive Bayes classifiers are a family of simple probabilistic classifiers based on applying 

# Bayes' theorem with strong (naive) independence assumptions between the features. 

# Naive Bayes classifiers are highly scalable, requiring a number of parameters 

# linear in the number of variables (features) in a learning problem.

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)



# Perceptron

# The perceptron is an algorithm for supervised learning of binary classifiers 

# (functions that can decide whether an input, represented by a vector of numbers, 

# belongs to some specific class or not). It is a type of linear classifier, i.e. a classification 

# algorithm that makes its predictions based on a linear predictor function combining a 

# set of weights with the feature vector. The algorithm allows for online learning, in that it processes 

# elements in the training set one at a time. 

perceptron = Perceptron(max_iter=1000, tol=0.003)

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)



# Linear SVC

linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)



# Stochastic Gradient Descent

sgd = SGDClassifier(max_iter=1000, tol=0.003)

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)



# Decision Tree

# A predictive model which maps features (tree branches) to conclusions about the target value 

# (tree leaves). Tree models where the target variable can take a finite set of values are called 

# classification trees; in these tree structures, leaves represent class labels and branches represent 

# conjunctions of features that lead to those class labels.

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)



# Random Forest

# r. Random forests or random decision forests are an ensemble learning method for classification, 

# regression and other tasks, that operate by constructing a multitude of decision trees (n_estimators=100)

# at training time and outputting the class that is the mode of the classes (classification) 

# or mean prediction (regression) of the individual trees.

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)



# Model eval

models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

print(models.sort_values(by='Score', ascending=False))



# Cross Validation

num_test = 0.20

X_train_cross, X_test_cross, Y_train_cross, Y_test_cross = train_test_split(X_train, Y_train, test_size=num_test, random_state=23)



# Choose the type of classifier. 

clf = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train_cross, Y_train_cross)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train_cross, Y_train_cross)



predictions = clf.predict(X_test_cross)

print(accuracy_score(Y_test_cross, predictions))



plot_variable_importance(X_train_cross, Y_train_cross)



print (clf.score( X_train_cross , Y_train_cross ) , clf.score( X_test_cross , Y_test_cross ))



rfecv = RFECV( estimator = clf , step = 1 , cv = StratifiedKFold(n_splits=2) , scoring = 'accuracy' )

#rfecv = RFECV( estimator = clf , step = 1 , scoring = 'accuracy' )

rfecv.fit( X_train_cross , Y_train_cross )



# KFold

def run_kfold(clf):

    kf = KFold(n_splits=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf.split(X_train):

        fold += 1

        X_train_cross, X_test_cross = X_train.values[train_index], X_train.values[test_index]

        Y_train_cross, Y_test_cross = Y_train.values[train_index], Y_train.values[test_index]

        clf.fit(X_train_cross, Y_train_cross)

        predictions = clf.predict(X_test_cross)

        accuracy = accuracy_score(Y_test_cross, predictions)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome)) 



run_kfold(clf)

run_kfold(rfecv)
#Predict and Output



predictions = rfecv.predict(test_df.drop('PassengerId', axis=1))



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": predictions

    })

submission.to_csv('titanic_submission_cross_val_with_removal.csv', index=False)



print(submission.to_string())