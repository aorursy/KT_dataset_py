# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()



SEED = 42



import warnings

warnings.filterwarnings('ignore')
""" Helper Functions"""

def univariate_analysis_continuous(df, feature):

    dataType = df[feature].dtype

    if dataType in ['int64','float64']:

        print(df[feature].describe())

        

        kurt = df[feature].kurt()

        skew = df[feature].skew()

        

        if skew < 0:

            print("\n{:<10} {:.6f}\n--------------------\nThe distribution is skewed left\n".format("Skew", skew))

        elif skew > 0:

            print("\n{:<10} {:.6f}\n--------------------\nThe distribution is skewed right\n".format("Skew", skew))

        else:

            print("No skew\n")

        

        if kurt < 0:

            print("{:<10} {:.6f}\n--------------------\nDistribution is flatter and possess thinner tails\nIt is flatter and less peaked compared to a normal distribution with few values in it's shorter tail ".format("Kurtosis", kurt))

        elif kurt > 0:

            print("{:<10} {:.6f}\n--------------------\nDistribution is peaked and possess thicker tails\nIt has higher peak and taller tails compared to a normal distribution".format("Kurtosis", kurt))

        else:

            print("{:<10} {:.6f}".format("Kurtosis", kurt))

        fig, axs = plt.subplots(ncols=3, figsize=(15,5))

        sns.boxplot(y = iris[feature], ax=axs[0])

        sns.distplot(df[feature], ax=axs[1])

        sns.kdeplot(df[feature], shade=True, ax=axs[2])

    

    elif dataType in ['object']:

        tab = pd.crosstab(index=df[feature], columns="count")

        print("{}\n\n{}".format(tab, tab/tab.sum()))

        data =  df[feature].value_counts()

        labels = df[feature].value_counts().index

        sns.barplot(x=labels, y=data)    



def missing_percentage(data):

    """

    Prints the count of missing values and overall percentage missing for each feature

    """

    rows, cols = data.shape

    num_missing = data.isnull().sum()

    missing_percent = (((data.isnull().sum())/data.shape[0]) * 100)

    print(pd.concat([num_missing, missing_percent], axis=1).rename(columns={0:'Num_Missing',1:'Missing_Percent'}).sort_values(by='Missing_Percent', ascending=False)) 

    

def color_pairplot(setosa, versicolor, virginica):

    """

    Given dataframes respective to each species, we plot pairplot that distinguishes between the speicies

    """

    fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(20,15))

    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

    fig.subplots_adjust(hspace=.5)



    i = j = 0

    for feature2 in features:

        for feature1 in features:

            if feature1 != feature2:

                axs[i,j].set_title(str(round(iris[feature1].corr(iris[feature2]), 4))) 

                sns.scatterplot(x=setosa[feature1], y=setosa[feature2], cmap='Red', label='setosa', ax=axs[i,j])

                sns.scatterplot(x=versicolor[feature1], y=versicolor[feature2], cmap='Purple', label='veriscolor', ax=axs[i,j])

                sns.scatterplot(x=virginica[feature1], y=virginica[feature2], cmap='Green', label='virginica', ax=axs[i,j])

                j += 1

                if j == 3:

                    i += 1

                    j = 0

    
iris = pd.read_csv("/kaggle/input/iris/Iris.csv")

iris.drop('Id', axis=1, inplace=True)
iris.dtypes
iris.head()
iris.describe()
missing_percentage(iris)
univariate_analysis_continuous(iris, 'SepalLengthCm')
univariate_analysis_continuous(iris, 'SepalWidthCm')

univariate_analysis_continuous(iris, 'PetalLengthCm')
univariate_analysis_continuous(iris, 'PetalWidthCm')
univariate_analysis_continuous(iris, 'Species')
sns.pairplot(iris)
sns.heatmap(iris.corr(), annot=True, square=True, cmap='coolwarm', annot_kws={'size': 12})
setosa = iris[iris['Species'] == 'Iris-setosa']

versicolor = iris[iris['Species'] == 'Iris-versicolor']

virginica = iris[iris['Species'] == 'Iris-virginica']

color_pairplot(setosa, versicolor, virginica)
import math

from pprint import pprint # pretty printing



# # Preprocessing

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler



# Model selection

from sklearn.model_selection import StratifiedKFold



# Models

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import svm



# Scoring

from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



from sklearn.model_selection import StratifiedKFold
def evaluate(model, features, target):

    """

    Given a model, the test features, and test target, prints the accuracy, confusion matrix, and log_loss metrics while returning accuracy

    """

    y_pred = model.predict(features)

    accuracy = model.score(X_test, y_test)

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    classification_report = metrics.classification_report(y_test, y_pred)

    

    # Note, log_loss takes in probabilities so categorical data must be converted

    log_loss = metrics.log_loss(y_test, pd.get_dummies(y_pred), eps=1e-15)



    print("Accuracy: {}\n\nConfusion Matrix:\n{}\n\nClassification Report:\n{}\nLog-Loss: {}\n".format(accuracy, confusion_matrix, classification_report, log_loss))

    return accuracy



def compare(base_model, new_model, features, target):

    """

    Given two models, test features, and test target, compares the model metrics of the base and new model

    """

    print("--------------\n| Base Model |\n--------------")

    base_accuracy = evaluate(base_model, features, target)

    print("------------------\n| 'Better' Model |\n------------------")

    new_accuracy = evaluate(new_model, features, target)

    

    print("---------------\n| Improvement |\n---------------\n{:4.3}%".format(100 * (new_accuracy - base_accuracy) / base_accuracy))
# Split Data

y = iris['Species']

X = iris.drop('Species', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2019)



# Pipeline Setup

numeric_transformer = Pipeline(steps=[

('imputer', SimpleImputer(strategy='median')),

('scaler', StandardScaler())])



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])



numeric_features = iris.select_dtypes(include=['int64','float64']).columns

categorical_features = iris.select_dtypes(include=['object']).drop('Species', axis=1).columns



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])
# List of potential classifers we will score for accuracy

classifiers = [

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    LogisticRegression(),

    KNeighborsClassifier(int(math.sqrt(iris.shape[0]))), # K = sqrt(N) where N is the number of samples

    svm.SVC(),

    GaussianNB()

    ]



# Score the classifiers and store them in a tuple containing the model and score they recieved

classifier_ranking = []

for classifier in classifiers:

        pipe = Pipeline(steps=[('preprocessor', preprocessor),

                               ('classifier', classifier)])

        pipe.fit(X_train, y_train)   

        classifier_ranking.append((classifier, pipe.score(X_test, y_test)))



# Sort the classifers in descending order

classifier_ranking.sort(key = lambda x: x[1], reverse=True)

for classifier in classifier_ranking:

        print("{}\nmodel score: {:.3}\n".format(classifier[0], classifier[1]))


split = 4

# Folds are made by preserving the percentage of samples for each class

skf = StratifiedKFold(n_splits=split)





# For each fold, I train KNN for varying values of K and record their scores

# Morever, I plot the accuracy of the model of each fold for each varying K value



fold = 1

fold_scores = []

fig, axs = plt.subplots(ncols=split, figsize=(25,split))

for train_index, test_index in skf.split(X, y):

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y.loc[train_index], y.loc[test_index]

    

    # Score the model with K where K range from 2 to 30 

    scores = []

    x = [x for x in range(2, 31)]

    for i in range(2, 31):

        pipe = Pipeline(steps=[('preprocessor', preprocessor),

                                ('classifier', KNeighborsClassifier(i))])

        pipe.fit(X_train, y_train)   

        

        y_pred = pipe.predict(X_test)

        scores.append(pipe.score(X_test, y_test))

        

    # Record all the scores for the given fold

    fold_scores.append(scores)

        

    # Plot the accuracy of KNN as it varies with K

    sns.lineplot(x, scores, label="fold " + str(fold), ax=axs[fold - 1])

    fold += 1
# A 2D Matrix containing acccuracy for each K in each fold where rows are K and columns are fold

score_matrix = np.array(fold_scores).T



# The average accuracy of all folds for each K 

k_averages = np.average(score_matrix, axis=1).T



# Find the index of K with the best accuracy among the averages of all folds

# Note since our range starts at 0 and K starts at 2, we add 2 to our index to get the correct K

K_index = np.argmax(k_averages)

K = K_index + 2



print("KNN's best average accuracy is {} when K = {}".format(k_averages[K_index], K))
# Default values for hyperparameters

rf = RandomForestClassifier()

pprint(rf.get_params())
# Fill in range for hyperparameters



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]



# Number of features to consider at every split

max_features = ['auto', 'sqrt', 'log2']



# Maximum depth of the tree

max_depth = [int(x) for x in np.linspace(10, 110, 11)]

max_depth.append(None)



# Minimum number of samples required to split a node

min_samples_split = [2, 5, 7, 10]



# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]



# Method of selecting samples for training each tree

bootstrap = [True, False]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

pprint(random_grid)
from sklearn.model_selection import RandomizedSearchCV



# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()



# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 4, verbose=2, n_jobs = -1, random_state = SEED)



# Fit the random search model

rf_random.fit(X_train, y_train)
rf_random.best_params_
# Fit our best random model

best_random = rf_random.best_estimator_

best_random.fit(X_train, y_train)



# Create a baseline model

base_model = RandomForestClassifier()

base_model.fit(X_train, y_train)



compare(base_model, best_random, X_test, y_test)
from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [False],

    'max_depth': [int(x) for x in np.linspace(start = 2, stop = 20, num = 2)],

    'max_features': ['sqrt'],

    'min_samples_leaf': [int(x) for x in np.linspace(start = 1, stop = 5, num = 1)],

    'min_samples_split': [int(x) for x in np.linspace(start = 2, stop = 20, num = 2)],

    'n_estimators': [int(x) for x in np.linspace(start = 800, stop = 1200, num = 20)]

}



# base model for grid search

rf = RandomForestClassifier()



grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 4, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)

compare(base_model, grid_search.best_estimator_, X_test, y_test)