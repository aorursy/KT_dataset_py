# Importing the required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# To ignore warnings

import warnings

warnings.filterwarnings("ignore")
# Reading the csv file and putting it into 'df' object.

df = pd.read_csv('../input/income-data-dt/adult_dataset.csv')
# Let's understand the type of values in each column of our dataframe 'df'.

df.info()
# Let's understand the data, how it look like.

df.head()
# rows with missing values represented as'?'.

df_1 = df[df.workclass == '?']

df_1
df_1.info()
# dropping the rows having missing values in workclass

df = df[df['workclass'] != '?']

df.head()
# select all categorical variables

df_categorical = df.select_dtypes(include=['object'])



# checking whether any other columns contain a "?"

df_categorical.apply(lambda x: x=="?", axis=0).sum()
# dropping the "?"s

df = df[df['occupation'] != '?']

df = df[df['native.country'] != '?']
# clean dataframe

df.info()
from sklearn import preprocessing





# encode categorical variables using Label Encoder



# select all categorical variables

df_categorical = df.select_dtypes(include=['object'])

df_categorical.head()
# apply Label encoder to df_categorical



le = preprocessing.LabelEncoder()

df_categorical = df_categorical.apply(le.fit_transform)

df_categorical.head()
# concat df_categorical with original df

df = df.drop(df_categorical.columns, axis=1)

df = pd.concat([df, df_categorical], axis=1)

df.head()
# look at column types

df.info()
# convert target variable income to categorical

df['income'] = df['income'].astype('category')
# Importing train-test-split 

from sklearn.model_selection import train_test_split
# Putting feature variable to X

X = df.drop('income',axis=1)



# Putting response variable to y

y = df['income']
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.30, 

                                                    random_state = 99)

X_train.head()
# Importing decision tree classifier from sklearn library

from sklearn.tree import DecisionTreeClassifier



# Fitting the decision tree with default hyperparameters, apart from

# max_depth which is 5 so that we can plot and read the tree.

dt_default = DecisionTreeClassifier(max_depth=5)

dt_default.fit(X_train, y_train)
# Let's check the evaluation metrics of our default model



# Importing classification report and confusion matrix from sklearn metrics

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# Making predictions

y_pred_default = dt_default.predict(X_test)



# Printing classification report

print(classification_report(y_test, y_pred_default))
# Printing confusion matrix and accuracy

print(confusion_matrix(y_test,y_pred_default))

print(accuracy_score(y_test,y_pred_default))
# Importing required packages for visualization

#from IPython.display import Image  

#from sklearn.externals.six import StringIO  

#from sklearn.tree import export_graphviz

#import pydotplus, graphviz



# Putting features

features = list(df.columns[1:])

features
# If you're on windows:

# Specifing path for dot file.

# import os

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'
# plotting tree with max_depth=3

#dot_data = StringIO()  

#export_graphviz(dt_default, out_file=dot_data,

#                feature_names=features, filled=True,rounded=True)



#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

#Image(graph.create_png())
# GridSearchCV to find optimal max_depth

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'max_depth': range(1, 40)}



# instantiate the model

dtree = DecisionTreeClassifier(criterion = "gini", 

                               random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",return_train_score=True)

tree.fit(X_train, y_train)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with max_depth

plt.figure()

plt.plot(scores["param_max_depth"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_max_depth"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_depth")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# GridSearchCV to find optimal max_depth

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_leaf': range(5, 200, 20)}



# instantiate the model

dtree = DecisionTreeClassifier(criterion = "gini", 

                               random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",return_train_score=True)

tree.fit(X_train, y_train)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with min_samples_leaf

plt.figure()

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_leaf")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# GridSearchCV to find optimal min_samples_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_split': range(5, 200, 20)}



# instantiate the model

dtree = DecisionTreeClassifier(criterion = "gini", 

                               random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",return_train_score=True)

tree.fit(X_train, y_train)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with min_samples_leaf

plt.figure()

plt.plot(scores["param_min_samples_split"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_min_samples_split"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_split")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# Create the parameter grid 

param_grid = {

    'max_depth': range(5, 15, 5),

    'min_samples_leaf': range(50, 150, 50),

    'min_samples_split': range(50, 150, 50),

    'criterion': ["entropy", "gini"]

}



n_folds = 5



# Instantiate the grid search model

dtree = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 

                          cv = n_folds, verbose = 1,return_train_score=True)



# Fit the grid search to the data

grid_search.fit(X_train,y_train)
# cv results

cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results
# printing the optimal accuracy score and hyperparameters

print("best accuracy", grid_search.best_score_)

print(grid_search.best_estimator_)
# model with optimal hyperparameters

clf_gini = DecisionTreeClassifier(criterion = "gini", 

                                  random_state = 100,

                                  max_depth=10, 

                                  min_samples_leaf=50,

                                  min_samples_split=50)

clf_gini.fit(X_train, y_train)
# accuracy score

clf_gini.score(X_test,y_test)
# plotting the tree

#dot_data = StringIO()  

#export_graphviz(clf_gini, out_file=dot_data,feature_names=features,filled=True,rounded=True)



#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

#Image(graph.create_png())
# tree with max_depth = 3

clf_gini = DecisionTreeClassifier(criterion = "gini", 

                                  random_state = 100,

                                  max_depth=3, 

                                  min_samples_leaf=50,

                                  min_samples_split=50)

clf_gini.fit(X_train, y_train)



# score

print(clf_gini.score(X_test,y_test))
# plotting tree with max_depth=3

#dot_data = StringIO()  

#export_graphviz(clf_gini, out_file=dot_data,feature_names=features,filled=True,rounded=True)



#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

#Image(graph.create_png())
# classification metrics

from sklearn.metrics import classification_report,confusion_matrix

y_pred = clf_gini.predict(X_test)

print(classification_report(y_test, y_pred))
# confusion matrix

print(confusion_matrix(y_test,y_pred))