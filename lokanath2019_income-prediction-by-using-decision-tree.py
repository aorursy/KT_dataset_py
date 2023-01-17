# Import libraries

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# To ignore warning messages

import warnings

warnings.filterwarnings('ignore')
# Adult dataset path

adult_dataset_path = "../input/adult_dataset.csv"



# Function for loading adult dataset

def load_adult_data(adult_path=adult_dataset_path):

    csv_path = os.path.join(adult_path)

    return pd.read_csv(csv_path)
# Calling load adult function and assigning to a new variable df

df = load_adult_data()

# load top 3 rows values from adult dataset

df.head(3)
print ("Rows     : " ,df.shape[0])

print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n" ,df.columns.tolist())

print ("\nMissing values :  ", df.isnull().sum().values.sum())

print ("\nUnique values :  \n",df.nunique())
# Let's understand the type of values present in each column of our adult dataframe 'df'.

df.info()
# Numerical feature of summary/description 

df.describe()
# pull top 5 row values to understand the data and how it's look like

df.head()
# checking "?" total values present in particular 'workclass' feature

df_check_missing_workclass = (df['workclass']=='?').sum()

df_check_missing_workclass
# checking "?" total values present in particular 'occupation' feature

df_check_missing_occupation = (df['occupation']=='?').sum()

df_check_missing_occupation
# checking "?" values, how many are there in the whole dataset

df_missing = (df=='?').sum()

df_missing
percent_missing = (df=='?').sum() * 100/len(df)

percent_missing
# Let's find total number of rows which doesn't contain any missing value as '?'

df.apply(lambda x: x !='?',axis=1).sum()
# dropping the rows having missing values in workclass

df = df[df['workclass'] !='?']

df.head()
# select all categorical variables

df_categorical = df.select_dtypes(include=['object'])



# checking whether any other column contains '?' value

df_categorical.apply(lambda x: x=='?',axis=1).sum()
# dropping the "?"s from occupation and native.country

df = df[df['occupation'] !='?']

df = df[df['native.country'] !='?']
# check the dataset whether cleaned or not?

df.info()
from sklearn import preprocessing



# encode categorical variables using label Encoder



# select all categorical variables

df_categorical = df.select_dtypes(include=['object'])

df_categorical.head()
# apply label encoder to df_categorical

le = preprocessing.LabelEncoder()

df_categorical = df_categorical.apply(le.fit_transform)

df_categorical.head()
# Next, Concatenate df_categorical dataframe with original df (dataframe)



# first, Drop earlier duplicate columns which had categorical values

df = df.drop(df_categorical.columns,axis=1)

df = pd.concat([df,df_categorical],axis=1)

df.head()
# look at column type

df.info()
# convert target variable income to categorical

df['income'] = df['income'].astype('category')
# check df info again whether everything is in right format or not

df.info()
# Importing train_test_split

from sklearn.model_selection import train_test_split
# Putting independent variables/features to X

X = df.drop('income',axis=1)



# Putting response/dependent variable/feature to y

y = df['income']
X.head(3)
y.head(3)
# Splitting the data into train and test

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=99)



X_train.head()
# Importing decision tree classifier from sklearn library

from sklearn.tree import DecisionTreeClassifier



# Fitting the decision tree with default hyperparameters, apart from

# max_depth which is 5 so that we can plot and read the tree.

dt_default = DecisionTreeClassifier(max_depth=5)

dt_default.fit(X_train,y_train)
# Let's check the evaluation metrics of our default model



# Importing classification report and confusion matrix from sklearn metrics

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score



# making predictions

y_pred_default = dt_default.predict(X_test)



# Printing classifier report after prediction

print(classification_report(y_test,y_pred_default))
# Printing confusion matrix and accuracy

print(confusion_matrix(y_test,y_pred_default))

print(accuracy_score(y_test,y_pred_default))
!pip install pydotplus
# Importing required packages for visualization

from IPython.display import Image  

from sklearn.externals.six import StringIO  

from sklearn.tree import export_graphviz

import pydotplus,graphviz



# Putting features

features = list(df.columns[1:])

features
# If you're on windows:

# Specifing path for dot file.

# import os

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'
# plotting tree with max_depth=3

dot_data = StringIO()  

export_graphviz(dt_default, out_file=dot_data,

                feature_names=features, filled=True,rounded=True)



graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
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

                   scoring="accuracy")

tree.fit(X_train, y_train)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
"""

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

"""
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

                   scoring="accuracy")

tree.fit(X_train, y_train)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
"""

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

"""
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

                   scoring="accuracy")

tree.fit(X_train, y_train)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
"""

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

"""
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

                          cv = n_folds, verbose = 1)



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

dot_data = StringIO()  

export_graphviz(clf_gini, out_file=dot_data,feature_names=features,filled=True,rounded=True)



graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
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

dot_data = StringIO()  

export_graphviz(clf_gini, out_file=dot_data,feature_names=features,filled=True,rounded=True)



graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
# classification metrics

from sklearn.metrics import classification_report,confusion_matrix

y_pred = clf_gini.predict(X_test)

print(classification_report(y_test, y_pred))
# confusion matrix

print(confusion_matrix(y_test,y_pred))