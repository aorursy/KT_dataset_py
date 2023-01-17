# Let's start by importing the data 

# importing modules that we'll need for the analysis

import pandas as pd

import numpy as np



df = pd.read_csv("../input/diabetes.csv") # importing our data into the notebook

print(df.head(5)) # Examine first columns

print ("") # spacing for better reading

print (df.columns) # All the columns, we'll work with, they are formatted correctly 
# Examining row object of columns, and a doing some summary statistics

print (df.info())

# we have 9 columns and 768 rows



print ("")



print (df.describe())

# there's something wrong with some columns, i think they are missing values for some columns such as BloodPressure since

# there are zeros in the minimum BMI as well as Glucose, BloodPressure, SkinThickness and Insulin

# Let's count the number of these occurences and draw some plots.
#print(df.BMI.value_counts()); print ("") # 11 missing BMI records

#print(df.Glucose.value_counts()); print ("") # 5 missing

#print(df.BloodPressure.value_counts()); print ("") # 35 missing records

#print(df.SkinThickness.value_counts()); print ("") # 227 missing records Not sure about this one.

#print(df.Insulin.value_counts()); print ("") # 374 missing records



# Saving up on space

# i've already indicated the answers to the number of missing values for some colums



# percentage of data missing

missing = 11 + 5 + 35 + 374 # adding up the missing value occurences

msg = "About {}% missing data points in the pima indians dataset if i'm not wrong."

print (msg.format(round(missing/768 * 100))) # finding the percentage of missing data on the columns.



# I'll use an imputation strategy in the workflow to deal with missing data
# Looking for missing values with NA and returns a boolean with columns with missing values

pd.isnull(df).any()



print ("They are no NaN/Na in this dataset.")
# exploratory visualisation to see if there is any correlation between columns

# Drawing all the columns at once seemed like a good idea in the start

# I noticed it was difficult to see the axes, therefore, i made subplots instead



# import necessary modules

import matplotlib.pyplot as plt

import seaborn as sns

sns.set() 



#plt.title("Pairplot between the different columns of the dataset.")

sns.pairplot(df, vars = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin"])



plt.show()
sns.pairplot(df, vars = ["BMI","DiabetesPedigreeFunction","Age"])



plt.show()
# some variables seem correlated. Let's try finding the pearson correlation coefficient for one.

# Let's try a correlation calcutation the correlation coefficient between Insulin and Glucose column 

# correlation coefficient (measure of strength of a linear relationship

# between two variables) 17

Glucoins = np.corrcoef(df.Insulin, df.Glucose)

print (Glucoins)

# that wasn't what i expected. 0.3 is really weak positive linear relationship
# Converting the dataframe into a numpy arrays

# this is a necessary step to proceed with the analysis of the data set for the algorithm in the library to work 



y = df['Outcome'].values # target

X = df.drop('Outcome', axis = 1).values # predictors



print (y)

print (X) ; print ("")

print (type(X))

# successful
# Needed for the next step in model parameter tuning

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# random forest test

# create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2)



# Instantiate classifier

forest = RandomForestClassifier(n_estimators = 10, random_state = 0) 



# fit on training data

forest.fit(X_train, y_train)



# Seeing the metrics

print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split



# create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2) # changed to 2 



# Instantiate classifier

svc = SVC(kernel = "linear") # changed to "linear"



# fit on training data

svc.fit(X_train, y_train)



# Seeing the metrics

print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
# import the necessary modules

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV



# I'll add a standard scaler since SVC works better if the data is scaled.

pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC(kernel = "linear"))])



# Next we'll tune hyperparameters of the estimators separately in the pipeline

param_grid = [

    {'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],

    'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],

    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},

    {'classifier': [RandomForestClassifier(n_estimators=100)],

    'preprocessing': [None], 'classifier__max_features': [1, 2, 3]}]



# Train-test split,instantiate,fit and predict paradigm

# create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2)



# grid search with cross-validation

grid = GridSearchCV(pipe, param_grid, cv = 5)

grid.fit(X_train, y_train)



print("Best params:\n{}\n".format(grid.best_params_))

print("Best cross-validation score: {:.2f}".format(grid.best_score_))

print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))
# Without imputer

# Setting up the pipeline



steps = [('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),

         ('SVM', SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',

  max_iter=-1, probability=False, random_state=None, shrinking=True,

  tol=0.001, verbose=False))]



pipeline = Pipeline(steps)



# Specifying the hyperparameter space

parameters = {'SVM__C':[1] ,

             'SVM__gamma':[0.01]}



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 2)



# Instantiate the GridSearchCV

grid2 = GridSearchCV(pipeline,parameters)



# Fit to the training set

grid2.fit(X_train, y_train)



# Predict the labels of the test set

y_pred = grid2.predict(X_test)



# Compute and print metrics

print("Accuracy: {}".format(grid2.score(X_test, y_test)))

print(classification_report(y_test, y_pred))

print("Tuned Model Parameters: {}".format(grid2.best_params_))

from sklearn.preprocessing import Imputer

from sklearn.model_selection import cross_val_score



# With an imputer



steps = [('Imputer', Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)),

         ('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),

         ('SVM', SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',

  max_iter=-1, probability=False, random_state=None, shrinking=True,

  tol=0.001, verbose=False))]



pipeline = Pipeline(steps)



# Specifying the hyperparameter space

parameters = {'SVM__C':[1] ,

            'SVM__gamma':[0.01]}



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 2)



# Instantiate the GridSearchCV

grid2 = GridSearchCV(pipeline,parameters)



# Fit to the training set

grid2.fit(X_train, y_train)



# Predict the labels of the test set

y_pred = grid2.predict(X_test)



# Compute and print metrics

print("Accuracy: {}".format(grid2.score(X_test, y_test)))

print(classification_report(y_test, y_pred))

print("Tuned Model Parameters: {}".format(grid2.best_params_))



score = cross_val_score(SVC(kernel = "rbf", C = 1, gamma = 0.01), X, y, cv = 5, scoring = 'roc_auc').mean()

print("Accuracy mean cross-validation score: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
# random forests

# without imputer

# rather slow



steps = [('classifier', RandomForestClassifier())]



pipeline = Pipeline(steps)



# Specify the hyperparameter space

parameters = {'classifier__max_features': [1,2,3,4,5],

              'classifier__max_leaf_nodes':[-2,-1,2,3,4],

              'classifier__n_estimators': [100,250,500,10000]}



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2)



# Instantiate the GridSearchCV

forest2 = GridSearchCV(pipeline, param_grid = parameters)



# Fit to the training set

forest2.fit(X_train, y_train)



# Predict the labels of the test set

y_pred = forest2.predict(X_test)



# Compute and print metrics

print("Accuracy: {}".format(forest2.score(X_test, y_test)))

print(classification_report(y_test, y_pred))

print("Tuned Model Parameters: {}".format(forest2.best_params_))

score2 = cross_val_score(RandomForestClassifier(n_estimators = 10000, max_features = 2, max_leaf_nodes =  -1), X, y, cv = 5, scoring = 'roc_auc').mean()

print("Accuracy mean cross-validation score: %0.2f (+/- %0.2f)" % (score2.mean(), score2.std() * 2))


