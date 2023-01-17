# Importing libraries - Data Wrangling

import numpy as np

import pandas as pd
# Importing libraries - Data Vusualizaton

import seaborn as sns

import matplotlib.pyplot as plt
import os

print(os.listdir("../input"))
## Importing Data

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',

                'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = pd.read_excel('../input/bostonhousing.xlsx',

                     header = None, names = column_names)
# Understanding/Exploring Data

data.head()
# Checking null values in data

data.isnull().sum()
# Checking data type

data.dtypes
# Checking number of columns and rows in data

data.shape
# More inforamation about data

data.describe()
# Some more detail about data

data.info()
# Checking correlation - between columns with each other

corr_table = data.corr('pearson')

plt.figure(figsize = (10,10))

plt.title("Correlation b/w columns")

plt.figure(1);

sns.heatmap(corr_table,annot = True);
# Correlation graph for all the Columns with MEDV

for var in data.columns[:-1]:

    corr_val = corr_table.ix[var,'MEDV']

    plt.figure(var);

    plt.title('Corr Value of MEDV & '+ var +' is '+ str(corr_val))

    sns.regplot(data[var],data['MEDV']);

    del corr_val

    del var
# Histogram for all the columns

plt.figure(2);

data.hist(figsize = (10,10), grid = False, edgecolor = 'w');

plt.show();
## Segregating features/highly correleated columns and MEDV column (i.e. Price)

prices = data['MEDV']

dataset = data[['CHAS', 'RM', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']]

features = data[['CHAS', 'RM', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
# Histogram

plt.figure(figsize = (7,7))

plt.title("Distribution Plot - MEDV");

sns.distplot(data['MEDV']);

plt.show();
# Box & Whiskers plots

plt.figure()

dataset.plot.box(subplots = True, sharex = False, sharey = False,

                 layout = (3,3), figsize = (10,10))

plt.show()
# Correlation graph for features columns with MEDV

dataset_corr = dataset.corr()

plt.figure(5, figsize=(8,7))

plt.title("Shortlisted Columns - Correlation")

sns.heatmap(dataset_corr, annot = True, cmap='YlGnBu');

plt.show()
## Developing Model - Defining X & Y

X = features

y = prices
# Shuffling & Spliting Data in training and testing

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    test_size = 0.10,

                                                    random_state = 100

                                                    )



# Importing library and randomly selected Max Depth = 5

from sklearn.tree import DecisionTreeRegressor

dtr_cls = DecisionTreeRegressor(max_depth = 5)



# Implementing a model

dtr_cls.fit(X_train, y_train)



# Implementing a model

dtr_cls.fit(X_train, y_train)



# Predicting values on the basis of training data

y_predct = dtr_cls.predict(X_test)



# Checking R2 Score - 1st Attempt

from sklearn.metrics import r2_score

R2_First = r2_score(y_test,y_predct)

print("1st Attempt R2 Score for Decision Tree Model is", r2_score(y_test,y_predct))
# Shuffling & Spliting Data in training and testing

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    test_size = 0.10,

                                                    random_state = 100

                                                    )



# Importing library and randomly selected Max Depth = 5

from sklearn.tree import DecisionTreeRegressor

dtr_cls = DecisionTreeRegressor(max_depth = 5)



# Implementing a model

dtr_cls.fit(X_train, y_train)



# Implementing a model

dtr_cls.fit(X_train, y_train)



# Predicting values on the basis of training data

y_predct = dtr_cls.predict(X_test)



# Checing R2 Score - 2nd Attempt

from sklearn.metrics import r2_score

R2_Second = r2_score(y_test,y_predct)

print("2nd Attempt R2 Score for Decision Tree Model is", r2_score(y_test,y_predct))
# Shuffling & Spliting Data in training and testing

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    test_size = 0.10,

                                                    random_state = 100

                                                    )



# Importing library and randomly selected Max Depth = 5

from sklearn.tree import DecisionTreeRegressor

dtr_cls = DecisionTreeRegressor(max_depth = 5)



# Implementing a model

dtr_cls.fit(X_train, y_train)



# Implementing a model

dtr_cls.fit(X_train, y_train)



# Predicting values on the basis of training data

y_predct = dtr_cls.predict(X_test)



# Checing R2 Score - 3rd Attempt

from sklearn.metrics import r2_score

R2_Third = r2_score(y_test,y_predct)

print("3rd Attempt R2 Score for Decision Tree Model is", r2_score(y_test,y_predct))
# R2 Score for all three attempts, every time we run our model we get different result.

    # Every sample is different and so every sample has different score some times simmilar.

print("1st Attempt R2 Score for Decision Tree Model is", R2_First)

print("2nd Attempt R2 Score for Decision Tree Model is", R2_Second)

print("3rd Attempt R2 Score for Decision Tree Model is", R2_Third)
##  Checking cross value score, K-fold cross validation (cv) = 3

from sklearn.model_selection import cross_val_score

score = cross_val_score(dtr_cls, X, y, cv = 3)

cv_score = score.mean()

print("K Cross Value Score we get when K-Folds = 3 is", cv_score)
# Cross value score for K-fold cross validation (cv) for range 2 to 15

cv_range = range(2, 27)

cv_score_list = []

for cvr in cv_range:

    score = cross_val_score(dtr_cls, X, y, cv = cvr)

    cv_score = score.mean()

    cv_score_list.append(cv_score)

plt.figure(7)

plt.title("K-fold cross validation (cv) for range 2 to 15")

plt.plot(cv_range, cv_score_list)

plt.xlabel("Range")

plt.ylabel("CV Score")

print("We run Cross Value Score from the K-Fold range 2 to 15 and Maximum Cross Value score we get is",

      max(cv_score_list))

##  Importing library Grid Search CV model

from sklearn.model_selection import GridSearchCV
# Defineing maximum depth range for decision tree model

max_depth_range = list(range(1,21))
# Creating parameter grid

param_grid = dict(max_depth = max_depth_range)
# Implementing the grid

grid = GridSearchCV(dtr_cls, param_grid, cv = 2)
# Implementing model

grid.fit(X, y)
# Parametere wise Grid Score

# Parametes

ScoreList = pd.DataFrame(grid.cv_results_['params'])

# scores list

ScoreList['Score'] = pd.DataFrame(grid.cv_results_['mean_test_score'])

print(ScoreList)
# Plotting Maximum Depth Range & Cross Value Mean Score

plt.figure(figsize = (8,4))

plt.title("Maximum Depth Range & Cross Value Mean Score")

plt.plot(ScoreList['max_depth'], ScoreList['Score'])

plt.xlabel("Maximum Depth - Of Decision Tree Model")

plt.ylabel("Cross Value Mean Score");
# Getting the best or best Maximum depth for our model

grid.best_estimator_
# Getting the best or best Maximum depth for our model

grid.best_params_
# Getting the best or best Maximum depth for our model

grid.best_score_