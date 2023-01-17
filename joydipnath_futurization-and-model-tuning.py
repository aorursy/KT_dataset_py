import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split

import numpy as np

import math



from matplotlib import pyplot as plt

from sklearn import metrics

from sklearn.metrics import mean_squared_error,accuracy_score,f1_score,recall_score,precision_score, confusion_matrix

%matplotlib inline
df = pd.read_csv('../input/concrete (1).csv')
# See what variables are tracked

df.columns
# Just see what the first 3 rows look like

df.head(3)
print('Number of empty entries by column')

df.isnull().sum()
# Split into inputs (x) and targets (y)

x = df.drop('strength', axis=1)



y = pd.DataFrame(df['strength'])

y.columns = ['concrete_compressive_str_MPa']
# The number of samples with which to work

x.shape[0]
df.describe().transpose()
plt.figure(figsize=(10,8))

sns.heatmap(df.drop(columns = 'age').corr(),

            annot=True,

            linewidths=.5,

            center=0,

            cbar=False,

            cmap="YlGnBu")

plt.show()
# Lets check for highly correlated variables

cor = df.corr()

cor.loc[:,:] = np.tril(cor,k=-1)

cor = cor.stack()

cor[(cor > 0.55) | (cor< -0.55)]
sns.pairplot(df, diag_kind= 'kde')

plt.show()
plt.figure(figsize=(25,10))

pos = 1

for i in df.columns:

    plt.subplot(3, 4, pos)

    sns.boxplot(df[i])

    pos += 1 
# X = df.drop(columns = 'age')

# y = df.age
# 80/20 split between testing and training

proportion_of_training = 0.8



# integer intervals [0, t_t_cutoff] and [t_t_cutoff+1, 1030]

train_test_cutoff = int(x.shape[0] * proportion_of_training) 



# train (and validation)

x_train = x.iloc[0:train_test_cutoff]

y_train = y.iloc[0:train_test_cutoff]



# test

x_test = x.iloc[train_test_cutoff+1:]

y_test = y.iloc[train_test_cutoff+1:]



# Now split x_train further into actual training and validation data

# fit on training; tune on validation

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)



x_train = x_train.reset_index(drop=True)

y_train = y_train.reset_index(drop=True)



x_val = x_val.reset_index(drop=True)

y_val = y_val.reset_index(drop=True)



x_test = x_test.reset_index(drop=True)

y_test = y_test.reset_index(drop=True)

for i in x_train.columns:

    q1, q2, q3 = x_train[i].quantile([0.25,0.5,0.75])

    IQR = q3 - q1

    a = x_train[i] > q3 + 1.5*IQR

    b = x_train[i] < q1 - 1.5*IQR

    x_train[i] = np.where(a | b, q2, x_train[i]) 
plt.figure(figsize=(15,10))

pos = 1

for i in x_train.columns:

    plt.subplot(3, 3, pos)

    sns.boxplot(x_train[i])

    pos += 1 
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

lr = LinearRegression()



lr.fit(x_train, y_train)



lr_predictions = lr.predict(x_val)

mse = mean_squared_error(y_val, lr_predictions)

print('{} had an MSE of {}'.format('linear regression', mse))

print('\t this means the average guess is off by {} mega Pascals'.format(math.sqrt(mse)))

print('{} had an R^2 of {}'.format('linear regression', r2_score(y_val, lr_predictions)))

for var_name, coeff in zip(x_train.columns.values, lr.coef_[0]):

    print(var_name, '\t\t', coeff)
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()



# try linear regression with each column standard scaled; THEN should be able 

# to see what factors are important (at least relative to each other) in the lin reg

mms.fit(x_train)

lr.fit(mms.transform(x_train), y_train)



lr_predictions = lr.predict(mms.transform(x_val))

mse = mean_squared_error(y_val, lr_predictions)

print('{} had an MSE of {}'.format('min-max scaled linear regression',

                                   mse))

print('\t this means the average guess is off by {} mega Pascals'.format(math.sqrt(mse)))



print('{} had an R^2 of {}'.format('min-max scaled linear regression',

                                   r2_score(y_val, lr_predictions)))

for var_name, coeff in zip(x_train.columns.values, lr.coef_[0]):

    print(var_name, '\t\t', coeff)
# Let's see the distributions of each of those variables



import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



matplotlib.rcParams['font.size'] = 22





num_cols = round(math.sqrt(x_train.shape[1]))

num_rows = round(math.sqrt(x_train.shape[1])) + 2

sorted_cols = sorted(x_train.columns)





fig = plt.figure(1, figsize=(26, 24))

num_plotted_subplots = 0

for col in sorted_cols:

    num_plotted_subplots += 1

    ax = fig.add_subplot(num_rows, num_cols, num_plotted_subplots)

    

    ax.hist(x_train[col].values, color='skyblue', bins=36)

    

    ax.grid(color='lightgray', linestyle='--', axis='y')

    ax.set_axisbelow(True)

    ax.set_facecolor(color='gray')

    ax.set_xlabel(col)

plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8,

        wspace=0.25, hspace=0.35)

plt.show()
# Let's see what all 2-variable combinations' scatter plots look like (if there's anything interesting)

fig = plt.figure(1, figsize=(30, 55))

matplotlib.rcParams['font.size'] = 22





num_cols = 3

num_rows = 10



num_plotted_subplots = 0

                     

# reverse because plots with `age' x-axis all look very similar

revrese_sorted_cols = [col for col in reversed(sorted_cols)]



for col_x_idx, col_x in enumerate(revrese_sorted_cols):

    # this way, plot all combinations, NOT all permutations

    for col_y in revrese_sorted_cols[col_x_idx:]:

        if col_x == col_y:

            continue

            

        num_plotted_subplots += 1

        ax = fig.add_subplot(num_rows, num_cols, num_plotted_subplots)



        ax.scatter(x_train[col_x].values, x_train[col_y].values, color='orange', s=30)



        ax.grid(color='lightgray', linestyle='--', axis='both')

        ax.set_axisbelow(True)

        ax.set_facecolor(color='gray')

        ax.set_xlabel(col_x)

        ax.set_ylabel(col_y)

plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8,

        wspace=0.25, hspace=0.35)

plt.show()
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=1000, max_depth=None)



rfr.fit(x_train, y_train.values.ravel()) # used ravel() to get rid of a warning message



mse = mean_squared_error(y_val, rfr.predict(x_val))

print('{} had an MSE of {}'.format('random forest regressor', mse))

print('\t this means the average guess is off by {} mega Pascals'.format(math.sqrt(mse)))
from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import zscore

### Number of nearest neighbors

knn_clf = KNeighborsClassifier(n_neighbors= 5 , weights = 'distance' )

# knn_clf.fit(x_train, y_train)



from sklearn import preprocessing

from sklearn import utils

lab_enc = preprocessing.LabelEncoder()

y_train = lab_enc.fit_transform(y_train)

# print(y_train)



x_trainScaled  = x_train.apply(zscore)

x_testScaled  = x_test.apply(zscore)



knn_clf.fit(x_trainScaled, y_train)

mse_knn = mean_squared_error(y_val, knn_clf.predict(x_val))

print('{} had an MSE of {}'.format('random forest regressor', mse_knn))

print('\t this means the average guess is off by {} mega Pascals'.format(math.sqrt(mse_knn)))
from sklearn.tree import DecisionTreeRegressor

from sklearn import tree

from sklearn.externals.six import StringIO  

# from IPython.display import Image  

# from sklearn.tree import export_graphviz

# import pydotplus

# import graphviz

# from os import system



dt = DecisionTreeRegressor(max_depth=3)



dt.fit(x_train, y_train)



mse = mean_squared_error(y_val, dt.predict(x_val))

print('{} had an mse of {}'.format('decision tree regressor', mse))

print('\t this means the average guess is off by {} mega Pascals'.format(math.sqrt(mse)))



# to display the decision tree, export to a .dot file and then convert .dot file to .png

# concrete_dt = open('concrete_dt.dot','w')

# tree.export_graphviz(dt, out_file=concrete_dt, feature_names = x_train.columns.values,

#                 filled=True, impurity=False, proportion=True, rounded=True,

#                 leaves_parallel=False,)

# concrete_dt.close()



# import pydot

# (graph,) = pydot.graph_from_dot_file('concrete_dt.dot')

# graph.write_png('concrete_dt.png')



# retCode = system("dot -Tpng concrete_dt.dot -o concrete_dt.png")

# if(retCode>0):

#     print("system command returning error: "+str(retCode))

# else:

#     display(Image("concrete_dt.png"))
from time import time

from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier
# build a classifier

clf = RandomForestClassifier(n_estimators=50)



param_dist = {"max_depth": [3, None],

              "max_features": [1, 1, 10],

              "min_samples_split": [2, 2, 10],

              "min_samples_leaf": [2, 1, 10],

              "bootstrap": [True, False],

              "criterion": ["gini", "entropy"]}

gs = GridSearchCV(clf,param_dist,cv=2)



# grid_search = GridSearchCV(clf, param_grid=param_dist)

start = time()

gs.fit(x_train, y_train)
# gs.fit(x_train, y_train)

print("Best parameter for Random Forest: ", gs.best_params_)

print("Best Estimator for Random Forest: ", gs.best_estimator_)

print("Mean score for Random Forest: ", gs.cv_results_['mean_test_score'])
## KNN 

param_grid = {'n_neighbors': list(range(1,9)),

             'algorithm': ('auto', 'ball_tree', 'kd_tree' , 'brute') }



gs_knn = GridSearchCV(knn_clf,param_grid,cv=3)

gs_knn.fit(x_train, y_train)
print("Best parameter for KNN: ", gs_knn.best_params_)

# print("Best parameter for KNN: ", gs_knn.cv_results_['params'])/

print("Best Estimator for KNN: ", gs_knn.best_estimator_)

print("Mean score for KNN: ", gs_knn.cv_results_['mean_test_score'])