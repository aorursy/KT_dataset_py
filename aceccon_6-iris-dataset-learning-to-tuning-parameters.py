# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Loading data from sklearn package

from sklearn import datasets

iris = datasets.load_iris()
from sklearn import tree        #importing subpackage from sklearn library

from sklearn.pipeline import make_pipeline

from sklearn import preprocessing

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score



tree_score_table = pd.DataFrame(columns=['Criterion', 'Accuracy'])    #creating table to store accuracy results

tree_crit = ['gini', 'entropy']         #criteria to be tested

tree_score_table['Criterion'] = tree_crit       #fill column 'Criterion' with test_crit



cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)        #determining where to separe test and train data



j = 0



for i in tree_crit:     #loop over criterion

    tree_model = make_pipeline(preprocessing.StandardScaler(), tree.DecisionTreeClassifier(criterion=i))       #initialize decision tree

    tree_score_table.iloc[j, 1] = cross_val_score(tree_model, iris.data, iris.target, cv = cv).mean()        #cross validation for decision tree

    j += 1      #update j

    

print(tree_score_table)

from sklearn.grid_search import GridSearchCV



criteria = ["gini", "entropy"]      #criteria to be tested

min_sample_split_range = [2,10, 20] #min sample split to be tested

max_depth_range = [None, 2, 5, 10]  #max depth to be tested

min_samples_leaf_range = [1, 5, 10] #min samples in the leaf to be tested

min_leaf_nodes_range = [None, 5, 10, 20]    #min leaf nodes to be tested



param_grid = {"criterion": criteria,

              "min_samples_split": min_sample_split_range,

              "max_depth": max_depth_range,

              "min_samples_leaf": min_samples_leaf_range,

              "max_leaf_nodes": min_leaf_nodes_range

                }



grid = GridSearchCV(estimator=tree.DecisionTreeClassifier(), 

                    param_grid=param_grid, 

                    cv = 5, 

                    scoring='accuracy', 

                    refit=True)     #setting grid with estimator



tree_model = make_pipeline(preprocessing.StandardScaler(), grid)    #creating preprocessing

tree_model.fit(iris.data, iris.target)      #fitting data



print("Accuracy of the tuned model: %.4f" %grid.best_score_)

print(grid.best_params_)
from sklearn import svm



kernel_types = ["linear", "poly", "rbf", "sigmoid"]     #types of kernels to be tested

C_range = [0.01, 0.1, 1, 10, 100, 1000]                 #range of C to be tested

degree_range = [1, 2, 3, 4, 5, 6]                       #degrees to be tested



param_grid = {"kernel": kernel_types,

              "C": C_range,

              "degree": degree_range,

              }         #setting grid of parameters



grid = GridSearchCV(estimator = svm.SVC(), 

                    param_grid = param_grid, 

                    cv = 5, 

                    scoring = 'accuracy', 

                    refit = True)   #setting grid with estimator



svm_model = make_pipeline(preprocessing.StandardScaler(), grid)     #creating preprocessing

svm_model.fit(iris.data, iris.target)       #fitting data



print("Accuracy of the tuned model: %.4f" %grid.best_score_)

print(grid.best_params_)
from sklearn.neighbors import KNeighborsClassifier



weight_functions = ["uniform", "distance"]

p_values = [1, 2]

n_range = list(range(1, 51))



param_grid = {"n_neighbors": n_range,

              "weights": weight_functions,

              "p": p_values

              }         #setting grid of parameters



grid = GridSearchCV(estimator = KNeighborsClassifier(), 

                    param_grid = param_grid, 

                    cv = 5, 

                    scoring = 'accuracy', 

                    refit = True)   #setting grid with estimator



knn_model = make_pipeline(preprocessing.StandardScaler(), grid)     #creating preprocessing

knn_model.fit(iris.data, iris.target)       #fitting data



print("Accuracy of the tuned model: %.4f" %grid.best_score_)

print(grid.best_params_)