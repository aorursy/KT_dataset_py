import pandas as pd
import numpy as np

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')
df
X = df.drop('diagnosis', axis=1)
Y = df['diagnosis']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
def summarize_classification(y_test, y_pred):
    
    acc = accuracy_score(y_test, y_pred, normalize=True)
    num_acc = accuracy_score(y_test, y_pred, normalize=False)

    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print("Test data count: ",len(y_test))
    print("accuracy_count : " , num_acc)
    print("accuracy_score : " , acc)
    print("precision_score : " , prec)
    print("recall_score : ", recall)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
parameters = {'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12]}

grid_search = GridSearchCV(DecisionTreeClassifier(), parameters, cv=3, return_train_score=True)
grid_search.fit(x_train, y_train)

grid_search.best_params_
decision_tree_model = DecisionTreeClassifier(max_depth = grid_search.best_params_['max_depth']).fit(x_train, y_train)
y_pred = decision_tree_model.predict(x_test)
summarize_classification(y_test, y_pred)

# example of bayesian optimization with scikit-optimize
from numpy import mean
from sklearn.model_selection import cross_val_score
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

import warnings
warnings.filterwarnings("ignore")
# define the model
model_tree = DecisionTreeClassifier()
# define the search space of hyperparameters to search
search_space = [Integer(1, 12, name='max_depth')]
# define the function used to evaluate a given configuration
@use_named_args(search_space)
def evaluate_model(**params):
    # something
    model_tree.set_params(**params)
    # calculate 10-fold cross validation
    result = cross_val_score(model_tree, x_train, y_train, cv=10, n_jobs=-1, scoring='accuracy')
    # calculate the mean of the scores
    estimate = mean(result)
    return 1.0 - estimate
# perform optimization
result = gp_minimize(evaluate_model, search_space)

print('Best Accuracy: %.f' % (1.0 - result.fun))
print('Best Parameters: max_depth=%d' % (result.x[0]))
model_tree= DecisionTreeClassifier( max_depth = result.x[0]).fit(x_train, y_train)
y_pred_tree = model_tree.predict(x_test)
summarize_classification(y_test, y_pred_tree)
from skopt.plots import plot_convergence
plot_convergence(result);
from sklearn.neighbors import KNeighborsClassifier
# define the model
model_kn =KNeighborsClassifier()


# define the search space of hyperparameters to search
search_space = [Integer(1, 12, name='n_neighbors'), Integer(1, 3, name='p')]

# define the function used to evaluate a given configuration
@use_named_args(search_space)
def evaluate_model(**params):
    # something
    model_kn.set_params(**params)
    # calculate 10-fold cross validation
    result = cross_val_score(model_kn, x_train, y_train, cv=10, n_jobs=-1, scoring='accuracy')
    # calculate the mean of the scores
    estimate = mean(result)
    return 1.0 - estimate

# perform optimization
result = gp_minimize(evaluate_model, search_space)
# summarizing finding:
print('Best Accuracy: %.f' % (1.0 - result.fun))
print('Best Parameters: n_neighbors=%d, p=%d' % (result.x[0], result.x[1]))
model_kn = KNeighborsClassifier( n_neighbors = result.x[0],p=result.x[1]).fit(x_train, y_train)
y_pred = model_kn.predict(x_test)
summarize_classification(y_test, y_pred)
from skopt.plots import plot_convergence
plot_convergence(result);

