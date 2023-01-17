import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#my_local_path = "B:/UPX docs/Machine Learning/Project_datasets/Project datasets modified/NYC Taxi Trip/NYC Taxi Trip/"

taxi_data = pd.read_csv('../input/Taxi_new.csv')

taxi_data.head(5)


y = taxi_data["trip_duration"].values



columns = ["hour_of_day", "day_of_month", "month_of_date", "day_of_week_num","distance"]

x = taxi_data[list(columns)].values

x


from sklearn import tree

my_tree_one = tree.DecisionTreeRegressor(criterion="friedman_mse", max_depth=10, random_state=42)

my_tree_one
my_tree_one = my_tree_one.fit(x,y)
print(my_tree_one.score(x, y))
x_pred=my_tree_one.predict(x, check_input=True)

print('This is the length of predicted values of duration:',len(x_pred))

print(x_pred)
# Visualize the decision tree graph



with open('tree.dot','w') as dotfile:

    tree.export_graphviz(my_tree_one, out_file=dotfile, feature_names=columns, filled=True)

    dotfile.close()

    

# You may have to install graphviz package using 

# conda install graphviz

# conda install python-graphviz



from graphviz import Source



with open('tree.dot','r') as f:

    text=f.read()

    plot=Source(text)

plot   
from sklearn import metrics
print ('MAE:', metrics.mean_absolute_error(y, x_pred))

print ('MSE:', metrics.mean_squared_error(y, x_pred))

print ('RMSE:', np.sqrt(metrics.mean_squared_error(y, x_pred)))
list(zip(columns,my_tree_one.feature_importances_))
from sklearn.model_selection import train_test_split



# define a function that accepts a list of features and returns testing RMSE

def train_test_rmse(columns):

    X = taxi_data[columns]

    Y = taxi_data.trip_duration

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123)

    DTreg = tree.DecisionTreeRegressor(criterion="friedman_mse", max_depth=10, random_state=123)

    DTreg.fit(X_train, y_train)

    y_pred = DTreg.predict(X_test)

    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print (train_test_rmse(['distance', 'hour_of_day', 'day_of_month']))

print (train_test_rmse(['distance', 'hour_of_day','day_of_week_num']))

print (train_test_rmse(['distance', 'hour_of_day']))

print (train_test_rmse(['distance', 'hour_of_day','day_of_week_num', 'day_of_month','month_of_date']))

print (train_test_rmse(['distance']))
max_depth = [10,15,20] 

criterion = ['mse', 'friedman_mse']

from sklearn.model_selection import GridSearchCV

#import GridSearchCV
DT_GS = tree.DecisionTreeRegressor()

grid = GridSearchCV(estimator = DT_GS, cv=3, 

                    param_grid = dict(max_depth = max_depth, criterion = criterion))
grid.fit(x,y)
grid.best_score_
# Best parameters for the model



grid.best_params_
new_DT_GS = tree.DecisionTreeRegressor(criterion= 'friedman_mse', max_depth= 10, random_state=42)
new_DT_GS.fit(x,y)
new_DT_GS.score(x,y)
grid_modelPred = new_DT_GS.predict(x)
from sklearn.metrics import mean_squared_error

from math import sqrt
grid_meanSquaredError=mean_squared_error(y, grid_modelPred)

print("MSE:", grid_meanSquaredError)
grid_rootMeanSquaredError = sqrt(grid_meanSquaredError)

print("RMSE:", grid_rootMeanSquaredError)