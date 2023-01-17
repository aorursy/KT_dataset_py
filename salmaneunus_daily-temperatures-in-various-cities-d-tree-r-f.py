# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
temp_path='../input/daily-temperature-of-major-cities/city_temperature.csv'
temp_data = pd.read_csv(temp_path)
temp_data.describe()
temp_data.head()
temp_data.tail()
temp_data
temp_data.describe
temp_data.columns
temp_data = pd.read_csv(temp_path,index_col="Year",parse_dates=True)
y= temp_data.AvgTemperature

feature_names = [ 'Month', 'Day']

X = temp_data[feature_names]
X.describe()
X.head()
from sklearn.tree import DecisionTreeRegressor

temp_model=DecisionTreeRegressor(random_state=1)

temp_model.fit(X,y)
print(X.head())
predictions = temp_model.predict(X)
print(predictions)
y.head()
from sklearn.model_selection import train_test_split

train_X, val_X, train_y,val_y = train_test_split(X,y,random_state=1)

temp_model = DecisionTreeRegressor(random_state=1)



temp_model.fit(train_X,train_y)



val_predictions = temp_model.predict(val_X)


# print the top few validation predictions

print(y.head())

# print the top few actual prices from validation data

print((val_y, val_predictions))
from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_y, val_predictions)

print(val_mae)
def get_mae(max_leaf_nodes, train_X,val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val= model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return mae
candidate_max_leaf_nodes = [5,25,50,100,250,500]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for max_leaf_nodes in [5,50,500,5000]:

    scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, val_mae))



    # Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)

best_tree_size = min(scores , key=scores.get)



    

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

final_model.fit(X,y)
final_model.predict(X)
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor()

rf_model.fit(train_X,train_y)



#calculate mean absolute error of your Random Forest model on your validation data

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y) 

print(rf_val_mae)
rf_model.predict(X)