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
import pandas as pd

from sklearn.model_selection import train_test_split



# Reading Training Data

file_path_Training= '../input/home-data-for-ml-course/train.csv'

X = pd.read_csv(file_path_Training) 



# Speicify Prediction Target [SalesPrice]

y = X.SalePrice



# Define features

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

#features = ['LotArea','TotRmsAbvGrd']

X = X[features]

#print(X.columns)



# Drop out missing values

X.dropna(axis=0)



# Create test/trian sizes

X_full_train, X_full_valid, y_training, y_validate =  train_test_split(X,y, train_size=0.80, test_size=0.20)

# Random Forest Model

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(X_full_train,y_training)

pred = forest_model.predict(X_full_valid)

print(mean_absolute_error(y_validate, pred))
# Random Forest Model with More Nodes

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



nodes = [5, 10, 20, 50, 100, 250, 500]

MAE = []

for n in nodes:

    my_mae = get_mae(n, X_full_train, X_full_valid, y_training, y_validate)

    print('Mean abs error is {} for {} nodes.'.format(my_mae,n))

    MAE.append(my_mae)



index = MAE.index(min(MAE))

best_node_size = nodes[index]

print('Best amount of nodes {}'.format(best_node_size))



# Final Model

final_model = RandomForestRegressor(max_leaf_nodes=best_node_size, random_state=1)

final_model.fit(X_full_train, y_training)

final_pred = final_model.predict(X_full_valid)

print(mean_absolute_error(y_validate, final_pred))
import pandas as pd

output = pd.DataFrame({'Id': X_full_valid.index,

                       'SalePrice': final_pred})

output.to_csv('submission.csv', index=False)