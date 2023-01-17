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
link_data = pd.read_csv('../input/legal-marketing-link-and-word-count-analysis/link_analysis.csv')
link_data.head()
import seaborn as sns

import matplotlib.pyplot as plt
sns.barplot(x=link_data['Rank'], y=link_data['Majestic External Backlinks - URL (Exact URL)'])
sns.barplot(x=link_data['Rank'], y=link_data['Majestic Referring Domains - URL (Exact URL)'])
sns.barplot(x=link_data['Rank'], y=link_data['Word Count'])
rank_1 = link_data['Rank'] == 1

means_1 = link_data.loc[rank_1, 'Word Count'].mean()

print('Average words for #1 organic position: ' + str(int(means_1)))



rank_2 = link_data['Rank'] == 2

means_2 = link_data.loc[rank_2, 'Word Count'].mean()

print('Average words for #2 organic position: ' + str(int(means_2))) 



rank_3 = link_data['Rank'] == 3

means_3 = link_data.loc[rank_3, 'Word Count'].mean()

print('Average words for #3 organic position: ' + str(int(means_3))) 

        
rank_1 = link_data['Rank'] == 1

means_1 = link_data.loc[rank_1, 'Majestic External Backlinks - URL (Exact URL)'].mean()

print('Average total backlinks for #1 organic position: ' + str(int(means_1)))



rank_2 = link_data['Rank'] == 2

means_2 = link_data.loc[rank_2, 'Majestic External Backlinks - URL (Exact URL)'].mean()

print('Average total backlinks for #2 organic position: ' + str(int(means_2))) 



rank_3 = link_data['Rank'] == 3

means_3 = link_data.loc[rank_3, 'Majestic External Backlinks - URL (Exact URL)'].mean()

print('Average total backlinks for #3 organic position: ' + str(int(means_3))) 
rank_1 = link_data['Rank'] == 1

means_1 = link_data.loc[rank_1, 'Majestic Referring Domains - URL (Exact URL)'].mean()

print('Average referring domains for #1 organic position: ' + str(int(means_1)))



rank_2 = link_data['Rank'] == 2

means_2 = link_data.loc[rank_2, 'Majestic Referring Domains - URL (Exact URL)'].mean()

print('Average referring domains for #2 organic position: ' + str(int(means_2))) 



rank_3 = link_data['Rank'] == 3

means_3 = link_data.loc[rank_3, 'Majestic Referring Domains - URL (Exact URL)'].mean()

print('Average referring domains for #3 organic position: ' + str(int(means_3))) 
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

link_features = ['Word Count', 'Majestic External Backlinks - URL (Exact URL)', 'Majestic Referring Domains - URL (Exact URL)']

y = link_data.Rank

X = link_data[link_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
link_model = DecisionTreeRegressor(random_state=0)

link_model.fit(train_X, train_y)
val_predictions = link_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)

ideal_model = 1.5

for max_leaf_nodes in range(2, 100, 1):

    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    if my_mae < ideal_model:

        ideal_model = my_mae

        print("Max leaf nodes: {} \n Mean Absolute Error:  {}".format(max_leaf_nodes, my_mae))
link_model = DecisionTreeRegressor(max_leaf_nodes=2, random_state=0)

link_model.fit(train_X, train_y)

val_predictions = link_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))
from sklearn.ensemble import RandomForestRegressor

random_link_model = RandomForestRegressor(random_state=1)

random_link_model.fit(train_X, train_y)

random_predict = random_link_model.predict(val_X)

print(mean_absolute_error(val_y, random_predict))
from xgboost import XGBRegressor

random_link_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.9, n_jobs=4)

random_link_model_2.fit(train_X, train_y, 

             early_stopping_rounds=5, 

             eval_set=[(val_X, val_y)],

             verbose=False)

predictions_2 = random_link_model_2.predict(val_X)

print(mean_absolute_error(val_y, predictions_2))
from xgboost import XGBClassifier

random_link_model_3 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.6, gamma=1, gpu_id=-1,

              importance_type='gain', interaction_constraints='',

              learning_rate=0.02, max_delta_step=0, max_depth=3,

              min_child_weight=5, monotone_constraints='()',

              n_estimators=1000, n_jobs=4, num_parallel_tree=1,

              objective='multi:softprob', random_state=0, reg_alpha=0,

              reg_lambda=1, scale_pos_weight=None, subsample=0.6,

              tree_method='exact', validate_parameters=1, verbosity=None)

random_link_model_3.fit(train_X, train_y, 

             early_stopping_rounds=5, 

             eval_set=[(val_X, val_y)],

             verbose=False)

predictions_3 = random_link_model_3.predict(val_X)

print(mean_absolute_error(val_y, predictions_3))

#MAE 1.0714285714285714
'''from sklearn.model_selection import GridSearchCV



params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }



grid = GridSearchCV(random_link_model_3, param_grid=params, n_jobs=4, cv=5, verbose=3 )

grid.fit(X, y)

print('\n All results:')

print(grid.cv_results_)

print('\n Best estimator:')

print(grid.best_estimator_)

print('\n Best score:')

print(grid.best_score_ * 2 - 1)

print('\n Best parameters:')

print(grid.best_params_)'''