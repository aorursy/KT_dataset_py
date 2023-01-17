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
train_data = pd.read_csv('../input/mlregression-cabbage-price/train_cabbage_price.csv')

test_data = pd.read_csv('../input/mlregression-cabbage-price/test_cabbage_price.csv')

submit_data = pd.read_csv('../input/mlregression-cabbage-price/sample_submit.csv')
train_data.head()
test_data.head()
print(train_data.shape, test_data.shape)
X_train = train_data.drop('avgPrice', axis=1)

y_train = train_data['avgPrice']

X_test = test_data
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV
knn_rgs = KNeighborsRegressor(weights='distance')
parameters = {'n_neighbors':[7, 8],  

              'leaf_size':[60, 70],

              'p':[1, 2]}



grid_knrg = GridSearchCV(knn_rgs, param_grid=parameters, scoring='neg_root_mean_squared_error', cv=6)

grid_knrg.fit(X_train, y_train)



print('The optimal paramters of GridSearchCV :', grid_knrg.best_params_)

print('The best score of GridSearchCV : {:.4f}'.format(grid_knrg.best_score_))

best_knrg = grid_knrg.best_estimator_
predictions = best_knrg.predict(X_test)
submit_data.head()
for i in range(len(predictions)):

    submit_data['Expected'][i]=predictions[i]
submit_data
submit_data.to_csv('submit.csv', mode='w', header= True, index= False)