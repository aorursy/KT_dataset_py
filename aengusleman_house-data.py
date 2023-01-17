# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train_data.head()
y = train_data['SalePrice']
features_numeric = ['LotArea','YearRemodAdd','BedroomAbvGr','TotRmsAbvGrd','PoolArea']

features_categorical = ['Utilities','SaleType','Condition1','MSSubClass']

x_num = train_data[features_numeric]

cat = pd.DataFrame(train_data[features_categorical])

ohe_cat = pd.get_dummies(cat)

X = pd.concat([x_num,ohe_cat], axis = 1)

X.info()
x_num_test = test_data[features_numeric]

cat_test = pd.DataFrame(test_data[features_categorical])

ohe_cat_test = pd.get_dummies(cat_test)

X_test = pd.concat([x_num_test,ohe_cat_test], axis = 1)

X_test['Utilities_NoSeWa'] = 0

X_test.head()
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state = 1)
gs_grid = {'n_estimators' : np.arange(10,150,10),

            'max_depth' : np.arange(1,10,2),

           }
rf_model = GridSearchCV(RandomForestRegressor(criterion = 'mse'),param_grid = gs_grid)

rf_model.fit(train_X,train_y)
rf_model.best_params_
rf_model_final = RandomForestRegressor(n_estimators = 120, max_depth = 9, random_state = 1)

rf_model_final.fit(train_X,train_y)
val_pred = rf_model.predict(val_X)
mae_val = mean_absolute_error(val_pred,val_y)

print("Mean Absolute Error for validation set = %d" %(mae_val))
def get_mae(n_estimators, train_X, val_X, train_y, val_y):

    model = RandomForestClassifier(n_estimators = n_estimators, max_depth = 11, random_state = 2)

    model.fit(train_X,train_y)

    pred_val = model.predict(val_X)

    mae = mean_absolute_error(pred_val,val_y)

    return mae

    

for n_estimators in [10,50,100]:

    final_mae = get_mae(n_estimators, train_X, val_X, train_y, val_y)

    print("Number of trees in forest: %d \t MAE: %d" %(n_estimators, final_mae))
final_data = rf_model.predict(X_test)
output = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': final_data})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")