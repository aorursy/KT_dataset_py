# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
ipl_data = pd.read_csv("/kaggle/input/ipldataset/IPL IMB381IPL2013.csv")
ipl_data.head()
ipl_data.info()
ipl_data.shape
ipl_data.describe()
X_features = ['AGE', 'COUNTRY', 'PLAYING ROLE',
'T-RUNS', 'T-WKTS', 'ODI-RUNS-S', 'ODI-SR-B',
'ODI-WKTS', 'ODI-SR-BL', 'CAPTAINCY EXP', 'RUNS-S',
'HS', 'AVE', 'SR-B', 'SIXERS', 'RUNS-C', 'WKTS',
'AVE-BL', 'ECON', 'SR-BL']
categorical_features = ['AGE', 'COUNTRY', 'PLAYING ROLE', 'CAPTAINCY EXP']
ipl_encoded_data = pd.get_dummies( ipl_data[X_features],columns = categorical_features,drop_first = True )
ipl_encoded_data.columns
X = ipl_encoded_data
Y = ipl_data['SOLD PRICE']
from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
Y = (Y - Y.mean())/Y.std()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled,Y,test_size=0.2,random_state = 42)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train,y_train)
print("train accuracy",linreg.score(X_train,y_train))
print("test accuracy",linreg.score(X_test,y_test))
linreg.coef_
columns_coef_df = pd.DataFrame( { 'columns': ipl_encoded_data.columns,
'coef': linreg.coef_ } )
sorted_coef_vals = columns_coef_df.sort_values( 'coef', ascending=False)
plt.figure( figsize = ( 8, 8 ))

## Creating a bar plot

ax = sn.barplot(x="coef", y="columns", data=sorted_coef_vals);
ax.set_xlabel("Coefficients from Linear Regression")
ax.set_ylabel("Features");
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Takes a model as a parameter
# Prints the RMSE on train and test set

def get_train_test_rmse( model ):
    # Predicting on training dataset
    y_train_pred = model.predict( X_train )
    
    # Compare the actual y with predicted y in the training dataset
    rmse_train = round(np.sqrt(mean_squared_error( y_train, y_train_pred )), 3)
    
    # Predicting on test dataset
    y_test_pred = model.predict( X_test )
    
    # Compare the actual y with predicted y in the test dataset
    rmse_test = round(np.sqrt(mean_squared_error( y_test, y_test_pred )), 3)
    
    # R square 
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
#     # Adjusted R square
#     train_ad_r2 = 1-(1-train_r2)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
#     test_ad_r2 = 1-(1-test_r2)*(len(y_test)-1)/(len(y_test)-X_train.shape[1]-1)
    
    print( "Train RMSE: ", rmse_train, " Test RMSE:", rmse_test )
    print("Train R^2:", round(train_r2,3), "Test R^2:", round(test_r2,3))
get_train_test_rmse( linreg )
### Ridge and Lasso###
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge,Lasso
ridge = Ridge(alpha = 1, max_iter = 500)
ridge.fit( X_train, y_train )
get_train_test_rmse( ridge )
param_grid_ridge = {'alpha': np.logspace(-3, 3, 13)}
print(param_grid_ridge)
grid_ridge = GridSearchCV(Ridge(), param_grid=param_grid_ridge, cv=10)
grid_ridge.fit(X_train, y_train)

print(grid_ridge.best_params_)
get_train_test_rmse( grid_ridge )
param_grid_lasso = {'alpha': np.logspace(-3, 0, 13)}
print(param_grid_lasso)
grid_lasso = GridSearchCV(Lasso(), param_grid=param_grid_lasso, cv=10)
grid_lasso.fit(X_train, y_train)

print(grid_lasso.best_params_)
get_train_test_rmse( grid_lasso )
from sklearn.ensemble import RandomForestRegressor
param_grid_RF = {'max_depth': [10, 15,20],
                 'n_estimators': [100,150, 200]}

print(param_grid_RF)
R_F = RandomForestRegressor(n_jobs=-1)
grid_RF = GridSearchCV(R_F, param_grid=param_grid_RF, cv=10)
grid_RF.fit(X_train, y_train)
print(grid_RF.best_params_)
get_train_test_rmse( grid_RF )
aram_grid_RF = {'max_depth': [5,10, 15],
                   'n_estimators': [30,50,75]}

print(param_grid_RF)
grid_RF = GridSearchCV(R_F, param_grid=param_grid_RF, cv=10)
grid_RF.fit(X_train, y_train)
print(grid_RF.best_params_)
get_train_test_rmse( grid_RF )
param_grid_RF_3 = {'max_depth': [3,5,10],
                   'n_estimators': [15,30,45]}
print(param_grid_RF_3)
grid_RF = GridSearchCV(R_F, param_grid_RF_3, cv=10)
grid_RF.fit(X_train, y_train)
print(grid_RF.best_params_)
get_train_test_rmse( grid_RF )
#GBM#
param_grid_GBM = {'n_estimators': [100,200,500], 
                  'max_depth': [1,2,4], 
                  'min_samples_split':[20,40,60] }
print(param_grid_GBM)
from sklearn.ensemble import GradientBoostingRegressor
grid_GBM = GridSearchCV(GradientBoostingRegressor(), param_grid=param_grid_GBM, cv=10)
grid_GBM.fit(X_train, y_train)

print(grid_GBM.best_params_)
get_train_test_rmse( grid_GBM )
param_grid_GBM_2 = {'n_estimators': [40,60,90], 
                    'max_depth': [1,2], 
                    'min_samples_split':[10,20] }
print(param_grid_GBM_2)
grid_GBM = GridSearchCV(GradientBoostingRegressor(), param_grid_GBM_2, cv=10)
grid_GBM.fit(X_train, y_train)
print(grid_GBM.best_params_)
get_train_test_rmse( grid_GBM )
#feature importance
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
def feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}).sort_values(by='imp', ascending=False)
model_RF = RandomForestRegressor(max_depth=5, max_features='auto', n_estimators=45)
model_RF.fit(X_train, y_train)
fi = feat_importance(model_RF, ipl_encoded_data)
plot_fi(fi[:10]);
model = GradientBoostingRegressor(max_depth= 1, min_samples_split=10, n_estimators= 90)
model.fit(X_train, y_train)
fi = feat_importance(model, ipl_encoded_data);
plot_fi(fi[:10]);