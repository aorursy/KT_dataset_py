# Package Imports
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold,train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 4
# Read Data
raw_data = pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')
raw_data.head()
# Convert a coumns to lower case
raw_data.columns= raw_data.columns.str.lower()
# Drop the car name feature
df = raw_data.drop('car_name', axis = 1)
df.head()
# Find categorical columns and create dummy variables
categorical_columns = list(df.columns[df.dtypes == 'object'])
for cols in categorical_columns:
    dummy_vals  = pd.get_dummies(df[cols], prefix=cols, prefix_sep='_', drop_first=True)
    df = pd.concat([df, dummy_vals], axis = 1)
#Convert year and create age of the car in years
df['years_old'] = 2020 - df['year']
# Drop original categorical variables
df = df.drop(['fuel_type','seller_type','transmission','year'], axis =1)

# Correllation
df.corr()
# Check for null values in the data set
df.isnull().sum().sum()
# Dependent and Independent Variables

X = df.drop('selling_price', axis =1)
y = df['selling_price']
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 42)
# XGBoost model
model_xgb = XGBRegressor()
# param Grid
param_grid = {'n_estimators': list(range(50,550,50)),
              'max_depth' : list(range(1,21)),
             'learning_rate': [0.001, 0.01, 0.1, 1.0],
             'reg_alpha': [0.001, 0.01, 0.1, 1.0, 1],
             'min_child_weight': list(range(20))
             }
# Fit the model with Randomized CV
model_rs_XGB = RandomizedSearchCV(estimator=model_xgb,param_distributions=param_grid,
                                  scoring='neg_mean_squared_error', cv=5, 
                                  random_state=42, verbose=2, n_iter = 20)
model_rs_XGB.fit(X_train, y_train)
# Model hyper parameters after tuning
model_rs_XGB.best_params_
# r square score
y_pred = model_rs_XGB.predict(X_test)
score = r2_score(y_test, y_pred)
print(score)
#RMSE
rmse = mean_squared_error(y_test,y_pred)
rmse
# Feature importance, individual sellers and present price are significant variables
feature_importance = pd.Series(model_rs_XGB.best_estimator_.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importance.plot(kind = 'bar',title = 'Feature Impotance')
plt.ylabel('Feature Importance Score')