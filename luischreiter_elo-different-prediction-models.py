# Importing necessary packages
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import lightgbm as lgb
# Importing the train test split data frames to execute them on different prediction models
# Load test train split
X_train = pd.read_csv("../input/X_train.csv")
X_test = pd.read_csv("../input/X_test.csv")
Y_train = pd.read_csv("../input/Y_train.csv")
Y_test = pd.read_csv("../input/Y_test.csv")
model_tree = DecisionTreeRegressor()
model_tree.fit(X_train, pd.DataFrame(Y_train))
prediction = model_tree.predict(X_test)
score_tree = model_tree.score(X_test, Y_test)
rmse_tree = np.sqrt(mean_squared_error(Y_test,prediction))
model_random = RandomForestRegressor()
model_random.fit(X_train, pd.DataFrame(Y_train))
prediction = model_random.predict(X_test)
score_random = model_random.score(X_test, Y_test)
rmse_random = np.sqrt(mean_squared_error(Y_test,prediction))
model_boost = GradientBoostingRegressor()
model_boost.fit(X_train, pd.DataFrame(Y_train))
prediction = model_boost.predict(X_test)
score_boost = model_boost.score(X_test, Y_test)
rmse_boost = np.sqrt(mean_squared_error(Y_test,prediction))
# Dataset definition LGB
train_data = lgb.Dataset(X_train, label=Y_train)
test_data = lgb.Dataset(X_test,label=Y_test)

# Parameters definition
lgb_params = {"objective" : "regression", 
"metric" : "rmse",
"max_depth": 7, 
"min_child_samples": 20, 
"reg_alpha": 1, 
"reg_lambda": 1,
"num_leaves" : 64, 
"learning_rate" : 0.005,
"subsample" : 0.8, 
"colsample_bytree" : 0.8, 
"verbosity": -1}

# Model training
lgb_model = lgb.train(lgb_params,train_data,valid_sets=test_data,num_boost_round=10000,early_stopping_rounds=50)
score_lgb = 3.66712
# ElasticNetCV
model_elasticcv = ElasticNetCV()
model_elasticcv.fit(X_train, pd.DataFrame(Y_train))
prediction = model_elasticcv.predict(X_test)
score_ecv = model_elasticcv.score(X_test, Y_test)
rmse_elasticcv = np.sqrt(mean_squared_error(Y_test,prediction))
# RidgeCV
model_ridgecv = RidgeCV()
model_ridgecv.fit(X_train, pd.DataFrame(Y_train))
prediction = model_ridgecv.predict(X_test)
score_rcv = model_ridgecv.score(X_test, Y_test)
rmse_ridgecv = np.sqrt(mean_squared_error(Y_test,prediction))
# LassoCV
model_lasso = LassoCV()
model_lasso.fit(X_train, pd.DataFrame(Y_train))
prediction = model_lasso.predict(X_test)
score_lasso = model_lasso.score(X_test, Y_test)
rmse_lasso = np.sqrt(mean_squared_error(Y_test,prediction))
# Exporting the result in order to import it in the main kernel
Labels = ['LassoCV', 'RidgeCV', 'ElasticNetCV', 'LightGBM', 'GradientBoostingRegressor',
          'RandomForestRegressor', 'DecisionTreeRegressor']

RMSEs = [rmse_lasso, rmse_ridgecv, rmse_elasticcv, score_lgb, rmse_boost, rmse_random, rmse_tree]

RMSE_df = pd.DataFrame({"Labels":Labels})
RMSE_df['RMSE']=RMSEs
RMSE_df.to_csv("RMSE_table.csv", index=False)