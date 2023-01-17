import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression,Ridge

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb 

from sklearn.metrics import r2_score,mean_absolute_error



#Load raw data

file_path = '../input/melb_data.csv'

melb_raw_data = pd.read_csv(file_path)



#drop NaN from the data

melb_data = melb_raw_data.dropna()



#set features as X (features selected numeric type)

features_lis = ["Rooms", "Distance", "Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt",

               "Lattitude", "Longtitude", "Propertycount"]

X = melb_data[features_lis]



#set target as y

y = melb_data.loc[:,["Price"]]





#Holdout

X_train,X_test,y_train,y_test = train_test_split(X.astype(float),   ###astype(float) for protect DataconversionWarning

                                                 y,

                                                 test_size=0.25,

                                                 random_state=1)



#selected XGBRegressor due to better score performance than others(see below cell)

xg_set = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05)

xg_set.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)],verbose=False)



print("XGBRegressor")

print("MAE Train_score:",mean_absolute_error(y_train, xg_set.predict(X_train)))

print("MAE  Test_score:",mean_absolute_error(y_test, xg_set.predict(X_test)))

#other candidates

##GradientBoostingRegressor has similar performance as well 

### not tried parameter tuning



pipelines = {

    "pipe_ols" : Pipeline([("scl", StandardScaler()),("est", LinearRegression())]),

    "pipe_ridge" : Pipeline([("scl", StandardScaler()),("est", Ridge(random_state=1))]),

    "pipe_rf" : Pipeline([("scl", StandardScaler()),("est", RandomForestRegressor(max_leaf_nodes=50, n_estimators=100,random_state=1))]),

    "pipe_gbr" : Pipeline([("scl", StandardScaler()),("est", GradientBoostingRegressor(random_state=1))]),

    "pipe_dt" : Pipeline([("scl", StandardScaler()),("est", DecisionTreeRegressor(random_state=1))])

    }



for pipe_name, pipeline in pipelines.items():

    pipeline.fit(X_train, y_train.values.ravel())

    print("--------------------------------")

    print(pipe_name)

    print("Train_score:", mean_absolute_error(y_train.values.ravel(), pipeline.predict(X_train)))

    print(" Test_score:", mean_absolute_error(y_test.values.ravel(), pipeline.predict(X_test)))

    print("--------------------------------")