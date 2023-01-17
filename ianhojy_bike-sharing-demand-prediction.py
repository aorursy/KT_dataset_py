import numpy as np 

import pandas as pd



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_log_error

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



from xgboost import XGBRegressor



from bayes_opt import BayesianOptimization
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")

submit = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")
# Function to get information about day, week, hour, year from datetime



def date_extractor(data):



    data['datetime'] = pd.to_datetime(data['datetime'])

    data["date"] = data["datetime"].apply(lambda x: x.date())

    data["hour"] = data["datetime"].apply(lambda x: x.hour)

    data["weekday"] = data["datetime"].apply(lambda x:x.isoweekday())

    data["month"] = data["datetime"].apply(lambda x:x.month)

    data["year"] = data["datetime"].apply(lambda x:x.year)



    data.drop(['datetime'], axis=1, inplace=True)

    

    return data
train = date_extractor(train)

submit = date_extractor(submit)
drop_col = ['casual', 'registered']

for col in drop_col:

    train.drop(col, axis=1, inplace=True)
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[('onehotencoder', OneHotEncoder(handle_unknown='ignore'))])



numeric_features = ['temp', 'atemp', 'humidity', 'windspeed']

categorical_features = ['season', 'holiday', 'workingday', 'weather', 'hour', 'weekday', 'month', 'year']



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])
y = train['count']
train.drop(['date', 'count'], axis=1, inplace=True)

submit.drop(['date'], axis=1, inplace=True)
X = preprocessor.fit_transform(train)

X_submit = preprocessor.transform(submit)
def RMSLE(y_true, y_pred):

    

    pairs = list(zip(np.log(y_true+1), np.log(y_pred+1)))

    

    return round(np.sqrt(sum(map(lambda x: (x[0] - x[1]) ** 2, pairs)) / len(y_true)), 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
def XGB_error(max_depth, min_child_weight, gamma, colsample_bytree, subsample, X_train, X_test, y_train, y_test):

    

    XGB = XGBRegressor(

        max_depth=max_depth, 

        min_child_weight=min_child_weight, 

        gamma=gamma, 

        colsample_bytree=colsample_bytree, 

        subsample=subsample

    )

    

    y_train = np.log(y_train)

    XGB.fit(X_train, y_train)

    y_pred = XGB.predict(X_test)

    y_pred = np.exp(y_pred)

    

    return -RMSLE(y_test, y_pred)





def optimize_XGB(X_train, X_test, y_train, y_test):

    

    def XGB_wrapper(max_depth, min_child_weight, gamma, colsample_bytree, subsample):

        

        return XGB_error(

            max_depth=int(max_depth), 

            min_child_weight=min_child_weight, 

            gamma=gamma, 

            colsample_bytree=colsample_bytree, 

            subsample=subsample,

            X_train=X_train, 

            X_test=X_test, 

            y_train=y_train, 

            y_test=y_test

        )

    

    optimizer = BayesianOptimization(

        f=XGB_wrapper,

        pbounds={

            "max_depth": (0, 20), 

            "min_child_weight": (1, 10), 

            "gamma": (0.2, 0.8), 

            "colsample_bytree": (0.2, 0.9), 

            "subsample": (0.2, 0.9)

        },

        random_state=0,

        verbose=2

    )

        

    optimizer.maximize(n_iter=50)



    print("Final result:", optimizer.max)

    print()

    final_params = optimizer.max['params']

    for k, v in final_params.items():

        print(k,'=',v,',')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)





optimize_XGB(X_train, X_test, y_train, y_test)
XGB = XGBRegressor(

    colsample_bytree = 0.8420564768063263 ,

    gamma = 0.3169832518324276 ,

    max_depth = 6 ,

    min_child_weight = 8.956069480250676 ,

    subsample = 0.8715769930248918 ,)



y_logged = np.log(y)

XGB.fit(X, y_logged)



y_submit = XGB.predict(X_submit)

y_submit = np.exp(y_submit)



submit = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

submission = pd.concat([submit['datetime'], pd.DataFrame(y_submit)], axis=1)

submission.columns = ['datetime', 'count']

submission.set_index('datetime', inplace=True)

submission.to_csv('submission.csv')