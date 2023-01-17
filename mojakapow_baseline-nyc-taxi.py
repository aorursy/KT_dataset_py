import numpy as np

import pandas as pd



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split





from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error
Xtest = pd.read_csv("../input/test.csv", parse_dates = ["pickup_datetime"], index_col=0)

Xtest.head()
# parse_dates



df = pd.read_csv("../input/train.csv", parse_dates = ["pickup_datetime", "dropoff_datetime"], index_col=0)

df.head()
df.shape
print(df.isnull().sum())
df.info()
df.describe()
df.head()
# Extract dates sur test





Xtest.loc[:,'pickup_dayofweek'] = Xtest['pickup_datetime'].dt.dayofweek

Xtest.loc[:,'pickup_dayofyear'] = Xtest['pickup_datetime'].dt.dayofyear

Xtest.loc[:,'pickup_hour'] = Xtest['pickup_datetime'].dt.hour

Xtest.loc[:,'pickup_month'] = Xtest['pickup_datetime'].dt.month
# Extract dates sur train



df.loc[:,'pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek

df.loc[:,'pickup_dayofyear'] = df['pickup_datetime'].dt.dayofyear

df.loc[:,'pickup_hour'] = df['pickup_datetime'].dt.hour

df.loc[:,'pickup_month'] = df['pickup_datetime'].dt.month

Dates = ['pickup_dayofweek', 'pickup_dayofyear', 'pickup_hour', 'pickup_month']

X_dates = df[Dates]

X_dates.head()
# transform store&flag sur test et train



Xtest['flag'] = np.where(Xtest['store_and_fwd_flag']=='N', 0, 1)

Xtest.drop(['store_and_fwd_flag'], axis = 1)



#df["store_and_fwd_flag"]

df['flag'] = np.where(df['store_and_fwd_flag']=='N', 0, 1)

df.drop(['store_and_fwd_flag'], axis = 1)

df.head()
# Selected_variables test



Num_Var = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'flag']



test = Xtest[Num_Var + Dates]

test.head()

#"vendor_id", "passenger_id", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "trip_duration"
test.columns
# Selected_variables



Num_Var = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'flag']

X_num = df[Num_Var]



X = df[Num_Var + Dates]

X.head()

#"vendor_id", "passenger_id", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "trip_duration"
y = df['trip_duration']

y.head()
# On subset



X_train, X_testing, y_train, y_testing = train_test_split(X,y, test_size=0.01, random_state=42)

X_train.shape, X_testing.shape, y_train.shape, y_testing.shape
y_train = np.log(y_train)

y_train.head()
# rf parameters



rf = RandomForestRegressor(random_state=42, n_estimators=10)
# training

rf.fit(X_train, y_train)
# pred = rf.predict(X_train)
y_pred = rf.predict(test)

test_id = pd.read_csv("../input/test.csv", parse_dates = ["pickup_datetime"])

test_id.head()
my_submission = pd.DataFrame({'id': test_id.id, 'trip_duration': np.expm1(y_pred)})

my_submission.shape
my_submission.to_csv('submission.csv',index=False)
# benchmark MSE:  0.1736042975124673

# loss

#cv_losses = - cross_val_score(rf, X_train, y_train, cv=5, scoring = "neg_mean_squared_error")

#cv_losses
# benchmark [0.33777047 0.33648306 0.33589602 0.33641725 0.33700657]

# 10% [0.69064132 0.67015777 0.67540274 0.67913618 0.67750758]

# 90% [0.42467579 0.42089421 0.4258463  0.41847506 0.42417022]

#for i in range(len(cv)):

#    cv[i] = np.sqrt((cv[i])

#print(cv)