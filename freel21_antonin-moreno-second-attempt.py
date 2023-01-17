import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#from sklearn.ensemble import RandomForestRegressor

#from sklearn.model_selection import cross_val_score, train_test_split

#from sklearn.metrics import r2_score, mean_squared_error as MSE

#from sklearn.linear_model import SGDRegressor, LinearRegression



import lightgbm as lgb



#all the previous algorithms from sklearn did a terrible job at predicting the trip duration on this dataset.

#I tried something new first with gbm, next with lightgbm which seemed to be faster and even more accurate



%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_train.size



# Holy Pepperoni, that's big
print(df_train.columns)



# in order to get an idea of the futur features to take in count for the prediction

# ... and because I love to print columns anyway, hope you don't mind
df_train.info()
df_train.isna().sum()
df_train.isnull().sum()



# a data with no NaN value, what is this sorcery
# just to be sure



df_train.duplicated().sum()



# sweet
# First things first, we need to get rid of outliers in the trip duration feature



plt.subplots(figsize=(20,10))

plt.title("Top Outliers repartition in the trip duration feature")

df_train.boxplot();
plt.subplots(figsize=(20,10))

plt.title("Top Outliers repartition in the trip duration feature - zoomed")

df_train.loc[df_train.trip_duration<5000,"trip_duration"].hist(bins=120);
# 1 minute of silence for the people who forgot to turn off the taxi counter

# We have to get rid of these values in order to make correct predictions



df_train = df_train[(df_train['trip_duration'] > 60) & (df_train['trip_duration'] < 4600)]

df_train['trip_duration'] = np.log(df_train['trip_duration'].values)



# The "top" outliers are the easiest to deal with. It gets more complicated with the "bottom" outliers
# Beware mortal, from this point of the kernel, you will find few lines of code put under commentary 

# because I made some wild test before getting a satisfactory result on the predictions



# I did not erase the code not giving useless information, not working correctly or even at all 

# so you can see by yourself what went wrong in the first steps. Sorry for the mess !
# we will consider any trip duration below 5 minutes as outliers

# haters gonna hate, I know, who would take the taxi for a trip duration less than 5 minutes anyway



#df_train = df_train.loc[(df_train['trip_duration'] > 300) & (df_train['trip_duration'] < 300000)]
# Now we also need to get rid of outliers in the geographical place data section where people are picked up



#df_train.plot.scatter(x='pickup_longitude',y='pickup_latitude');
#df_train = df_train.loc[(df_train['pickup_longitude']> -90) & (df_train['pickup_latitude']< 46)]
# Same goes with the place people are dropped off



#df_train.plot.scatter(x='dropoff_longitude',y='dropoff_latitude');
#df_train = df_train.loc[(df_train['dropoff_longitude']< -70) & (df_train['dropoff_latitude']> 35)]
# I tried to create those two features but it turned out to be useless in the end, mostly because the output information was not precise enough



#df_train['Time'] = df_train['pickup_datetime'].apply(lambda x: int(x.split()[1][0:2]))



#df_train['Distance'] = np.sqrt((df_train['pickup_latitude']-df_train['dropoff_latitude'])**2 + (df_train['pickup_longitude']-df_train['dropoff_longitude'])**2) 
# In prevision of the prediction model, we are going to create a new feature : DateTime

# The DateTime module gives classes to manipulate times and dates.
from datetime import datetime



df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])

df_train['dropoff_datetime'] = pd.to_datetime(df_train['dropoff_datetime'])

df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])
# In order to use at best this feature, we will add 4 mores columns in the dataset, both Train and Test :

# Month, Day, Hour, Minute
df_train['month'] = df_train.pickup_datetime.dt.month

df_train['day'] = df_train.pickup_datetime.dt.dayofweek

df_train['hour'] = df_train.pickup_datetime.dt.hour

df_train['minute'] = df_train.pickup_datetime.dt.minute



df_test['month'] = df_test.pickup_datetime.dt.month

df_test['day'] = df_test.pickup_datetime.dt.dayofweek

df_test['hour'] = df_test.pickup_datetime.dt.hour

df_test['minute'] = df_test.pickup_datetime.dt.minute
# Now we will add some mathematics in the cauldron to get the distance of any trip by using the pickup & dropoff lattitude & longitude data.
df_train['d_latitude'] = df_train['pickup_latitude'] - df_train['dropoff_latitude']

df_train['d_longitude'] = df_train['pickup_longitude'] - df_train['dropoff_longitude']



df_test['d_latitude'] = df_test['pickup_latitude'] - df_test['dropoff_latitude']

df_test['d_longitude'] = df_test['pickup_longitude'] - df_test['dropoff_longitude']
# And now, for the final result :
df_train['distance'] = np.sqrt(np.square(df_train['d_longitude']) + np.square(df_train['d_latitude']))

df_test['distance'] = np.sqrt(np.square(df_test['d_longitude']) + np.square(df_test['d_latitude']))
df_train.shape, df_test.shape



# Good boy
# Next, we are going to use every feature and column we added in the dataset to prepare the training
#y = df_train["trip_duration"] # <-- target

#X = df_train[["passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","Time","Distance","minute","hour","day","month"]] # <-- features



Features = ["vendor_id","passenger_count","pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","distance","month","hour","day"]

Target = "trip_duration"



#X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2, random_state=42)

#X_train.shape, X_valid.shape, y_train.shape, y_valid.shape



X_train = df_train[Features]

y_train = df_train[Target]



lgb_train = lgb.Dataset(X_train,y_train)



lgb_params = {'learning_rate': 0.1,

                'max_depth': 25,

                'num_leaves': 1000, 

                'objective': 'regression',

                'feature_fraction': 0.9,

                'bagging_fraction': 0.5,

                'max_bin': 1000

             }   
#m1 = RandomForestRegressor()

#m1.fit(X_train, y_train)
#model_lgb = lgb.train(lgb_params,lgb_train)

#model_lgb = lgb.train(lgb_params,lgb_train,num_boost_round=1000)



model_lgb = lgb.train(lgb_params,lgb_train,num_boost_round=500)
#m1_scores = cross_val_score(m1, X, y, cv=5, scoring ="neg_mean_squared_log_error")

#m1_scores
#for i in range(len(m1_scores)):

#    m1_scores[i] = np.sqrt(abs(m1_scores[i]))

#print(m1_scores)

#print(np.mean(m1_scores))
#df_test['Time'] = df_test['pickup_datetime'].apply(lambda x: int(x.split()[1][0:2]))



#df_test['Distance'] = np.sqrt((df_test['pickup_latitude']-df_test['dropoff_latitude'])**2 + (df_test['pickup_longitude']-df_test['dropoff_longitude'])**2) 

#df_test.head()
#X_test = df_test[["passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","Time","Distance"]]

#predicted_duration = m1.predict(X_test)

#print(predicted_duration)
X_prediction = df_test[Features]

prediction = np.exp(model_lgb.predict(X_prediction))

print(prediction)
My_Submission = pd.DataFrame({'id': df_test.id, 'trip_duration': prediction})

print(My_Submission)
My_Submission.to_csv('submission.csv', index=False)