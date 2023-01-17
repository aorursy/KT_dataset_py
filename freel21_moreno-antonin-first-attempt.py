import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score, train_test_split



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
# 1 minute of silence for the people who forgot to turn off the taxi counter

# We have to get rid of these values in order to make correct predictions



df_train = df_train.loc[df_train['trip_duration']< 300000]



# The "top" outliers are the easiest to deal with. It gets more complicated with the "bottom" outliers

# In the following, we will consider any trip duration below 5 minutes as outliers

# haters gonna hate, I know, who would take the taxi for a trip duration less than 5 minutes anyway



df_train = df_train.loc[(df_train['trip_duration'] > 300) & (df_train['trip_duration'] < 300000)]
# Now we also need to get rid of outliers in the geographical place data section where people are picked up



df_train.plot.scatter(x='pickup_longitude',y='pickup_latitude');
df_train = df_train.loc[(df_train['pickup_longitude']> -90) & (df_train['pickup_latitude']< 46)]
# Same goes with the place people are dropped off



df_train.plot.scatter(x='dropoff_longitude',y='dropoff_latitude');
df_train = df_train.loc[(df_train['dropoff_longitude']< -70) & (df_train['dropoff_latitude']> 35)]
# In prevision of the prediction model, we are going to create 7 more features : DateTime, Month, Day, Hour, Minute, Time, Distance



df_train['Time'] = df_train['pickup_datetime'].apply(lambda x: int(x.split()[1][0:2]))



df_train['Distance'] = np.sqrt((df_train['pickup_latitude']-df_train['dropoff_latitude'])**2 + (df_train['pickup_longitude']-df_train['dropoff_longitude'])**2) 
y = df_train["trip_duration"] # <-- target

X = df_train[["passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","Time","Distance"]] # <-- features



X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2, random_state=42)

X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
m1 = RandomForestRegressor()

m1.fit(X_train, y_train)
m1_scores = cross_val_score(m1, X, y, cv=5, scoring ="neg_mean_squared_log_error")

m1_scores
for i in range(len(m1_scores)):

    m1_scores[i] = np.sqrt(abs(m1_scores[i]))

print(m1_scores)

print(np.mean(m1_scores))
df_test['Time'] = df_test['pickup_datetime'].apply(lambda x: int(x.split()[1][0:2]))



df_test['Distance'] = np.sqrt((df_test['pickup_latitude']-df_test['dropoff_latitude'])**2 + (df_test['pickup_longitude']-df_test['dropoff_longitude'])**2) 

df_test.head()
X_test = df_test[["passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","Time","Distance"]]

predicted_duration = m1.predict(X_test)

print(predicted_duration)
My_Submission = pd.DataFrame({'id': df_test.id, 'trip_duration': predicted_duration})

print(My_Submission)
My_Submission.to_csv('submission.csv', index=False)