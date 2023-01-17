import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.preprocessing import scale

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense

from sklearn.metrics import mean_squared_error

from keras.layers import BatchNormalization





def process_data(df):



    # get pickup times

    df['year'] = df['pickup_datetime'].dt.year

    df['month'] = df['pickup_datetime'].dt.month

    df['day'] = df['pickup_datetime'].dt.day

    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

    df['hour'] = df['pickup_datetime'].dt.hour

    

    def create_time_features(df):

        df['year'] = df['pickup_datetime'].dt.year

        df['month'] = df['pickup_datetime'].dt.month

        df['day'] = df['pickup_datetime'].dt.day

        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

        df['hour'] = df['pickup_datetime'].dt.hour

        df = df.drop(['pickup_datetime'], axis=1)

        df = df.drop(['key'], axis=1)

        return df

    

    # preprocess data

    def preprocessing(df):

        print(df.isnull().sum())

        df = df.dropna()

        print(df.describe())

        return df

            

    # remove fare outliers

    def remove_fare_outliers(df,lower_bound, upper_bound):

        df = df[ (df['fare_amount'] > lower_bound) & (df['fare_amount'] <= upper_bound)]

        return df

        

    # replace rider outliers 

    def replace_passenger_outliers(df):

        df.loc[ df['passenger_count']==0, 'passenger_count']= 1 

        return df



    # check for geolocation outliers

    def check_geolocation(df):

        df.plot.scatter('pickup_longitude', 'pickup_latitude')

        plt.show()



    # only consider locations inside nyc

    def remove_lat_lon_outliers(df):

        nyc_min_lon = -74.05

        nyc_max_lon = -73.75



        nyc_min_lat = 40.63

        nyc_max_lat = 40.85

        

        for long in ['pickup_longitude', 'dropoff_longitude']:

            df = df[(df[long] > nyc_min_lon) & (df[long] < nyc_max_lon) ]



        for lat in ['pickup_latitude', 'dropoff_latitude']:

            df = df[(df[lat] > nyc_min_lat) & (df[lat] < nyc_max_lat)]

        return df

    

    # create distance feature

    def make_distance(df):

        def euc_distance(lat1, lon1, lat2, lon2):

            return (((lat1-lat2)**2 +(lon1-lon2)**2)**0.5)

        df['distance'] = euc_distance(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])

        #df.plot.scatter('fare_amount', 'distance')

        #plt.show()

        return df



    # copy main data

    #df2 = df.copy(deep=True)

    # dr

    print('Removing Null values')

    df = preprocessing(df)   

    print('Removing fare outliers') 

    df = remove_fare_outliers(df, lower_bound=0, upper_bound=100)

    print('Replacing passenger count outliers')

    df = replace_passenger_outliers(df)

    print('Removing location outliers')

    df = remove_lat_lon_outliers(df)

    print('Creating time features')

    df = create_time_features(df)

    print('Creating distance features')

    df = make_distance(df)

    return df



def visualize(df):

    # dict of landmarks in NYC to overlay rides 

    landmarks = {

        'JFK':(-73.78, 40.643),

        'LGA':(-73.87, 40.77),

        'Midtown':(-73.98, 40.76),

        'Lower Manhattan':(-74.00, 40.72),

        'Upper Manhattan':(-73.94, 40.82),

        'Brooklyn':(-73.94, 40.66)

        }   

    # hour

    def hour(df):

        df['hour'].plot.hist(bins=24, ec='blue')

        plt.title('Rides by Hour')

        plt.xlabel("Ride per Hour")

        plt.show()



    # day of week

    def day_of_week(df):

        df['day_of_week'].plot.hist(bins=np.arange(8)-0.5, ylim= (60000, 75000), ec='black')

        plt.title("Rides by Day of the Week")

        plt.xlabel("Day of the Week: 0=Monday 6=Sunday")

        plt.show()



    # fare amount

    def fare_amount(df):

        df['fare_amount'].hist(bins=500)

        plt.title("Fare Amount")

        plt.xlabel("Fare")

        plt.show()

            

    # pickup locations

    def pickup_locations(df):

        df.plot.scatter('pickup_longitude', 'pickup_latitude')

        plt.show()



    # drop off

    def drop_off(df):

        df.plot.scatter('dropoff_longitude', 'dropoff_latitude')

        plt.show()



    # Scatter Plot

    def plot_lat_lon(df, landmarks, points='Pickup'):

        plt.figure(figsize=(12,12))

        if points == 'pickup':

            plt.plot(list(df.pickup_longitude), list(df.pickup_latitude), '.', markersize=1)

        else:

            plt.plot(list(df.dropoff_longitude), list(df.dropoff_latitude), '.', markersize=1)

        

        for landmark in landmarks:

            plt.plot(landmarks[landmark][0], landmarks[landmark][1], '*', markersize=15, alpha=1, color='r')

            plt.annotate(landmark, (landmarks[landmark][0]+.005,

            landmarks[landmark][1]+.005), color='r', backgroundcolor='w')

            

        plt.title("{} Locations in NYC Illustrated".format(points))

        plt.grid(None)

        plt.xlabel('Lat')

        plt.ylabel('lon')

        plt.show()



    plot_lat_lon(df, landmarks, points='Pickup')

    plot_lat_lon(df, landmarks, points='Dropoff')



def scale_data(df):

    df_prescaled = df.copy()

    df_scaled = df.drop(['fare_amount'], axis=1)

    # scale features

    df_scaled = scale(df_scaled)

    # transform into pd data frame 

    # add in fare amount from prescale

    cols = df.columns.tolist()

    cols.remove('fare_amount')

    df_scaled = pd.DataFrame(df_scaled, columns=cols, index=df.index)

    df_scaled = pd.concat([df_scaled, df['fare_amount']], axis=1)

    df = df_scaled.copy()

    return df, df_prescaled

    

# feature engineering    



print('loading data')

df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", parse_dates=['pickup_datetime'], nrows=5000000)

# clean and feature engineering

df = process_data(df)

# visualize(df)

# scale data for network

df, df_prescaled = scale_data(df) 



# Neural Network

# independent and dependent variables

X = df.loc[:, df.columns != 'fare_amount']

y = df.loc[:, 'fare_amount']

# split data

print('Splitting data into training and testing sets')

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# build model

model = Sequential()

model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))

model.add(Dense(64, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(X_train, y_train, epochs=1)

data = model.predict(X_test)

# TODO: GET KEY TO ADD TO SUBMISSION DF

submission = pd.DataFrame(data)

print(submission.head())

print(X_train.head())



# # Results

# train_pred = model.predict(X_train)

# train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

# test_pred = model.predict(X_test)

# test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

# print("Train RMSE: {:0.2f}".format(train_rmse))

# print("Test RMSE: {:0.2f}".format(test_rmse))

# print('------------------------')

# 

# def predict_random(df_prescaled, X_test, model):

#     sample = X_test.sample(n=1, random_state=np.random.randint(low=0, high=10000))

#     idx = sample.index[0]

# 

#     actual_fare = df_prescaled.loc[idx,'fare_amount']

#     day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

#     day_of_week = day_names[df_prescaled.loc[idx,'day_of_week']]

#     hour = df_prescaled.loc[idx,'hour']

#     predicted_fare = model.predict(sample)[0][0]

#     rmse = np.sqrt(np.square(predicted_fare-actual_fare))

# 

#     print("Trip Details: {}, {}:00hrs".format(day_of_week, hour))  

#     print("Actual fare: ${:0.2f}".format(actual_fare))

#     print("Predicted fare: ${:0.2f}".format(predicted_fare))

#     print("RMSE: ${:0.2f}".format(rmse))

# 

# predict_random(df_prescaled, X_test, model)






