'''

@author: Glenn-Galang

'''

from numpy import genfromtxt

from geopy.distance import great_circle



train = genfromtxt('train.csv', delimiter=',')



N1 = train.shape[0]



for i in range(1,N1):

	pick_up = (train[i,6],train[i,5])

	drop_off = (train[i,8],train[i,7])

	distance = 1000*(great_circle(pick_up,drop_off).km)

	with open("meter_train.csv", "a") as myfile:

			myfile.write(str(distance)+",\n")

		

test = genfromtxt('test.csv', delimiter=',')



N2 = test.shape[0]



for i in range(1,N2):

	pick_up = (test[i,5],test[i,4])

	drop_off = (test[i,7],test[i,6])

	distance = 1000*(great_circle(pick_up,drop_off).km)

	with open("meter_test.csv", "a") as myfile:

		myfile.write(str(distance)+",\n")
'''

@author: Glenn-Galang

'''

from numpy import genfromtxt

from geopy.distance import great_circle

    

central = (40.781277,-73.966622)

	

train = genfromtxt('train.csv', delimiter=',')



N1 = train.shape[0]



for i in range(1,N1):

	pick_up = (train[i,6],train[i,5])

	drop_off = (train[i,8],train[i,7])

	distance_pickup = 1000*(great_circle(central,pick_up).km)

	distance_dropoff = 1000*(great_circle(central,drop_off).km)

	with open("distance_from_central_train.csv", "a") as myfile:

			myfile.write(str(distance_pickup)+","+str(distance_dropoff)+",\n")

		

test = genfromtxt('test.csv', delimiter=',')



N2 = test.shape[0]



for i in range(1,N2):

	pick_up = (test[i,5],test[i,4])

	drop_off = (test[i,7],test[i,6])

	distance_pickup = 1000*(great_circle(central,pick_up).km)

	distance_dropoff = 1000*(great_circle(central,drop_off).km)

	with open("distance_from_central_test.csv", "a") as myfile:

			myfile.write(str(distance_pickup)+","+str(distance_dropoff)+",\n")       
'''

@author: Glenn-Galang

'''

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt



# import original train dataset

train_df = pd.read_csv('train.csv')



# import original train dataset

test_df = pd.read_csv('test.csv')



# import weather dataset

weat_data = pd.read_csv('weather_data_nyc_centralpark_2016.csv')



# function to add snow weather by pickup date

weat_data['date'] = pd.to_datetime( weat_data['date'] ).dt.date

weat_data.set_index('date', inplace = True)

def addWeather( df ):

    

    df['date'] =  pd.to_datetime(df['pickup_datetime']).dt.date

    

    dates = df['date'].unique()

    

    df.set_index('date', inplace = True)

   

    weat_cols = ['precipitation', 'snow_fall', 'snow_depth']

    

    for col in weat_cols:

        df[col] = np.nan

        

        for date in dates:

            val = weat_data.loc[date, col]

            

            if( 'T' != val ):

                df.loc[date, col] = float(val)

        

    df.reset_index(drop = True, inplace = True)

        

    return df



#1. Processing train dataset



# import weather data

train_df = addWeather(train_df)



# import distance data, no header, then attach it to train_df

meter_train_df = pd.read_csv('meter_train.csv', header=None)

train_df['meter'] = meter_train_df[0]



# import distance to central data, no header, then attach it to train_df

distance_from_central_train_df = pd.read_csv('distance_from_central_train.csv', header=None)

train_df['pickup_distance'] = distance_from_central_train_df[0]

train_df['dropoff_distance'] = distance_from_central_train_df[1]



# compute pickup hour, date, month for each ride

train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])

train_df['pickup_hour'] = train_df.pickup_datetime.dt.hour

train_df['pickup_date'] = train_df.pickup_datetime.dt.day

train_df['pickup_month'] = train_df.pickup_datetime.dt.month



# comput pickup day of the week for each ride from 0 (Monday) to 6 (Sunday) 

train_df['day_week'] = train_df.pickup_datetime.dt.weekday



# remove columns that we don't need

del train_df['id']

del train_df['pickup_datetime']

del train_df['dropoff_datetime']

del train_df['pickup_longitude']

del train_df['pickup_latitude']

del train_df['dropoff_longitude']

del train_df['dropoff_latitude']



# get the list of column names

list(train_df)



# checking our target Y: trip_duration and plot

plt.scatter(range(train_df.shape[0]), np.sort(train_df.trip_duration.values))

plt.xlabel('index')

plt.ylabel('trip duration')

plt.show()



# remove some unsual long or short trips and plot again

q1 = train_df.trip_duration.quantile(0.001)

q2 = train_df.trip_duration.quantile(0.999)

train_df = train_df[(train_df.trip_duration > q1) & (train_df.trip_duration < q2)]

plt.scatter(range(train_df.shape[0]), np.sort(train_df.trip_duration.values))

plt.xlabel('index')

plt.ylabel('trip duration')

plt.show()



# checking our meter data field and plot

plt.scatter(range(train_df.shape[0]), np.sort(train_df.meter.values))

plt.xlabel('index')

plt.ylabel('meter')

plt.show()



# remove some trip shorter than 100 meters and longer than 80 km and plot again

train_df = train_df[(train_df.meter > 100) & (train_df.meter < 80000)]

plt.scatter(range(train_df.shape[0]), np.sort(train_df.meter.values))

plt.xlabel('index')

plt.ylabel('meter')

plt.show()



# count values in other X columns to detect unusual values

train_df['vendor_id'].value_counts()

train_df['passenger_count'].value_counts()

train_df['store_and_fwd_flag'].value_counts()

train_df['pickup_hour'].value_counts()

train_df['pickup_date'].value_counts()

train_df['pickup_month'].value_counts()

train_df['day_week'].value_counts()



# remove trips with 0, 8 or 9 passenger(s) and check again

train_df = train_df[(train_df.passenger_count < 8) & (train_df.passenger_count != 0)]

train_df['passenger_count'].value_counts()



# (optional) move trip_duration to end column

train_df['duration'] = train_df['trip_duration']

del train_df['trip_duration']



# split train and valuation

train=train_df.sample(frac=0.8,random_state=200)

val=train_df.drop(train.index)



# write the processed train and valuation dataset to csv

train.to_csv('train_df.csv', index=False)

val.to_csv('val_df.csv', index=False)



#2. Processing test dataset



# import weather data

test_df = addWeather(test_df)



# import distance data, no header, then attach it to train_df

meter_test_df = pd.read_csv('meter_test.csv', header=None)

test_df['meter'] = meter_test_df[0]



# import distance to central data, no header, then attach it to train_df

distance_from_central_test_df = pd.read_csv('distance_from_central_test.csv', header=None)

test_df['pickup_distance'] = distance_from_central_test_df[0]

test_df['dropoff_distance'] = distance_from_central_test_df[1]



# get the list of column names

list(test_df)



# compute pickup hour, date, month for each ride

test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])

test_df['pickup_hour'] = test_df.pickup_datetime.dt.hour

test_df['pickup_date'] = test_df.pickup_datetime.dt.day

test_df['pickup_month'] = test_df.pickup_datetime.dt.month



# comput pickup day of the week for each ride from 0 (Monday) to 6 (Sunday) 

test_df['day_week'] = test_df.pickup_datetime.dt.weekday



# remove columns that we don't need

del test_df['id']

del test_df['pickup_datetime']

del test_df['pickup_longitude']

del test_df['pickup_latitude']

del test_df['dropoff_longitude']

del test_df['dropoff_latitude']



# write the processed test_df to csv

test_df.to_csv('test_df.csv', index=False)
import pandas as pd

import numpy as np

#from sklearn import metrics

import itertools

import tensorflow as tf



# (Optional) Extra logging 

tf.logging.set_verbosity(tf.logging.ERROR)

import warnings

warnings.filterwarnings("ignore")

tf.logging.set_verbosity(tf.logging.INFO)



# Import data

train_df = pd.read_csv('train_df.csv')

evaluate_df = pd.read_csv('val_df.csv')

test_df = pd.read_csv('test_df.csv')



# define Root Mean Squared Logarithmic Error for evaluation

def rmsle(real,predicted):

    sum=0.000

    length=len(predicted)

    for x in range(length):

        p = np.log(predicted[x]+1)

        r = np.log(real[x]+1)

        sum = sum + (p - r)**2

    return (sum/length)**0.5

        



MODEL_DIR = "tf_model_full"



categorical_features = ['vendor_id', 'passenger_count', 'store_and_fwd_flag', 'pickup_hour', 'pickup_date', 'pickup_month', 'day_week']

continuous_features = ['meter', 'pickup_distance', 'dropoff_distance', 'precipitation', 'snow_fall', 'snow_depth']

LABEL_COLUMN = 'duration'



# convert types of categorical features to string

for k in categorical_features:

    train_df[k] = train_df[k].apply(str)

    evaluate_df[k] = evaluate_df[k].apply(str)

    test_df[k] = test_df[k].apply(str)



# Converting Data into Tensors

def input_fn(df, training = True):

    # Creates a dictionary mapping from each continuous feature column name (k) to

    # the values of that column stored in a constant Tensor.

    continuous_cols = {k: tf.constant(df[k].values)

                       for k in continuous_features}



    # Creates a dictionary mapping from each categorical feature column name (k)

    # to the values of that column stored in a tf.SparseTensor.

    categorical_cols = {k: tf.SparseTensor(

        indices=[[i, 0] for i in range(df[k].size)],

        values=df[k].values,

        dense_shape=[df[k].size, 1])

        for k in categorical_features}



    # Merges the two dictionaries into one.

    feature_cols = dict(list(continuous_cols.items()) +

                        list(categorical_cols.items()))



    if training:

        # Converts the label column into a constant Tensor.

        label = tf.constant(df[LABEL_COLUMN].values)



        # Returns the feature columns and the label.

        return feature_cols, label

    

    # Returns the feature columns    

    return feature_cols



def train_input_fn():

    return input_fn(train_df)



def eval_input_fn():

    return input_fn(evaluate_df)



def test_input_fn():

    return input_fn(test_df, False)

    

# engineering features

engineered_features = []



for continuous_feature in continuous_features:

    engineered_features.append(

        tf.contrib.layers.real_valued_column(continuous_feature))





for categorical_feature in categorical_features:

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(

        categorical_feature, hash_bucket_size=1000)



    engineered_features.append(tf.contrib.layers.embedding_column(sparse_id_column=sparse_column, dimension=16,

                                                                  combiner="sum"))

# defining model

regressor = tf.contrib.learn.DNNRegressor(

    feature_columns=engineered_features, hidden_units=[10,10,10], model_dir=MODEL_DIR)



# Fit the model

wrap = regressor.fit(input_fn=train_input_fn, steps=1000)

    

# Evaluate model with rmsle metric

val_df = regressor.predict_scores(input_fn=eval_input_fn)

val_prediction = list(itertools.islice(val_df,evaluate_df['duration'].size))

val_prediction_array = np.asfarray(val_prediction)

val_y_array = np.asfarray(evaluate_df['duration']) 

print(rmsle(val_y_array, val_prediction_array))

'''

# Evaluating Our Model    

print('Evaluating ...')

results = regressor.evaluate(input_fn=eval_input_fn, steps=1)

for key in sorted(results):

    print("%s: %s" % (key, results[key]))



# Other evaluation metrics for reference

print(metrics.explained_variance_score(val_y_array, val_prediction_array))

print(metrics.mean_absolute_error(val_y_array, val_prediction_array))

print(metrics.mean_squared_error(val_y_array, val_prediction_array))

print(metrics.median_absolute_error(val_y_array, val_prediction_array))

print(metrics.r2_score(val_y_array, val_prediction_array))

    

# Predict with Our Model

predicted_output = regressor.predict_scores(input_fn=test_input_fn)

predictions = list(itertools.islice(predicted_output,test_df['vendor_id'].size))

prediction_array = np.asfarray(predictions)



# write predictions to csv

np.savetxt("prediction.csv", prediction_array, delimiter=",")

'''
