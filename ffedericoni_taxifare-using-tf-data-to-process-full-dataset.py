from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os; os.environ['OMP_NUM_THREADS'] = '4'

import tensorflow as tf
import numpy as np
import shutil
import pandas as pd

print("Tensorflow v",tf.__version__)
assert tf.__version__ >= "1.8" or tf.__version__ >= "1.10"
tf.logging.set_verbosity(tf.logging.INFO)

# List the CSV columns
CSV_COLUMNS = ['fare_amount', 'pickup_datetime','pickup_longitude','pickup_latitude',
               'dropoff_longitude','dropoff_latitude', 'passenger_count', 'key']

#Choose which column is your label
LABEL_COLUMN = 'fare_amount'
TRAIN_LINES = 55423856
from contextlib import contextmanager
import time
#Utility generator to time operations like training and evaluation
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
#This is just to have a look at the input data
PATH = '../input'
train_df = pd.read_csv(f'{PATH}/train.csv', nrows=10000)
train_df.head()
#train_df.describe()
BATCH_SIZE=8 
dataset = tf.contrib.data.make_csv_dataset(
    file_pattern=f'{PATH}/train.csv',
    batch_size=BATCH_SIZE,
    column_names=None,
    column_defaults=None,
    label_name='fare_amount',
    select_columns=[1, 2, 3, 4, 5, 6, 7],
    field_delim=',',
    use_quote_delim=True,
    na_value='',
    header=True,
    num_epochs=None,
    shuffle=True,
    shuffle_buffer_size=10000,
    shuffle_seed=None,
    prefetch_buffer_size=1,
    num_parallel_reads=1,
    num_parallel_parser_calls=2,
    sloppy=False,
    num_rows_for_inference=100
)

next_element = dataset.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    features, label = sess.run(next_element)
    print("Features:\n", features, "\n\nLabel:\n", label)


def pd_weekDay(year, month, day):
    df = pd.DataFrame({'year': year,
                       'month': month,
                       'day': day})
    date_df = pd.to_datetime(df)
    return date_df.dt.weekday.astype(np.int32)

#Let's check that the function is working correctly
years=np.array([2018, 2018, 2018])
months=np.array([8, 11, 1])
days=np.array([20, 6, 8])
print(pd_weekDay(years, months, days))

def tf_isAirport(latitude,longitude,airport_name='JFK'):
    jfkcoord = tf.constant([-73.8352, -73.7401, 40.6195, 40.6659])
    ewrcoord = tf.constant([-74.1925, -74.1531, 40.6700, 40.7081])
    lgucoord = tf.constant([-73.8895, -73.8550, 40.7664, 40.7931])
    if airport_name=='JFK':
        coord = jfkcoord
    elif airport_name=='EWR':
        coord = ewrcoord
    elif airport_name=='LGU':
        coord = lgucoord
    else:
        raise ValueError( f'Unknown NYC Airport {airport_name}' )
        
    is_airport = \
    tf.logical_and(
        tf.logical_and(
            tf.greater(latitude, coord[0]), tf.less(latitude, coord[1])
        ),
        tf.logical_and(
            tf.greater(longitude, coord[2]), tf.less(longitude, coord[3])
        )
    )
    return is_airport

def feat_eng_func(features, label=None):
    print("Feature Engineered Label:", label)
    #New features based on pickup datetime
    features['pickup_year'] = tf.string_to_number(tf.substr(features['pickup_datetime'], 0, 4), tf.int32)
    features['pickup_month'] = tf.string_to_number(tf.substr(features['pickup_datetime'], 5, 2), tf.int32)
    features['pickup_day'] = tf.string_to_number(tf.substr(features['pickup_datetime'], 8, 2), tf.int32)
    features['pickup_hour'] = tf.string_to_number(tf.substr(features['pickup_datetime'], 11, 2), tf.int32)
    #TODO is there an easy way to perform below calculation using TF APIs?
    features['pickup_weekday'] = tf.py_func(pd_weekDay,
                                            [features['pickup_year'], features['pickup_month'], features['pickup_day']],
                                            tf.int32,
                                            stateful=False,
                                            name='Weekday'
                                           )
    #TODO features['pickup_dayofyear'] = tf.cast(features['pickup_month'] * 30 + features['pickup_day'], tf.int32 )
    #Normalize year and add decimals for months. This is because fares increase with time
    features['pickup_dense_year'] = (
                tf.cast(features['pickup_year'], tf.float32) + \
                tf.cast(features['pickup_month'], tf.float32) / tf.constant(12.0, tf.float32) -  \
                 tf.constant(2009.0, tf.float32) ) /  \
                 tf.constant(6.0, tf.float32) 
   
    #Clip latitudes and longitudes
    minlat = tf.constant(38.0)
    maxlat = tf.constant(42.0)
    minlon = tf.constant(-76.0)
    maxlon = tf.constant(-72.0)
    features['pickup_longitude'] = tf.clip_by_value(features['pickup_longitude'], minlon, maxlon)
    features['pickup_latitude'] = tf.clip_by_value(features['pickup_latitude'], minlat, maxlat)
    features['dropoff_longitude'] = tf.clip_by_value(features['dropoff_longitude'], minlon, maxlon)
    features['dropoff_latitude'] = tf.clip_by_value(features['dropoff_latitude'], minlat, maxlat)
    #Clip passengers 
    minpass = tf.constant(1.0)
    maxpass = tf.constant(6.0)
    features['passenger_count'] = tf.clip_by_value(tf.cast(features['passenger_count'], tf.float32), minpass, maxpass)
    #Clip fare_amount
    #TODO normalize or tf.log the fare_amount
    if label != None:
        minfare = tf.constant(1.0)
        maxfare = tf.constant(300.0)
        label = tf.clip_by_value(label,  minfare, maxfare) 
    #New features based on pickup and dropoff position
    features['longitude_dist'] = tf.abs(features['pickup_longitude'] - features['dropoff_longitude'])
    features['latitude_dist'] = tf.abs(features['pickup_latitude'] - features['dropoff_latitude'])
    #compute euclidean distance of the trip 
    features['distance'] = tf.sqrt(features['longitude_dist']**2 + features['latitude_dist']**2)
    #features for airport locations
    features['is_JFK_pickup'] = tf_isAirport(features['pickup_latitude'], 
                                             features['pickup_longitude'],
                                             airport_name='JFK')
    features['is_JFK_dropoff'] = tf_isAirport(features['dropoff_latitude'], 
                                             features['dropoff_longitude'],
                                             airport_name='JFK')
    features['is_EWR_pickup'] = tf_isAirport(features['pickup_latitude'], 
                                             features['pickup_longitude'],
                                             airport_name='EWR')
    features['is_EWR_dropoff'] = tf_isAirport(features['dropoff_latitude'], 
                                             features['dropoff_longitude'],
                                             airport_name='EWR')
    features['is_LGU_pickup'] = tf_isAirport(features['pickup_latitude'], 
                                             features['pickup_longitude'],
                                             airport_name='LGU')
    features['is_LGU_dropoff'] = tf_isAirport(features['dropoff_latitude'], 
                                             features['dropoff_longitude'],
                                             airport_name='LGU')
    features['is_NYC_airport'] = tf.logical_or(
        tf.logical_or(
            tf.logical_or(features['is_JFK_pickup'], features['is_JFK_dropoff']),
            tf.logical_or(features['is_EWR_pickup'], features['is_EWR_dropoff'])),
        tf.logical_or(features['is_LGU_pickup'], features['is_LGU_dropoff'])
    )
    
    if label == None:
        return features
    return (features, label)
def read_dataset(filename, mode, batch_size = 512):
    def _input_fn():    
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            shuffle = False   # I assume the train records are already shuffled. Not sure this is correct.
        else:
            num_epochs = 1 # end-of-input after this
            shuffle = False

        if mode == tf.estimator.ModeKeys.PREDICT:
            label_name=None
            select_columns=[1, 2, 3, 4, 5, 6]
        else:
            label_name ='fare_amount'
            select_columns = [1, 2, 3, 4, 5, 6, 7]

        # Create list of files that match pattern
        file_list = tf.gfile.Glob(filename)
        # Create Dataset from the CSV files
        dataset = tf.contrib.data.make_csv_dataset(
            file_pattern=file_list,
            batch_size=batch_size, 
            column_names=None,
            column_defaults=None,
            label_name=label_name,
            select_columns=select_columns,
            field_delim=',',
            use_quote_delim=True,
            na_value='',
            header=True,
            num_epochs=num_epochs,
            shuffle=shuffle,
            shuffle_buffer_size=128*batch_size,
            shuffle_seed=None,
            prefetch_buffer_size=1,
            num_parallel_reads=1,
            num_parallel_parser_calls=3,
            sloppy=False,
            num_rows_for_inference=100
        )
#This is necessary to split train and eval
        skip_train_lines = TRAIN_LINES // batch_size // 100 * 10 #skip first 10% lines of train data set
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.skip(skip_train_lines) #this is very slow. I don't know if there are better ways.
        elif mode == tf.estimator.ModeKeys.EVAL:
            dataset = dataset.take(skip_train_lines) 

        dataset = dataset.map(feat_eng_func) #do all the feature engineering
        #TODO filter the dataset removing outliers and dirty data: 
        #TODO I tried with dataset.filter
#        dataset = dataset.repeat(3)
        return dataset.make_one_shot_iterator().get_next()
    return _input_fn

train_input_fn = read_dataset(f'{PATH}/train.csv', tf.estimator.ModeKeys.EVAL, batch_size = 8)
with timer('Evaluating'):
    with tf.Session() as sess:
        features, label = sess.run(train_input_fn())
        print("Features:\n", features, "\n\nLabel:\n", label)

# Define your feature columns
def create_feature_cols():
    hour_cat = tf.feature_column.categorical_column_with_identity('pickup_hour', 24 )
    weekday_cat = tf.feature_column.categorical_column_with_identity('pickup_weekday', 7)
    hour_X_weekday = tf.feature_column.crossed_column([hour_cat, weekday_cat], 500)
#    days_list = range(367)
#    yearday = tf.feature_column.categorical_column_with_vocabulary_list('pickup_dayofyear', days_list)
    return [
#    tf.feature_column.numeric_column('pickup_longitude'),
#    tf.feature_column.numeric_column('pickup_latitude'),
#    tf.feature_column.numeric_column('dropoff_longitude'),
#    tf.feature_column.numeric_column('dropoff_latitude'),
    tf.feature_column.numeric_column('passenger_count'),
    tf.feature_column.numeric_column('pickup_dense_year'),
#TODO    tf.feature_column.numeric_column('pickup_dayofyear'),
#TODO    tf.feature_column.embedding_column(yearday, 2),
#    tf.feature_column.numeric_column('pickup_year'),
#    tf.feature_column.numeric_column('pickup_month'),
#    tf.feature_column.numeric_column('pickup_day'),
    #TODO use embeddings for the hour
    #tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('pickup_hour', (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    #                                                                        11, 12, 13, 14, 15, 16, 17, 18,
    #                                                                         19, 20, 21, 22, 23) )
    #                                  ),
    #tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('pickup_weekday', (0, 1, 2, 3, 4, 5, 6)
    #                                                                                            )),
    tf.feature_column.embedding_column(hour_X_weekday, 2),
    tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_list('pickup_month', (0, 1, 2, 3, 4, 5, 6, 
                                                                                   7, 8, 9, 10, 11, 12)),
        2),
    tf.feature_column.numeric_column('longitude_dist'),
    tf.feature_column.numeric_column('latitude_dist'),
    tf.feature_column.numeric_column('distance'),
    tf.feature_column.numeric_column('is_JFK_pickup'),
    tf.feature_column.numeric_column('is_JFK_dropoff'),
    tf.feature_column.numeric_column('is_EWR_pickup'),
    tf.feature_column.numeric_column('is_EWR_dropoff'),
    tf.feature_column.numeric_column('is_LGU_pickup'),
    tf.feature_column.numeric_column('is_LGU_dropoff'),
    tf.feature_column.numeric_column('is_NYC_airport'),
#    tf.feature_column.numeric_column('is_long_distance')
  ]

# Define the sparse feature columns to be used with DNNLinearCombinedRegressor
def create_sparse_feature_cols():
    return [
    tf.feature_column.numeric_column('is_JFK_pickup'),
    tf.feature_column.numeric_column('is_JFK_dropoff'),
    tf.feature_column.numeric_column('is_EWR_pickup'),
    tf.feature_column.numeric_column('is_EWR_dropoff'),
    tf.feature_column.numeric_column('is_LGU_pickup'),
    tf.feature_column.numeric_column('is_LGU_dropoff'),
    tf.feature_column.numeric_column('is_NYC_airport'),
#    tf.feature_column.numeric_column('is_long_distance')
  ]

# Define the dense feature columns DNNLinearCombinedRegressor
def create_dense_feature_cols():
    hour_cat = tf.feature_column.categorical_column_with_identity('pickup_hour', 24 )
    weekday_cat = tf.feature_column.categorical_column_with_identity('pickup_weekday', 7)
    hour_X_weekday = tf.feature_column.crossed_column([hour_cat, weekday_cat], 500)
    month_cat = tf.feature_column.categorical_column_with_identity('pickup_month', 13 )
    return [
    tf.feature_column.embedding_column(hour_X_weekday, 2),
    tf.feature_column.embedding_column(month_cat, 2),
    tf.feature_column.numeric_column('pickup_longitude'),
    tf.feature_column.numeric_column('pickup_latitude'),
    tf.feature_column.numeric_column('dropoff_longitude'),
    tf.feature_column.numeric_column('dropoff_latitude'),
    tf.feature_column.numeric_column('passenger_count'),
    tf.feature_column.numeric_column('pickup_dense_year'),
    tf.feature_column.numeric_column('longitude_dist'),
    tf.feature_column.numeric_column('latitude_dist'),
    tf.feature_column.numeric_column('distance'),
  ]

BATCH_SIZE = 512
OUTDIR = './trained_model'
train_input_fn = read_dataset(f'{PATH}/train.csv', tf.estimator.ModeKeys.TRAIN, batch_size = BATCH_SIZE)
eval_input_fn = read_dataset(f'{PATH}/train.csv', tf.estimator.ModeKeys.EVAL, batch_size = BATCH_SIZE)
shutil.rmtree(OUTDIR, ignore_errors = True)
#estimator = tf.estimator.LinearRegressor(model_dir = OUTDIR, feature_columns = create_feature_cols())
estimator = tf.estimator.DNNRegressor(model_dir = OUTDIR, feature_columns = create_feature_cols(),
                                     hidden_units=[177, 73, 32],
                                     optimizer='Ftrl', 
                                     batch_norm=False, 
                                     dropout=0.1) 
#TODO I haven't got good results with the Wide and Deep model, but may be I have to experiment even more                                     
#estimator = tf.estimator.DNNLinearCombinedRegressor(model_dir = OUTDIR, 
#                                                    linear_feature_columns=create_sparse_feature_cols(),
#                                                    dnn_feature_columns=create_dense_feature_cols(),
#                                                    dnn_hidden_units=[128, 64, 32],
#                                                    dnn_dropout=None
#                                                   )
with timer('Training...'):
#increasing the steps you can increase the amount of processed records that is = BATCH_SIZE * max_steps
    estimator.train(train_input_fn, max_steps=120000) 
with timer('Evaluating'):
    evaluation = estimator.evaluate(eval_input_fn, name='train_eval')
print(evaluation)
avg_loss = evaluation['average_loss']
predict_input_fn = read_dataset(f'{PATH}/test.csv', tf.estimator.ModeKeys.PREDICT, batch_size=128)
predictions = estimator.predict(predict_input_fn)

test_df = pd.read_csv(f'{PATH}/test.csv', nrows=10000)
#test_df.head()

s = pd.Series()
for i, p in enumerate(predictions):
    s.at[i] = p['predictions'][0]
#s.describe()
test_df['fare_amount'] = s
sub = test_df[['key', 'fare_amount']]
sub.to_csv(f'DNNregr-{avg_loss:4.4}.csv', index=False)
#    print("Prediction %s: %s" % (i + 1, p))
s.describe()
