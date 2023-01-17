from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os; os.environ['OMP_NUM_THREADS'] = '4'
import sys
import tensorflow as tf
import numpy as np
import shutil
import pandas as pd

print(tf.__version__)
assert tf.__version__ >= "1.8" or tf.__version__ >= "1.10"
tf.logging.set_verbosity(tf.logging.INFO)

# List the CSV columns
CSV_COLUMNS = ['fare_amount', 'pickup_datetime','pickup_longitude','pickup_latitude',
               'dropoff_longitude','dropoff_latitude', 'passenger_count', 'key']

#Choose which column is your label
LABEL_COLUMN = 'fare_amount'
TRAIN_LINES = 55423856
#import os
#print(os.listdir("../input"))
DEBUG=False

from contextlib import contextmanager
import time
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
#This is just to have a look at the data
PATH = '../input'
train_df = pd.read_csv(f'{PATH}/train.csv', nrows=100000)
train_df['distance'] = np.sqrt(np.abs(train_df['pickup_longitude']-train_df['dropoff_longitude'])**2 +
                        np.abs(train_df['pickup_latitude']-train_df['dropoff_latitude'])**2)
train_df.head()
train_df.describe()
BATCH_SIZE=1 #Filtering works only with size 1 batches!!
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
#    num_parallel_parser_calls=2,
    sloppy=False,
    num_rows_for_inference=100
)

if DEBUG:
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

def pd_dayofYear(year, month, day):
    df = pd.DataFrame({'year': year,
                       'month': month,
                       'day': day})
    date_df = pd.to_datetime(df)
    return date_df.dt.dayofyear.astype(np.int32)

if DEBUG:
    years=np.array([2018, 2018, 2018])
    months=np.array([8, 11, 1])
    days=np.array([20, 6, 8])
    print(pd_dayofYear(years, months, days))

nyccenter = tf.constant([-74.0063889, 40.7141667])

def outer_product(x, y, x1, y1, x2, y2):
#    x = tf.Print(x, [x, y, x1, y1, x2, y2])
    outer_product = (x-x1)*(y2-y1)-(y-y1)*(x2-x1)
    return outer_product

def river_side(x, y, river_name='EAS'):
    '''The function takes coordinates as input and calculates if the point is one side or another of the river.
        I tried to use the information but I got no benefit.
    '''
    if river_name=='EAS': # East River
        river_line = tf.constant([-74.07, 40.6, -73.84, 40.9])
    elif river_name=='HUD': # East River
        river_line = tf.constant([-74.0356, 40.6868, -73.9338, 40.8823])
    else:
        raise ValueError( f'Unknown NYC River {river_name}' )
    tf.reshape(river_line, [1,4])
    if DEBUG:
        print(x,y, river_line)
    side = tf.sign(outer_product(x, y, river_line[0], river_line[1], river_line[2], river_line[3]))
    return side

def distance_from_loc(x, y, x1, y1):
    '''Euclidean Distance between location (x,y) and Tensor of locations (x1,y1)'''
    return ( tf.sqrt(tf.abs(x1-x)**2 + tf.abs(y1-y)**2) )

def angle_from_nyc(x, y):
    x1 = x - nyccenter[0]
    y1 = y - nyccenter[1]
    angle = tf.atan2(y1, x1)
    return angle
    
if DEBUG:
    long=np.array([-73.94, -73.95, -73.96])
    lat=np.array([40.7613, 40.7613, 40.7613])
    side = river_side(long, lat, river_name='EAS')
    with tf.Session() as sess:
        result = sess.run(side)
        print("Side:\n", result)
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
            tf.greater(longitude, coord[0]), tf.less(longitude, coord[1])
        ),
        tf.logical_and(
            tf.greater(latitude, coord[2]), tf.less(latitude, coord[3])
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
    #TODO is there an easy way to perform below functions using TF APIs?
    features['pickup_weekday'] = tf.py_func(pd_weekDay,
                                            [features['pickup_year'], features['pickup_month'], features['pickup_day']],
                                            tf.int32,
                                            stateful=False,
                                            name='Weekday'
                                           )
    #no advantage features['pickup_dayofyear'] = tf.cast(features['pickup_month'] * 31 + features['pickup_day'], tf.int32 ) #not precise, but good enough
    #Normalize year and add decimals for months. This is because fares increase with time
#    features['pickup_dense_year'] = (
#                tf.cast(features['pickup_year'], tf.float32) + \
#                tf.cast(features['pickup_month'], tf.float32) / tf.constant(12.0, tf.float32) -  \
#                 tf.constant(2009.0, tf.float32) ) /  \
#                 tf.constant(6.0, tf.float32) 
    features['night'] = tf.cast(
        tf.logical_or( tf.greater(features['pickup_hour'], 19),  tf.less(features['pickup_hour'], 7)),
        tf.float32)
    #Clip latitudes and longitudes
    minlat = tf.constant(38.0)
    maxlat = tf.constant(42.0)
    minlon = tf.constant(-76.0)
    maxlon = tf.constant(-72.0)
    features['pickup_longitude'] = tf.clip_by_value(features['pickup_longitude'], minlon, maxlon)
    features['pickup_latitude'] = tf.clip_by_value(features['pickup_latitude'], minlat, maxlat)
    features['dropoff_longitude'] = tf.clip_by_value(features['dropoff_longitude'], minlon, maxlon)
    features['dropoff_latitude'] = tf.clip_by_value(features['dropoff_latitude'], minlat, maxlat)
    #Clip (normalize passengers didn't work?)
    minpass = tf.constant(1.0)
    maxpass = tf.constant(6.0)
    features['passenger_count'] = tf.clip_by_value(tf.cast(features['passenger_count'], tf.float32), minpass, maxpass)
    #Clip fare_amount
    #TODO normalize or tf.log the fare_amount
    if label != None:
        minfare = tf.constant(3.0)
        maxfare = tf.constant(300.0)
        label = tf.clip_by_value(label,  minfare, maxfare) 
    #TODO feature for bridge passing
    #TODO new feature for distance and angle from city center
    #New features based on pickup and dropoff position
    features['longitude_dist'] = tf.abs(features['pickup_longitude'] - features['dropoff_longitude'])
    features['latitude_dist'] = tf.abs(features['pickup_latitude'] - features['dropoff_latitude'])
    #compute euclidean distance of the trip (multiply by 10 to slightly normalize)
    features['distance'] = tf.sqrt(features['longitude_dist']**2 + features['latitude_dist']**2)
#    features['pick_dist_center'] = distance_from_loc(nyccenter[0], nyccenter[1], 
#                                                     features['pickup_longitude'], features['pickup_latitude'])
#    features['drop_dist_center'] = distance_from_loc(nyccenter[0], nyccenter[1], 
#                                                     features['dropoff_longitude'], features['dropoff_latitude'])
    features['pick_angle'] = angle_from_nyc(features['pickup_longitude'],features['pickup_latitude'])
    features['drop_angle'] = angle_from_nyc(features['dropoff_longitude'],features['dropoff_latitude'])
    features['angle'] = features['pick_angle'] - features['drop_angle']
#    features['ortdistance'] = features['longitude_dist'] + features['latitude_dist']
#    long_distance = tf.constant(0.7)
#    features['is_long_distance'] = tf.less(long_distance, features['distance'])
#    features['is_JFK_pickup'] = tf_isAirport(features['pickup_latitude'], 
#                                             features['pickup_longitude'],
#                                             airport_name='JFK')
#    features['is_JFK_dropoff'] = tf_isAirport(features['dropoff_latitude'], 
#                                             features['dropoff_longitude'],
#                                             airport_name='JFK')
#    features['is_EWR_pickup'] = tf_isAirport(features['pickup_latitude'], 
#                                             features['pickup_longitude'],
#                                             airport_name='EWR')
#    features['is_EWR_dropoff'] = tf_isAirport(features['dropoff_latitude'], 
#                                             features['dropoff_longitude'],
#                                             airport_name='EWR')
#    features['is_LGU_pickup'] = tf_isAirport(features['pickup_latitude'], 
#                                             features['pickup_longitude'],
#                                             airport_name='LGU')
#    features['is_LGU_dropoff'] = tf_isAirport(features['dropoff_latitude'], 
#                                             features['dropoff_longitude'],
#                                             airport_name='LGU')
#    features['is_NYC_airport'] = tf.logical_or(
#        tf.logical_or(
#            tf.logical_or(features['is_JFK_pickup'], features['is_JFK_dropoff']),
#            tf.logical_or(features['is_EWR_pickup'], features['is_EWR_dropoff'])),
#        tf.logical_or(features['is_LGU_pickup'], features['is_LGU_dropoff'])
#    )
#    BOOL_COLUMNS = ['is_JFK_pickup', 'is_JFK_dropoff', 'is_EWR_pickup', 'is_EWR_dropoff',
#                   'is_LGU_pickup', 'is_LGU_dropoff', 'is_NYC_airport' ]
#    for key in BOOL_COLUMNS:
#        features[key] = tf.cast(features[key], tf.int32)
#
#    features['same_side_EAS'] = tf.equal(river_side(features['pickup_longitude'], features['pickup_latitude'], river_name='EAS'),
#                                         river_side(features['dropoff_longitude'], features['dropoff_latitude'], river_name='EAS'))
#    features['same_side_HUD'] = tf.equal(river_side(features['pickup_longitude'], features['pickup_latitude'], river_name='HUD'),
#                                         river_side(features['dropoff_longitude'], features['dropoff_latitude'], river_name='HUD'))
#

#    features['pickup_minute'] = tf.substr(features['pickup_datetime'], 14, 2)
#TODO normalize long and lat
#TODO remove outliers on passenger_count and fare_amount
#    print(features)
    if label == None:
        return features
    return (features, label)
# Create an input function that stores your data into a dataset
def read_dataset(filename, mode, batch_size = 512):
    def _input_fn():    
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            shuffle = False
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
            batch_size=batch_size, #for filtering
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
#            num_parallel_parser_calls=3,
            sloppy=False,
            num_rows_for_inference=100
        )
#This is necessary to split train and eval
        skip_train_lines = TRAIN_LINES // batch_size // 100 * 10 #skip first 10% lines of train data set
        if mode == tf.estimator.ModeKeys.TRAIN:
#        dataset = dataset.filter(filter_data)
            dataset = dataset.skip(skip_train_lines)
        elif mode == tf.estimator.ModeKeys.EVAL:
            dataset = dataset.take(skip_train_lines) 

        dataset = dataset.map(feat_eng_func)
#        dataset = dataset.repeat(3)
#        dataset = dataset.batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()
    return _input_fn

if DEBUG:
    train_input_fn = read_dataset(f'{PATH}/train.csv', tf.estimator.ModeKeys.EVAL, batch_size = 8)
    with timer('Debugging'):
        with tf.Session() as sess:
            features, label = sess.run(train_input_fn())
            print("Features:\n", features, "\n\nLabel:\n", label)
NUMERIC_COLUMNS = ['passenger_count', 'pickup_dense_year', 
                   'longitude_dist', 'latitude_dist',
                   'distance', 'pick_dist_center', 'drop_dist_center', 'angle',
                   'pick_angle','drop_angle'
#                   , 'night'
                  ]
BOOL_COLUMNS = ['is_JFK_pickup', 'is_JFK_dropoff', 'is_EWR_pickup', 'is_EWR_dropoff',
                   'is_LGU_pickup', 'is_LGU_dropoff', 'is_NYC_airport'
                  ]
# Define your feature columns
def create_bucket_feature_cols():
    hour_buckets = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('pickup_hour'), list(np.linspace(0, 23, 24)))
    weekday_buckets = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('pickup_weekday'), list(np.linspace(0, 7, 8)))
    month_buckets = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('pickup_month'), list(np.linspace(0, 11, 12)))
    year_buckets = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('pickup_year'), list(np.linspace(2009, 2017, 9)))
    passenger_buckets = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('passenger_count'), list(np.linspace(0, 8, 9)))
    distance_buckets = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('distance'), list(np.linspace(0, 0.05, 100)))
    night_buckets = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('night'), list(np.linspace(0, 1, 2)))
    angle_buckets = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('angle'), list(np.linspace(-3.14, 3.14, 90)))
    #hour_X_weekday = tf.feature_column.crossed_column([hour_cat, weekday_cat], 500)
    #days_list = range(367)
    #yearday = tf.feature_column.categorical_column_with_vocabulary_list('pickup_dayofyear', days_list)
    NUM_BUCKETS = 27
    long_list = list(np.linspace(-74.2, -73.7, NUM_BUCKETS))
    lat_list = list(np.linspace(40.55, 41.0, NUM_BUCKETS))
    p_lon = tf.feature_column.numeric_column('pickup_longitude')
    p_lat = tf.feature_column.numeric_column('pickup_latitude')
    d_lon = tf.feature_column.numeric_column('dropoff_longitude')
    d_lat = tf.feature_column.numeric_column('dropoff_latitude')
    buck_p_lon = tf.feature_column.bucketized_column(p_lon, long_list)
    buck_p_lat = tf.feature_column.bucketized_column(p_lat, lat_list)
    buck_d_lon = tf.feature_column.bucketized_column(d_lon, long_list)
    buck_d_lat = tf.feature_column.bucketized_column(d_lat, lat_list)
########################################################################    
    return [
        hour_buckets,
#        weekday_buckets,
        month_buckets,
        year_buckets,
        passenger_buckets,
        distance_buckets,
        angle_buckets,
        night_buckets,
        buck_p_lon,
        buck_p_lat,
        buck_d_lon,
        buck_d_lat
           ]


if DEBUG:
    feat = create_bucket_feature_cols()
    print("Number of features=", len(feat))
    print(feat)
BATCH_SIZE = 1024
train_input_fn = read_dataset(f'{PATH}/train.csv', tf.estimator.ModeKeys.TRAIN, batch_size = BATCH_SIZE)
eval_input_fn = read_dataset(f'{PATH}/train.csv', tf.estimator.ModeKeys.EVAL, batch_size = BATCH_SIZE)
# Create estimator train and evaluate function
def train_and_evaluate(output_dir, num_train_steps):
#    estimator = tf.estimator.LinearRegressor(model_dir = output_dir, feature_columns = create_feature_cols())
    runconfig = tf.estimator.RunConfig(model_dir = OUTDIR, keep_checkpoint_max=1, 
                                   save_summary_steps=5000, log_step_count_steps=5000,
                                   save_checkpoints_steps=10000,
                                   tf_random_seed = 42
                                  )
    estimator = tf.estimator.BoostedTreesRegressor(model_dir = output_dir, 
                                        feature_columns = create_bucket_feature_cols(),
                                        n_batches_per_layer=128,
                                        n_trees=400,
                                        max_depth=6,
                                        learning_rate=0.05,
                                        l1_regularization=0.01,
                                        l2_regularization=0.0,
                                        tree_complexity=0.0,
                                        min_node_weight=0.0,
                                        center_bias = True,
                                        config=runconfig)
    train_spec = tf.estimator.TrainSpec(input_fn = train_input_fn, 
                                      max_steps = num_train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn = eval_input_fn, 
                                    steps = None, 
                                    start_delay_secs = 0, # start evaluating after N seconds, 
                                    throttle_secs = 60)  # evaluate every N seconds
    evaluation, result = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    return estimator, evaluation, result
    

OUTDIR = './trained_model'
shutil.rmtree(OUTDIR, ignore_errors = True)
with timer('Train and Evaluate...'):
    estimator, evaluation, result = train_and_evaluate(OUTDIR, 240000)
print(evaluation)
print(evaluation)
print(evaluation, file = sys.stderr)
avg_loss = evaluation['average_loss']
predict_input_fn = read_dataset(f'{PATH}/test.csv', tf.estimator.ModeKeys.PREDICT, batch_size=1)
predictions = estimator.predict(predict_input_fn)

test_df = pd.read_csv(f'{PATH}/test.csv', nrows=10000)
#test_df.head()

s = pd.Series()
for i, p in enumerate(predictions):
    if i < 9915:
        s.at[i] = p['predictions'][0]
    else:
        break
test_df['fare_amount'] = s
sub = test_df[['key', 'fare_amount']]
sub.to_csv(f'DNNregr-{avg_loss:4.4}.csv', index=False, float_format='%.1f')
#    print("Prediction %s: %s" % (i + 1, p))
if DEBUG:
    s.describe()


