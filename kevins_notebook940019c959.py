# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR) # supress warning messages

tf.__version__
data = pd.read_csv('../input/data.csv')

data.set_index('shot_id', inplace=True)
# prepare data

data_cl = data.copy()

target = data_cl['shot_made_flag'].copy()



# drop unnecessary columns

data_cl.drop('team_id', inplace=True, axis=1) #only 1 category

data_cl.drop('lat', inplace=True, axis=1) # correlated with loc_x

data_cl.drop('lon', inplace=True, axis=1) # correlated with loc_y

data_cl.drop('game_id', inplace=True, axis=1) # should not be dependent on game id, furthermore it's contained in opponent/match

data_cl.drop('game_event_id', inplace=True, axis=1) # independent, unique for every shots in a game

data_cl.drop('team_name', inplace=True, axis=1) # always LA Lakers

data_cl.drop('shot_made_flag', inplace=True, axis=1) # target variables
# Time remaining

data_cl['seconds_from_period_end'] = 60 * data_cl['minutes_remaining'] + data_cl['seconds_remaining']

data_cl['last_5_sec_in_period'] = data_cl['seconds_from_period_end'] < 5

# drop redundant features

data_cl.drop('minutes_remaining', axis=1, inplace=True)

data_cl.drop('seconds_remaining', axis=1, inplace=True)



# Matchup -- away/home

data_cl['home_play'] = data_cl['matchup'].str.contains('vs').astype('int')

data_cl.drop('matchup', axis=1, inplace=True)



# Extract year and month from date of game

data_cl['game_date'] = pd.to_datetime(data_cl['game_date'])

data_cl['game_year'] = data_cl['game_date'].dt.year

data_cl['game_month'] = data_cl['game_date'].dt.month

data_cl.drop('game_date', axis=1, inplace=True)



# Replace 20 least common action types with value 'Other'

rare_action_types = data_cl['action_type'].value_counts().sort_values().index.values[:20]

data_cl.loc[data_cl['action_type'].isin(rare_action_types), 'action_type'] = 'Other'

categorial_cols = [

    'action_type', 'combined_shot_type', 'period', 'season', 'shot_type',

    'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'game_year',

    'game_month', 'opponent']



for cc in categorial_cols:

    dummies = pd.get_dummies(data_cl[cc])

    dummies = dummies.add_prefix("{}_".format(cc))

    data_cl.drop(cc, axis=1, inplace=True)

    data_cl = data_cl.join(dummies)
data_cl.shape
# Train/test mask

unknown_mask = data['shot_made_flag'].isnull()

data_submit = data_cl[unknown_mask]

# Separate dataset for training

X = data_cl[~unknown_mask]

Y = target[~unknown_mask]



# Train/validation mask

val_mask = np.random.rand(len(X)) < 0.7 # 70/30 split

X_train = X[val_mask]

Y_train = Y[val_mask]

X_val = X[~val_mask]

Y_val = Y[~val_mask]
# define two feature columns with real values

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=161)]



# create a neural network

dnnc = tf.contrib.learn.DNNClassifier(

  feature_columns=feature_columns,

  hidden_units=[],

  n_classes=2)



dnnc
# print the accuracy of the neural network

def print_accuracy():

  loss = dnnc.evaluate(x=X_val, y=Y_val)['loss']

  print(loss)

  

# train the model just for 1 step and print the accuracy

dnnc.fit(x=X_train, y=Y_train, steps=1)

print_accuracy()
tf.logging.set_verbosity(tf.logging.INFO)

steps = 500

for i in range (1, 6):

  dnnc.fit(x=X_train, y=Y_train, steps=steps)

  print('Steps: ' + str(i * steps))

  print_accuracy()

  

print('\nTraining Finished.')