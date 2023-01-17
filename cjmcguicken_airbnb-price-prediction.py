import pandas as pd



# Get entire airbnb dataset and store in dataframe

dataset_path = '../input/airbnb-nyc/AB_NYC_2019_full.csv'

data = pd.read_csv(dataset_path)



# Print head of data to show column names

data.head()
# Show columns with NA values

data.isna().sum()



# For simplicity, remove NA values

data = data.dropna()
# Drop all columns that aren't required

data = data.drop('id', 1)



# Numerical columns have been added for neighbourhood and room type so can remove categorical type columns

data = data.drop('neighbourhood_group', 1)

data = data.drop('neighbourhood', 1)

data = data.drop('room_type', 1)



# Convert date into numerical type

import datetime as dt

data['last_review'] = pd.to_datetime(data['last_review'])

data['last_review']=data['last_review'].map(dt.datetime.toordinal)
data.head()
# Split data into test / train in a 20 / 80 split

train_dataset = data.sample(frac=0.8,random_state=0)

test_dataset = data.drop(train_dataset.index)
# Use seaborn to visualise relationships between features

import matplotlib.pyplot as plt

import seaborn as sns



sns.pairplot(train_dataset[["price", "number_of_reviews", "calculated_host_listings_count", "reviews_per_month"]], diag_kind="kde")

plt.show()
# Display statistics of features

train_stats = train_dataset.describe()

train_stats = train_stats.transpose()

train_stats
# Separate predictor from features

train_labels = train_dataset.pop('price')

test_labels = test_dataset.pop('price')
# Normalise data - as different scales used across the features

def norm(x):

  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)

normed_test_data = norm(test_dataset)
normed_train_data = normed_train_data.drop(columns=['price'])

normed_test_data = normed_test_data.drop(columns=['price'])

normed_train_data.head()
# Use keras with Tensorflow to build a sequential model 

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



def build_model():

  model = keras.Sequential([

    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(normed_train_data.keys())]),

    layers.Dense(64, activation=tf.nn.relu),

    layers.Dense(1)

  ])



  # Use Keras optimiser RMSProp with learning rate set

  optimizer = tf.keras.optimizers.RMSprop(0.001)



  # Compile model using means squared error as loss function (model will aim to minimise)

  model.compile(loss='mean_squared_error',

                optimizer=optimizer,

                metrics=['mean_absolute_error', 'mean_squared_error'])

  return model
# Build model

model = build_model()
# Display summary of model, broken down by layers

model.summary()
# Display training progress by printing a single dot for each completed epoch

class PrintDot(keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs):

    if epoch % 100 == 0: print('')

    print('.', end='')

    

# Number of iterations

EPOCHS = 200



history = model.fit(

  normed_train_data, train_labels,

  epochs=EPOCHS, validation_split = 0.2, verbose=0,

  callbacks=[PrintDot()])
# Show tail of iteration results

hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
# Plot error over iterations

def plot_history(history):

  hist = pd.DataFrame(history.history)

  hist['epoch'] = history.epoch



  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Abs Error [Price]')

  plt.plot(hist['epoch'], hist['mean_absolute_error'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],

           label = 'Val Error')

  plt.ylim([0,100])

  plt.legend()



  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Square Error [$Price^2$]')

  plt.plot(hist['epoch'], hist['mean_squared_error'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mean_squared_error'],

           label = 'Val Error')

  plt.ylim([0,50000])

  plt.legend()

  plt.show()





plot_history(history)
# Evaluate model against test data and get quality metrics (MAE and MSE)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)



print("Testing set Mean Abs Error: ${:5.2f}, Mean Squared Error: $^2:{:5.2f}".format(mae, mse))
# Plot actual vs. predicted on graph

test_predictions = model.predict(normed_test_data).flatten()



plt.scatter(test_labels, test_predictions)

plt.xlabel('True Values [$]')

plt.ylabel('Predictions [$]')

plt.axis('equal')

plt.axis('square')

plt.xlim([0,1000])

plt.ylim([0,1000])

_ = plt.plot([-100, 100], [-100, 100])