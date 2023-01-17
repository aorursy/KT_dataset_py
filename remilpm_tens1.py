#Multiple linear regression using tensorflow

from __future__ import print_function



import math



from IPython import display

from matplotlib import cm

from matplotlib import gridspec

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

from sklearn import metrics

import tensorflow as tf

from tensorflow.python.data import Dataset

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import tensorflow as tf

import pandas as pd

import matplotlib.pyplot as plt

import time

tf.logging.set_verbosity(tf.logging.ERROR)

pd.options.display.max_rows = 10

pd.options.display.float_format = '{:.1f}'.format



House_Price1=pd.read_csv("../input/data.csv")

House_Price1.tail()
#Check for null valuess

House_Price1.isnull().sum()
#Randomizing the data to make sure that no pathological ordering effects  the performance of Stochastic Gradient Descent.

House_Price2 = House_Price1.reindex(np.random.permutation(House_Price1.index))

House_Price2.head()
#Examine the data

House_Price2.describe()
# Define the input feature

House_Price_Feature=House_Price2[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above',

                           'sqft_basement']]



# Configure a numeric feature column for total_rooms.

House_Price_Feature_columns = [tf.feature_column.numeric_column("bedrooms"),tf.feature_column.numeric_column("bathrooms"),

                               tf.feature_column.numeric_column("sqft_living"),tf.feature_column.numeric_column("sqft_lot"),

                               tf.feature_column.numeric_column("floors"),tf.feature_column.numeric_column("waterfront"),

                               tf.feature_column.numeric_column("view"),tf.feature_column.numeric_column("condition"),

                               tf.feature_column.numeric_column("sqft_above"),tf.feature_column.numeric_column("sqft_basement")]

House_Price_Feature_columns
# Define the label.

House_Price_Targets = House_Price2["price"]

House_Price_Targets
# Use gradient descent as the optimizer for training the model.

my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)

my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)



# Configure the linear regression model with our feature columns and optimizer.

# Set a learning rate of 0.0000001 for Gradient Descent.

linear_regressor = tf.estimator.LinearRegressor(

    feature_columns=House_Price_Feature_columns,

    optimizer=my_optimizer

)

linear_regressor
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    """Trains a linear regression model of one feature.

  

    Args:

      features: pandas DataFrame of features

      targets: pandas DataFrame of targets

      batch_size: Size of batches to be passed to the model

      shuffle: True or False. Whether to shuffle the data.

      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely

    Returns:

      Tuple of (features, labels) for next data batch

    """

  

    # Convert pandas data into a dict of np arrays.

    features = {key:np.array(value) for key,value in dict(features).items()}                                           

 

    # Construct a dataset, and configure batching/repeating.

    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit

    ds = ds.batch(batch_size).repeat(num_epochs)

    

    # Shuffle the data, if specified.

    if shuffle:

      ds = ds.shuffle(buffer_size=10000)

    

    # Return the next batch of data.

    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels
_ = linear_regressor.train(

    input_fn = lambda:my_input_fn(House_Price_Feature, House_Price_Targets),

    steps=100

)
# Create an input function for predictions.

# Note: Since we're making just one prediction for each example, we don't 

# need to repeat or shuffle the data here.

prediction_input_fn =lambda: my_input_fn(House_Price_Feature, House_Price_Targets, num_epochs=1, shuffle=False)



# Call predict() on the linear_regressor to make predictions.

predictions = linear_regressor.predict(input_fn=prediction_input_fn)



# Format predictions as a NumPy array, so we can calculate error metrics.

predictions = np.array([item['predictions'][0] for item in predictions])



# Print Mean Squared Error and Root Mean Squared Error.

mean_squared_error = metrics.mean_squared_error(predictions, House_Price_Targets)

root_mean_squared_error = math.sqrt(mean_squared_error)

print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)

print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
min_house_value = House_Price2["price"].min()

max_house_value = House_Price2["price"].max()

min_max_difference = max_house_value - min_house_value



print("Min. Median House Value: %0.3f" % min_house_value)

print("Max. Median House Value: %0.3f" % max_house_value)

print("Difference between Min. and Max.: %0.3f" % min_max_difference)

print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)
