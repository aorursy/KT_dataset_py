import theano

from theano import tensor



# declare two symbolic floating-point scalars

a = tensor.dscalar()

b = tensor.dscalar()



# create a simple expression

c = a + b



# convert the expression into a callable object that takes (a,b)

# values as input and computes a value for c

f = theano.function([a,b], c)



# bind 7.5 to 'a', 3.5 to 'b', and evaluate 'c'

result = f(7.5, 3.5)

print(result)
import tensorflow as tf



# declare two symbolic floating-point scalars

a = tf.placeholder(tf.float32)

b = tf.placeholder(tf.float32)



# create a simple symbolic expression using the add function

add = tf.add(a, b)



# bind 7.5 to ' a ' , 3.5 to ' b ' , and evaluate ' c '

sess = tf.Session()

binding = {a: 7.5, b: 3.5}

c = sess.run(add, feed_dict=binding)

print(c)
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



tf.logging.set_verbosity(tf.logging.ERROR)

pd.options.display.max_rows = 10

pd.options.display.float_format = '{:.1f}'.format
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(

        np.random.permutation(california_housing_dataframe.index))

california_housing_dataframe["median_house_value"] /= 1000.0

california_housing_dataframe
california_housing_dataframe.describe()
# Define the input feature: total_rooms.

my_feature = california_housing_dataframe[["total_rooms"]]



# Configure a numeric feature column for total_rooms.

feature_columns = [tf.feature_column.numeric_column("total_rooms")]
targets = california_housing_dataframe["median_house_value"]
# Use gradient descent as the optimizer for training the model.

my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)

my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)



# Configure the linear regression model with our feature columns and optimizer.

# Set a learning rate of 0.0000001 for Gradient Descent.

linear_regressor = tf.estimator.LinearRegressor(

    feature_columns=feature_columns,

    optimizer=my_optimizer

)
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

    input_fn = lambda:my_input_fn(my_feature, targets),

    steps=100

)
# Create an input function for predictions.

# Note: Since we're making just one prediction for each example, we don't 

# need to repeat or shuffle the data here.

prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)



# Call predict() on the linear_regressor to make predictions.

predictions = linear_regressor.predict(input_fn=prediction_input_fn)



# Format predictions as a NumPy array, so we can calculate error metrics.

predictions = np.array([item['predictions'][0] for item in predictions])



# Print Mean Squared Error and Root Mean Squared Error.

mean_squared_error = metrics.mean_squared_error(predictions, targets)

root_mean_squared_error = math.sqrt(mean_squared_error)

print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)

print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
min_house_value = california_housing_dataframe["median_house_value"].min()

max_house_value = california_housing_dataframe["median_house_value"].max()

min_max_difference = max_house_value - min_house_value



print("Min. Median House Value: %0.3f" % min_house_value)

print("Max. Median House Value: %0.3f" % max_house_value)

print("Difference between Min. and Max.: %0.3f" % min_max_difference)

print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)
calibration_data = pd.DataFrame()

calibration_data["predictions"] = pd.Series(predictions)

calibration_data["targets"] = pd.Series(targets)

calibration_data.describe()
sample = california_housing_dataframe.sample(n=300)
# Get the min and max total_rooms values.

x_0 = sample["total_rooms"].min()

x_1 = sample["total_rooms"].max()



# Retrieve the final weight and bias generated during training.

weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]

bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')



# Get the predicted median_house_values for the min and max total_rooms values.

y_0 = weight * x_0 + bias 

y_1 = weight * x_1 + bias



# Plot our regression line from (x_0, y_0) to (x_1, y_1).

plt.plot([x_0, x_1], [y_0, y_1], c='r')



# Label the graph axes.

plt.ylabel("median_house_value")

plt.xlabel("total_rooms")



# Plot a scatter plot from our data sample.

plt.scatter(sample["total_rooms"], sample["median_house_value"])



# Display graph.

plt.show()
def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):

  """Trains a linear regression model of one feature.

  

  Args:

    learning_rate: A `float`, the learning rate.

    steps: A non-zero `int`, the total number of training steps. A training step

      consists of a forward and backward pass using a single batch.

    batch_size: A non-zero `int`, the batch size.

    input_feature: A `string` specifying a column from `california_housing_dataframe`

      to use as input feature.

  """

  

  periods = 10

  steps_per_period = steps / periods



  my_feature = input_feature

  my_feature_data = california_housing_dataframe[[my_feature]]

  my_label = "median_house_value"

  targets = california_housing_dataframe[my_label]



  # Create feature columns.

  feature_columns = [tf.feature_column.numeric_column(my_feature)]

  

  # Create input functions.

  training_input_fn = lambda:my_input_fn(my_feature_data, targets, batch_size=batch_size)

  prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

  

  # Create a linear regressor object.

  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

  linear_regressor = tf.estimator.LinearRegressor(

      feature_columns=feature_columns,

      optimizer=my_optimizer

  )



  # Set up to plot the state of our model's line each period.

  plt.figure(figsize=(15, 6))

  plt.subplot(1, 2, 1)

  plt.title("Learned Line by Period")

  plt.ylabel(my_label)

  plt.xlabel(my_feature)

  sample = california_housing_dataframe.sample(n=300)

  plt.scatter(sample[my_feature], sample[my_label])

  colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]



  # Train the model, but do so inside a loop so that we can periodically assess

  # loss metrics.

  print("Training model...")

  print("RMSE (on training data):")

  root_mean_squared_errors = []

  for period in range (0, periods):

    # Train the model, starting from the prior state.

    linear_regressor.train(

        input_fn=training_input_fn,

        steps=steps_per_period

    )

    # Take a break and compute predictions.

    predictions = linear_regressor.predict(input_fn=prediction_input_fn)

    predictions = np.array([item['predictions'][0] for item in predictions])

    

    # Compute loss.

    root_mean_squared_error = math.sqrt(

        metrics.mean_squared_error(predictions, targets))

    # Occasionally print the current loss.

    print("  period %02d : %0.2f" % (period, root_mean_squared_error))

    # Add the loss metrics from this period to our list.

    root_mean_squared_errors.append(root_mean_squared_error)

    # Finally, track the weights and biases over time.

    # Apply some math to ensure that the data and line are plotted neatly.

    y_extents = np.array([0, sample[my_label].max()])

    

    weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]

    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')



    x_extents = (y_extents - bias) / weight

    x_extents = np.maximum(np.minimum(x_extents,

                                      sample[my_feature].max()),

                           sample[my_feature].min())

    y_extents = weight * x_extents + bias

    plt.plot(x_extents, y_extents, color=colors[period]) 

  print("Model training finished.")



  # Output a graph of loss metrics over periods.

  plt.subplot(1, 2, 2)

  plt.ylabel('RMSE')

  plt.xlabel('Periods')

  plt.title("Root Mean Squared Error vs. Periods")

  plt.tight_layout()

  plt.plot(root_mean_squared_errors)



  # Output a table with calibration data.

  calibration_data = pd.DataFrame()

  calibration_data["predictions"] = pd.Series(predictions)

  calibration_data["targets"] = pd.Series(targets)

  display.display(calibration_data.describe())



  print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
train_model(

    learning_rate=0.00002,

    steps=500,

    batch_size=5

)
train_model(

    learning_rate=0.00002,

    steps=1000,

    batch_size=5,

    input_feature="population"

)
california_housing_dataframe["rooms_per_person"] = (

    california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"])



calibration_data = train_model(

    learning_rate=0.05,

    steps=500,

    batch_size=5,

    input_feature="rooms_per_person")
plt.subplot(1, 2, 2)

_ = california_housing_dataframe["rooms_per_person"].hist()
# delete the outliers

california_housing_dataframe["rooms_per_person"] = (

    california_housing_dataframe["rooms_per_person"]).apply(lambda x: min(x, 5))



_ = california_housing_dataframe["rooms_per_person"].hist()
calibration_data = train_model(

    learning_rate=0.05,

    steps=500,

    batch_size=5,

    input_feature="rooms_per_person")
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



tf.logging.set_verbosity(tf.logging.ERROR)

pd.options.display.max_rows = 10

pd.options.display.float_format = '{:.1f}'.format



california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")



def preprocess_features(california_housing_dataframe):

  """Prepares input features from California housing data set.



  Args:

    california_housing_dataframe: A Pandas DataFrame expected to contain data

      from the California housing data set.

  Returns:

    A DataFrame that contains the features to be used for the model, including

    synthetic features.

  """

  selected_features = california_housing_dataframe[

    ["latitude",

     "longitude",

     "housing_median_age",

     "total_rooms",

     "total_bedrooms",

     "population",

     "households",

     "median_income"]]

  processed_features = selected_features.copy()

  # Create a synthetic feature.

  processed_features["rooms_per_person"] = (

    california_housing_dataframe["total_rooms"] /

    california_housing_dataframe["population"])

  return processed_features



def preprocess_targets(california_housing_dataframe):

  """Prepares target features (i.e., labels) from California housing data set.



  Args:

    california_housing_dataframe: A Pandas DataFrame expected to contain data

      from the California housing data set.

  Returns:

    A DataFrame that contains the target feature.

  """

  output_targets = pd.DataFrame()

  # Scale the target to be in units of thousands of dollars.

  output_targets["median_house_value"] = (

    california_housing_dataframe["median_house_value"] / 1000.0)

  return output_targets
training_examples = preprocess_features(california_housing_dataframe.head(12000))

training_examples.describe()
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

training_targets.describe()
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))

validation_examples.describe()
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

validation_targets.describe()
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    """Trains a linear regression model of multiple features.

  

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

      ds = ds.shuffle(10000)

    

    # Return the next batch of data.

    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels
def construct_feature_columns(input_features):

  """Construct the TensorFlow Feature Columns.



  Args:

    input_features: The names of the numerical input features to use.

  Returns:

    A set of feature columns

  """ 

  return set([tf.feature_column.numeric_column(my_feature)

              for my_feature in input_features])
def train_model(

    learning_rate,

    steps,

    batch_size,

    training_examples,

    training_targets,

    validation_examples,

    validation_targets):

  """Trains a linear regression model of multiple features.

  

  In addition to training, this function also prints training progress information,

  as well as a plot of the training and validation loss over time.

  

  Args:

    learning_rate: A `float`, the learning rate.

    steps: A non-zero `int`, the total number of training steps. A training step

      consists of a forward and backward pass using a single batch.

    batch_size: A non-zero `int`, the batch size.

    training_examples: A `DataFrame` containing one or more columns from

      `california_housing_dataframe` to use as input features for training.

    training_targets: A `DataFrame` containing exactly one column from

      `california_housing_dataframe` to use as target for training.

    validation_examples: A `DataFrame` containing one or more columns from

      `california_housing_dataframe` to use as input features for validation.

    validation_targets: A `DataFrame` containing exactly one column from

      `california_housing_dataframe` to use as target for validation.

      

  Returns:

    A `LinearRegressor` object trained on the training data.

  """



  periods = 10

  steps_per_period = steps / periods

  

  # Create a linear regressor object.

  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

  linear_regressor = tf.estimator.LinearRegressor(

      feature_columns=construct_feature_columns(training_examples),

      optimizer=my_optimizer

  )

  

  # Create input functions.

  training_input_fn = lambda: my_input_fn(

      training_examples, 

      training_targets["median_house_value"], 

      batch_size=batch_size)

  predict_training_input_fn = lambda: my_input_fn(

      training_examples, 

      training_targets["median_house_value"], 

      num_epochs=1, 

      shuffle=False)

  predict_validation_input_fn = lambda: my_input_fn(

      validation_examples, validation_targets["median_house_value"], 

      num_epochs=1, 

      shuffle=False)



  # Train the model, but do so inside a loop so that we can periodically assess

  # loss metrics.

  print("Training model...")

  print("RMSE (on training data):")

  training_rmse = []

  validation_rmse = []

  for period in range (0, periods):

    # Train the model, starting from the prior state.

    linear_regressor.train(

        input_fn=training_input_fn,

        steps=steps_per_period,

    )

    # Take a break and compute predictions.

    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)

    training_predictions = np.array([item['predictions'][0] for item in training_predictions])

    

    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)

    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

    

    

    # Compute training and validation loss.

    training_root_mean_squared_error = math.sqrt(

        metrics.mean_squared_error(training_predictions, training_targets))

    validation_root_mean_squared_error = math.sqrt(

        metrics.mean_squared_error(validation_predictions, validation_targets))

    # Occasionally print the current loss.

    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))

    # Add the loss metrics from this period to our list.

    training_rmse.append(training_root_mean_squared_error)

    validation_rmse.append(validation_root_mean_squared_error)

  print("Model training finished.")



  # Output a graph of loss metrics over periods.

  plt.ylabel("RMSE")

  plt.xlabel("Periods")

  plt.title("Root Mean Squared Error vs. Periods")

  plt.tight_layout()

  plt.plot(training_rmse, label="training")

  plt.plot(validation_rmse, label="validation")

  plt.legend()



  return linear_regressor
linear_regressor = train_model(

    learning_rate=0.00003,

    steps=500,

    batch_size=5,

    training_examples=training_examples,

    training_targets=training_targets,

    validation_examples=validation_examples,

    validation_targets=validation_targets)
california_housing_test_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv", sep=",")



test_examples = preprocess_features(california_housing_test_data)

test_targets = preprocess_targets(california_housing_test_data)



predict_test_input_fn = lambda: my_input_fn(

      test_examples, 

      test_targets["median_house_value"], 

      num_epochs=1, 

      shuffle=False)



test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)

test_predictions = np.array([item['predictions'][0] for item in test_predictions])



root_mean_squared_error = math.sqrt(

    metrics.mean_squared_error(test_predictions, test_targets))



print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)
import tensorflow as tf

from tensorflow.keras import layers



print(tf.VERSION)

print(tf.keras.__version__)



mnist = tf.keras.datasets.mnist



# Load and pepare the MNIST data

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.Sequential()

# Adds a densely-connected layer with 64 units to the model:

model.add(layers.Dense(64, activation='relu'))

# Add another:

model.add(layers.Dense(64, activation='relu'))

# Add a softmax layer with 10 output units:

model.add(layers.Dense(10, activation='softmax'))
# Create a sigmoid layer:

layers.Dense(64, activation='sigmoid')  

# Or: layers.Dense(64, activation=tf.sigmoid)



# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:

layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))



# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:

layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))



# A linear layer with a kernel initialized to a random orthogonal matrix:

layers.Dense(64, kernel_initializer='orthogonal')



# A linear layer with a bias vector initialized to 2.0s:

layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))
model = tf.keras.Sequential([

# Adds a densely-connected layer with 64 units to the model:

layers.Dense(64, activation='relu', input_shape=(32,)),

# Add another:

layers.Dense(64, activation='relu'),

# Add a softmax layer with 10 output units:

layers.Dense(10, activation='softmax')])



model.compile(optimizer=tf.train.AdamOptimizer(0.001),

              loss='categorical_crossentropy',

              metrics=['accuracy'])
# Configure a model for mean-squared error regression.

model.compile(optimizer=tf.train.AdamOptimizer(0.01),

              loss='mse',       # mean squared error

              metrics=['mae'])  # mean absolute error



# Configure a model for categorical classification.

model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),

              loss=tf.keras.losses.categorical_crossentropy,

              metrics=[tf.keras.metrics.categorical_accuracy])
import numpy as np



def random_one_hot_labels(shape):

  n, n_class = shape

  classes = np.random.randint(0, n_class, n)

  labels = np.zeros((n, n_class))

  labels[np.arange(n), classes] = 1

  return labels



data = np.random.random((1000, 32))

labels = random_one_hot_labels((1000, 10))



model.fit(data, labels, epochs=10, batch_size=32)
import numpy as np



data = np.random.random((1000, 32))

labels = random_one_hot_labels((1000, 10))



val_data = np.random.random((100, 32))

val_labels = random_one_hot_labels((100, 10))



model.fit(data, labels, epochs=10, batch_size=32,

          validation_data=(val_data, val_labels))
# Instantiates a toy dataset instance:

dataset = tf.data.Dataset.from_tensor_slices((data, labels))

dataset = dataset.batch(32)

dataset = dataset.repeat()



# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.

model.fit(dataset, epochs=10, steps_per_epoch=30)
dataset = tf.data.Dataset.from_tensor_slices((data, labels))

dataset = dataset.batch(32).repeat()



val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))

val_dataset = val_dataset.batch(32).repeat()



model.fit(dataset, epochs=10, steps_per_epoch=30,

          validation_data=val_dataset,

          validation_steps=3)
data = np.random.random((1000, 32))

labels = random_one_hot_labels((1000, 10))



model.evaluate(data, labels, batch_size=32)



model.evaluate(dataset, steps=30)
result = model.predict(data, batch_size=32)

print(result.shape)