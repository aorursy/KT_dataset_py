import pandas as pd

import tensorflow as tf

import numpy as np

from tensorflow.python.data import Dataset

import math

from matplotlib import cm

from matplotlib import gridspec

from matplotlib import pyplot as plt

from sklearn import metrics

import os

tf.logging.set_verbosity(tf.logging.ERROR)

data = pd.read_csv("../input/Admission_Predict.csv")
data.head()
data.shape
data.describe()
def list_coloumns_have_nan_values(data):

    return data.columns[data.isna().any()].tolist()
list_coloumns_have_nan_values(data)
data.columns = ['serial_num', 'gre_score', 'tofel_score', 'university_rating', 'sop', 'lor', 'cgpa', 'research', 'chance']
data.apply(lambda s: data.corrwith(s))
feature_cols = ['gre_score', 'tofel_score', 'university_rating', 'sop', 'lor', 'cgpa', 'research']

target_col = "chance";
data.columns
for f in feature_cols:

    data.plot.scatter(x=f, y=target_col);


for f in feature_cols:

    data.hist(f);
def shuffle(data_frame):

     return data_frame.reindex(np.random.permutation(data_frame.index))
data = shuffle(data)
def split_training_and_test(data_frame, training_percentage):

    training_number = data_frame.shape[0] * training_percentage / 100

    test_number = data_frame.shape[0] - training_number

    return data_frame.head(int(training_number)), data_frame.tail(int(test_number))
training_data, test_data = split_training_and_test(data, 90)
training_data, validation_data = split_training_and_test(training_data, 80)
print(training_data.shape)

print(validation_data.shape)

print(test_data.shape)
training_features = training_data[feature_cols]

training_labels = training_data[[target_col]]



validation_features = validation_data[feature_cols]

validation_labels = validation_data[[target_col]]



test_features = test_data[feature_cols]

test_labels = test_data[[target_col]]
print(training_features.shape)

print(training_labels.shape)
# here we only pass the name of the coloumns.

def construct_feature_columns(input_features):

    return set([tf.feature_column.numeric_column(my_feature)for my_feature in input_features])
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    

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
def train_nn_regression_model(my_optimizer, steps, batch_size, hidden_units, training_examples,

        training_targets, validation_examples, validation_targets, label_col_name):

    



    periods = 10

    steps_per_period = steps / periods



    # Create a DNNRegressor object.

    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    dnn_regressor = tf.estimator.DNNRegressor(

        feature_columns=construct_feature_columns(training_examples),

        hidden_units=hidden_units,

        optimizer=my_optimizer

    )

    # Create input functions.

    training_input_fn = lambda: my_input_fn(training_examples,

                                            training_targets[label_col_name],

                                            batch_size=batch_size)

    predict_training_input_fn = lambda: my_input_fn(training_examples,

                                                    training_targets[label_col_name],

                                                    num_epochs=1,

                                                    shuffle=False)

    predict_validation_input_fn = lambda: my_input_fn(validation_examples,

                                                      validation_targets[label_col_name],

                                                      num_epochs=1,

                                                      shuffle=False)



    # Train the model, but do so inside a loop so that we can periodically assess

    # loss metrics.

    print("Training model...")

    print("RMSE (on training data):")

    training_rmse = []

    validation_rmse = []

    for period in range(0, periods):

        # Train the model, starting from the prior state.

        dnn_regressor.train(

            input_fn=training_input_fn,

            steps=steps_per_period

        )

        # Take a break and compute predictions.

        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)

        training_predictions = np.array([item['predictions'][0] for item in training_predictions])



        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)

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



    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)

    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)



    return dnn_regressor, training_rmse, validation_rmse

learning_rate = 0.0003

steps=500

batch_size = 10

hisdden_units = [6, 4, 3]


model = train_nn_regression_model(

    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),

    steps=steps,

    batch_size=batch_size,

    hidden_units=hisdden_units,

    training_examples=training_features,

    training_targets=training_labels,

    validation_examples=validation_features,

    validation_targets=validation_labels,

    label_col_name=target_col)
def linear_scale(series):

    min_val = series.min()

    max_val = series.max()

    scale = (max_val - min_val) / 2.0

    return series.apply(lambda x: ((x - min_val) / scale) - 1.0)
def normalize_linear_scale(examples_data_frame):

    for col in examples_data_frame:

        examples_data_frame[col] = linear_scale(examples_data_frame[col])

    return examples_data_frame
training_features = normalize_linear_scale(training_features)

validation_features = normalize_linear_scale(validation_features)

test_features= normalize_linear_scale(test_features)
training_features.head()
validation_features.head()
learning_rate = 0.003

steps=3500

batch_size = 10

hisdden_units = [6]
dnn_regressor, training_rmse, validation_rmse = train_nn_regression_model(

    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),

    steps=steps,

    batch_size=batch_size,

    hidden_units=hisdden_units,

    training_examples=training_features,

    training_targets=training_labels,

    validation_examples=validation_features,

    validation_targets=validation_labels,

    label_col_name=target_col)
prediction = dnn_regressor.predict(input_fn=lambda: my_input_fn(test_features, 

                                                  test_labels[target_col], 

                                                  num_epochs=1, 

                                                  shuffle=False))
prediction = np.array([item['predictions'][0] for item in prediction])
print(math.sqrt(metrics.mean_squared_error(prediction, test_labels)))