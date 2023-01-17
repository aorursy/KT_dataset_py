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
dataframe = pd.read_csv("../input/Mall_Customers_Data.csv", sep=",")

dataframe = dataframe.reindex(np.random.permutation(dataframe.index))

dataframe.head()
dataframe.describe()
#Get general info about data

print ("data shape :",dataframe.shape,"\n")

print ("data info  :",dataframe.info())

print ("\ncolumns  :",dataframe.columns)

print ("\nmissing values :",dataframe.isnull().sum())
import seaborn as sns

import warnings

import itertools

warnings.filterwarnings("ignore")

%matplotlib inline

correlation = dataframe[["Annual Income (k$)", "Age", 'Spending Score (1-100)']].corr()

plt.figure(figsize=(9,7))

sns.heatmap(correlation,annot=True,cmap="coolwarm",linewidth=2,edgecolor="k")

plt.title("CORRELATION BETWEEN VARIABLES")
import seaborn as sns

sns.set(style="ticks")

sns.pairplot(dataframe, vars=["Annual Income (k$)", "Age", 'Spending Score (1-100)'],hue="Gender", markers=["o", "s"])
plt.scatter(dataframe['Age'],dataframe['Annual Income (k$)'], cmap="coolwarm", c=dataframe['Spending Score (1-100)'])

plt.ylabel("Annual Income (k$)")

plt.xlabel("Age")

plt.title("Spending Score (1-100)")

plt.show()



dataframe_female = dataframe[dataframe["Gender"]=="Female"]

plt.scatter(dataframe_female['Age'],dataframe_female['Annual Income (k$)'], cmap="coolwarm", c=dataframe_female['Spending Score (1-100)'])

plt.ylabel("Annual Income (k$)")

plt.xlabel("Age")

plt.title("Female Spending Score (1-100)")

plt.show()



dataframe_male = dataframe[dataframe["Gender"]=="Male"]

plt.scatter(dataframe_male['Age'],dataframe_male['Annual Income (k$)'], cmap="coolwarm", c=dataframe_male['Spending Score (1-100)'])

plt.ylabel("Annual Income (k$)")

plt.xlabel("Age")

plt.title("Male Spending Score (1-100)")

plt.show()
def preprocess_features(dataframe):

  selected_features = dataframe[

    ["Gender",          

     "Age",           

     "Annual Income (k$)"      

    ]]

  processed_features = selected_features

  return processed_features



def preprocess_targets(dataframe):

  output_targets = pd.DataFrame()

  output_targets["Spending Score (1-100)"] = dataframe["Spending Score (1-100)"]/100

  return output_targets



def linear_scale(series):

  min_val = series.min()

  max_val = series.max()

  scale = (max_val - min_val) / 2.0

  return series.apply(lambda x:((x - min_val) / scale) - 1.0)



def normalize(dataframe):

  processed_features = pd.DataFrame()

  processed_features["Gender"] = dataframe["Gender"]

  processed_features["Age"] = linear_scale(dataframe["Age"])

  processed_features["AnnualIncome"] = linear_scale(dataframe["Annual Income (k$)"])

  return processed_features



def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):    

    features = {key:np.array(value) for key,value in dict(features).items()}                                           

    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit

    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:

      ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels
total_records = dataframe.shape[0]

training_percentage = 0.7

validation_percentage = 1 - training_percentage



normalized_preprocessed_features = normalize(preprocess_features(dataframe))

preprocessed_targets = preprocess_targets(dataframe)

# Choose the first 80% examples for training.

training_examples = normalized_preprocessed_features.head(int(total_records*training_percentage))

training_targets = preprocessed_targets.head(int(total_records*training_percentage))

# Choose the last 20% examples for validation.

validation_examples = normalized_preprocessed_features.tail(int(total_records*validation_percentage))

validation_targets = preprocessed_targets.tail(int(total_records*validation_percentage))

feature_columns = [tf.feature_column.numeric_column("Age"),

             tf.feature_column.numeric_column("AnnualIncome"),

             tf.feature_column.indicator_column(

                   tf.feature_column.categorical_column_with_vocabulary_list("Gender",["Male","Female"]))

                  ]
def train_model(

    learning_rate,

    steps,

    batch_size,

    training_examples,

    training_targets,

    validation_examples,

    validation_targets):



  periods = 10

  steps_per_period = steps / periods



  # Create a linear regressor object.

  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

  linear_regressor = tf.estimator.LinearRegressor(

      feature_columns=feature_columns,

      optimizer=my_optimizer

  )

    

  # Create input functions

  training_input_fn = lambda: my_input_fn(training_examples, 

                                          training_targets, 

                                          batch_size=batch_size)

  predict_training_input_fn = lambda: my_input_fn(training_examples, 

                                                  training_targets, 

                                                  num_epochs=1, 

                                                  shuffle=False)

  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 

                                                    validation_targets, 

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

    learning_rate=0.001,

    steps=100,

    batch_size=100,

    training_examples=training_examples,

    training_targets=training_targets,

    validation_examples=validation_examples,

    validation_targets=validation_targets)
predict_training_input_fn = lambda: my_input_fn(training_examples, 

                                                  training_targets, 

                                                  num_epochs=1, 

                                                  shuffle=False)

training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)

training_predictions = np.array([item['predictions'][0] for item in training_predictions]) 

training_predictions = pd.DataFrame(training_predictions)



x2 = np.linspace(0.0, 1.0)

plt.scatter(training_targets,training_predictions)

plt.plot(x2, x2)

plt.show()
Age = tf.feature_column.numeric_column("Age")

AnnualIncome = tf.feature_column.numeric_column("AnnualIncome")

Gender = tf.feature_column.categorical_column_with_vocabulary_list("Gender",["Male","Female"])

Age_buckets = tf.feature_column.bucketized_column(Age, np.arange(-1.0,1.0,0.05).tolist())

AnnualIncome_buckets = tf.feature_column.bucketized_column(AnnualIncome, np.arange(-1.0,1.0,0.05).tolist())

feature_columns = [Age_buckets, AnnualIncome_buckets, Gender]
linear_regressor = train_model(

    learning_rate=0.01,

    steps=100,

    batch_size=100,

    training_examples=training_examples,

    training_targets=training_targets,

    validation_examples=validation_examples,

    validation_targets=validation_targets)
predict_training_input_fn = lambda: my_input_fn(training_examples, 

                                                  training_targets, 

                                                  num_epochs=1, 

                                                  shuffle=False)

training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)

training_predictions = np.array([item['predictions'][0] for item in training_predictions]) 

training_predictions = pd.DataFrame(training_predictions)



x2 = np.linspace(0.0, 1.0)

plt.scatter(training_targets,training_predictions)

plt.plot(x2, x2)

plt.show()
predict_validation_input_fn = lambda: my_input_fn(validation_examples, 

                                                  validation_targets, 

                                                  num_epochs=1, 

                                                  shuffle=False)



validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)

validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

validation_predictions = pd.DataFrame(validation_predictions)



x1 = np.linspace(0.0, 1.0)

plt.scatter(validation_targets,validation_predictions)

plt.plot(x1, x1)

plt.show()