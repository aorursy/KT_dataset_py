import math
import numpy as np
import pandas as pd 
import matplotlib
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import seaborn as sns
import os
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
%matplotlib inline
print(os.listdir("../input/"))
df_train = pd.read_csv('../input/train.csv', index_col=0)
df_test = pd.read_csv('../input/test.csv', index_col=0)
df_train.columns
df_train['SalePrice'].describe()
print('train shape:', df_train.shape, '\n', 'test shape:', df_test.shape)
df_train.head(5)
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() #just checking that there's no missing data missing...
print(df_train.shape)
df_train.head(2)
#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
df_train = df_train.reindex(
    np.random.permutation(df_train.index))
def preprocess_features(housing_dataframe):
  selected_features = housing_dataframe[
    ["OverallQual",
     "GrLivArea",
     "GarageCars",
     "TotalBsmtSF",
     "1stFlrSF",
     "FullBath",
     "TotRmsAbvGrd",
     "YearBuilt"
     ]]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  processed_features["luxury_index"] = (
    housing_dataframe["GarageCars"] *
    housing_dataframe["FullBath"])
  return processed_features

def preprocess_targets(housing_dataframe):
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["SalePrice"] = housing_dataframe["SalePrice"]
  return output_targets
from IPython import display

num_validate_data = 200
num_train_data = df_train.shape[0] - num_validate_data
# training.
# training_examples = preprocess_features(df_train)
# training_targets = preprocess_targets(df_train)
X_train = preprocess_features(df_train.head(num_train_data))
Y_train = preprocess_targets(df_train.head(num_train_data)).SalePrice

# validation.
X_validate = preprocess_features(df_train.tail(200))
Y_validate = preprocess_targets(df_train.tail(200)).SalePrice

# testing.
X_test = preprocess_features(df_test)

# Double-check that we've done the right thing.
print("X_train summary:")
display.display(X_train.describe())
# print("Validation examples summary:")
# display.display(validation_examples.describe())
# print("Testing examples summary:")
# display.display(testing_examples.describe())
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
lm = LinearRegression()
lm_model = lm.fit(X_train,Y_train)
# pred = pd.DataFrame(np.exp(lm_model.predict(X_test)))
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
sns.regplot(np.exp(Y_train), np.exp(lm_model.predict(X_train)) )
def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """
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
def train_model(
    learning_rate,
    steps,
    batch_size,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model.
  Returns:
    A `LinearRegressor` object trained on the training data.
  """
  periods = 10
  steps_per_period = steps / periods

  # Create a linear regressor object.
  my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
  )

  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets,#["SalePrice"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets,#["SalePrice"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets,#["SalePrice"], 
                                                    num_epochs=1, 
                                                    shuffle=False)


  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  testing_rmse = []  
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
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
linear_classifier = train_model(
    learning_rate=0.001,
    steps=500,
    batch_size=20,
    feature_columns=construct_feature_columns(X_train),
    training_examples=X_train,
    training_targets=Y_train,
    validation_examples=X_validate,
    validation_targets=Y_validate)
def train_random_forest_model(training_examples, training_targets, validation_examples, validation_targets):
    forest_regressor = RandomForestRegressor()
    forest_regressor.fit(training_examples, np.ravel(training_targets))
    training_predictions = forest_regressor.predict(training_examples)
    validation_predictions = forest_regressor.predict(validation_examples)
    training_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(validation_predictions, validation_targets))
    print("training set RMSE: %0.2f" % (training_root_mean_squared_error))
    print("validation set RMSE: %0.2f" % (validation_root_mean_squared_error))
    return forest_regressor
from sklearn.ensemble import RandomForestRegressor
forest_regressor = train_random_forest_model(X_train, Y_train, 
                                             X_validate, Y_validate)
forest_model = RandomForestRegressor()
forest_model.fit(X_train, Y_train)
#pred = pd.DataFrame(np.exp(forest_model.predict(X_test)))
sns.regplot(np.exp(Y_train), np.exp(forest_model.predict(X_train)) )
def train_model(
    learning_rate,
    steps,
    batch_size,
    hidden_units,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a Tree regression model.
  Returns:
    A `BoostedTreesRegressor` object trained on the training data.
  """
  periods = 10
  steps_per_period = steps / periods

  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  classifer = tf.estimator.DNNRegressor(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units
  )
  
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets,#["SalePrice"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets,#["SalePrice"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets,#["SalePrice"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  testing_rmse = []  
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    classifer.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = classifer.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    validation_predictions = classifer.predict(input_fn=predict_validation_input_fn)
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

  return classifer
_ = train_model(
    learning_rate=0.001,
    steps=500,
    batch_size=20,
    hidden_units=[10, 2],
    feature_columns=construct_feature_columns(X_train),
    training_examples=X_train,
    training_targets=Y_train,
    validation_examples=X_validate,
    validation_targets=Y_validate)
