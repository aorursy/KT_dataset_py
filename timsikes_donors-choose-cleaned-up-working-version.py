# Imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import math
import nltk
from matplotlib import pyplot as plt
from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.data import Dataset
from IPython import display
from sklearn import metrics
# Load the data

# Filepath to main training dataset.
train_file_path = '../input/train.csv'

# Read data and store in DataFrame.
train_data = pd.read_csv(train_file_path, sep=',')
train_data = train_data.reindex(
    np.random.permutation(train_data.index))

#... I just realized there's a whole nother data set to deal with

# Filepath to main training dataset.
resources_file_path = '../input/resources.csv'

# Read data and store in DataFrame.
train_data_resources = pd.read_csv(resources_file_path, sep=',')
# Utility Methods

def getCorpus(text):
    file = open('data.txt', 'w')
    file.write(text)
    file.close()
    corpusReader = nltk.corpus.PlaintextCorpusReader("", ".*\.txt")
    return (len(corpusReader.sents()), len(corpusReader.words()), len([char for sentence in corpusReader.sents() for word in sentence for char in word]))

def get_essay_information(data):
    data["project_essay_1_corpus_lengths"] = data["project_essay_1"].apply(lambda val: getCorpus(val))
    data["project_essay_1_num_sentences"] = data["project_essay_1_corpus_lengths"].apply(lambda val: val[0])
    data["project_essay_1_num_words"] = data["project_essay_1_corpus_lengths"].apply(lambda val: val[1])
    data["project_essay_1_num_characters"] = data["project_essay_1_corpus_lengths"].apply(lambda val: val[2])
    return data
    
def get_resource_information(data, data_resources):
    data = data.merge(data_resources.groupby('id')['price'].agg('sum').reset_index(), left_on='id', right_on='id', how='left')
    data = data.merge(data_resources.groupby('id')['quantity'].agg('sum').reset_index(), left_on='id', right_on='id', how='left')
    data.reset_index()
    return data
    
def get_binary_encoded_categories(data):
    data["project_subject_categories_split"] = data["project_subject_categories"].apply(lambda val: [x.strip() for x in val.split(',')])
    data["project_subject_subcategories_split"] = data["project_subject_subcategories"].apply(lambda val: [x.strip() for x in val.split(',')])
    
    print (data.columns)

    mlb = MultiLabelBinarizer()
    mlb2 = MultiLabelBinarizer()

    data = data.join(pd.DataFrame(mlb.fit_transform(data.pop('project_subject_categories_split')),
                          columns=mlb.classes_,
                          index=data.index).add_prefix("cat_"))
    
    print (data.columns)
    
    data = data.join(pd.DataFrame(mlb2.fit_transform(data.pop('project_subject_subcategories_split')),
                          columns=mlb2.classes_,
                          index=data.index).add_prefix("sub_"))
    
    return data

def transform_data_set(data, data_resources):
    data = get_essay_information(data)
    data = get_resource_information(data, data_resources)
    data = get_binary_encoded_categories(data)
    data.reset_index()
    return data

def get_feature_columns(data):
    #start to build feature columns
    sub_cat_filter_col = [col for col in data if col.startswith('sub_')]
    cat_filter_col = [col for col in data if col.startswith('cat_')]

    sub_cat_cross_features = tf.feature_column.crossed_column(
      set(sub_cat_filter_col), hash_bucket_size=1000)

    cat_and_sub_cat_cross_features = tf.feature_column.crossed_column(
      set(sub_cat_filter_col + cat_filter_col), hash_bucket_size=1000)

    essay_1_complexity_cross_features = tf.feature_column.crossed_column(
      set(["project_essay_1_num_sentences", "project_essay_1_num_words", "project_essay_1_num_characters"]), hash_bucket_size=1000)

    project_essay_1_num_sentences = tf.feature_column.numeric_column("project_essay_1_num_sentences")
    project_essay_1_num_words = tf.feature_column.numeric_column("project_essay_1_num_words")
    project_essay_1_num_characters = tf.feature_column.numeric_column("project_essay_1_num_characters")
    price = tf.feature_column.numeric_column("price")
    quantity = tf.feature_column.numeric_column("quantity")

    return set([
        #sub_cat_cross_features, 
        #cat_and_sub_cat_cross_features,
        project_essay_1_num_sentences,
        project_essay_1_num_words,
        project_essay_1_num_characters,
        price,
        quantity])
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
    if (targets is None):
        ds = Dataset.from_tensor_slices(features) # warning: 2GB limit
    else:
        ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    if (targets is None):
        features = ds.make_one_shot_iterator().get_next()
        return features
    else:
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
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    feature_columns: A `set` specifying the input feature columns to use.
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
  my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )
  
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["project_is_approved"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["project_is_approved"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["project_is_approved"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print ("Training model...")
  print ("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
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
    print ("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print ("Model training finished.")

  
  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  return linear_regressor
def train_classifier_2_model(
    learning_rate,
    steps,
    batch_size,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    feature_columns: A `set` specifying the input feature columns to use.
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
  my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  classifier = tf.estimator.LinearClassifier(
      feature_columns=feature_columns,
      n_classes=2,
      optimizer=my_optimizer,
      config=tf.estimator.RunConfig(keep_checkpoint_max=1)
  )
  
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["project_is_approved"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["project_is_approved"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["project_is_approved"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print ("Training model...")
  print ("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
    training_probabilities = np.array([item['probabilities'] for item in training_predictions])
    training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
    training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id,2)
        
    validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
    validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])    
    validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
    validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id,2)    
    
    # Compute training and validation loss.
    training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
    validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
    # Occasionally print the current loss.
    print ("  period %02d : %0.2f" % (period, validation_log_loss))
    
  print ("Model training finished.")

  
  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  return classifier
def train_linear_classification_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear classification model for the MNIST digits dataset.
  
  In addition to training, this function also prints training progress information,
  a plot of the training and validation loss over time, and a confusion
  matrix.
  
  Args:
    learning_rate: An `int`, the learning rate to use.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing the training features.
    training_targets: A `DataFrame` containing the training labels.
    validation_examples: A `DataFrame` containing the validation features.
    validation_targets: A `DataFrame` containing the validation labels.
      
  Returns:
    The trained `LinearClassifier` object.
  """

  periods = 10

  steps_per_period = steps / periods  
  # Create the input functions.
  predict_training_input_fn = lambda: my_input_fn(
    training_examples, training_targets, batch_size)
  predict_validation_input_fn = lambda: my_input_fn(
    validation_examples, validation_targets, batch_size)
  training_input_fn = lambda: my_input_fn(
    training_examples, training_targets, batch_size)
  
  # Create a LinearClassifier object.
  my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  classifier = tf.estimator.LinearClassifier(
      feature_columns=get_feature_columns(train_data),
      n_classes=2,
      optimizer=my_optimizer,
      config=tf.estimator.RunConfig(keep_checkpoint_max=1)
  )

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print ("Training model...")
  print ("LogLoss error (on validation data):")
  training_errors = []
  validation_errors = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
  
    # Take a break and compute probabilities.
    training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
    training_probabilities = np.array([item['probabilities'] for item in training_predictions])
    training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
    training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id,10)
        
    validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
    validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])    
    validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
    validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id,10)    
    
    # Compute training and validation errors.
    training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
    validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
    # Occasionally print the current loss.
    print ("  period %02d : %0.2f" % (period, validation_log_loss))
    # Add the loss metrics from this period to our list.
    training_errors.append(training_log_loss)
    validation_errors.append(validation_log_loss)
  print ("Model training finished.")
  # Remove event files to save disk space.
  _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))
  
  # Calculate final predictions (not probabilities, as above).
  final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
  final_predictions = np.array([item['class_ids'][0] for item in final_predictions])
  
  
  accuracy = metrics.accuracy_score(validation_targets, final_predictions)
  print ("Final accuracy (on validation data): %0.2f" % accuracy)

  # Output a graph of loss metrics over periods.
  plt.ylabel("LogLoss")
  plt.xlabel("Periods")
  plt.title("LogLoss vs. Periods")
  plt.plot(training_errors, label="training")
  plt.plot(validation_errors, label="validation")
  plt.legend()
  plt.show()
  
  # Output a plot of the confusion matrix.
  cm = metrics.confusion_matrix(validation_targets, final_predictions)
  # Normalize the confusion matrix by row (i.e by the number of samples
  # in each class).
  cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  ax = sns.heatmap(cm_normalized, cmap="bone_r")
  ax.set_aspect(1)
  plt.title("Confusion matrix")
  plt.ylabel("True label")
  plt.xlabel("Predicted label")
  plt.show()

  return classifier

def create_training_input_fn(features, labels, batch_size, num_epochs=None, shuffle=True):
  """A custom input_fn for sending MNIST data to the estimator for training.

  Args:
    features: The training features.
    labels: The training labels.
    batch_size: Batch size to use during training.

  Returns:
    A function that returns batches of training features and labels during
    training.
  """
  def _input_fn(num_epochs=None, shuffle=True):
    # Input pipelines are reset with each call to .train(). To ensure model
    # gets a good sampling of data, even when number of steps is small, we 
    # shuffle all the data before creating the Dataset object
    idx = np.random.permutation(features.index)
    raw_features = {"pixels":features.reindex(idx)}
    raw_targets = np.array(labels[idx])
   
    ds = Dataset.from_tensor_slices((raw_features,raw_targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
    return feature_batch, label_batch

  return _input_fn

def create_predict_input_fn(features, labels, batch_size):
  """A custom input_fn for sending mnist data to the estimator for predictions.

  Args:
    features: The features to base predictions on.
    labels: The labels of the prediction examples.

  Returns:
    A function that returns features and labels for predictions.
  """
  def _input_fn():
    raw_features = {"pixels": features.values}
    raw_targets = np.array(labels)
    
    ds = Dataset.from_tensor_slices((raw_features, raw_targets)) # warning: 2GB limit
    ds = ds.batch(batch_size)
    
        
    # Return the next batch of data.
    feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
    return feature_batch, label_batch

  return _input_fn
train_data = transform_data_set(train_data, train_data_resources)
# Set Training and validation

#feature_columns = get_feature_columns(train_data)
#train_data.columns
#train_data[list(feature_columns)].describe()
feature_columns = [
    "project_essay_1_num_sentences",
    "project_essay_1_num_words",
    "project_essay_1_num_characters",
    "price",
    "quantity"]
# Choose the first 12000 (out of 17000) examples for training.
training_examples = train_data[list(feature_columns)].head(12000)
training_targets = pd.DataFrame(train_data["project_is_approved"].head(12000))

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = train_data[list(feature_columns)].tail(5000)
validation_targets = pd.DataFrame(train_data["project_is_approved"].tail(5000))

# Double-check that we've done the right thing.
print ("Training examples summary:")
display.display(training_examples.describe())
print ("Validation examples summary:")
display.display(validation_examples.describe())

print ("Training targets summary:")
display.display(training_targets.describe())
print ("Validation targets summary:")
display.display(validation_targets.describe())

model = train_model(
    learning_rate=.5,
    steps=500,
    batch_size=100,
    feature_columns=get_feature_columns(train_data),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
classification_model = train_classifier_2_model(learning_rate=.5,
    steps=500,
    batch_size=100,
    feature_columns=get_feature_columns(train_data),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
test_file_path = '../input/test.csv'

# Read data and store in DataFrame.
test_data = pd.read_csv(test_file_path, sep=',')

test_data = transform_data_set(test_data, train_data_resources)
test_data.describe()
print("Is this working?")
#test_data.columns
#test_data[list(feature_columns)].describe()
test_input_fn = lambda: my_input_fn(test_data[list(feature_columns)], 
                                                    None, 
                                                    num_epochs=1, 
                                                    shuffle=False)

test_predictions = model.predict(input_fn=test_input_fn)

#print(test_predictions)
#for item in test_predictions:
#    print(item)

test_predictions2 = np.array([item['predictions'][0] for item in test_predictions])
test_predictions2
train_data.describe()
print("literally antying")
test_input_fn = lambda: my_input_fn(test_data[list(feature_columns)], 
                                                    None, 
                                                    num_epochs=1, 
                                                    shuffle=False)

test_classifier_predictions = classification_model.predict(input_fn=test_input_fn)

#print(test_classifier_predictions)
#for item in test_classifier_predictions:
#    print(item)

#test_classifier_predictions.head(10)
test_classifier_predictions2 = [item['probabilities'][0] for item in test_classifier_predictions]
test_classifier_predictions2.head(100)
#test_classifier_predictions2 = np.array(index, [item['probabilities'][0] for index, item in test_classifier_predictions])
#np.array([item['predictions'][0] for item in test_classifier_predictions])
#test_classifier_predictions2.head(10)
#test_classifier_predictions2[:10]

test_data.iloc[[0]]["id"]

predictions = [("id","project_is_approved")]
for index, item in enumerate(test_classifier_predictions2):
    #print(item)
    #print("thing")
    #print(test_data.iloc[[index]]["id"])
    #print(test_data.loc[0].iat[0])
    #test_data.iloc[[index]]["id"]
    predictions.append((test_data.loc[index].iat[0], item))
    
predictions[:10]

import csv
with open('submission.csv','w') as out:
    #csv_out=csv.writer(out)
    for row in predictions:
        out.write(f'{row[0]},{row[1]}\n')