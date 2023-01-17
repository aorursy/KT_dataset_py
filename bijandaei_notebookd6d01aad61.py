#%tensorflow_version 2.x
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
# tf.keras.backend.set_floatx('float32')

train_df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index))
train_df = train_df.drop(columns=['id', 'Unnamed: 32'])

diagnosis = train_df['diagnosis']
train_df = train_df.drop(columns=['diagnosis'])
train_df_mean = train_df.mean()
train_df_std = train_df.std()
train_df_norm = (train_df - train_df_mean)/train_df_std

train_df_norm['diagnosis'] = (diagnosis == "M").astype(float) 
train_df_norm['concave_points_mean'] = train_df_norm['concave points_mean']
train_df_norm['concave_points_worst'] = train_df_norm['concave points_worst']
train_df_norm['concave_points_se'] = train_df_norm['concave points_se']
train_df_norm = train_df_norm.drop(columns=['concave points_mean', 'concave points_worst', 'concave points_se'])

train_df_norm.head()
def create_model(my_learning_rate, feature_layer, my_metrics):
  """Create and compile a simple classification model."""
  # Most simple tf.keras models are sequential.
  model = tf.keras.models.Sequential()

  # Add the feature layer (the list of features and how they are represented)
  # to the model.
  model.add(feature_layer)

  # Funnel the regression value through a sigmoid function.
  model.add(tf.keras.layers.Dense(units=1, input_shape=(1,),
                                  activation=tf.sigmoid),)

  # Call the compile method to construct the layers into a model that
  # TensorFlow can execute.  Notice that we're using a different loss
  # function for classification than for regression.    
  model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),                                                   
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=my_metrics)

  return model        


def train_model(model, dataset, epochs, label_name,
                batch_size=None, shuffle=True):
  """Feed a dataset into the model in order to train it."""

  # The x parameter of tf.keras.Model.fit can be a list of arrays, where
  # each array contains the data for one feature.  Here, we're passing
  # every column in the dataset. Note that the feature_layer will filter
  # away most of those columns, leaving only the desired columns and their
  # representations as features.
  features = {name:np.array(value) for name, value in dataset.items()}
  label = np.array(features.pop(label_name)) 
  history = model.fit(x=features, y=label, batch_size=batch_size,
                      epochs=epochs, shuffle=shuffle)
  
  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch

  # Isolate the classification metric for each epoch.
  hist = pd.DataFrame(history.history)

  return epochs, hist  

def plot_curve(epochs, hist, list_of_metrics):
  """Plot a curve of one or more classification metrics vs. epoch."""  
  # list_of_metrics should be one of the names shown in:
  # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics  

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()
feature_columns = []
feature_columns.append(tf.feature_column.numeric_column("radius_mean"))
feature_columns.append(tf.feature_column.numeric_column("texture_mean"))
feature_columns.append(tf.feature_column.numeric_column("perimeter_mean"))
feature_columns.append(tf.feature_column.numeric_column("area_mean"))
feature_columns.append(tf.feature_column.numeric_column("smoothness_mean"))
feature_columns.append(tf.feature_column.numeric_column("compactness_mean"))
feature_columns.append(tf.feature_column.numeric_column("concavity_mean"))
feature_columns.append(tf.feature_column.numeric_column("concave_points_mean"))
feature_columns.append(tf.feature_column.numeric_column("symmetry_mean"))
feature_columns.append(tf.feature_column.numeric_column("fractal_dimension_mean"))

# You can also add all of the available columns
#for col in train_df_norm.columns:
#    if col != 'diagnosis':
#        feature_columns.append(tf.feature_column.numeric_column(col))

feature_layer = layers.DenseFeatures(feature_columns)
feature_layer(dict(train_df_norm))


learning_rate = 0.001
epochs = 100
batch_size = 10
classification_threshold = 0.35
label_name = "diagnosis"

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=classification_threshold),
      tf.keras.metrics.Precision(thresholds=classification_threshold, name='precision'),
      tf.keras.metrics.Recall(thresholds=classification_threshold, name="recall")
      ]

my_model = create_model(learning_rate, feature_layer, METRICS)

epochs, hist = train_model(my_model, train_df_norm, epochs, label_name, batch_size)

list_of_metrics_to_plot = ['accuracy', "precision", "recall"] 
plot_curve(epochs, hist, list_of_metrics_to_plot)