# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')
print(df.head(10)) 
print(len(df))
# print(df.describe())
print(df.columns)
print(test_df.head(10)) 
print(len(test_df))
# print(df.describe())
print(test_df.columns)
df['Cover_Type'].describe()
pd.value_counts(df['Cover_Type'])
df.describe()
import matplotlib.pyplot as plt
df.drop(axis=1, columns=['Soil_Type7','Soil_Type15'], inplace=True)
# Convert the Wilderness Area one hot encoded to single column
columns = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4']
wilderness_types = []
for index, row in df.iterrows():
    dummy = 'Wilderness_Area_NA'
    for col in columns:
        if row[col] == 1:
            dummy = col
            break
    wilderness_types.append(dummy)
df['Wilderness_Areas'] = wilderness_types
# Convert the Soil Type one hot encoded to single column
columns = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
soil_types = []
for index, row in df.iterrows():
    dummy = 'Soil_Type_NA'
    for col in columns:
        if row[col] == 1:
            dummy = col
            break
    soil_types.append(dummy)
df['Soil_Types'] = soil_types


print(pd.value_counts(df['Soil_Types']))
ax = df['Soil_Types'].value_counts().plot(kind='bar',
                                    figsize=(8,5),
                                    title="Number for each Soli Type")
ax.set_xlabel("Soil Types")
ax.set_ylabel("Frequency")
plt.show()

print(pd.value_counts(df['Wilderness_Areas']))
ax1 = df['Wilderness_Areas'].value_counts().plot(kind='bar',
                                    figsize=(8,5),
                                    title="Number for each Soli Type")
ax1.set_xlabel("Wilderness_Areas")
ax1.set_ylabel("Frequency")
plt.show()
df.drop(axis=1, columns=['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4'], inplace=True)
df.drop(axis=1, columns=['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6',  'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'], inplace=True)
df['Soil_Types'].replace(to_replace={'Soil_Type8': 'Soil_Type_NA', 'Soil_Type25': 'Soil_Type_NA'}, inplace=True)
print(pd.value_counts(df['Soil_Types']))
## Apply to test_df
test_df.drop(axis=1, columns=['Soil_Type7','Soil_Type15'], inplace=True)
# Convert the Wilderness Area one hot encoded to single column
columns = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4']
wilderness_types = []
for index, row in test_df.iterrows():
    dummy = 'Wilderness_Area_NA'
    for col in columns:
        if row[col] == 1:
            dummy = col
            break
    wilderness_types.append(dummy)
test_df['Wilderness_Areas'] = wilderness_types
# Convert the Soil Type one hot encoded to single column
columns = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6',  'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
soil_types = []
for index, row in test_df.iterrows():
    dummy = 'Soil_Type_NA'
    for col in columns:
        if row[col] == 1:
            dummy = col
            break
    soil_types.append(dummy)
test_df['Soil_Types'] = soil_types


print(pd.value_counts(test_df['Soil_Types']))

print(pd.value_counts(test_df['Wilderness_Areas']))

test_df.drop(axis=1, columns=['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4'], inplace=True)
test_df.drop(axis=1, columns=['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6',  'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'], inplace=True)

test_df['Soil_Types'].replace(to_replace={'Soil_Type8': 'Soil_Type_NA', 'Soil_Type25': 'Soil_Type_NA', 'Soil_Type7': 'Soil_Type_NA', 'Soil_Type15': 'Soil_Type_NA'}, inplace=True)
print(pd.value_counts(test_df['Soil_Types']))
from sklearn.model_selection import train_test_split
# Split the data into train, validation and test
train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')
print("Train dataset :")
print(pd.value_counts(train['Cover_Type']))
print("Val dataset :")
print(pd.value_counts(val['Cover_Type']))
print("Test dataset :")
print(pd.value_counts(test['Cover_Type']))
# Normalize the feature columns
from sklearn import preprocessing
import matplotlib.pyplot as plt


# Remove Id column as well
train.drop(axis=1, columns=['Id'], inplace=True)
val.drop(axis=1, columns=['Id'], inplace=True)
test.drop(axis=1, columns=['Id'], inplace=True)

# Create a min max processor object
min_max_scaler = preprocessing.MinMaxScaler()
# Use the fit transform function on processor object
feature_columns = ['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
for column in feature_columns:
    print("Transforming column {}:".format(column))
    train[column] = min_max_scaler.fit_transform(train[[column]].values.astype(float))
    test[column] = min_max_scaler.fit_transform(test[[column]].values.astype(float))
    val[column] = min_max_scaler.fit_transform(val[[column]].values.astype(float))
    print(train[column].describe())
import tensorflow as tf
print('TF: {}'.format(tf.__version__))

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Cover_Type')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 30
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)



for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of Slope:', feature_batch['Slope'])
  print('A batch of targets:', label_batch )

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import feature_column

feature_cols_for_training = []

# numeric cols
for header in ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']:
  feature_cols_for_training.append(feature_column.numeric_column(header))

# indicator cols
wilderness = feature_column.categorical_column_with_vocabulary_list(
      'Wilderness_Areas', ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4'])
wilderness_one_hot = feature_column.indicator_column(wilderness)
feature_cols_for_training.append(wilderness_one_hot)

soil_types = feature_column.categorical_column_with_vocabulary_list(
      'Soil_Types', ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Soil_Type_NA'])
soil_types_one_hot = feature_column.indicator_column(soil_types)
feature_cols_for_training.append(soil_types_one_hot)

# Dense layer as input to the model
feature_layer = tf.keras.layers.DenseFeatures(feature_cols_for_training)

def build_model():
  model = keras.Sequential([
    feature_layer,
    layers.Dense(100, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(8)
  ])

  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer='adam',
                metrics=['accuracy'])
  return model

model = build_model()

model.fit(train_ds,
          validation_data=val_ds,
          epochs=100)
model.summary()
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
import numpy as np
predictions = np.argmax(model.predict(test_ds), axis=-1)
print(predictions)
original_labels = tf.Variable(test['Cover_Type'], tf.int32)
predictions =  tf.Variable(predictions, tf.int32)
print(type(predictions))
tf.math.confusion_matrix(original_labels, predictions, num_classes=8)
from sklearn.metrics import classification_report
import time
test_features = []
start_time = time.time()
test_predictions = np.argmax(model.predict(test_ds), axis=-1)
# Comparing the predictions to actual actual forest cover types for the sentences
print(classification_report(test['Cover_Type'],test_predictions))
print("Time taken to predict the model " + str(time.time() - start_time))
# Extract the Id column because we need it
ids = test_df['Id']
test_df.drop(axis=1, columns=['Id'], inplace=True)

# Normalize the values for few columns
feature_columns = ['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
for column in feature_columns:
    print("Transforming column {}:".format(column))
    test_df[column] = min_max_scaler.fit_transform(test_df[[column]].values.astype(float))
    
def df_to_dataset_test(dataframe, shuffle=True, batch_size=32):
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe)))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds
test_ds = df_to_dataset_test(test_df, shuffle=False, batch_size=batch_size)
final_predictions = np.argmax(model.predict(test_ds), axis=-1)
print(len(final_predictions))
print(len(ids))
final_df = pd.DataFrame()
final_df = final_df.from_dict({'Id': ids, 'Cover_Type': final_predictions})
print(final_df.head())
pd.value_counts(final_df['Cover_Type'])
final_df.to_csv('tensorflow_ffnn_output.csv', index=False)