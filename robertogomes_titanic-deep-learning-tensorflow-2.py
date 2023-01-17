# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import random
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

device_name = tf.test.gpu_device_name()
print(f"Found GPU at:{device_name}")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataframe_train = pd.read_csv('/kaggle/input/titanic/train.csv')
dataframe_test = pd.read_csv('/kaggle/input/titanic/test.csv')

def clean_process_dataset(dataset):
    freq_port = dataset['Embarked'].dropna().mode()[0]
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset = dataset.drop(['Cabin','Ticket'], axis=1)
    #dataframe = dataframe.drop(['Cabin','Ticket'], axis=1)
    print(dataset[-100:])
    return dataset

dataframe_train = clean_process_dataset(dataframe_train)
dataframe_test = clean_process_dataset(dataframe_test)
# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32, has_label=True):
  dataframe = dataframe.copy()
  if has_label:
      labels = dataframe.pop('Survived')
      ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  else:
      ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

age = feature_column.numeric_column("Age")
isalone = feature_column.numeric_column("IsAlone")
fare = feature_column.numeric_column("Fare")

age_buckets = feature_column.bucketized_column(age, boundaries=[16, 32, 48, 64, 200])
fare_buckets = feature_column.bucketized_column(fare, boundaries=[7, 14, 31, 512, 2000])

sex = feature_column.categorical_column_with_vocabulary_list(
      'Sex', ['male', 'female'])
sex_one_hot = feature_column.indicator_column(sex)
title = feature_column.categorical_column_with_vocabulary_list(
      'Title', ['Master', 'Miss', 'Mr', 'Mrs', 'Rare'])
title_one_hot = feature_column.indicator_column(title)

embarked = feature_column.categorical_column_with_vocabulary_list(
      'Embarked', ['C', 'S', 'Q'])
embarked_one_hot = feature_column.indicator_column(embarked)
pclass = feature_column.categorical_column_with_identity(
      'Pclass', 3)
pclass_one_hot = feature_column.indicator_column(pclass)
crossed_feature_1 = feature_column.crossed_column([sex, pclass], hash_bucket_size=6)
crossed_feature_1_one_hot = feature_column.indicator_column(crossed_feature_1)

crossed_feature_2 = feature_column.crossed_column([sex, title], hash_bucket_size=10)
crossed_feature_2_one_hot = feature_column.indicator_column(crossed_feature_2)
def optimize_model(config, split_test=True):
    dropout_rate = config['dropout_rate']
    cells_total = config['cells_total']
    adam = tf.keras.optimizers.Adam(config['learning_rate'])
    
    if split_test:
        train, test = train_test_split(dataframe_train, test_size=0.2)
        train, val = train_test_split(train, test_size=0.2)
    else:
        train, val = train_test_split(dataframe_train, test_size=0.2)
    feature_columns = []

# numeric cols
    for header in ['Fare']:
      feature_columns.append(feature_column.numeric_column(header))

    feature_columns.append(isalone)
    feature_columns.append(age_buckets)
    feature_columns.append(fare_buckets)
    feature_columns.append(sex_one_hot)
    feature_columns.append(embarked_one_hot)
    feature_columns.append(pclass_one_hot)
    feature_columns.append(title_one_hot)
    feature_columns.append(crossed_feature_1_one_hot)
    feature_columns.append(crossed_feature_2_one_hot)
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    batch_size = 64
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    if split_test:
        test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    model = tf.keras.Sequential([
      feature_layer,
      layers.Dropout(dropout_rate),
      layers.Dense(cells_total, activation='relu'),
      layers.Dropout(dropout_rate),
      layers.Dense(cells_total, activation='relu'),
      layers.Dropout(dropout_rate),
      layers.Dense(cells_total, activation='relu'),
      layers.Dropout(dropout_rate),
      layers.Dense(1, activation='sigmoid')
    ])

    earlystop_callback = EarlyStopping(
      monitor='val_accuracy',
         restore_best_weights=True,
      patience=200)

    model.compile(optimizer=adam,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_ds,
              validation_data=val_ds,
              verbose=False,
              callbacks=[earlystop_callback],
              epochs=1000)
    if split_test:
        loss, accuracy = model.evaluate(test_ds)
    else:
        loss, accuracy = model.evaluate(val_ds)
    best_loss_val, best_accuracy_val = model.evaluate(val_ds)
    print("Accuracy test", accuracy)
    print("Accuracy validation best", best_accuracy_val)
    return model, accuracy
range_dropout = [0.1, 0.15, 0.2]
range_cells = [16, 24, 32, 48, 64]
range_learning_rate = [0.001, 0.0005, 0.0008]
total_optimize_iterations = 20
total_run_per_config = 5
configs_maps = {}
for iteration_optimize in range(total_optimize_iterations):
    dropout_value = random.choice(range_dropout)
    cell_value = random.choice(range_cells)
    learning_rate = random.choice(range_learning_rate)
    a_config = {"dropout_rate": dropout_value, "cells_total": cell_value, 'learning_rate': learning_rate}
    print(f"Using config:{a_config} iteration_optimize:{iteration_optimize}")
    accuracies = [x[1] for x in [optimize_model(a_config) for _ in range(total_run_per_config)]]
    mean_accuracy = sum(accuracies) / total_run_per_config
    print(f"Result config:{a_config} accuracy:{mean_accuracy}")
    configs_maps[tuple(a_config.values())] = mean_accuracy
submit_ds = df_to_dataset(dataframe_test, shuffle=False, batch_size=32, has_label=False)

list_config = list(configs_maps.items())
list_config.sort(key=lambda x:x[1])
print(f"10 best config:{list_config[-10:]}")
best_model = list_config[-1][0]
print(f"best model:{best_model}")
best_config = {"dropout_rate": best_model[0], "cells_total": best_model[1], 'learning_rate': best_model[2]}
model, accuracy = optimize_model(a_config)
predictions = model.predict(submit_ds)
predictions_binary = [1 if x > 0.50 else 0 for x in predictions]
print(predictions_binary)

submission = pd.DataFrame({
        "PassengerId": dataframe_test["PassengerId"],
        "Survived": predictions_binary
    })

submission.to_csv("/kaggle/working/submission.csv", index=False)

