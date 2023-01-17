# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import tensorflow as tf 
import pandas as pd
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_dataset = pd.read_csv("../input/fashion-mnist_train.csv")
test_dataset = pd.read_csv("../input/fashion-mnist_test.csv")
train_dataset.head()
train_dataset.describe()
# hyper parameters
percentage_of_validation_data = 20
batch_size = 100
steps = 10000
percentage_of_validation_data = 20
learning_rate = 0.0009
l1_regularization = 0.003
total_rows = train_dataset.shape[0]
training_rows = (int) ((total_rows * (100 - percentage_of_validation_data))/ 100)
validation_rows = total_rows - training_rows
#data seperation
feature_cols = train_dataset.columns.drop('label');
label_cols = ['label']

validation_dataset = train_dataset.tail(validation_rows)
train_dataset = train_dataset.head(training_rows)
my_feature_columns = []
for key in feature_cols:
    my_feature_columns.append(tf.feature_column.numeric_column(key=key, dtype=tf.int32))
# features : dataframe, labrls : dataframe
def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset

def valid_input_fn(features, labels):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset
model = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, 
                                       hidden_units=[512, 248,124 ,40], 
                                       n_classes=10,
                                       optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate, 
                                                                                   l1_regularization_strength=l1_regularization))
TrainResult = model.train(input_fn=lambda:train_input_fn(train_dataset[feature_cols], train_dataset[label_cols],batch_size),steps=steps)
model.evaluate(input_fn=lambda:train_input_fn(validation_dataset[feature_cols], validation_dataset[label_cols],batch_size),
               steps=1)