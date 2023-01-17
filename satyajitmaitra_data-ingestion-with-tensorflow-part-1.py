#modules
import functools
import numpy as np
import tensorflow as tf
import IPython.display as display
from PIL import Image
import matplotlib.pyplot as plt
import os

#checking installed tensorflow versions
print("Current tensorflow version {}".format(tf.version.VERSION))
TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
!head{train_file_path}
LABEL_COLUMN = 'survived'
LABELS = [0, 1]
def get_dataset(file_path, **kwargs):
    
    dataset = tf.data.experimental.make_csv_dataset(
                                                   file_path,
                                                   batch_size=5,
                                                   label_name=LABEL_COLUMN,
                                                   na_value="?",
                                                   ignore_errors=True,
                                                   **kwargs)
    return dataset
raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)
def show_dataset(dataset):
    for batch,label in dataset.take(1):
        for key,value in batch.items():
            print("{}:{}".format(key,value.numpy()))
show_dataset(raw_train_data)
def get_dataset32(file_path, **kwargs):
    
    dataset = tf.data.experimental.make_csv_dataset(
                                                   file_path,
                                                   batch_size=32,
                                                   label_name=LABEL_COLUMN,
                                                   na_value="?",
                                                   ignore_errors=True,
                                                   **kwargs)
    return dataset
raw_train_data32 = get_dataset32(train_file_path)
raw_test_data32 = get_dataset32(test_file_path)
show_dataset(raw_train_data32)

CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']

temp_dataset = get_dataset(train_file_path, column_names=CSV_COLUMNS) #remember we have used **kwargs

show_dataset(temp_dataset)
SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'class', 'deck', 'alone']

temp_dataset = get_dataset(train_file_path, select_columns=SELECT_COLUMNS)
show_dataset(temp_dataset)
SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']
DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]
temp_dataset = get_dataset(train_file_path,
                          select_columns=SELECT_COLUMNS,
                          column_defaults = DEFAULTS)
show_dataset(temp_dataset)
example_batch, labels_batch = next(iter(temp_dataset))
for key,value in example_batch.items():
    print("{}:{}".format(key,value))
def pack(features,label):
    return tf.stack(list(features.values()),axis=-1),label
packed_dataset = temp_dataset.map(pack)

for features,labels in packed_dataset.take(1):
    print(features.numpy())
    print("--------------------------")
    print(labels.numpy())
class PackNumericFeatures(object):
    def __init__(self,names):
        self.names = names
        
    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features
        
        return features, labels
        
    
NUMERIC_FEATURES = ['age','n_siblings_spouses','parch', 'fare']

packed_train_data = raw_train_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

packed_test_data = raw_test_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

show_dataset(packed_train_data)
import pandas as pd
desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
desc
MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])
def normalize_numeric_data(data, mean, std):
    return (data - mean)/std
normalizer = functools.partial(normalize_numeric_data,mean=MEAN,std=STD)
numeric_column = tf.feature_column.numeric_column("numeric",normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
numeric_column
numeric_columns
example_batch, labels_batch = next(iter(packed_train_data))
example_batch['numeric']
CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
  categorical_columns.append(tf.feature_column.indicator_column(cat_col))
categorical_columns
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)
print(preprocessing_layer(example_batch).numpy()[0])