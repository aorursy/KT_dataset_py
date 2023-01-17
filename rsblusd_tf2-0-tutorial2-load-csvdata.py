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
import numpy as np

import tensorflow as tf



np.set_printoptions(precision=3, suppress=True)
TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"

TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"



train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)

test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
train_file_path
LABEL_COLUMN = 'survived'

LABELS = [0, 1]
def get_dataset(file_path, **kwargs):

  dataset = tf.data.experimental.make_csv_dataset(

      file_path,

      batch_size=5, # Artificially small to make examples easier to show.

      label_name=LABEL_COLUMN,

      na_value="?",

      num_epochs=1,

      ignore_errors=True, 

      **kwargs)

  return dataset



raw_train_data = get_dataset(train_file_path)

raw_test_data = get_dataset(test_file_path)
def show_batch(dataset):

  for batch, label in dataset.take(1):

    for key, value in batch.items():

      print("{:20s}: {}".format(key,value.numpy()))

    print("{:20s}: {}".format("label",label.numpy()))

show_batch(raw_train_data)
class PackNumericFeatures(object):

  def __init__(self, names):

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
show_batch(packed_train_data)
example_batch, labels_batch = next(iter(packed_train_data)) 
import pandas as pd

desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()

desc
MEAN = np.array(desc.T['mean'])

STD = np.array(desc.T['std'])



def normalize_numeric_data(data, mean, std):

  # Center the data

  return (data-mean)/std
# See what you just created.

import functools

normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)



numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])

numeric_columns = [numeric_column]

numeric_column
numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)

numeric_layer(example_batch).numpy()
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



categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)

print(categorical_layer(example_batch).numpy()[0])

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)

print(preprocessing_layer(example_batch).numpy())
model = tf.keras.Sequential([

  preprocessing_layer,

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(1),

])



model.compile(

    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

    optimizer='adam',

    metrics=['accuracy'])



train_data = packed_train_data.shuffle(500)

test_data = packed_test_data
model.fit(train_data, epochs=20)
test_loss, test_accuracy = model.evaluate(test_data)
