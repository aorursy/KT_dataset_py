import tensorflow as tf

from tensorflow import keras



import functools

import numpy as np

import pandas as pd
train_file_path = "../input/titanic/train.csv"

test_file_path = "../input/titanic/test.csv"
np.set_printoptions(3, suppress=True)
!head {train_file_path}
!head {test_file_path}
def getCabin(s):

    ss = str(s)

    if ss != "nan":

        return ss[0]

    return s
df = pd.read_csv(test_file_path)

df['Survived'] = 0

df['Cabin'] = df['Cabin'].apply(getCabin)

df.to_csv('test.csv', index=False)
df = pd.read_csv(train_file_path)

df['Cabin'] = df['Cabin'].apply(getCabin)

df.to_csv('train.csv', index=False)
!head {'train.csv'}

train_file_path = 'train.csv'
!head {'test.csv'}

test_file_path = 'test.csv'
df['Cabin'].value_counts()
LABEL_COLUMN = 'Survived'

LABELS = [0, 1]
def get_dataset(file_path, **kwargs):

    dataset = tf.data.experimental.make_csv_dataset(

        file_path,

        batch_size=5,

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

      print("{:20s}: {}".format(key, value.numpy()))
show_batch(raw_train_data)
def pack(features, label):

    return tf.stack(list(features.values()), axis=-1), label
class PackNumericFeatures(object):

    def __init__(self, names):

        self.names = names

    

    def __call__(self, features, labels):

        numeric_features = [features.pop(name) for name in self.names]

        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]

        numeric_features = tf.stack(numeric_features, axis=-1)

        features['numeric'] = numeric_features

        

        return features, labels
NUMERIC_FEATURES = ['Age', 'SibSp', 'Parch', 'Fare']



packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))



packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))
show_batch(packed_train_data)
desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()

desc
MEAN = np.array(desc.T['mean'])

STD = np.array(desc.T['std'])
def normalize_numeric_data(data, mean, std):

    return (data-mean)/std
normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)



numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])

numeric_columns = [numeric_column]

numeric_column
numeric_layer = keras.layers.DenseFeatures(numeric_columns)
CATEGORIES = {

    'Pclass': [1, 2, 3],

    'Sex': ['female', 'male'],

    'Cabin': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'],

    'Embarked': ['C', 'Q', 'S']

}
categorical_columns = []

for feature, vocab in CATEGORIES.items():

    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(

        key=feature, vocabulary_list=vocab)

    categorical_columns.append(tf.feature_column.indicator_column(cat_col))
categorical_columns
categorical_layer = keras.layers.DenseFeatures(categorical_columns)
preprocessing_layer = keras.layers.DenseFeatures(categorical_columns+numeric_columns)
model = keras.Sequential([

    preprocessing_layer,

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(1),

])



model.compile(

    loss=keras.losses.BinaryCrossentropy(from_logits=True),

    optimizer='adam',

    metrics=['accuracy'])
train_data = packed_train_data.shuffle(500)

test_data = packed_test_data
model.fit(train_data, epochs=20)
predictions = tf.round(tf.sigmoid(model.predict(test_data)).numpy())
df = pd.read_csv(test_file_path)
output = pd.DataFrame({"PassengerId":df['PassengerId'] , "Survived" : predictions.numpy().astype(int)})

output.to_csv("Submission.csv",index = False)
output