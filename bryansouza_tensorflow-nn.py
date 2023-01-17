# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf



from tensorflow import feature_column

from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
!pip install -q git+https://github.com/tensorflow/docs



import tensorflow_docs as tfdocs

import tensorflow_docs.modeling

import tensorflow_docs.plots
import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.info()
train.tail()
test.info()
test.tail()
train_labels = train.pop('Survived')
train.info()
age_binned, age_bins = pd.qcut(train['Age'], 

                               q=4,

                               retbins=True)
fare_binned, fare_bins = pd.qcut(train['Fare'], 

                               q=4,

                               retbins=True)
train.Age.fillna(train.Age.median(), inplace=True)
# transform null values to new value 

train.Cabin.fillna(train.Cabin.mode()[0].split()[0], inplace=True)
train.Embarked.fillna(train.Embarked.mode()[0], inplace=True)
train.info()
test.Age.fillna(test.Age.median(), inplace=True)
# transform null values to new value 

test.Cabin.fillna(test.Cabin.mode()[0].split()[0], inplace=True)
test.Fare.fillna(test.Fare.median(), inplace=True)
test.info()
test['Survived'] = np.nan
test.tail()
test_labels = test.pop('Survived')
test_features = test.copy()
train.dtypes
SEED = 65536

BATCH = 32
tf.random.set_seed(SEED)
train_features, val_features, train_labels, val_labels = train_test_split(train, 

                                                                          train_labels, 

                                                                          test_size=0.2,

                                                                          random_state=SEED,

                                                                          shuffle=True,

                                                                          stratify=train_labels)





print(len(train_features), 'train examples')

print(len(val_features), 'validation examples')

print(len(test_features), 'test examples')
train_features.info()
desc = train_features.describe()
train_ds = tf.data.Dataset.from_tensor_slices((dict(train_features), train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((dict(val_features), val_labels))
test_ds = tf.data.Dataset.from_tensor_slices((dict(test_features), test_labels))
feature_columns = []



# numeric columns

for header in ['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Age']:

  feature_columns.append(feature_column.numeric_column(header)) 
# bucketized columns

age = feature_column.numeric_column('Age')

age_buckets = feature_column.bucketized_column(age, boundaries=[*age_bins])

feature_columns.append(age_buckets)



fare = feature_column.numeric_column('Fare')

fare_buckets = feature_column.bucketized_column(fare, boundaries=[*fare_bins])

feature_columns.append(fare_buckets)
# indicator_columns

indicator_column_names = ['Sex', 'Embarked']



for col_name in indicator_column_names:

  categorical_column = feature_column.categorical_column_with_vocabulary_list(

      col_name, train[col_name].unique())

  indicator_column = feature_column.indicator_column(categorical_column)

  feature_columns.append(indicator_column)
# embedding columns

embedding_column_names = ['Ticket', 'Name', 'Cabin']



for col_name in embedding_column_names:

  categorical_column = feature_column.categorical_column_with_vocabulary_list(

      col_name, train[col_name].unique())

  embedding_column = feature_column.embedding_column(categorical_column, dimension=64)

  feature_columns.append(embedding_column)
gender = feature_column.categorical_column_with_vocabulary_list('Sex', train['Sex'].unique())
port = feature_column.categorical_column_with_vocabulary_list('Embarked', train['Embarked'].unique())
class_pass = feature_column.categorical_column_with_vocabulary_list('Pclass', train['Pclass'].unique())
# crossed columns

age_gender_feature = feature_column.crossed_column([age_buckets, gender], hash_bucket_size=64)

feature_columns.append(feature_column.indicator_column(age_gender_feature))





age_port_feature = feature_column.crossed_column([age_buckets, port], hash_bucket_size=64)

feature_columns.append(feature_column.indicator_column(age_port_feature))





age_class_feature = feature_column.crossed_column([age_buckets, class_pass], hash_bucket_size=64)

feature_columns.append(feature_column.indicator_column(age_class_feature))





fare_gender_feature = feature_column.crossed_column([fare_buckets, gender], hash_bucket_size=64)

feature_columns.append(feature_column.indicator_column(fare_gender_feature))





# fare_port_feature = feature_column.crossed_column([fare_buckets, port], hash_bucket_size=64)

# feature_columns.append(feature_column.indicator_column(fare_port_feature))





fare_class_feature = feature_column.crossed_column([fare_buckets, class_pass], hash_bucket_size=64)

feature_columns.append(feature_column.indicator_column(fare_class_feature))
feature_layer = tf.keras.layers.DenseFeatures(feature_columns, trainable=True)
model = tf.keras.Sequential([

                  feature_layer,

                  layers.Dense(128, activation='elu'),

                  layers.Dense(128, activation='elu'),

                  layers.Dropout(.2),

                  layers.Dense(1)

])



model.compile(optimizer='adamax',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=[tf.keras.metrics.BinaryAccuracy(),

                       'accuracy'])
train_ds = train_ds.shuffle(len(train_features)).batch(BATCH).cache()

val_ds = val_ds.batch(BATCH).cache()

test_ds = test_ds.batch(BATCH).cache()
stopping = tf.keras.callbacks.EarlyStopping(patience=25,

                                             monitor='val_accuracy')
history = {}
history['model_0'] = model.fit(train_ds,

                                  validation_data=val_ds,

                                  epochs=200,

                                  workers=4,

                                  use_multiprocessing=True,

                                  callbacks=[stopping])
model.summary()
plotter = tfdocs.plots.HistoryPlotter(metric='accuracy', smoothing_std=2)

plotter.plot(history)

plt.ylim([0.5, 1.05]);
preds = model.predict(test_ds)
preds[-20:]
predictions = [1 if i >= 1.5 else 0 for i in preds]

predictions[-20:]
sub = pd.DataFrame(predictions, columns=['Survived'], index=test.PassengerId)

sub.tail(20)
sub.to_csv('submission.csv')