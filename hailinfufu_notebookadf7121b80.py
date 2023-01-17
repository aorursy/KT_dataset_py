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
import pathlib
import time

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

__SELECT_COLUMN_NAMES = ['age', 'education', 'income_bracket']


def get_train_test_pandas_data():
    census = pd.read_csv("/kaggle/input/adult-census-income/adult.csv")

    census['income_bracket'] = census['income'].apply(lambda label: 0 if label == ' <=50K' else 1)
    census = census[__SELECT_COLUMN_NAMES]

    y_labels = census.pop('income_bracket')
    x_data = census

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.3)

    return x_train, x_test, y_train, y_test


def get_feature_columns():
    age = tf.feature_column.numeric_column("age", dtype=tf.int64)
    education = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000),
        dimension=100)

    feat_cols = [age, education]

    return feat_cols
if (tf.__version__ < '2.0'):
    tf.enable_eager_execution()

x_train, _, y_train, _ = get_train_test_pandas_data()

dataset = tf.data.Dataset.from_tensor_slices((dict(x_train), y_train))

dataset = dataset.shuffle(len(x_train)).batch(4)

feat_cols = get_feature_columns()
class mymodel(tf.keras.Model):
    def __init__(self):
        super(mymodel, self).__init__()
        self.layer1 = tf.keras.layers.DenseFeatures(feature_columns=feat_cols)
        self.layer2 = tf.keras.layers.Dense(10, activation='relu')
        self.layer3 = tf.keras.layers.Dense(10, activation='relu')
        self.layer4 = tf.keras.layers.Dense(1, activation='sigmoid')

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
model = mymodel()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(dataset, epochs=1)
__SAVED_MODEL_DIR = './saved_models/census_keras/{}'.format(int(time.time()))
pathlib.Path(__SAVED_MODEL_DIR).mkdir(parents=True, exist_ok=True)

tf.saved_model.save(model, export_dir=__SAVED_MODEL_DIR)
loaded_model = tf.keras.models.load_model(__SAVED_MODEL_DIR)
y_pred = loaded_model.call({"age": [39],
                            "education": ["Bachelors"]})
print(y_pred)
