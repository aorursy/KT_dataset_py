# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_cluster = pd.read_csv('../input/cluster-data-nn-example/train.csv')
train_cluster.head()
train_cluster.color.unique()
colordict = { 'red': 0, 'blue': 1, 'green' : 2 , 'teal' : 3, 'orange' : 4, 'purple' : 5}
train_cluster['color'] = train_cluster.color.apply(lambda x: colordict[x])
train_cluster.color.unique()
np.random.shuffle(train_cluster.values)
model = keras.Sequential([
    keras.layers.Dense(32, input_shape = [2], activation = 'relu'),
    keras.layers.Dropout(rate = 0.15),
    keras.layers.Dense(32, activation = 'relu'),
    keras.layers.Dense(6, activation = 'sigmoid')
])
model.compile(
    optimizer = 'adam',
     loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
     metrics = ['accuracy'],
)
x = np.column_stack((train_cluster.x.values, train_cluster.y.values))
history = model.fit(x, train_cluster.color.values, batch_size = 4, epochs = 10)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss']].plot();
test_cluster = pd.read_csv('../input/cluster-data-nn-example/test.csv')
test_cluster.head()
test_cluster['color'] = test_cluster.color.apply(lambda x: colordict[x])
test_x = np.column_stack((test_cluster.x.values,test_cluster.y.values))
model.evaluate(test_x, test_cluster.color.values)
print("Prediction", np.round(model.predict(np.array([[0,3]]))))
