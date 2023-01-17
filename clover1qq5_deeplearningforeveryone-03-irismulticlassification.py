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
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from sklearn.preprocessing import LabelEncoder



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf

np.random.seed(3)

tf.random.set_seed(3)
df = pd.read_csv('../input/iris.csv', names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

df.head()
sns.pairplot(df, hue='species');

plt.show()
dataset = df.values

X = dataset[:, 0:4].astype(float)

Y_obj = dataset[:,4]


e = LabelEncoder()

e.fit(Y_obj)

Y = e.transform(Y_obj)

Y_encoded = tf.keras.utils.to_categorical(Y)
model = Sequential()

model.add(Dense(16, input_dim=4, activation='relu'))

model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])
model.fit(X,Y_encoded, epochs=50, batch_size=1)
print("Accuarcy: %.4f" % (model.evaluate(X,Y_encoded)[1]))