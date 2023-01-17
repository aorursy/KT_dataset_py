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
import pandas as pd

import tensorflow as tf
train = pd.read_csv('../input/digit-recognizer/train.csv')

train.head()
train.iloc[:,1:]
train_x, train_y = train.iloc[:,1:], train['label']
train_y.value_counts()
train_x, train_y = tf.cast(train_x/255.0, tf.float32), tf.cast(train_y, tf.int64)
train_x
model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10, activation=tf.nn.softmax),

])
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=64, epochs=1000)
test = pd.read_csv('../input/digit-recognizer/test.csv')
len(test)
pd.DataFrame(model.predict_classes(test))
predictions = model.predict_classes(test)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions})

# Generate csv file

submissions.to_csv("submission.csv", index=False, header=True)