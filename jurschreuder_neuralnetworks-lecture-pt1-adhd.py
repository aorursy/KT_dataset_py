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

print("done!")
data = pd.DataFrame(np.random.randint(0,2,size=(1000, 6)), columns=['ACC/mPFC', 'PFC/OFC', 'Striatum', 'Parietal & Sensory cortices', 'HPC', 'Thalamus'])

data.head(20) # see what we made, the first 20 rows
labels = pd.DataFrame(np.zeros(shape=(1000, 2)), columns=["yes", "no"])



# check all the data

for i, row in data.iterrows():

    # find 0 1 0 * * *

    if row['PFC/OFC'] == 0 and row['Parietal & Sensory cortices'] == 1 and row['Thalamus'] == 0:

        # set the label to 1, 0 (yes)

        labels["yes"][i] = 1

        labels["no"][i] = 0

    # find 1 0 1 * * *

    elif row['PFC/OFC'] == 1 and row['Parietal & Sensory cortices'] == 0 and row['Thalamus'] == 1:

        # set the label to 1, 0 (yes)

        labels["yes"][i] = 1

        labels["no"][i] = 0

    # find 1 1 1 * * *

    elif row['PFC/OFC'] == 1 and row['Parietal & Sensory cortices'] == 1 and row['Thalamus'] == 1:

        # set the label to 1, 0 (yes)

        labels["yes"][i] = 1

        labels["no"][i] = 0

    # find 0 0 0 * * *

    elif row['PFC/OFC'] == 0 and row['Parietal & Sensory cortices'] == 0 and row['Thalamus'] == 0:

        # set the label to 1, 0 (yes)

        labels["yes"][i] = 1

        labels["no"][i] = 0

    else:

        # anything brain scan not these 4 patterns is not ADHD, set the label to 0, 1 (no)

        labels["yes"][i] = 0

        labels["no"][i] = 1





labels.head(20) # see what we made, the first 20 rows
print("times yes:", labels["yes"].sum(), "times no:", labels["no"].sum())
import tensorflow as tf

from tensorflow.keras import layers



model = tf.keras.Sequential([

  layers.Dense(7),

  layers.Dense(2)

])

model.compile(loss = tf.losses.MeanSquaredError(),

                      optimizer = tf.optimizers.Adam())

print("done!")
model.fit(data, labels, epochs=5)
output = model(np.array([[1,1,1,1,1,1]]), training=False)

print(np.around(output, 2))
model2 = tf.keras.Sequential([

  layers.Dense(6),

  layers.Dense(10, activation='sigmoid'),

  layers.Dense(2)

])

model2.compile(loss = tf.losses.MeanSquaredError(),

                      optimizer = tf.optimizers.Adam())



model2.fit(data, labels, epochs=100)
#output = model(np.array([[1,0,1,0,1,0]]), training=False)

output = model2.predict(np.array([[0,0,0,1,1,1]]))

print(np.around(output, 2))