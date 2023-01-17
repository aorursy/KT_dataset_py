# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

train.head(5)
train_labels = train['label'].to_numpy()

train_features = train.drop(['label'],axis=1).to_numpy()



train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.2)

test_features = test.to_numpy()
train_features = train_features / 255

test_features = test_features / 255

max(train_features[0])
train_features.shape, train_labels.shape, val_features.shape, val_labels.shape
i = np.random.choice(len(train_features))

img = np.reshape(train_features[i], (28, 28))

plt.imshow(img)
model = keras.Sequential([

    keras.layers.Dense(784),

    keras.layers.Dense(784/2, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])
model.compile(optimizer='Adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_features, train_labels, epochs = 10)
model.summary()
val_features = val_features / 255

validation_loss, validaion_acc = model.evaluate(val_features, val_labels, verbose=2)

val_predictions = model.predict(val_features)
i = np.random.choice(len(val_features))

img = np.reshape(val_features[i], (28, 28))

pred = val_predictions[i]



fig, axs = plt.subplots(1,2)



axs[0].imshow(img)

axs[1].barh(np.arange(0,10), pred);

print(f'Model Prediction: {np.argmax(pred)}')