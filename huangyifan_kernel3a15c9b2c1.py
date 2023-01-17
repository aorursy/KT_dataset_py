# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.python import keras 

import csv

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





#sub=pd.read_csv()
train_df = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test_df = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')

pred_df = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

train_df = train_df.append(test_df)



X_train = train_df.drop(['label'], axis = 1)

Y_train = train_df['label']

X_pred = pred_df.drop(['id'], axis = 1)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.20)

X_train, X_test, X_pred = X_train.apply(lambda x: x/255), X_test.apply(lambda x: x/255), X_pred.apply(lambda x: x/255)



X_train = X_train.values.reshape(-1, 28, 28, 1)

X_test = X_test.values.reshape(-1, 28, 28, 1)
model = keras.Sequential([keras.layers.Conv2D(32, 5, 1, input_shape=(28, 28, 1), padding="same", activation="relu"),

							  keras.layers.MaxPooling2D(),

							  keras.layers.Dropout(0.5),

							  keras.layers.Conv2D(64, 5, 1, padding="same", activation="relu"),

							  keras.layers.MaxPooling2D(),

							  keras.layers.Dropout(0.5),

							  keras.layers.Flatten(),

							  keras.layers.Dense(1024, activation="relu"),

							  keras.layers.Dropout(0.5),

							  keras.layers.Dense(10, activation="softmax")])

model.summary()
model.compile(optimizer=keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, epochs=20,

						validation_data=(X_test,Y_test))
def save_result(filename, results):

	with open(filename, 'w', newline='') as f:

		myWriter = csv.writer(f)

		myWriter.writerow(['id', 'label'])

		for i, result in enumerate(results):

			myWriter.writerow([i , result])

X_pred = X_pred.values.reshape(-1, 28, 28, 1)

result1 = model.predict(X_pred, batch_size=64)

result1 = np.argmax(result1, axis=-1)

save_result("nnResult.csv", result1)