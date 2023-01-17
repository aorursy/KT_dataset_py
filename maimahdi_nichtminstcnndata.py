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

from tensorflow import keras

from tensorflow.keras import datasets, layers, models

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

import numpy as np

import matplotlib.pyplot as plt
train_df = pd.read_csv('../input/nicht-mnist/train.csv', header = None , index_col = 0)

train_df
test_df = pd.read_csv('../input/nicht-mnist/test.csv', header = None , index_col = 0)

test_df
print(train_df.isnull().any().sum())

print(test_df.isnull().any().sum())
target_col = train_df[1]

print(target_col , '\n' , len(target_col))
train_df[1] = pd.Categorical(train_df[1])

train_df[1] = train_df[1].cat.codes

train_df[1]
target_col
df_test = train_df.sample(frac=0.3, random_state=7)

df_train = train_df.drop(df_test.index)
y_train = df_train.iloc[:,0]

x_train = df_train.iloc[:,1:]

y_val = df_train.iloc[:,0]

x_val = df_train.iloc[:,1:]
x_train = x_train / 255.0 

x_val = x_val / 255.0 
x_train
x_train_np = np.vstack([[np.array(r).astype('uint8').reshape(28,28, 1) for i, r in x_train.iterrows()] ] )

x_val_np = np.vstack([[np.array(r).astype('uint8').reshape(28,28, 1) for i, r in x_val.iterrows()] ] )
model =Sequential()





model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.25))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
history = model.fit(x_train_np, y_train, epochs=30, 

                    validation_data=(x_val_np, y_val))
model.evaluate(x_val_np,  y_val, verbose=2)
test_df = test_df / 255.0
x_test_np = np.vstack([[np.array(r).astype('uint8').reshape(28,28, 1) for i, r in test_df.iterrows()] ] )
model.predict(x_test_np)
preds = np.argmax(model.predict(x_test_np), axis=1).tolist()
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

pred_labes = pd.Series([class_labels[p] for p in preds])

pred_labes
out_df = pd.DataFrame({

    'Id': test_df.index,

    'target': pred_labes

})

out_df
out_df.to_csv('my_submission.csv', index=False)