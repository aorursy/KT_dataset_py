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

   

df = pd.read_csv('../input/nicht-mnist/train.csv',header=None)



df.isna().any().sum()
df
df.shape
target=df.pop(1)
df.drop(0,axis=1,inplace=True)


classes=target.unique()

classes
from sklearn.preprocessing import LabelEncoder

label_encoder=LabelEncoder()

target = pd.DataFrame(label_encoder.fit_transform(target))

target_val=target.sample(frac=.2,random_state=1)

target_train=target.drop(target_val.index)
target_val = target_val.values

target_train = target_train.values
target_val.shape
df
df_val=df.sample(frac=.2,random_state=1)

df_train=df.drop(df_val.index)

df_train
df_val = df_val.values

df_train = df_train.values
X_train = df_train.reshape(-1, 28, 28,1 ).astype('float32')

X_val = df_val.reshape(-1, 28, 28,1 ).astype('float32')
df_train = df_train.reshape(-1, 28, 28,1 ).astype('float32')

#y_train=target_train.reshape(-1, 28, 28,1 ).astype('float32')
df_val = df_val.reshape(-1, 28, 28,1 ).astype('float32')

#target_val=target_val.reshape(-1, 28, 28,1 ).astype('float32')
X_train[0].shape

X_train = X_train/255.0

X_train[0].shape
X_val=X_val/255.0

X_val
X_train[0].shape



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline

sns.set(style='white', context='notebook', palette='deep')

plt.figure(figsize=(5,5))

g = plt.imshow(X_train[50][:,:,0])

'''model = keras.Sequential([ keras.layers.Flatten(input_shape=(28, 28)),

keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(128, activation='relu'),

   



    keras.layers.Dense(10)

])'''
model = models.Sequential()

model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28,28,1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(10))

model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])

history = model.fit(df_train, target_train, epochs=20, 

                    validation_data=(df_val, target_val))
#model.fit(df_train, target_train, epochs=50)
test_loss, test_acc = model.evaluate(df_val,  target_val, verbose=2)



print('\nTest accuracy:', test_acc)
test_data = pd.read_csv('../input/nicht-mnist/test.csv',header=None)
test_data
test_data.drop(0,axis=1,inplace=True)
probability_model = tf.keras.Sequential([model, 

                                         tf.keras.layers.Softmax()])
test_data=test_data.values
test_data = test_data.reshape(-1, 28, 28,1 ).astype('float32')
predictions = probability_model.predict(test_data)
results = np.argmax(predictions,axis = 1)
results
results = pd.Series(results,name="target")

submission = pd.concat([pd.Series(range(0,9364),name = "Id"),results],axis = 1)
submission['target'] = label_encoder.inverse_transform(submission['target'])
submission.head()

predictions.shape
submission.target.value_counts()
submission.to_csv('sub_ensemble_10_cnn.csv', index=False)