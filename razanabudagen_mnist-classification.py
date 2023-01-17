import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator



import tensorflow as tf

from tensorflow import keras

train_df = pd.read_csv('../input/nicht-mnist/train.csv',header=None)



test_df = pd.read_csv('../input/nicht-mnist/test.csv',header=None)
train_df.head()
train_df.isna().any().sum()
train_df.shape

target= train_df.pop(1)

train_df.drop(0,axis=1,inplace=True)



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
df_val=train_df.sample(frac=.2,random_state=1)

df_train=train_df.drop(df_val.index)



df_train
df_val = df_val.values

df_train = df_train.values
X_train = df_train.reshape(-1, 28, 28,1 ).astype('float32')

X_val = df_val.reshape(-1, 28, 28,1 ).astype('float32')



X_train = X_train / 255.0

X_val = X_val / 255.0
X_val
sns.set(style='white', context='notebook', palette='deep')

plt.figure(figsize=(5,5))

g = plt.imshow(X_train[80][:,:,0])
model = keras.Sequential([ keras.layers.Flatten(input_shape=(28, 28)),

keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(64, activation='relu'),

    keras.layers.Dense(64, activation='relu'),

    keras.layers.Dense(64, activation='relu'),



    keras.layers.Dense(10)

])
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.fit(df_train, target_train, epochs=40)
test_loss, test_acc = model.evaluate(df_val,  target_val, verbose=3)



print('\nTest accuracy:', test_acc)
test_data = pd.read_csv('../input/nicht-mnist/test.csv',header=None)
test_data.drop(0,axis=1,inplace=True)
probability_model = tf.keras.Sequential([model, 

                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_data)



results = np.argmax(predictions,axis = 1)

results = pd.Series(results,name="target")



submission = pd.concat([pd.Series(range(0,9364),name = "Id"),results],axis = 1)

submission['target'] = label_encoder.inverse_transform(submission['target'])

submission.head()
submission.target.value_counts()
submission.to_csv('sub_file.csv', index=False)