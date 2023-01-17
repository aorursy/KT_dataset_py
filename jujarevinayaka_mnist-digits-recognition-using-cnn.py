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
import time
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
IMG_SHAPE = (28, 28)
NUM_IMAGES_PER_CLASS = 6000
df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
df = df.sample(frac=1)
df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
df.head()
df_count = df.groupby('label', as_index=False).count().rename({'pixel0': 'count'}, axis=1)[['label', 'count']]
df_count
plt.bar(df_count['label'], df_count['count'])
df.drop(['label'], axis=1).values.shape
df = df.sample(frac=1)
X = df.drop(['label'], axis=1).values
X = X.reshape(-1, IMG_SHAPE[0], IMG_SHAPE[1], 1)
y = df['label'].values
# Agument data by rotating it to 10 degree towards right
datagen = ImageDataGenerator()
df_1000 = df[df['label'].isin(df_count['label'])].groupby('label').head(NUM_IMAGES_PER_CLASS - df_count['count'].min())

X_1000 = df_1000.drop(['label'], axis=1).values
X_1000 = X_1000.reshape(-1, IMG_SHAPE[0], IMG_SHAPE[1], 1)
y_ag = df_1000['label'].values

X_ag = []
for i in X_1000:
    X_ag.append(datagen.apply_transform(i, {'theta': 10}))
    
X_ag = np.array(X_ag)
print(np.array(X_ag).shape, y_ag.shape)
plt.subplot(2, 3, 1)
plt.imshow(X_1000[10].reshape(IMG_SHAPE[0], IMG_SHAPE[1]))
plt.subplot(2, 3, 2)
plt.imshow(X_ag[10].reshape(IMG_SHAPE[0], IMG_SHAPE[1]))
plt.subplot(2, 3, 3)
plt.imshow(X_ag[10 + int(X_ag.shape[0] / 2)].reshape(IMG_SHAPE[0], IMG_SHAPE[1]))
plt.subplot(2, 2, 1)
plt.imshow(X[5].reshape(IMG_SHAPE[0], IMG_SHAPE[1]))
X = np.concatenate((X, X_ag))
X = tf.keras.utils.normalize(X)
y = np.concatenate((y, y_ag))
plt.subplot(2, 2, 2)
plt.imshow(X[5].reshape(IMG_SHAPE[0], IMG_SHAPE[1]))
indices = []
for idx in set(y):
    indices += list(np.where(y == idx)[0][:NUM_IMAGES_PER_CLASS])
np.random.shuffle(indices)
X = np.array([X[idx] for idx in indices])
y = np.array([y[idx] for idx in indices])
print(X.shape, y.shape)
ind = 0
print(y[ind])
plt.subplot(2, 2, 1)
plt.imshow(X[ind].reshape(IMG_SHAPE[0], IMG_SHAPE[1]))
plt.subplot(2, 2, 2)
plt.imshow(X[ind + 1].reshape(IMG_SHAPE[0], IMG_SHAPE[1]))
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(input_shape=X.shape[1:], filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.5))

#model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(532, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(x=X, y=y, validation_split=0.15, epochs=15, batch_size=32)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_loss'], label='test loss')
plt.plot(history.history['val_accuracy'], label='test acc')
plt.legend()
def visualize_conv_layer(layer_name, index=0):
  layer_output=model.get_layer(layer_name).output
  intermediate_model=tf.keras.models.Model(inputs=model.input,outputs=layer_output)
  intermediate_prediction=intermediate_model.predict(X[index].reshape(1,28,28,1))
  row_size=4
  col_size=8
  img_index=0
  print(np.shape(intermediate_prediction))
  fig,ax=plt.subplots(row_size,col_size,figsize=(10,8))
  for row in range(0,row_size):
    for col in range(0,col_size):
      ax[row][col].imshow(intermediate_prediction[0, :, :, img_index], cmap='gray')
      img_index=img_index+1
for layer_name in [i.name for i in model.layers if 'conv' in i.name]:
    print("--------------------- {} ----------------------".format(layer_name))
    visualize_conv_layer(layer_name, 10)
X_test = df_test.values
X_test = X_test.reshape(-1, IMG_SHAPE[0], IMG_SHAPE[1], 1)
X_test = tf.keras.utils.normalize(X_test)
y_pred = model.predict(X_test)
df_pred = pd.DataFrame(np.argmax(y_pred, axis=1), columns=['Label'])
df_pred['ImageId'] = [i+1 for i in df_pred.index]
ind = 210
print(df_pred.iloc[ind].Label)
plt.imshow(X_test[ind].reshape(28, 28))

df_pred.to_csv("predictions.csv", index=False)
model.save("model_2C_0.3-0.5.model")

