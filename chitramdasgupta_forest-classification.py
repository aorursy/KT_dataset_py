import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import tensorflow as tf

from tensorflow import keras
train_path = '../input/forest-cover-type-kernels-only/train.csv.zip'



df = pd.read_csv(train_path)

df.head()
df.columns
print(df.iloc[0])
df.shape  # 15,120 rows and 56 columns
corr = df.corr()

corr.style.background_gradient(cmap='coolwarm', axis=None)
df = df.drop(['Aspect', 'Soil_Type7', 'Soil_Type15'], axis=1)

df.head(1)
# temp = df.drop(['soil', 'wild', 'Cover_Type', 'Id'], axis=1)

# temp

temp = df.drop(['Cover_Type', 'Id'], axis=1)

temp.iloc[0]
temp = (temp-temp.min())/(temp.max()-temp.min()).astype(np.float32)  # Using min-max standardization

temp.describe()
temp
from keras.utils import to_categorical



all_data = temp.values

print(all_data[0])



all_labels = to_categorical(df['Cover_Type'])

print(all_labels[: 2])
assert(len(all_labels) == len(all_data))
print(len(all_data))



train_size = 12096



train_data = all_data[: train_size]

valid_data = all_data[train_size: ]



train_labels = all_labels[: train_size]

valid_labels = all_labels[train_size: ]

print(len(train_data[0]))
model = keras.models.Sequential([

    keras.layers.Dense(64),

    keras.layers.LeakyReLU(),

    keras.layers.Dense(8, 'softmax'),

])



model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
my_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)



history = model.fit(train_data, train_labels, epochs=200,

              validation_data=(valid_data, valid_labels), 

              callbacks=[my_cb])
print(history.history.keys())

epochs = len(history.history['loss'])

epochs
y1 = history.history['loss']

y2 = history.history['val_loss']

x = np.arange(1, epochs+1)



plt.plot(x, y1, y2)

plt.legend(['loss', 'val_loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.tight_layout()
y1 = history.history['acc']

y2 = history.history['val_acc']

x = np.arange(1, epochs+1)



plt.plot(x, y1, y2)

plt.legend(['acc', 'val_acc'])

plt.xlabel('Epochs')

plt.ylabel('Acc')

plt.tight_layout()
model.evaluate(valid_data, valid_labels)
test_path = '../input/forest-cover-type-kernels-only/test.csv.zip'



test_df = pd.read_csv(test_path)
test_df.head()
test_df = test_df.drop(['Aspect', 'Soil_Type7', 'Soil_Type15'], axis=1)



id_nums = test_df['Id']
temp = test_df.drop(['Id'], axis=1)

temp = (temp-temp.min())/(temp.max()-temp.min()).astype(np.float32)



all_data = temp.values

all_data
answer = model.predict(all_data)
classes = answer.argmax(axis=-1)

classes
my_final = pd.DataFrame({'id': id_nums, 'Cover_Type': classes})

my_final.head()
kaggle_output_path = './submission.csv'

my_final.to_csv(kaggle_output_path, index=False)