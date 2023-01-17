import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

sns.set_style('dark')

import sklearn

import tensorflow as tf

from tensorflow import keras
data_path = '../input/dont-call-me-turkey/train.json'

test_path = '../input/dont-call-me-turkey/test.json'



df = pd.read_json(data_path)

test_df = pd.read_json(test_path)
df.head()
df.shape
test_df.head()
test_df.shape
df.info()
df.describe()
sns.countplot(df['is_turkey'])

plt.tight_layout()
length = df['audio_embedding'].apply(len)



y = length

x = np.arange(1, len(length)+1)



sns.countplot(length)

plt.tight_layout()
plt.yscale('log')

sns.countplot(length)

plt.tight_layout()
df.head()
df['duration'] = df['end_time_seconds_youtube_clip'] = df['start_time_seconds_youtube_clip']

df.head()
corr = df.corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5)

plt.tight_layout()
from keras.preprocessing.sequence import pad_sequences

maxlen = 10



data = pad_sequences(df['audio_embedding'], maxlen=maxlen, padding='post')

labels = df['is_turkey'].values
from keras.utils import to_categorical



train_size = int((80/100) * df.shape[0])



train_data = data[: train_size]

train_labels = to_categorical(labels[: train_size])



valid_data = data[train_size: ]

valid_labels = to_categorical(labels[train_size: ])



assert(len(train_data) == len(train_labels))

assert(len(valid_data) == len(valid_labels))
train_data[0]
num_features = len(train_data[0][0])

num_features
train_labels[0]
test_data = pad_sequences(test_df['audio_embedding'], maxlen=maxlen, padding='post')

test_data[0]
def build_model():

    inp = keras.layers.Input(shape=(maxlen, num_features))

    x = keras.layers.BatchNormalization()(inp)

    

    x = keras.layers.Bidirectional( keras.layers.LSTM(128, return_sequences=True) )(x)

    x = keras.layers.Bidirectional( keras.layers.LSTM(64, return_sequences=True) )(x)

    

    avg_pool = keras.layers.GlobalAveragePooling1D()(x)

    max_pool = keras.layers.GlobalMaxPooling1D()(x)

    

    concat = keras.layers.concatenate([avg_pool, max_pool])

    

    hidden_1 = keras.layers.Dense(64)(concat)

    hidden_2 = keras.layers.Dropout(0.5)(hidden_1)

    hidden_activ = keras.layers.LeakyReLU()(hidden_2)

    

    output = keras.layers.Dense(2, activation="softmax")(hidden_activ)

    

    model = keras.Model(inputs=inp, outputs=output)

    

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    

    return model
model = build_model()

keras.utils.plot_model(model, dpi=62)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=2, verbose=1, min_lr=1e-8)



my_cb = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)



history = model.fit(train_data, train_labels, batch_size=64, epochs=100, 

                    validation_data=(valid_data, valid_labels), callbacks=[reduce_lr, my_cb], verbose=2)
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
predictions = np.argmax(model.predict(test_data), axis=1)

predictions
submission = pd.DataFrame({'vid_id': test_df['vid_id'], 'is_turkey': predictions})

submission.head()
submission.to_csv('submission.csv', index=False)