!pip install tensorflow_datasets > /dev/null
import time
import numpy as np
import gc
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from keras import backend as K

from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, concatenate, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS
from sklearn import metrics

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

start = time.time()
pd.options.display.max_colwidth = 1500
BATCH_SIZE = 512
SEED=42
train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"], 
                                  batch_size=-1, as_supervised=True)

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)
np.save("train_examples", train_examples)
np.save("train_labels", train_labels)

np.save("test_examples", test_examples)
np.save("test_labels", test_labels)
print("Training entries: {}, test entries: {}".format(len(train_examples), len(test_examples)))
input_len = [len(x) for x in np.concatenate((train_examples, test_examples), axis=0)]
print("Input Lengths:\nAverage {:.1f} +/- {:.1f}\nMax {} Min {}".format(np.mean(input_len), np.std(input_len), np.max(input_len), np.min(input_len)))
train_examples[:10]
train_labels[:10]
# Look at class balance..
unique_elements, counts_elements = np.unique(train_labels, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))
%%time
model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(model, output_shape=[], input_shape=[], 
                           dtype=tf.string, trainable=True, name='gnews_embedding')
def build_model(embed):
    
    model = Sequential([
        Input(shape=[], dtype=tf.string),
        embed,
        Dropout(.2),
        Dense(16, activation='relu'),
        Dropout(.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

model = build_model(hub_layer)
model.summary()
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min')
checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(
                    train_examples,
                    train_labels,
                    epochs=40,
                    batch_size=BATCH_SIZE,
                    validation_split = .2,
                    shuffle = True,
                    callbacks = [checkpoint, es],
                    verbose=1)

model.load_weights('model.h5')
results = model.evaluate(test_examples, test_labels)
print(results)
history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

f, ax = plt.subplots(1,2, figsize = [11,4])

ax[0].plot(epochs, loss, 'r', label='Training loss')
ax[0].plot(epochs, val_loss, 'b', label='Validation loss')
ax[0].set_title('Training and validation loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[1].plot(epochs, acc, 'r', label='Training acc')
ax[1].plot(epochs, val_acc, 'b', label='Validation acc')
ax[1].set_title('Training and validation accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

plt.tight_layout()

plt.show()
test_pred = model.predict(test_examples, batch_size = BATCH_SIZE)
results_pd = pd.DataFrame.from_dict({'text': test_examples, 'pred': test_pred[:,0], 'ground_truth': test_labels})
results_pd['error'] = results_pd['ground_truth'] - results_pd['pred']

display(results_pd.sort_values(by = 'error', ascending=False).iloc[:10])

display(results_pd.sort_values(by = 'error', ascending=True).iloc[:10])
K.clear_session()

del history
del model
_ = gc.collect()
%%time
module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/4'
USE_embed = hub.KerasLayer(module_url, trainable=False, name='USE_embedding')
USE_embed(train_examples[:3])
def build_model(embed):
    
    model = Sequential([
        Input(shape=[], dtype=tf.string),
        embed,
        Dropout(.2),
        Dense(16, activation='relu'),
        Dropout(.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

model = build_model(USE_embed)
model.summary()
MAX_LEN = 2058

small_train_examples = np.array([x[:MAX_LEN] for x in train_examples])
small_test_examples = np.array([x[:MAX_LEN] for x in test_examples])
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min')
checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(
                    small_train_examples,
                    train_labels,
                    epochs=40,
                    batch_size=BATCH_SIZE,
                    validation_split = .2,
                    shuffle = True,
                    callbacks = [checkpoint, es],
                    verbose=1)

model.load_weights('model.h5')
results = model.evaluate(small_test_examples, test_labels)
print(results)
history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

f, ax = plt.subplots(1,2, figsize = [11,4])

ax[0].plot(epochs, loss, 'r', label='Training loss')
ax[0].plot(epochs, val_loss, 'b', label='Validation loss')
ax[0].set_title('Training and validation loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[1].plot(epochs, acc, 'r', label='Training acc')
ax[1].plot(epochs, val_acc, 'b', label='Validation acc')
ax[1].set_title('Training and validation accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

plt.tight_layout()

plt.show()
test_pred = model.predict(small_test_examples, batch_size = BATCH_SIZE)
results_pd = pd.DataFrame.from_dict({'text': test_examples, 'pred': test_pred[:,0], 'ground_truth': test_labels})
results_pd['error'] = results_pd['ground_truth'] - results_pd['pred']

display(results_pd.sort_values(by = 'error', ascending=False).iloc[:10])

display(results_pd.sort_values(by = 'error', ascending=True).iloc[:10])
USE_embed([small_train_examples[0]])['outputs'].numpy().shape
%%time
full_labels = np.concatenate((train_labels, test_labels))
full_txt = np.concatenate((small_train_examples, small_test_examples))

batch_size = 500
embeddings = []

for b in range(0, full_txt.shape[0] // batch_size):
    embeddings.extend(USE_embed(full_txt[batch_size*b: batch_size*(b+1)])['outputs'].numpy())