# For data manipulation and visualization

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random



# For deep learning

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
# loading the data

df = pd.read_csv("../input/devanagari-character-set/data.csv")

df.head()
train_df, test_df = train_test_split(df, test_size=0.2)

train_df = train_df.reset_index(drop=True)

test_df = test_df.reset_index(drop=True)



x_train, x_test = train_df.drop('character', axis=1), test_df.drop('character', axis=1)

y_train, y_test = train_df['character'], test_df['character']
all_chars = df['character'].unique()



print(f'All characters: {all_chars}')

print(f"Number of characters: {len(all_chars)}")
# Converting data frame to image arrays

x_train, x_test = x_train.values.reshape(x_train.values.shape[0], 32, 32), x_test.values.reshape(x_test.values.shape[0], 32, 32)



print(f'Shape of Training Image array: {x_train.shape}')

print(f'Shape of Testing Image array: {x_test.shape}')
# Plotting some training images along with their labels

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12,6))

idxs = y_train.index



for row in ax:

    for col in row:

        idx = random.choice(idxs)

        col.axis('off')

        col.set_title(y_train[idx])

        col.imshow(x_train[idx], cmap=plt.get_cmap('gray'))
# Plotting distribution of output classes

data = df.character.value_counts()

plt.figure(figsize=(15, 10))

sns.barplot(y = data.index, x = data, orient='h')

plt.show()
# Label encoding the output classes

labelencoder = LabelEncoder()



Y_train = labelencoder.fit_transform(y_train)

X_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)



Y_test = labelencoder.fit_transform(y_test)

X_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
# Detect hardware

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection

except ValueError: # If TPU not found

    tpu = None
# Select appropriate distribution strategy

if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])  

else:

    strategy = tf.distribute.get_strategy() # Default strategy that works on CPU and single GPU

    print('Running on CPU instead')



print("Number of accelerators: ", strategy.num_replicas_in_sync)
with strategy.scope():

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))



    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(46))

    

    model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])

    

    # Display the architecture

    model.summary()
if tpu:

    BATCH_SIZE = 400 * strategy.num_replicas_in_sync

else:

    BATCH_SIZE = 400



EPOCHS = 30

VALIDATION_SPLIT = 0.25



# Train the model

history = model.fit(X_train, Y_train, epochs=EPOCHS, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE)



# Save the model weights

model.save('./devnagari_model.h5', overwrite=True)
plt.plot(history.history['accuracy'], label='accuracy')

plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0.5, 1])

plt.legend(loc='lower right')



# Show testing loss and accuracy

test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12,6))

idxs = y_test.index



for row in ax:

    for col in row:

        idx = random.choice(idxs)

        x = X_test[idx].reshape(1, 32, 32, 1)

        logit_y = model(x)

        pred_y = labelencoder.inverse_transform(np.argmax(logit_y, axis=-1))

        col.axis('off')

        col.set_title(pred_y[0])

        col.imshow(x_test[idx], cmap=plt.get_cmap('gray'))