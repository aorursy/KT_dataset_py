import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import tensorflow as tf
train_dir = '../input/lyme-disease-rashes/RashData/Train/Train_2_Cases'

val_dir = '../input/lyme-disease-rashes/RashData/Validation/Validation_2_Cases'
batch_size = 32



img_height = 128

img_width = 128
train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    train_dir,

    validation_split=0.2,

    seed=42,

    subset='training',

    batch_size=batch_size,

    image_size=(img_height, img_width)

)



val_ds = tf.keras.preprocessing.image_dataset_from_directory(

    train_dir,

    validation_split=0.2,

    seed=42,

    subset='validation',

    batch_size=batch_size,

    image_size=(img_height, img_width)

)



test_ds = tf.keras.preprocessing.image_dataset_from_directory(

    val_dir,

    batch_size=batch_size,

    image_size=(img_height, img_width)

)
class_names = train_ds.class_names

print(class_names)
plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

        plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype('uint8'))

        plt.title(class_names[labels[i]])

        plt.axis('off')

    plt.show()
AUTOTUNE = tf.data.experimental.AUTOTUNE



train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
inputs = tf.keras.Input(shape=(img_height, img_width, 3))

x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)

x = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(x)

x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)

x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.Dense(128, activation='relu')(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)







model = tf.keras.Model(inputs, outputs)





model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=[

        'accuracy',

        tf.keras.metrics.AUC(name='auc')

    ]

)





epochs = 45



history = model.fit(

    train_ds,

    validation_data=val_ds,

    batch_size=batch_size,

    epochs=epochs,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()],

    verbose=0

)
plt.figure(figsize=(20, 10))



epochs_range = range(1, epochs + 1)

train_loss = history.history['loss']

val_loss = history.history['val_loss']

train_auc = history.history['auc']

val_auc = history.history['val_auc']



plt.subplot(1, 2, 1)

plt.plot(epochs_range, train_loss, label="Training Loss")

plt.plot(epochs_range, val_loss, label="Validation Loss")



plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training and Validation Loss")

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(epochs_range, train_auc, label="Training AUC", color='b')

plt.plot(epochs_range, val_auc, label="Validation AUC", color='r')



plt.xlabel("Epoch")

plt.ylabel("AUC")

plt.title("Training and Validation AUC")

plt.legend()



plt.show()
np.argmin(val_loss)
np.argmax(val_auc)
model.evaluate(test_ds)