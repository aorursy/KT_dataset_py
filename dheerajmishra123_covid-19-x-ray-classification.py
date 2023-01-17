import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf

 

from tensorflow import keras

from tensorflow.keras import layers,preprocessing

from tensorflow.keras.models import Sequential
batch_size = 32

image_size = (180,180)
base_dir = '../input/covid19-radiography-database/COVID-19 Radiography Database'

train_data=tf.keras.preprocessing.image_dataset_from_directory(

    base_dir,

    label_mode="int",

    batch_size=batch_size,

    image_size=image_size,

    shuffle=True,

    seed=101,

    validation_split=0.2,

    subset='training',

)

val_data = tf.keras.preprocessing.image_dataset_from_directory(

  base_dir,

  validation_split=0.2,

  subset="validation",

  seed=101,

  image_size=image_size,

  batch_size=batch_size,

  shuffle=False)

class_names = train_data.class_names
plt.figure(figsize=(12, 12))

for images, labels in train_data.take(1):

  for i in range(12):

    ax = plt.subplot(4, 3, i + 1)

    plt.imshow(images[i].numpy().astype("uint8"))

    plt.title(class_names[labels[i]])

    plt.axis("off")

from tensorflow.keras import layers,preprocessing

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Flatten
AUTOTUNE = tf.data.experimental.AUTOTUNE



train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_data.map(lambda x, y: (normalization_layer(x), y))

image_batch, labels_batch = next(iter(normalized_ds))

first_image = image_batch[0]

print(np.min(first_image), np.max(first_image))
data_augmentation = keras.Sequential(

  [

    layers.experimental.preprocessing.RandomFlip("horizontal", 

                                                 input_shape=(180, 

                                                              180,

                                                              3)),

    layers.experimental.preprocessing.RandomRotation(0.1),

    layers.experimental.preprocessing.RandomZoom(0.1),

  ]

)

num_classes = 3



model = Sequential([

  data_augmentation,

  layers.experimental.preprocessing.Rescaling(1./255),

  layers.Conv2D(16, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Conv2D(32, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Conv2D(64, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Dropout(0.4),

  layers.Flatten(),

  layers.Dense(128, activation='relu'),

  layers.Dense(num_classes)

])



from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',patience=2)
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])

model.summary()

history = model.fit(

  train_data,

  validation_data=val_data,

  epochs=5

)

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(5)



plt.figure(figsize=(15, 10))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()

test_dir='../input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19/COVID-19 (109).png'

img = keras.preprocessing.image.load_img(

    test_dir, target_size=(180,180)

)

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0)



predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])



print(

    "This image most likely belongs to {} with a {:.2f} percent confidence."

    .format(class_names[np.argmax(score)], 100 * np.max(score))

)
