!pip install --upgrade --force-reinstall --no-deps kaggle
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import os
for dirname, _, filenames in os.walk('../input/super-ai-image-classification'):
    for filename in filenames:
        print(os.path.join(dirname, filename));

import pathlib
#dataset_url = "https://drive.google.com/drive/folders/1zFy9b0biylRn4kHwtVaJ0GsP1_xBg9QL/train"
#data_dir = tf.keras.utils.get_file('train', origin=dataset_url, untar=True)
data_dir = '../input/super-ai-image-classification/train/train/images/'
data_dir = pathlib.Path(data_dir)
data_dir
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
import pathlib
val_dir = '../input/super-ai-image-classification/val/val/images/'
val_dir = pathlib.Path(val_dir)

val_dir
image_count = len(list(val_dir.glob('*/*.jpg')))
print(image_count)
batch_size = 32
img_height = 180
img_width = 180
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=.95,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  val_dir,
  validation_split=.95,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image)) 
num_classes = 2  #  0, 1 

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
#  layers.Conv2D(128, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(256, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(512, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
!nvidia-smi
epochs=100
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
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
###
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
#  layers.Conv2D(128, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(256, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(512, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
epochs = 100
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
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
#val_url = "/content/drive/My Drive/Colab Notebooks/img-classification/images/test/test/000af6a7-8302-4085-b448-6eae2c1a60e5.jpg
#val_path = tf.keras.utils.get_file('test', origin=val_url)

#list(data_dir.glob('*/*.jpg'))

#list(data_dir.glob('0/*'))
#list(data_dir.glob('1/*'))
#val_path = '/content/drive/My Drive/Colab Notebooks/img-classification/images/train/0/021c1128-f985-4570-b59e-2a0a6b90506b.jpg'
#val_path = '/content/drive/My Drive/Colab Notebooks/img-classification/images/train/1/4cf12f9c-105f-457d-8179-082580a47fa5.jpg'
#val_path = '/content/drive/My Drive/Colab Notebooks/img-classification/images/train/1/00da017a-5ee9-49e5-85c8-c0baf0e466cd.jpg'
#val_path = '/content/drive/My Drive/Colab Notebooks/img-classification/images/train/1/00da017a-5ee9-49e5-85c8-c0baf0e466cd.jpg'

col = ['id','category']
test_df = pd.DataFrame(columns=col)

for val_path in list(val_dir.glob('*/*.jpg')):  #default >> list(val_dir.glob('*/*.jpg'))
    img = keras.preprocessing.image.load_img(
        val_path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
   
    print(str(val_path.name)+'->'+class_names[np.argmax(score)])
   
    new_row = {'id':str(val_path.name),'category':class_names[np.argmax(score)]}
    test_df = test_df.append(new_row,ignore_index=True)
test_df
test_df.to_csv('submission.csv', index=False)