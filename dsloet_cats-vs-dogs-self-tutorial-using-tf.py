
!unzip /kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip
%matplotlib inline
import os
from glob import glob
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
import shutil
images_path = '../working/train/'
pets = os.listdir(images_path)
print(pets[0])
print(len(pets))

# multipleImages = glob('../working/train/**')
# def plotImages2():
#     r = random.sample(multipleImages, 9)
#     plt.figure(figsize=(20,20))
#     plt.subplot(331)
#     plt.imshow(cv2.imread(r[0])); plt.axis('off')
#     plt.subplot(332)
#     plt.imshow(cv2.imread(r[1])); plt.axis('off')
#     plt.subplot(333)
#     plt.imshow(cv2.imread(r[2])); plt.axis('off')
#     plt.subplot(334)
#     plt.imshow(cv2.imread(r[3])); plt.axis('off')
#     plt.subplot(335)
#     plt.imshow(cv2.imread(r[4])); plt.axis('off')
#     plt.subplot(336)
#     plt.imshow(cv2.imread(r[5])); plt.axis('off')
#     plt.subplot(337)
#     plt.imshow(cv2.imread(r[6])); plt.axis('off')
#     plt.subplot(338)
#     plt.imshow(cv2.imread(r[7])); plt.axis('off')
#     plt.subplot(339)
#     plt.imshow(cv2.imread(r[8])); plt.axis('off')
# plotImages2()
print(tf.__version__)
# create dir and move pics to their own folder
os.mkdir(os.path.join(images_path, "dogs"))
os.mkdir(os.path.join(images_path, "cats"))
cats = os.path.join(images_path, "cats")
dogs = os.path.join(images_path, "dogs")

for file in os.listdir(images_path):
    print(file)
    label = str.split(file, ".")[0]
    if label == "dog":
        shutil.move(os.path.join(images_path, file), dogs)
        
    if label == "cat":
        shutil.move(os.path.join(images_path, file), cats)
   
class_names = np.array(["dogs", "cats"])
batch_size = 128
img_height = 118
img_width = 118
image_count = len(pets)
print(image_count)
str(images_path+'*')
list_ds = tf.data.Dataset.list_files(str(images_path+'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
for f in list_ds.take(5):
    print(f.numpy())
# train test split
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)
print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())
def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)

def decode_img(img):
    
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])

for f in list_ds.take(1):
    parts = tf.strings.split(f, os.path.sep)
    print(parts)
    one_hot = parts[-2] == class_names
    print(one_hot)
#     path = tf.strings.split(parts[2], os.path.sep)
# #     print(path[3])
#     one_hot = path[3] == class_names
#     print(one_hot)

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
image, label = next(iter(train_ds))
print(label.numpy())

for image, label in train_ds.take(5):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
image_batch, label_batch = next(iter(train_ds))
print(label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    label = label_batch[i]
    plt.title(class_names[label])
    plt.axis("off")
from tensorflow.keras import layers

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image)) 
num_classes = 2

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])
model.compile(
  optimizer='adam',
  loss=tf.losses.BinaryCrossentropy(from_logits=False),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=30
)

