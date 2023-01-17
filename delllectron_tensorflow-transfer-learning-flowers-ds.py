import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

import os
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras.layers as layers

import PIL
import PIL.Image as pim

from sklearn.model_selection import train_test_split
#get the location of images
base_url = '../input/flowers-recognition/flowers'
CATEGORIES = 'daisy dandelion rose sunflower tulip'.split()
files_count = []
for i,f in enumerate(CATEGORIES):
    folder_path = os.path.join(base_url, f)
    for path in os.listdir(os.path.join(folder_path)):
        files_count.append(['{}/{}'.format(folder_path,path), f, i])
flowers_df = pd.DataFrame(files_count, columns=['filepath', 'class_name', 'label'])
flowers_df.head()
flowers_df.class_name.value_counts()
SAMPLE_PER_CATEGORY = 500
flowers_df = pd.concat([flowers_df[flowers_df['class_name']== i][:SAMPLE_PER_CATEGORY] for i in CATEGORIES])
flowers_df.class_name.value_counts()
pim.open(flowers_df.filepath[0])
#split the data
X = flowers_df['filepath']
y = flowers_df['label']

train, test, label_train, label_test = train_test_split(X, y, test_size=0.2, random_state=101)
#convert data as a tensor
train_paths = tf.convert_to_tensor(train.values, dtype=tf.string)
train_labels = tf.convert_to_tensor(label_train.values)

test_paths = tf.convert_to_tensor(test.values, dtype=tf.string)
test_labels = tf.convert_to_tensor(label_test.values)
#create a tensor data set
train_data = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
test_data = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
#function to load the image and convert them as an array
def map_fn(path, label):
    image = tf.image.decode_jpeg(tf.io.read_file(path))

    return image, label

#apply the function
train_data = train_data.map(map_fn)
test_data = test_data.map(map_fn)
fig, ax = plt.subplots(1,2, figsize = (15,5))
for i,l in train_data.take(1):
    ax[0].set_title('SAMPLE IMAGE FROM TRAIN DATA');
    ax[0].imshow(i);
for i,l in test_data.take(1):
    ax[1].set_title('SAMPLE IMAGE FROM TEST DATA');
    ax[1].imshow(i);
IMAGE_SIZE = 150
#image preprocessing
def preprocessing(image, label):
    """
    returns a image that is reshaped and normalized
    """
    image = tf.cast(image, tf.float32)
    image = image / 255.
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    
    return image, label

#apply the function
train_data = train_data.map(preprocessing)
test_data = test_data.map(preprocessing)
#show the processed images
fig, ax = plt.subplots(1,2, figsize = (15,5))
for i,l in train_data.take(1):
    ax[0].set_title('SAMPLE IMAGE FROM TRAIN DATA');
    ax[0].imshow(i);
for i,l in test_data.take(1):
    ax[1].set_title('SAMPLE IMAGE FROM TEST DATA');
    ax[1].imshow(i);
#batch the images
BATCH_SIZE = 32

train_batches = train_data.batch(BATCH_SIZE)
test_batches = test_data.batch(BATCH_SIZE)

for i, l in train_batches.take(1):
    print('Train Data Shape',i.shape)
for i, l in test_batches.take(1):
    print('Test Data Shape',i.shape)
#define input shape
inp_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

#load a pretrained model for feature map extraction of images
base_model = tf.keras.applications.InceptionV3(input_shape=inp_shape,
                                               include_top=False,
                                               weights='imagenet')
base_model.summary()
#let's try to pass an image to the model to verify the output shape
for i,l in train_batches.take(1):
    pass
base_model(i).shape
#disable the training property (we dont want to change the convolutional base that was already trained)
base_model.trainable = False
model = tf.keras.models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(len(CATEGORIES), activation = 'sigmoid'))
model.summary()
#compile the model
model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
#fit the model
model.fit(train_batches,
          epochs=20,
          validation_data=(test_batches))
#lets create a function to communicate with the model

def predict(filepath, model):
    #image processing
    img = tf.image.decode_image(tf.io.read_file(filepath))
    img = tf.cast(img, tf.float32)
    img = img / 255.
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    
    #convert to tensor
    img = tf.convert_to_tensor(img)
    img = tf.expand_dims(img, axis=0)
    
    #make a prediction
    prediction = model.predict_classes(img)[0]
    return ("THAT'S A " + str(CATEGORIES[prediction])).capitalize()
#test the model
predict(test.iloc[12], model)