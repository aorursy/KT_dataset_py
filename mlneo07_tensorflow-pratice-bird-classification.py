import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import tensorflow as tf

import cv2

from PIL import Image

import matplotlib.pyplot as plt
os.listdir('../input/100-bird-species')

BASE_DIR ='../input/100-bird-species/'

TRAIN_DIR = os.path.join(BASE_DIR,'train')

TEST_DIR = os.path.join(BASE_DIR,'test')

VALID_DIR = os.path.join(BASE_DIR,'valid')

CONSOLIDATED = os.path.join(BASE_DIR, 'consolidated')
BATCH_SIZE=32

IMAGE_SIZE=[112,112]

AUTOTUNE = tf.data.experimental.AUTOTUNE
TCLASS_NAME = np.array([item for item in os.listdir(TEST_DIR)]) #classnames @train

VCLASS_NAME = np.array([item for item in os.listdir(VALID_DIR)]) #VLIDATION CLASS NAMES

CONSO_CLASS_NAME = np.array([item for item in os.listdir(CONSOLIDATED)])

print('Total Numbers of Training calsses:',len(TCLASS_NAME))

print('Total Number of Valid calsses:',len(VCLASS_NAME))
TRAIN_LS_DS = tf.data.Dataset.list_files(str(TRAIN_DIR +'/*/*')) #LIST ALL THE TRAINING FILES

VALID_LS_DS = tf.data.Dataset.list_files(str(VALID_DIR+'/*/*')) #LIST OF ALL VLAIDATION FILES

CONSO_LS_DS = tf.data.Dataset.list_files(str(CONSOLIDATED+'/*/*'))

TEST_LS_DS = tf.data.Dataset.list_files(str(TEST_DIR+'/*/*'))
TOTAL_DS = TRAIN_LS_DS.concatenate(CONSO_LS_DS)

TOTAL_DS = TOTAL_DS.concatenate(VALID_LS_DS)
#len(list(TOTAL_DS.as_numpy_iterator()))

DATASET_SIZE = 58006



#sice our total data size is 58006 and then spliting it into train and valid files



train_size = int(0.7 * DATASET_SIZE)

val_size = int(0.15 * DATASET_SIZE)
'''for i in range(1,9):

    print(TRAIN_DIR + '/'+ TCLASS_NAME[i] + f'/00{i}.jpg')'''
w=10

h=10

fig=plt.figure(figsize=(15, 15))

columns = 3

rows = 3

for i in range(1, columns*rows +1):

    img = TOTAL_DS.take(20)

    fig.add_subplot(rows, columns, i)

    k = cv2.imread(TRAIN_DIR + '/'+ TCLASS_NAME[i] + f'/00{i}.jpg')

    plt.imshow(k)

plt.show()
def get_label(file_path):

    parts =tf.strings.split(file_path, os.path.sep)

    return parts[-2] == TCLASS_NAME



def decode_img(img):

    img = tf.image.decode_jpeg(img,3)

    img = tf.image.convert_image_dtype(img,tf.float32)

    return tf.image.resize(img, ([*IMAGE_SIZE]))

def augment(img):

    img = decode_img(img)

    img = tf.image.random_flip_left_right(img)

    img = tf.image.random_flip_up_down(img)

    img = tf.image.random_crop(img ,(*IMAGE_SIZE,3))

    return img

def process_path(file_path):

    label = get_label(file_path)

    img =tf.io.read_file(file_path)

    img = augment(img)

    return img, label
DS= TOTAL_DS.map(process_path, num_parallel_calls=AUTOTUNE)
for image, label in DS.take(1):

  print("Image shape: ", image.numpy().shape)

  print("Label: ", label.numpy())
def get_dataset(ds, cache=True):

    ds = ds.cache()

    ds = ds.shuffle(1000)

    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds
full_train_ds = get_dataset(DS)



train_dataset = full_train_ds.take(train_size)

valid_dataset = full_train_ds.take(val_size)



image_batch, label_batch = next(iter(train_dataset))

def show_batch(image_batch, label_batch):

  plt.figure(figsize=(10,10))

  for n in range(25):

      ax = plt.subplot(5,5,n+1)

      plt.imshow(image_batch[n])

      plt.title(TCLASS_NAME[label_batch[n]==1][0].title())

      plt.axis('off')
show_batch(image_batch.numpy(), label_batch.numpy())
base_model_2 = tf.keras.applications.DenseNet121(

    include_top=False, weights='imagenet', input_shape=(112,112,3))
'''base_model_1 = tf.keras.applications.ResNet101(input_shape=(*IMAGE_SIZE,3),include_top = False,

                                             weights ='imagenet')

base_model_1.trainable = True'''



'''model = tf.keras.Sequential([base_model_1,

                            tf.keras.layers.GlobalAveragePooling2D(),

                            tf.keras.layers.Dense(len(TCLASS_NAME), activation='softmax')])

model.summary()'''



'''base_learning_rate = 0.001

model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),

             loss =tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])'''



'''loss0,accuracy0 = model.evaluate(valid_dataset, steps = validation_steps)'''



'''history = model.fit(train_dataset,

                    epochs=initial_epochs,steps_per_epoch = 20,

                    validation_data=valid_dataset,validation_steps = 10)'''
initial_epochs = 20

validation_steps=10

base_learning_rate = 0.0001
model_2 = tf.keras.Sequential([base_model_2,

                            tf.keras.layers.GlobalAveragePooling2D(),

                            tf.keras.layers.Dense(len(TCLASS_NAME), activation='softmax')])
model_2.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),

             loss =['categorical_crossentropy'],metrics=['accuracy'])
history_1 = model_2.fit(train_dataset,

                    epochs=initial_epochs,steps_per_epoch = initial_epochs,

                    validation_data=valid_dataset,validation_steps = validation_steps)
acc = history_1.history['accuracy']

val_acc = history_1.history['val_accuracy']



loss = history_1.history['loss']

val_loss = history_1.history['val_loss']



plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.ylim([min(plt.ylim()),1])

plt.title('Training and Validation Accuracy')
#fine tuning the model
len(base_model_2.layers)
fine_tune = 201

base_model_2.trainable = True

for layer in base_model_2.layers[:fine_tune]:

    layer.trainable = False
model_2.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),

             loss =['categorical_crossentropy'],

             metrics=['accuracy'])
fine_tune_epochs = 20

total_epochs =  initial_epochs + fine_tune_epochs



history_fine = model_2.fit(train_dataset,

                         epochs=total_epochs,

                         initial_epoch =  history_1.epoch[-1],

                         validation_data=valid_dataset,

                        steps_per_epoch=45,

                    validation_steps=15)
def proces_test_path(filename):

        img = tf.io.read_file(filename)

        img = tf.image.decode_jpeg(img)

        return (img)

    
TEST_DS = TEST_LS_DS.map(proces_test_path, num_parallel_calls=AUTOTUNE)