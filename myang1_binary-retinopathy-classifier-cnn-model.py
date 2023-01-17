from tensorflow import lite

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import numpy as np

import pandas as pd

import random, os

import shutil

import matplotlib.pyplot as plt

from matplotlib.image import imread

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.metrics import categorical_accuracy

from sklearn.model_selection import train_test_split
# Add an additional column, mapping to the type

df = pd.read_csv('../input/diabetic-retinopathy-224x224-gaussian-filtered/train.csv')



diagnosis_dict_binary = {

    0: 'No_DR',

    1: 'DR',

    2: 'DR',

    3: 'DR',

    4: 'DR'

}



diagnosis_dict = {

    0: 'No_DR',

    1: 'Mild',

    2: 'Moderate',

    3: 'Severe',

    4: 'Proliferate_DR',

}





df['binary_type'] =  df['diagnosis'].map(diagnosis_dict_binary.get)

df['type'] = df['diagnosis'].map(diagnosis_dict.get)

df.head()
df['type'].value_counts().plot(kind='barh')
df['binary_type'].value_counts().plot(kind='barh')
# Split into stratified train, val, and test sets

train_intermediate, val = train_test_split(df, test_size = 0.15, stratify = df['type'])

train, test = train_test_split(train_intermediate, test_size = 0.15 / (1 - 0.15), stratify = train_intermediate['type'])



print(train['type'].value_counts(), '\n')

print(test['type'].value_counts(), '\n')

print(val['type'].value_counts(), '\n')

# Create working directories for train/val/test

base_dir = ''



train_dir = os.path.join(base_dir, 'train')

val_dir = os.path.join(base_dir, 'val')

test_dir = os.path.join(base_dir, 'test')



if os.path.exists(base_dir):

    shutil.rmtree(base_dir)



if os.path.exists(train_dir):

    shutil.rmtree(train_dir)

os.makedirs(train_dir)



if os.path.exists(val_dir):

    shutil.rmtree(val_dir)

os.makedirs(val_dir)



if os.path.exists(test_dir):

    shutil.rmtree(test_dir)

os.makedirs(test_dir)

# Copy images to respective working directory

src_dir = '../input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images/'

for index, row in train.iterrows():

    diagnosis = row['type']

    binary_diagnosis = row['binary_type']

    id_code = row['id_code'] + ".png"

    srcfile = os.path.join(src_dir, diagnosis, id_code)

    dstfile = os.path.join(train_dir, binary_diagnosis)

    os.makedirs(dstfile, exist_ok = True)

    shutil.copy(srcfile, dstfile)



for index, row in val.iterrows():

    diagnosis = row['type']

    binary_diagnosis = row['binary_type']

    id_code = row['id_code'] + ".png"

    srcfile = os.path.join(src_dir, diagnosis, id_code)

    dstfile = os.path.join(val_dir, binary_diagnosis)

    os.makedirs(dstfile, exist_ok = True)

    shutil.copy(srcfile, dstfile)

 

for index, row in test.iterrows():

    diagnosis = row['type']

    binary_diagnosis = row['binary_type']

    id_code = row['id_code'] + ".png"

    srcfile = os.path.join(src_dir, diagnosis, id_code)

    dstfile = os.path.join(test_dir, binary_diagnosis)

    os.makedirs(dstfile, exist_ok = True)

    shutil.copy(srcfile, dstfile)

# Setting up ImageDataGenerator for train/val/test 



train_path = 'train'

val_path = 'val'

test_path = 'test'



train_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(train_path, target_size=(224,224), shuffle = True)

val_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(val_path, target_size=(224,224), shuffle = True)

test_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(test_path, target_size=(224,224), shuffle = False)

# Building the model



model = tf.keras.Sequential([

    layers.Conv2D(8, (3,3), padding="valid", input_shape=(224,224,3), activation = 'relu'),

    layers.MaxPooling2D(pool_size=(2,2)),

    layers.BatchNormalization(),

    

    layers.Conv2D(16, (3,3), padding="valid", activation = 'relu'),

    layers.MaxPooling2D(pool_size=(2,2)),

    layers.BatchNormalization(),

    

    layers.Conv2D(32, (4,4), padding="valid", activation = 'relu'),

    layers.MaxPooling2D(pool_size=(2,2)),

    layers.BatchNormalization(),

 

    layers.Flatten(),

    layers.Dense(32, activation = 'relu'),

    layers.Dropout(0.15),

    layers.Dense(2, activation = 'softmax')

])



model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-5),

              loss=tf.keras.losses.BinaryCrossentropy(),

              metrics=['acc'])



history = model.fit(train_batches,

                    epochs=12,

                    validation_data=val_batches)
loss, acc = model.evaluate_generator(test_batches, verbose=1)

print("Loss: ", loss)

print("Accuracy: ", acc)

plt.subplot(1,2,1)

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.ylabel('acc')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='lower right')



plt.subplot(1,2,2)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='lower right')

converter = lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

open('model.tflite', 'wb').write(tflite_model)
# Compare TFlite vs. Keras model predictions



interpreter = lite.Interpreter(model_path="model.tflite") 

interpreter.allocate_tensors()



input_details = interpreter.get_input_details() 

output_details = interpreter.get_output_details()



img = random.choice(os.listdir("test/DR/"))

input_shape = input_details[0]['shape']

im = imread("test/DR/" + img)

im = im.reshape(input_shape)



interpreter.set_tensor(input_details[0]['index'], im)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])



print("DR:")

print("TFLite: ", output_data)

print("Keras: ", model.predict(im))



img = random.choice(os.listdir("test/No_DR/"))

input_shape = input_details[0]['shape']

im = imread("test/No_DR/" + img)

im = im.reshape(input_shape)



interpreter.set_tensor(input_details[0]['index'], im)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])



print("\nNo_DR:")

print("TFLite: ", output_data)

print("Keras: ", model.predict(im))



shutil.rmtree(train_dir)

shutil.rmtree(val_dir)

shutil.rmtree(test_dir)