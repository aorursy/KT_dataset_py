import tensorflow as tf

import multiprocessing

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image



#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpu_name = tf.test.gpu_device_name()

#print(gpu_name)
#dog = prime

#cat = select

PATH = os.path.join('..', 'input', 'primenselect2','primenselect') 

train_dir = os.path.join(PATH, 'train')

validation_dir = os.path.join(PATH, 'validation')

# directory with our training cat pictures

train_cats_dir = os.path.join(train_dir, 'select')

# directory with our training dog pictures

train_dogs_dir = os.path.join(train_dir, 'prime')

# directory with our validation cat pictures

validation_cats_dir = os.path.join(validation_dir, 'select')

# directory with our validation dog pictures

validation_dogs_dir = os.path.join(validation_dir, 'prime')
num_cats_tr = len(os.listdir(train_cats_dir))

num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))

num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr

total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)

print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)

print('total validation dog images:', num_dogs_val)

print('—')

print('Total training images:', total_train)

print('Total validation images:', total_val)
batch_size = 1#128

epochs = 1 #15

IMG_HEIGHT = 150

IMG_WIDTH = 150
# Generator for our training data

train_image_generator = ImageDataGenerator(rescale=1./255)

# Generator for our validation data

validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(

    batch_size=batch_size, 

    directory=train_dir,

    shuffle=True,

    target_size=(IMG_HEIGHT, IMG_WIDTH),

    class_mode='binary')



val_data_gen = validation_image_generator.flow_from_directory(

batch_size=batch_size,

directory=validation_dir,

target_size=(IMG_HEIGHT, IMG_WIDTH),

class_mode='binary')
sample_training_images, _ = next(train_data_gen)

def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()

#plotImages(sample_training_images[:5])
model = Sequential([

    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

    MaxPooling2D(),

    Conv2D(32, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(1)

])



# Compile model

model.compile(

    optimizer='adam',

    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

    metrics=['accuracy'])



# Train the model

history = model.fit_generator(

    train_data_gen,

    steps_per_epoch=55, #total_train // batch_size

    epochs=epochs,

    validation_data=val_data_gen,

    validation_steps=19 #total_v

)
"""

def attempt(l1,l2,l3,l11,l12,l13,s1,s2,s3,s4):

    model = 0

    model = Sequential([

        Conv2D(l1, l11, padding='same', activation=s1, input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

        MaxPooling2D(),

        Conv2D(l2, l12, padding='same', activation=s2),

        MaxPooling2D(),

        Conv2D(l3, l13, padding='same', activation=s3),

        MaxPooling2D(),

        Flatten(),

        Dense(512, activation=s4),

        Dense(1)

    ])



    # Compile model

    model.compile(

        optimizer='adam',

        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

        metrics=['accuracy'])



    # Train the model

    history = model.fit_generator(

        train_data_gen,

        steps_per_epoch=55, #total_train // batch_size

        epochs=epochs,

        validation_data=val_data_gen,

        validation_steps=19 #total_v

    )

"""

#@@用很多for循环把attempt函数用各种变量的排列组合都跑一遍，找到'accuracy'最高的存入winner
"""

maxacc=0

llist = [16,32,64]

l1list = [1,5,9]

slist = ['relu','sigmoid']

for l1 in llist:

    for l2 in llist:

        for l3 in llist:

            for l11 in l1list:

                for l12 in l1list:

                    for l13 in l1list:

                        for s1 in slist: 

                            for s2 in slist: 

                                for s3 in slist: 

                                    for s4 in slist: 

                                        print(l1,l2,l3,l11,l12,l13,s1,s2,s3,s4)

                                        attempt(l1,l2,l3,l11,l12,l13,s1,s2,s3,s4)

                                        if maxacc<(history.history['accuracy'][0]):

                                            maxacc=history.history['accuracy'][0]

                                            winner=[l1,l2,l3,l11,l12,l13,s1,s2,s3,s4,maxacc]

                                            print("%%")

                                            

                                            """
# Model summary

#model.summary()

#attempt(16,16,16,3,3,3,'relu','relu','relu','relu')

#print(winner)
#img = Image.open('../input/primenselect1/primenselect/train/prime/0001111001970.jfif')
# Predict



#results = model.predict(sample_training_images[:5])
model.save('my_model.h5')

# 重新创建完全相同的模型，包括其权重和优化程序

new_model = tf.keras.models.load_model('my_model.h5')

# 显示网络结构

new_model.summary()
!pip install tensorflowjs
!tensorflowjs_converter --input_format=keras ./my_model.h5 tfjs_model
##!tensorflowjs_converter --input_format=keras ../input/dog-or-cat-image-detection-with-transfer-learning/dog_or_cat_with_inceptionv3.h5 inception
!zip -r beef150.zip tfjs_model
#import tensorflowjs as tfjs
#tfjs.converters.save_keras_model(model, "tfjsmodel")

#tfjs.converters.save_keras_model(model, "tfjsnewmodel")
#!ls -sh
!sha1sum beef150.zip
