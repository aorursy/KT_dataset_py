import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import glob
print('Tensorflow version: {} GPUing:{}'.format(tf.__version__,tf.test.is_gpu_available()))
train_image_path = glob.glob('../input/data/*')

#this func is loading train image path

print('picture number:{}'.format(len(train_image_path)))
train_image_label = [int(train_image_path[0].split('.')[-1]== 'jpg') for p in train_image_path]

#We need to label the iamge
def load_preprosess_image(path, label):

    image = tf.io.read_file(path)

    #load file path

    image = tf.image.decode_jpeg(image, channels=3)

    #load jpeg,because jpeg's channels is 3

    image = tf.image.resize(image, [256, 256])

    #load jpeg and than,we must resize

    image = tf.image.random_flip_left_right(image)

    #Data augmentation

    image = tf.image.random_flip_up_down(image)

    image = tf.image.random_brightness(image, 0.5)

    image = tf.image.random_contrast(image, 0, 1)

    

    #uint8 cast float32

    image = tf.cast(image, tf.float32)

    image = image/255

    return image, label
#The tf.data.Dataset API supports writing descriptive and efficient input pipelines. Dataset usage follows a common pattern:



#Create a source dataset from your input data.

#Apply dataset transformations to preprocess the data.

#Iterate over the dataset and process the elements.

#Iteration happens in a streaming fashion, so the full dataset does not need to fit into memory.



train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))



#Speed up data stream input to memory

AUTOTUNE = tf.data.experimental.AUTOTUNE



#map is function

train_image_ds = train_image_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)
for img, label in train_image_ds.take(1):

    plt.imshow(img)

    print(label)

#show image
BATCH_SIZE = 16

train_count = len(train_image_path)
train_image_ds = train_image_ds.shuffle(train_count).repeat().batch(BATCH_SIZE)
covn_base  = tf.keras.applications.MobileNetV2(

                    weights='imagenet', 

                    include_top=False, 

                    input_shape=(256, 256, 3)

)

#covn_base model
covn_base.summary()
model = tf.keras.Sequential()

model.add(covn_base)

model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(512, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#model 
model.summary()
#omtimizer and loss

model.compile(optimizer=tf.keras.optimizers.Adam(),

              loss='binary_crossentropy',

              metrics=['acc'])
history = model.fit(

    train_image_ds,

    steps_per_epoch=train_count//BATCH_SIZE,

    epochs=3)

#train
plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('loss'), label='loss')
def load_image(path):

    image = tf.io.read_file(path)

    #load file path

    image = tf.image.decode_jpeg(image, channels=3)

    #load jpeg,because jpeg's channels is 3

    image = tf.image.resize(image, [256, 256])

    #load jpeg and than,we must resize

    image = tf.image.random_flip_left_right(image)

    #Data augmentation

    image = tf.image.random_flip_up_down(image)

    image = tf.image.random_brightness(image, 0.5)

    image = tf.image.random_contrast(image, 0, 1)

    

    #uint8 cast float32

    image = tf.cast(image, tf.float32)

    #

    image = tf.expand_dims(image, 0)

    image = image/255

    return image
#prediction image

img = load_image(train_image_path[5])
plt.imshow(img[0])

plt.show()
float(model.predict(img)[0])

#one is true