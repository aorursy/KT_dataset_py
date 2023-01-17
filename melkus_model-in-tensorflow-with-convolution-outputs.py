import os

import zipfile

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

%matplotlib inline
train_cats_dir = os.path.join('../input/training_set/training_set/cats')

train_dogs_dir = os.path.join('../input/training_set/training_set/dogs')



test_cats_dir = os.path.join('../input/test_set/test_set/cats')

test_dogs_dir = os.path.join('../input/test_set/test_set/dogs')
train_cats_names = os.listdir(train_cats_dir)

print(train_cats_names[:10])
train_dogs_names = os.listdir(train_dogs_dir)

print(train_dogs_names[:10])
test_cats_names = os.listdir(test_cats_dir)

print(test_cats_names[:10])
test_cats_names = os.listdir(test_cats_dir)

print(test_cats_names[:10])
print('total training cats images:', len(os.listdir(train_cats_dir)))

print('total training dogs images:', len(os.listdir(train_dogs_dir)))
print('total test cats images:', len(os.listdir(test_cats_dir)))

print('total test dogs images:', len(os.listdir(test_dogs_dir)))
rows = 4

cols = 4

index = 0



fig = plt.gcf()

fig.set_size_inches(cols * 4, rows * 4)



index += 8

next_cats_pix = [os.path.join(train_cats_dir, fname) 

                for fname in train_cats_names[index-8:index]]

next_dogs_pix = [os.path.join(train_dogs_dir, fname) 

                for fname in train_dogs_names[index-8:index]]



for i, img_path in enumerate(next_cats_pix+next_dogs_pix):

  sp = plt.subplot(rows, cols, i + 1)

  sp.axis('Off')



  img = mpimg.imread(img_path)

  plt.imshow(img)



plt.show()
model = tf.keras.models.Sequential([

    

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),





    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),





    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),





    tf.keras.layers.Flatten(),



    tf.keras.layers.Dense(512, activation='relu'),

  

    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.summary()
model.compile(loss='binary_crossentropy',

              optimizer=RMSprop(lr=0.001),

              metrics=['acc'])
train_datagen = ImageDataGenerator(rescale=1/255,

                                   rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True,

                                   fill_mode='nearest')





train_generator = train_datagen.flow_from_directory(

        '../input/training_set/training_set',  

        target_size=(150, 150),  

        batch_size=128,

        class_mode='binary')



validation_datagen = ImageDataGenerator(rescale=1/255)



validation_generator = validation_datagen.flow_from_directory(

        '../input/test_set/test_set',  

        target_size=(150, 150),  

        batch_size=32,

        class_mode='binary')
history = model.fit_generator(

      train_generator, 

      epochs=20,

      verbose=1,

      validation_data = validation_generator)
import numpy as np

import random

from tensorflow.keras.preprocessing.image import img_to_array, load_img



successive_outputs = [layer.output for layer in model.layers[1:]]



visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)



cats_img_files = [os.path.join(train_cats_dir, f) for f in train_cats_names]

dogs_img_files = [os.path.join(train_dogs_dir, f) for f in train_dogs_names]

img_path = random.choice(cats_img_files + dogs_img_files)



img = load_img(img_path, target_size=(150, 150))  

x = img_to_array(img)  

x = x.reshape((1,) + x.shape) 



x /= 255





successive_feature_maps = visualization_model.predict(x)



layer_names = [layer.name for layer in model.layers]





for layer_name, feature_map in zip(layer_names, successive_feature_maps):

  if len(feature_map.shape) == 4:

    

    n_features = feature_map.shape[-1] 

    size = feature_map.shape[1]

    display_grid = np.zeros((size, size * n_features))

   

    for i in range(n_features):

      x = feature_map[0, :, :, i]

      x -= x.mean()

      x /= x.std()

      x *= 64

      x += 128

      x = np.clip(x, 0, 255).astype('uint8')

      

      display_grid[:, i * size : (i + 1) * size] = x

    

    scale = 20. / n_features

    plt.figure(figsize=(scale * n_features, scale))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')
acc      = history.history['acc']

val_acc  = history.history['val_acc']

loss     = history.history['loss']

val_loss = history.history['val_loss']

epochs   = range(len(acc))
plt.plot  ( epochs, acc )

plt.plot  ( epochs, val_acc )

plt.title ('Training and Validation accuracy')

plt.figure()
plt.plot  ( epochs, loss )

plt.plot  ( epochs, val_loss )

plt.title ('Training and Validation loss')

plt.figure()