import zipfile

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
base_dir = '../input/cats_and_dogs_filtered'

train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'validation')
# Directory with our training cat pictures

train_cats_dir = os.path.join(train_dir, 'cats')



# Directory with our training dog pictures

train_dogs_dir = os.path.join(train_dir, 'dogs')



# Directory with our validation cat pictures

validation_cats_dir = os.path.join(validation_dir, 'cats')



# Directory with our validation dog pictures

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
train_cat_fnames = os.listdir( train_cats_dir )

train_dog_fnames = os.listdir( train_dogs_dir )



print(train_cat_fnames[:10])

print(train_dog_fnames[:10])
print('total training cat images :', len(os.listdir(train_cats_dir)))

print('total training dog images :', len(os.listdir(train_dogs_dir)))



print('total validation cat images :', len(os.listdir(validation_cats_dir)))

print('total validation dog images :', len(os.listdir(validation_dogs_dir)))
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.compile(loss='binary_crossentropy',

              optimizer=RMSprop(lr=1e-4),

              metrics=['accuracy'])
#Now instead of the ImageGenerator just rescaling

# the image, we also rotate and do other operations

# Updated to do image augmentation

train_datagen = ImageDataGenerator(

      rescale=1./255,

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
# Flow training images in batches of 20 using train_datagen generator

train_generator = train_datagen.flow_from_directory(

        train_dir,  # This is the source directory for training images

        target_size=(150, 150),  # All images will be resized to 150x150

        batch_size=20,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')
#os.listdir(filename)
# Flow validation images in batches of 20 using test_datagen generator

validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')
%matplotlib inline



import matplotlib.image as mpimg

import matplotlib.pyplot as plt



# Parameters for our graph; we'll output images in a 4x4 configuration

nrows = 4

ncols = 4



pic_index = 0 # Index for iterating over images
# Set up matplotlib fig, and size it to fit 4x4 pics

fig = plt.gcf()

fig.set_size_inches(ncols*4, nrows*4)



pic_index+=8



next_cat_pix = [os.path.join(train_cats_dir, fname) 

                for fname in train_cat_fnames[ pic_index-8:pic_index] 

               ]



next_dog_pix = [os.path.join(train_dogs_dir, fname) 

                for fname in train_dog_fnames[ pic_index-8:pic_index]

               ]



for i, img_path in enumerate(next_cat_pix+next_dog_pix):

  # Set up subplot; subplot indices start at 1

  sp = plt.subplot(nrows, ncols, i + 1)

  sp.axis('Off') # Don't show axes (or gridlines)



  img = mpimg.imread(img_path)

  plt.imshow(img)



plt.show()

history = model.fit(

      train_generator,

      steps_per_epoch=100,  # 2000 images = batch_size * steps

      epochs=100,

      validation_data=validation_generator,

      validation_steps=50,  # 1000 images = batch_size * steps

      verbose=2)
print("Model Accuracy - " , model.evaluate_generator(validation_generator)[1]*100 , "%")
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'b', label='Training Accuracy',color='black')

plt.plot(epochs, val_acc, 'b', label='Validation Accuracy',color='gray')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'b', label='Training Loss',color='black')

plt.plot(epochs, val_loss, 'b', label='Validation Loss',color='gray')

plt.title('Training and Validation Loss')

plt.legend()



plt.show()