!pip install tensorflow-gpu==2.0.0-beta1
import tensorflow as tf
print(tf.__version__)


# small data 2000

import os

import zipfile

import random

import numpy as np

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile
import urllib.request

urllib.request.urlretrieve("https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip","cats_and_dogs_filtered.zip")
local_zip = 'cats_and_dogs_filtered.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('../output/')
cwd = os.getcwd()

print(cwd)
!ls
import os

os.listdir('../output/cats_and_dogs_filtered')
import os

os.listdir("../output/cats_and_dogs_filtered/train")
base_dir = "../output/cats_and_dogs_filtered"



train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'validation')



# Directory with our training cat/dog pictures

train_cats_dir = os.path.join(train_dir, 'cats')

train_dogs_dir = os.path.join(train_dir, 'dogs')



# Directory with our validation cat/dog pictures

validation_cats_dir = os.path.join(validation_dir, 'cats')

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
train_cat_fnames = os.listdir( train_cats_dir )

train_dog_fnames = os.listdir( train_dogs_dir )



print(train_cat_fnames[:10])

print(train_dog_fnames[:10])


print('total training cat images :', len(os.listdir(      train_cats_dir ) ))

print('total training dog images :', len(os.listdir(      train_dogs_dir ) ))



print('total validation cat images :', len(os.listdir( validation_cats_dir ) ))

print('total validation dog images :', len(os.listdir( validation_dogs_dir ) ))
%matplotlib inline



import matplotlib.image as mpimg

import matplotlib.pyplot as plt



# Parameters for our graph; we'll output images in a 4x4 configuration

nrows = 2

ncols = 2



pic_index = 0 # Index for iterating over images

# Set up matplotlib fig, and size it to fit 2x2 pics

fig = plt.gcf()

fig.set_size_inches(ncols*2, nrows*2)



pic_index+=4



next_cat_pix = [os.path.join(train_cats_dir, fname) 

                for fname in train_cat_fnames[ pic_index-4:pic_index] 

               ]





for i, img_path in enumerate(next_cat_pix):

  # Set up subplot; subplot indices start at 1

  sp = plt.subplot(nrows, ncols, i + 1)

  sp.axis('Off') # Don't show axes (or gridlines)



  img = mpimg.imread(img_path)

  plt.imshow(img)



plt.show()




next_dog_pix = [os.path.join(train_dogs_dir, fname) 

                for fname in train_dog_fnames[ pic_index-4:pic_index]

               ]



for i, img_path in enumerate(next_dog_pix):

  # Set up subplot; subplot indices start at 1

  sp = plt.subplot(nrows, ncols, i + 1)

  sp.axis('Off') # Don't show axes (or gridlines)



  img = mpimg.imread(img_path)

  plt.imshow(img)



plt.show()
# clean data

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# All images will be rescaled by 1./255.

train_datagen = ImageDataGenerator( rescale = 1.0/255. )

test_datagen  = ImageDataGenerator( rescale = 1.0/255. )



# --------------------

# Flow training images in batches of 20 using train_datagen generator

# --------------------

train_generator = train_datagen.flow_from_directory('../output/cats_and_dogs_filtered/train',

                                                    batch_size=20,

                                                    class_mode='binary',

                                                    target_size=(150, 150))     

# --------------------

# Flow validation images in batches of 20 using test_datagen generator

# --------------------

validation_generator =  test_datagen.flow_from_directory('../output/cats_and_dogs_filtered/validation',

                                                         batch_size=20,

                                                         class_mode  = 'binary',

                                                         target_size = (150, 150))



# define model

import tensorflow as tf



model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 150x150 with 3 bytes color

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2), 

    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 

    #tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(), 

    # 512 neuron hidden layer

    tf.keras.layers.Dense(128, activation='relu'), 

    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')

    tf.keras.layers.Dense(1, activation='sigmoid')  

])
# model summary    

model.summary()


# compile model    

from tensorflow.keras.optimizers import RMSprop



model.compile(optimizer=RMSprop(lr=0.001),

              loss='binary_crossentropy',

              metrics = ['acc'])
# train model

# Note that this may take some time.

# use model.fit_generator not model.fit  because the data is generator data

history = model.fit_generator(train_generator,

                              validation_data=validation_generator,

                              steps_per_epoch=100,

                              epochs=10,

                              validation_steps=50,

                              verbose=1)


# model result

#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

import matplotlib.image  as mpimg

import matplotlib.pyplot as plt

acc      = history.history[     'acc' ]

val_acc  = history.history[ 'val_acc' ]

loss     = history.history[    'loss' ]

val_loss = history.history['val_loss' ]



epochs   = range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot  ( epochs,     acc )

plt.plot  ( epochs, val_acc )

plt.title ('Training and validation accuracy')

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot  ( epochs,     loss )

plt.plot  ( epochs, val_loss )

plt.title ('Training and validation loss'   )



# evaluate model



model.evaluate(validation_generator)


# small data 2000

import os

import zipfile

import random

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile



# clean data with Augmentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# All images will be rescaled by 1./255.

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



# --------------------

# Flow training images in batches of 20 using train_datagen generator

# --------------------

train_generator = train_datagen.flow_from_directory('../output/cats_and_dogs_filtered/train',

                                                    batch_size=20,

                                                    class_mode='binary',

                                                    target_size=(150, 150))     

# --------------------

# Flow validation images in batches of 20 using test_datagen generator

# --------------------

validation_generator =  test_datagen.flow_from_directory('../output/cats_and_dogs_filtered/validation',

                                                         batch_size=20,

                                                         class_mode  = 'binary',

                                                         target_size = (150, 150))




# define model

import tensorflow as tf



model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 150x150 with 3 bytes color

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2), 

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 

    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(), 

    # 512 neuron hidden layer

    tf.keras.layers.Dense(256, activation='relu'), 

    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')

    tf.keras.layers.Dense(1, activation='sigmoid')  

])

# model summary    

model.summary()

        


# compile model    

from tensorflow.keras.optimizers import RMSprop



model.compile(optimizer=RMSprop(lr=0.001),

              loss='binary_crossentropy',

              metrics = ['acc'])
# train model

# Note that this may take some time.

# use model.fit_generator not model.fit  because the data is generator data

history = model.fit_generator(train_generator,

                              validation_data=validation_generator,

                              steps_per_epoch=100,

                              epochs=15,

                              validation_steps=50,

                              verbose=1)



# model result

#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

import matplotlib.image  as mpimg

import matplotlib.pyplot as plt

acc      = history.history[     'acc' ]

val_acc  = history.history[ 'val_acc' ]

loss     = history.history[    'loss' ]

val_loss = history.history['val_loss' ]



epochs   = range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot  ( epochs,     acc )

plt.plot  ( epochs, val_acc )

plt.title ('Training and validation accuracy')

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot  ( epochs,     loss )

plt.plot  ( epochs, val_loss )

plt.title ('Training and validation loss'   )



# evaluate model



model.evaluate(validation_generator)



import os



from tensorflow.keras import layers

from tensorflow.keras import Model









# small data 2000

import os

import zipfile

import random

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile

from tensorflow.keras import Model

from tensorflow.keras.applications.inception_v3 import InceptionV3



# clean data with Augmentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# All images will be rescaled by 1./255.

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



# --------------------

# Flow training images in batches of 20 using train_datagen generator

# --------------------

train_generator = train_datagen.flow_from_directory('../output/cats_and_dogs_filtered/train',

                                                    batch_size=20,

                                                    class_mode='binary',

                                                    target_size=(150, 150))     

# --------------------

# Flow validation images in batches of 20 using test_datagen generator

# --------------------

validation_generator =  test_datagen.flow_from_directory('../output/cats_and_dogs_filtered/validation',

                                                         batch_size=20,

                                                         class_mode  = 'binary',

                                                         target_size = (150, 150))
# download model weight

import urllib.request

urllib.request.urlretrieve("https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",'../output/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')



#!wget --no-check-certificate \

#    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 
!ls
local_weights_file = '../output/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 

                                include_top = False, 

                                weights = None)



pre_trained_model.load_weights(local_weights_file)



for layer in pre_trained_model.layers:

  layer.trainable = False

  

pre_trained_model.summary()
for i, layer in enumerate(pre_trained_model.layers): 

    print(i, layer.name)


last_layer = pre_trained_model.get_layer('mixed7')

print('last layer output shape: ', last_layer.output_shape)

last_output = last_layer.output
from tensorflow.keras.optimizers import RMSprop



# Flatten the output layer to 1 dimension

x = layers.Flatten()(last_output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation

x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2

x = layers.Dropout(0.2)(x)                  

# Add a final sigmoid layer for classification

x = layers.Dense  (1, activation='sigmoid')(x)           



model = Model( pre_trained_model.input, x) 



model.compile(optimizer = RMSprop(lr=0.0001), 

              loss = 'binary_crossentropy', 

              metrics = ['acc'])
model.summary()
history = model.fit_generator(

            train_generator,

            validation_data = validation_generator,

            steps_per_epoch = 100,

            epochs = 10,

            validation_steps = 50,

            verbose = 1)
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()
# download data

import urllib.request

urllib.request.urlretrieve("https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip","../output/rps.zip")

urllib.request.urlretrieve("https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip","../output/rps-test-set.zip")

import os

cwd = os.getcwd()

print(cwd)
!ls
import os

import zipfile

data_location='../output'





local_zip = data_location+'/rps.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall(data_location)

zip_ref.close()



local_zip = data_location+'/rps-test-set.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall(data_location)

zip_ref.close()





rock_dir = data_location+'/rps/rock'

paper_dir = data_location+'/rps/paper'

scissors_dir = data_location+'/rps/scissors'



print('total training rock images:', len(os.listdir(rock_dir)))

print('total training paper images:', len(os.listdir(paper_dir)))

print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)

print(rock_files[:10])



paper_files = os.listdir(paper_dir)

print(paper_files[:10])



scissors_files = os.listdir(scissors_dir)

print(scissors_files[:10])




# show picture

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



pic_index = 2



next_rock = [os.path.join(rock_dir, fname) 

                for fname in rock_files[pic_index-2:pic_index]]

next_paper = [os.path.join(paper_dir, fname) 

                for fname in paper_files[pic_index-2:pic_index]]

next_scissors = [os.path.join(scissors_dir, fname) 

                for fname in scissors_files[pic_index-2:pic_index]]



for i, img_path in enumerate(next_rock+next_paper+next_scissors):

  #print(img_path)

  img = mpimg.imread(img_path)

  plt.imshow(img)

  plt.axis('Off')

  plt.show()

# input data

  

import tensorflow as tf

import keras_preprocessing

from keras_preprocessing import image

from keras_preprocessing.image import ImageDataGenerator



TRAINING_DIR = data_location+"/rps"

training_datagen = ImageDataGenerator(

      rescale = 1./255,

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')



VALIDATION_DIR = data_location+"/rps-test-set"

validation_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = training_datagen.flow_from_directory(

    TRAINING_DIR,

    target_size=(150,150),

    class_mode='categorical'

)



validation_generator = validation_datagen.flow_from_directory(

    VALIDATION_DIR,

    target_size=(150,150),

    class_mode='categorical'

)


# define model

model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 150x150 with 3 bytes color

    # This is the first convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The third convolution

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The fourth convolution

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(512, activation='relu'),

    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')

    tf.keras.layers.Dense(3, activation='softmax')

])

# model summary

model.summary()


# compile model

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



# train model

history = model.fit_generator(train_generator, epochs=5, validation_data = validation_generator, verbose = 1)

# save model

model.save("rps.h5")


# resutl summary 

import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()

# predicting images

import numpy as np

path = data_location+'/rps-test-set/paper/testpaper01-00.png'

img = image.load_img(path, target_size=(150, 150))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)



images = np.vstack([x])

classes = model.predict(images, batch_size=10)

print(path)

print(classes)
img = mpimg.imread(path)

plt.imshow(img)

plt.axis('Off')

plt.show()
