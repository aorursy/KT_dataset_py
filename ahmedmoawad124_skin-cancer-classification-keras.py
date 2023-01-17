import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image
# All images will be rescaled by 1./255  ===> data normalization  

train_datagen = ImageDataGenerator(# Augmentation parameters:

                                   

                                   rescale=1.0/255.0, 

                                   rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True,

                                   fill_mode='nearest')

test_datagen  = ImageDataGenerator( rescale = 1.0/255.0)



# Flow training images in batches of 128 using train_datagen generator

'''

What does flow_from_directory give you on the ImageGenerator?

- The ability to easily load images for training

- The ability to pick the size of training images

- The ability to automatically label images based on their directory name

'''

# --------------------

# Flow training images in batches of 20 using train_datagen generator

# --------------------

train_generator = train_datagen.flow_from_directory(

        '../input/skin-cancer-malignant-vs-benign/train/',  # This is the source directory for training images

        target_size=(64, 64),  # All images will be resized to 150x150

        batch_size=64,  # Integer or None. Number of samples per gradient update. 

                        # If unspecified, batch_size will default to 32.

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')

# --------------------

# Flow validation images in batches of 20 using test_datagen generator

# --------------------

validation_generator =  test_datagen.flow_from_directory('../input/skin-cancer-malignant-vs-benign/test/',

                                                         batch_size=20,

                                                         class_mode  = 'binary',

                                                         target_size = (64, 64))





train_generator.class_indices
# We then add convolutional layers, 

# and flatten the final result to feed into the densely connected layers.

# Finally we add the densely connected layers.



model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 150x150 with 3 bytes color

    # This is the first convolution

    tf.keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, input_shape=(64, 64, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

    tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),

    tf.keras.layers.MaxPooling2D(2,2),

    # The third convolution

    tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),

    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')

    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

])
# The model.summary() method call prints a summary of the NN

model.summary()
from tensorflow.keras.optimizers import RMSprop



model.compile(loss='binary_crossentropy',

              optimizer=RMSprop(lr=0.001),

              metrics=['accuracy'])
class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('accuracy')>0.95):

          print("\nReached 95% accuracy so cancelling training!")

          self.model.stop_training = True
callbacks = myCallback()

history = model.fit(

      train_generator,

    

      #steps_per_epoch=100, # Total number of steps (batches of samples) before declaring one epoch finished 

                           # and starting the next epoch.  

                            # 2,000 samples / 20 batch size = 100 batches

    

      epochs=150,  # Integer. Number of epochs to train the model. 

                  # An epoch is an iteration over the entire data provided. 

    

      verbose=1,  # Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

    

      validation_data=validation_generator, # Data on which to evaluate the loss and any model metrics at 

                                           # the end of each epoch. The model will not be trained on this data.

    

      #validation_steps=50,  # Only relevant if validation_data is provided and is a generator. 

                            # Total number of steps (batches of samples) to draw before stopping when performing 

                            # validation at the end of every epoch.

    

      callbacks=[callbacks] # List of keras.callbacks.Callback instances. 

                            # List of callbacks to apply during training and validation (if ).

      )
#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

acc      = history.history[     'accuracy' ]

val_acc  = history.history[ 'val_accuracy' ]

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
import numpy as np



test_image = image.load_img('../input/skin-cancer-malignant-vs-benign/test/benign/10.jpg', target_size = (64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = model.predict(test_image)



if result[0][0] == 0:

    prediction = 'benign'

else:

    prediction = 'malignant'



prediction