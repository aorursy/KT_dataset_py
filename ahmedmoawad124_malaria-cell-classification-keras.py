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

                                   zoom_range=0.2)



# Flow training images in batches of 128 using train_datagen generator

'''

What does flow_from_directory give you on the ImageGenerator?

- The ability to easily load images for training

- The ability to pick the size of training images

- The ability to automatically label images based on their directory name

'''

# --------------------

# Flow training images in batches of 128 using train_datagen generator

# --------------------

train_generator = train_datagen.flow_from_directory(

        '../input/cell-images-for-detecting-malaria/cell_images/cell_images/',  # This is the source directory for training images

        target_size=(64, 64),  # All images will be resized to 64x64

        batch_size=128,  # Integer or None. Number of samples per gradient update. 

                        # If unspecified, batch_size will default to 32.

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')



train_generator.class_indices
# We then add convolutional layers, 

# and flatten the final result to feed into the densely connected layers.

# Finally we add the densely connected layers.



model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 150x150 with 3 bytes color

    # This is the first convolution

    tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(64, 64, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

    tf.keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu),

    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    # 128 neuron hidden layer

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dropout(0.2),

    # 256 neuron hidden layer

    tf.keras.layers.Dense(256, activation=tf.nn.relu),

    tf.keras.layers.Dropout(0.2),

    # 64 neuron hidden layer

    tf.keras.layers.Dense(64, activation=tf.nn.relu),

    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')

    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

])
# The model.summary() method call prints a summary of the NN

model.summary()
model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('accuracy') > 0.94):

          print("\nReached 94% accuracy so cancelling training!")

          self.model.stop_training = True
callbacks = myCallback()

history = model.fit(

      train_generator,

    

      #steps_per_epoch=100, # Total number of steps (batches of samples) before declaring one epoch finished 

                           # and starting the next epoch.  

                            # 2,000 samples / 20 batch size = 100 batches

    

      epochs=20,  # Integer. Number of epochs to train the model. 

                  # An epoch is an iteration over the entire data provided. 

    

      verbose=1,  # Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

    

    

      callbacks=[callbacks] # List of keras.callbacks.Callback instances. 

                            # List of callbacks to apply during training and validation (if ).

      )
import numpy as np



test_image = image.load_img('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png', target_size = (64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = model.predict(test_image)

if result[0][0] == 0:

    prediction = 'parasitized'

else:

    prediction = 'Uninfected'



print(prediction)