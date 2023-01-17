import tensorflow as tf

import numpy as np

from tensorflow import keras



from keras.models import Sequential

from keras.layers import Dense

from keras.utils.vis_utils import plot_model



import matplotlib.pyplot as plt


model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error',metrics=['mae'])
tf.keras.utils.plot_model(model, to_file='model_combined.png')


xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
history=model.fit(xs, ys, epochs=100);
print(model.predict([10.0]))
# Plot training & validation accuracy values

plt.plot(history.history['loss'])

#plt.plot(history.history['val_acc'])

plt.title('Model mean_squared_error')

plt.ylabel('mean_squared_error')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
import tensorflow as tf

print(tf.__version__)
# input data

mnist = tf.keras.datasets.fashion_mnist



(training_images, training_labels), (test_images, test_labels) = mnist.load_data()



import matplotlib.pyplot as plt

plt.imshow(training_images[0])

#print(training_labels[0])

#print(training_images[0])





# data cleaning

training_images  = training_images / 255.0

test_images = test_images / 255.0



# define model

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),  # flatten

                                    

                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),  # 128 cell 

                                    

                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)]) # 10 output



# compile model

model.compile(optimizer = 'Adam', # AdamOptimizer

              loss = 'sparse_categorical_crossentropy', # loss function

              metrics=['accuracy']) # accuracy metrics



# train model

history=model.fit(training_images, training_labels, epochs=20);
# evaluate model

model.evaluate(test_images, test_labels)
# Plot training & validation accuracy values

#plt.plot(history.history['accuracy'])

plt.plot(history.history['acc'])

#plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation accuracy values

plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint
callbacks = [

  # Interrupt training if `val_loss` stops improving for over 2 epochs

  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),

  # Write TensorBoard logs to `./logs` directory

  tf.keras.callbacks.TensorBoard(log_dir='./logs')

]



# callbacks = [tf.keras.callbacks.EarlyStopping(monitor='acc',mode='auto',baseline=0.95)]


# input data

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()





# data clean

training_images=training_images.reshape(60000, 28, 28, 1)

training_images=training_images / 255.0

test_images = test_images.reshape(10000, 28, 28, 1)

test_images=test_images/255.0







# define model

model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)), # 32 filter with 3 by 3

  tf.keras.layers.MaxPooling2D(2, 2), # max pooling with 2 by 2 select the max value

  tf.keras.layers.Flatten(), # flatten

  tf.keras.layers.Dense(128, activation='relu'), # 128 cell

  tf.keras.layers.Dense(10, activation='softmax') # 10 output

])

    

    

# complie model   

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train model

model.fit(training_images, training_labels, validation_split = 0.3,epochs=20, callbacks=callbacks);

# evaluate model

model.evaluate(test_images, test_labels)


import tensorflow as tf

import os

import zipfile



DESIRED_ACCURACY = 0.999

#!wget --no-check-certificate \"https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \-O "/tmp/happy-or-sad.zip"


# download data







import urllib.request



urllib.request.urlretrieve("https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip",'horse-or-human.zip')



urllib.request.urlretrieve("https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip", "validation-horse-or-human.zip")





import os

import zipfile


local_zip = 'horse-or-human.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('../output/horse-or-human')



local_zip = 'validation-horse-or-human.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('../output/validation-horse-or-human')

zip_ref.close()







# call back

class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('acc')>DESIRED_ACCURACY):

      print("\nReached 99.9% accuracy so cancelling training!")

      self.model.stop_training = True



callbacks = myCallback()


# input data 



# Directory with our training horse pictures

#train_horse_dir = os.path.join('../output/horse-or-human/horses')



# Directory with our training human pictures

#train_human_dir = os.path.join('../output/horse-or-human/humans')



# Directory with our training horse pictures

#validation_horse_dir = os.path.join('../output/validation-horse-or-human/validation-horses')



# Directory with our training human pictures

#validation_human_dir = os.path.join('../output/validation-horse-or-human/validation-humans')







# data cleaning

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale=1/255)

validation_datagen = ImageDataGenerator(rescale=1/255)



# Flow training images in batches of 128 using train_datagen generator

train_generator = train_datagen.flow_from_directory(

        '../output/horse-or-human/',  # This is the source directory for training images

        target_size=(150, 150),  # All images will be resized to 150x150

        batch_size=128,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')



# Flow training images in batches of 128 using train_datagen generator

validation_generator = validation_datagen.flow_from_directory(

        '../output/validation-horse-or-human/',  # This is the source directory for training images

        target_size=(150, 150),  # All images will be resized to 150x150

        batch_size=32,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')





# define model

import tensorflow as tf



model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 150x150 with 3 bytes color

    # This is the first convolution

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The third convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The fourth convolution

    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    #tf.keras.layers.MaxPooling2D(2,2),

    # The fifth convolution

    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    #tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(512, activation='relu'),

    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')

    tf.keras.layers.Dense(1, activation='sigmoid')

])

    

    

    

# model summary    

model.summary()

    



# compile model

from tensorflow.keras.optimizers import RMSprop





model.compile(loss='binary_crossentropy',

              optimizer=RMSprop(lr=0.001), # RMSprop optimizer with 0.1% learning rate

              metrics=['acc'])





# train model

history = model.fit_generator(

      train_generator,

      steps_per_epoch=8,  

      epochs=5,

      verbose=1,

      validation_data = validation_generator,

      validation_steps=8);







# evaluate model

#model.evaluate(test_images, test_labels)
# Plot training & validation accuracy values

plt.plot(history.history['acc'])

#plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation accuracy values

plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()