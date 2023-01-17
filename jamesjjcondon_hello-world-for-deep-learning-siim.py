print("")    # insert the text "Hello, world!" between the quotes, then run the cell

"Hello world"
# This notebook is built around using tensorflow as the backend for keras

!KERAS_BACKEND=tensorflow python -c "from keras import backend"

# Import the appropriate Keras modules

from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

from keras.models import Model

from keras.optimizers import Adam



%matplotlib inline

import os



# Set path variable to the directory where the data is located

# hard-code path if adding Keras pre-trained models dataset (avoids error due to multiple datasets in 'input/')

path = ('../input/hello-world-deep-learning-siim/data/')

print(path)

# Command line "magic" command to show directory contents

#!ls {path}/*/*



# set variables for paths to directories for training & validation data

# adjust join() call due to path changes:

train_dir = os.path.join(path + 'train')

val_dir = os.path.join(path + 'val')



# set variables for number of samples in each data set

num_train = 65

num_val = 10



# we'll need to import additional modules to look at an example image

import numpy as np    # this is a standard convention

from keras.preprocessing import image

import matplotlib.pyplot as plt    # also by convention



# these are the dimensions of our images

img_width, img_height = 299, 299



# set the path to a chest radiograph, then load it and show

img_path = os.path.join(train_dir, 'chst/chst_train_001.png')

img = image.load_img(img_path, target_size=(img_width, img_height))

plt.imshow(img)

plt.title('Example chest radiograph')

plt.show()



# set the path to an abdominal radiograph, then load it and show

img2_path = os.path.join(train_dir, 'abd/abd_train_001.png')

img2 = image.load_img(img2_path, target_size=(img_width, img_height))

plt.imshow(img2)

plt.title("Example abdominal radiograph")

plt.show()

# randomize to groups 1 and 2

group = np.random.randint(1, 3)

print("You have been assigned to Group", group)

# set the batch size for each training step

batch_size = 8



# create training data generator object

# initialize values for image augmentation

# rescaling is done to normalize the image pixel values into the [0, 1] range

train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        rotation_range=20,

        width_shift_range=0.2,

        height_shift_range=0.2,

        horizontal_flip=True

)



print('Success!')

# finalize training generator

print("Training generator: ", end="")

train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='binary'

)



# create validation data generator object

# no image augmentation here, as we are not training our model on this data

val_datagen = ImageDataGenerator(rescale=1./255)

print("Validation generator: ", end="")

val_generator = train_datagen.flow_from_directory(

    val_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='binary'

)

# changed weights to dir for Keras-pretrained-models dataset 

#(1. click 'add data' in draft environment right panel, 

# 2. search for keras pretrained weights, 

# 3. copy file path for inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 in place of 'imagenet')

backbone = applications.InceptionV3(weights='../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(img_width, img_height, 3))



# Freeze the pretrained weights in the model backbone

for layer in backbone.layers:

    layer.trainable = False



print('Model backbone initialized!')    # this will print when this step is complete

# randomize to subgroups A & B

subgroup = np.random.randint(1, 3)

print("You have been assigned to Subgroup", "A" if subgroup == 1 else "B")

# create the top layers of the model

model_top = Sequential()    # initialize as a Sequential model (i.e. no recurrent layers)

model_top.add(GlobalAveragePooling2D(input_shape=backbone.output_shape[1:], data_format=None)) 

model_top.add(Dense(256, activation='relu'))

model_top.add(Dropout(0.5))

model_top.add(Dense(1, activation='sigmoid')) # the output will be a probability



# connect "model_top" to "backbone"

model = Model(inputs=backbone.input, outputs=model_top(backbone.output))



# compile the model with the Adam optimizer, binary cross-entropy loss, and accuracy

model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])



print('Setup Complete!')

# set the number of epochs for training

epochs = 5



# train the model and save the training/validation results for each epoch to "history"

history = model.fit_generator(

            train_generator,

            steps_per_epoch=num_train // batch_size,

            epochs=epochs,

            validation_data=val_generator,

            validation_steps=num_val // batch_size)



print('Training complete!')

print(history.history.keys())



fig, ax = plt.subplots(2, 1)

ax[0].plot(history.history['acc'], 'orange', label='Training accuracy')

ax[0].plot(history.history['val_acc'], 'blue', label='Validation accuracy')

ax[1].plot(history.history['loss'], 'red', label='Training loss')

ax[1].plot(history.history['val_loss'], 'green', label='Validation loss')

ax[0].legend()

ax[1].legend()

plt.show()

# Unfreeze the model backbone before we train a little more

for layer in model.layers:

    layer.trainable = True



# When you make a change to the model, you have to compile it again prior to training

model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])



print('Setup Complete!')

# set the number of epochs for training

epochs = 3



# train the model and save the training/validation results for each epoch to "history"

history = model.fit_generator(

            train_generator,

            steps_per_epoch=num_train // batch_size,

            epochs=epochs,

            validation_data=val_generator,

            validation_steps=num_val // batch_size)



print('Training complete!')

print(history.history.keys())



plt.figure()

plt.plot(history.history['acc'], 'orange', label='Training accuracy')

plt.plot(history.history['val_acc'], 'blue', label='Validation accuracy')

plt.plot(history.history['loss'], 'red', label='Training loss')

plt.plot(history.history['val_loss'], 'green', label='Validation loss')

plt.legend()

plt.show()

# load example chest and abdominal radiographs from validation data set 

img_path = os.path.join(val_dir, 'chst/chst_val_001.png')

img_path2 = os.path.join(val_dir, 'abd/abd_val_001.png')

img = image.load_img(img_path, target_size=(img_width, img_height))

img2 = image.load_img(img_path2, target_size=(img_width, img_height))



# show the chest radiograph

plt.imshow(img)

plt.show()



# evaluate the chest radiograph with the model, then print the model's prediction

img = image.img_to_array(img)

x = np.expand_dims(img, axis=0) * 1./255

score = model.predict(x)

print('Predicted:', score[0][0], 'Chest radiograph' if score > 0.5 else 'Abdominal radiograph')



# show the abdominal radiograph

plt.imshow(img2)

plt.show()



# evaluate the abdominal radiograph with the model, then print the model's prediction

img2 = image.img_to_array(img2)

x = np.expand_dims(img2, axis=0) * 1./255

score2 = model.predict(x)

print('Predicted:', score2[0][0], 'Chest radiograph' if score2 > 0.5 else 'Abdominal radiograph')
