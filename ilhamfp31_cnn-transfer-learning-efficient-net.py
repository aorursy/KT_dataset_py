!pip install git+https://github.com/qubvel/efficientnet
import pandas as pd

import os
# directory list

train_path = '../input/kue-indonesia/train/'

test_path = '../input/kue-indonesia/test/'

validation_path = '../input/kue-indonesia/validation/'

items = os.listdir(train_path)

items
# create items path

train_dir = []

test_dir = []

validation_dir = []

for item in items :

    train_dir.append(train_path + item + '/')

    test_dir.append(test_path + item + '/')

    validation_dir.append(validation_path + item + '/')
%matplotlib inline



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import cv2



labels = [(" ").join(x.split('_')) for x in items]



fig = plt.figure(figsize=(10,10))

fig.suptitle("Some examples of the dataset", fontsize=16)

i = 1

for path in train_dir :

    for j in range(2) :

        img = mpimg.imread(path+os.listdir(path)[j])

        size = min(img.shape[0], img.shape[1])

        img = cv2.resize(img, (size,size))

        

        plt.subplot(4,4,i)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(img, cmap=plt.cm.binary)

        plt.xlabel(labels[int((i-1)/2)])

        i += 1

plt.show()
from tensorflow.keras.preprocessing.image import ImageDataGenerator



TRAINING_DIR = train_path

train_datagen = ImageDataGenerator( rescale = 1.0/255. ,

                                  rotation_range=40,

                                  width_shift_range=0.2,

                                  height_shift_range=0.2,

                                  shear_range=0.2,

                                  zoom_range=0.2,

                                  horizontal_flip=True,

                                  fill_mode='nearest')



train_generator = train_datagen.flow_from_directory(TRAINING_DIR,

                                                    batch_size=16,

                                                    class_mode='categorical',

                                                    target_size=(150, 150))



VALIDATION_DIR = validation_path

validation_datagen = ImageDataGenerator( rescale = 1.0/255. ,

                                  rotation_range=40,

                                  width_shift_range=0.2,

                                  height_shift_range=0.2,

                                  shear_range=0.2,

                                  zoom_range=0.2,

                                  horizontal_flip=True,

                                  fill_mode='nearest')



validation_generator = train_datagen.flow_from_directory(VALIDATION_DIR,

                                                    batch_size=16,

                                                    class_mode='categorical',

                                                    target_size=(150, 150))





TEST_DIR = test_path

test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=(150,150),

    class_mode='categorical',

    batch_size=10)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping



def callbacks(model_name):

    cb = []



    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',  

                                       factor=0.5, patience=1, 

                                       verbose=1, mode='min', 

                                       epsilon=0.0001, min_lr=0,

                                       restore_best_weights=True)

    cb.append(reduceLROnPlat)

    

    log = CSVLogger('log_{}.csv'.format(model_name))

    cb.append(log)

    

    es = EarlyStopping(monitor='val_loss', patience=4, verbose=0,

                       mode='min', restore_best_weights=True)

    

    cb.append(es)

    

    return cb
from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras import layers

from tensorflow.keras import Model

from tensorflow.keras.optimizers import RMSprop

from efficientnet.tfkeras import EfficientNetB7



pre_trained_model = EfficientNetB7(

    include_top=False,

    weights="imagenet",

    input_shape=(150, 150, 3),

    pooling='avg',

)

for layer in pre_trained_model.layers:

    layer.trainable = False





last_layer = pre_trained_model.get_layer('avg_pool')

print('last layer output shape: ', last_layer.output_shape)

last_output = last_layer.output

# Flatten the output layer to 1 dimension

x = layers.Flatten()(last_output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation

x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2

x = layers.Dropout(0.2)(x)                  

x = layers.Dense  (8, activation='softmax')(x)           



model = Model( pre_trained_model.input, x) 



model.compile(optimizer = 'adam', 

              loss = 'categorical_crossentropy', 

              metrics = ['accuracy'])
history = model.fit(

            train_generator,

            validation_data = validation_generator,

            epochs = 200,

            verbose = 1,

            callbacks = callbacks('model_efficientnetb7'))
results = model.evaluate(test_generator)
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(acc)) # Get number of epochs



plt.figure()

#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot(epochs, acc, label='Training accuracy')

plt.plot(epochs, val_acc, label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot  ( epochs,     loss, label='Training loss' )

plt.plot  ( epochs, val_loss, label='Validation loss' )

plt.title ('Training and validation loss')

plt.legend(loc=0)



plt.show()
model.save_weights('model_efficientnetb7.h5')
from efficientnet.tfkeras import EfficientNetB5



pre_trained_model = EfficientNetB5(

    include_top=False,

    weights="imagenet",

    input_shape=(150, 150, 3),

    pooling='avg',

)
for layer in pre_trained_model.layers:

    layer.trainable = False





last_layer = pre_trained_model.get_layer('avg_pool')

print('last layer output shape: ', last_layer.output_shape)

last_output = last_layer.output

# Flatten the output layer to 1 dimension

x = layers.Flatten()(last_output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation

x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2

x = layers.Dropout(0.2)(x)                  

x = layers.Dense  (8, activation='softmax')(x)           



model = Model( pre_trained_model.input, x) 



model.compile(optimizer = 'adam', 

              loss = 'categorical_crossentropy', 

              metrics = ['accuracy'])
history = model.fit(

            train_generator,

            validation_data = validation_generator,

            epochs = 200,

            verbose = 1,

            callbacks = callbacks('model_efficientnetb5'))
results = model.evaluate(test_generator)
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(acc)) # Get number of epochs



plt.figure()

#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot(epochs, acc, label='Training accuracy')

plt.plot(epochs, val_acc, label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot  ( epochs,     loss, label='Training loss' )

plt.plot  ( epochs, val_loss, label='Validation loss' )

plt.title ('Training and validation loss')

plt.legend(loc=0)



plt.show()
model.save("model_efficientnetb5.h5")
from efficientnet.tfkeras import EfficientNetB3



pre_trained_model = EfficientNetB3(

    include_top=False,

    weights="imagenet",

    input_shape=(150, 150, 3),

    pooling='avg',

)
for layer in pre_trained_model.layers:

    layer.trainable = False





last_layer = pre_trained_model.get_layer('avg_pool')

print('last layer output shape: ', last_layer.output_shape)

last_output = last_layer.output

# Flatten the output layer to 1 dimension

x = layers.Flatten()(last_output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation

x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2

x = layers.Dropout(0.2)(x)                  

x = layers.Dense  (8, activation='softmax')(x)           



model = Model( pre_trained_model.input, x) 



model.compile(optimizer = 'adam', 

              loss = 'categorical_crossentropy', 

              metrics = ['accuracy'])
history = model.fit(

            train_generator,

            validation_data = validation_generator,

            epochs = 200,

            verbose = 1,

            callbacks = callbacks('model_efficientnetb3'))
results = model.evaluate(test_generator)
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(acc)) # Get number of epochs



plt.figure()

#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot(epochs, acc, label='Training accuracy')

plt.plot(epochs, val_acc, label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot  ( epochs,     loss, label='Training loss' )

plt.plot  ( epochs, val_loss, label='Validation loss' )

plt.title ('Training and validation loss')

plt.legend(loc=0)



plt.show()
model.save_weights('model_efficientnetb3.h5')