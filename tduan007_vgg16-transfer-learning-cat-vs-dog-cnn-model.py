import tensorflow as tf

print(tf.__version__)
import numpy as np

import matplotlib.pyplot as plt

import os

import cv2

from tqdm import tqdm

import urllib.request



import os

import zipfile

import numpy as np

import pandas as pd

from sklearn.metrics import roc_auc_score

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator,load_img

from keras.models import Sequential

from keras.layers import Activation, Dropout, Flatten, Dense,Conv2D, MaxPooling2D,GlobalMaxPooling2D

from keras.optimizers import Adam

from keras.models import Model

from keras import optimizers
urllib.request.urlretrieve("https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip", "cat_dog.zip")

#zip = ZipFile('cat_dog.zip')

#zip.extractall()
import zipfile

zip_ref = zipfile.ZipFile('cat_dog.zip', 'r')

zip_ref.extractall('../output/')

zip_ref.close()



import os

os.listdir('../output/PetImages')



print('total  dog images :', len(os.listdir('../output/PetImages/Dog') ))

print('total  cat images :', len(os.listdir('../output/PetImages/Cat') ))



os.remove('../output/PetImages/Cat/666.jpg')

os.remove('../output/PetImages/Dog/11702.jpg')
os.listdir('../output/PetImages/Dog')[1:10]
image = load_img('../output/PetImages/Dog/4644.jpg')

plt.imshow(image)
os.listdir('../output/PetImages/Cat')[1:10]
image = load_img('../output/PetImages/Cat/8962.jpg')

plt.imshow(image)
img_width=150

img_height=150

batch_size=20

input_shape = (img_width, img_height, 3)
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# All images will be rescaled by 1./255.

train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.3,

    zoom_range=[0.6,1.0],

    brightness_range=[0.6,1.0],

    rotation_range=90,

    horizontal_flip=True,

    validation_split=0.2

)



# --------------------

# Flow training images in batches of 20 using train_datagen generator

# --------------------

train_generator = train_datagen.flow_from_directory(

    '../output/PetImages',

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='categorical',

    seed = 42,

    subset='training'

    

)

valid_generator = train_datagen.flow_from_directory(

    '../output/PetImages',

    target_size=(img_width, img_height),

    batch_size=batch_size,

    #class_mode='binary',

    class_mode='categorical',

    seed = 42,

    subset='validation'

    

)

#X, y = next(train_generator)
plt.figure(figsize=(12, 12))

for i in range(0, 15):

    plt.subplot(5, 3, i+1)

    for X_batch, Y_batch in train_generator:

        image = X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
####################### VGG16

from keras.applications import VGG16

pre_trained_model = VGG16(input_shape=(150, 150, 3), include_top=False, weights="imagenet")



for layer in pre_trained_model.layers[:15]:

    layer.trainable = False



for layer in pre_trained_model.layers[15:]:

    layer.trainable = True

    

last_layer = pre_trained_model.get_layer('block5_pool')

last_output = last_layer.output
pre_trained_model.summary()
# Flatten the output layer to 1 dimension

x = GlobalMaxPooling2D()(last_output)

# Add a fully connected layer with 512 hidden units and ReLU activation

x = Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5

x = Dropout(0.5)(x)

# Add a final sigmoid layer for classification

#x = Dense(1, activation='sigmoid')(x)

x = Dense(2, activation='softmax')(x)

model = Model(pre_trained_model.input, x)
model.compile(loss='binary_crossentropy',

              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

              metrics=['acc'])
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#earlystop = EarlyStopping(patience=5)

earlystop=EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=0, mode='auto')

callbacks = [earlystop]
history = model.fit_generator(

            train_generator,

            validation_data = valid_generator,

            steps_per_epoch = 100,

            epochs = 50,

            validation_steps = 50,

            verbose = 1

            ,callbacks=callbacks

)
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

#plt.plot  ( epochs,     loss )

#plt.plot  ( epochs, val_loss )

#plt.title ('Training and validation loss'   )
!pip install -q pyyaml h5py
# save model

from keras.models import load_model

model.save('VGG16_dog_cat_cnn_model.h5')
from IPython.display import FileLink

FileLink(r'VGG16_dog_cat_cnn_model.h5')
import keras

new_model = keras.models.load_model('VGG16_dog_cat_cnn_model.h5')
val_loss, val_acc = model.evaluate(valid_generator)  # evaluate the out of sample data with model

#print(val_loss)  # model's loss (error)

print(val_acc)  # model's accuracy
urllib.request.urlretrieve('https://dcist.com/wp-content/uploads/sites/3/2019/04/Gem2-768x689.jpg', "image.jpg")
import numpy as np

import pandas as pd

#from keras_preprocessing import image

#import PIL.Image as Image

import tensorflow as tf

#import cv2

import PIL.Image as Image

x = Image.open('image.jpg').resize((150, 150))

x = np.array(x)/255.0

new_model = tf.keras.models.load_model ('VGG16_dog_cat_cnn_model.h5')

result = new_model.predict(x[np.newaxis, ...])

df = pd.DataFrame(data =result,columns=['cat','dog'])

df