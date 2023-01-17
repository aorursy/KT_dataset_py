# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


base_folder = "/kaggle/input/dogs-cats-images/dataset/training_set/"

input_shape = (1280,720,3) # Width, Height , RGB, Picture will be resized to this before any computations

tf.__version__
def basic_classification_model(input_shape, model_name='basic_model'):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, 3, padding='same')(inputs) # Applies a Convulation layer to input matrix , padded to ensure that the Height and Width remains the same from input to output. Uses "inputs" as the input, produces "x"
    x = keras.layers.BatchNormalization()(x)               # Normalises the Batch of data. Uses the newly defined "x" as input, produces an updated "x" 
    x = keras.layers.ReLU()(x)                             # Turns any elements with negative numbers into 0, uses the newly updated "x" as input, produces an even more updated "x". This updating of "x" repeats itself
    x = keras.layers.Conv2D(64, 3,padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    block1_output = keras.layers.MaxPooling2D(2)(x)      # Max-pools the matrix with a window of 2x2, forming a matrix output shape of 640,360,64
    
    x = keras.layers.Conv2D(64, 3, padding='same')(block1_output)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.add([x, block1_output])             # For residual connection, possible because the current "x" output and block1_output has the same shape
    x = keras.layers.Conv2D(128, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    block2_output = keras.layers.MaxPooling2D(2)(x)      # 320,180,128
    
    x = keras.layers.Conv2D(128, 3, padding='same')(block2_output)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.add([x, block2_output])             
    x = keras.layers.Conv2D(256, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    block3_output = keras.layers.MaxPooling2D(2)(x)      # 160,90,256
    
    x = keras.layers.Conv2D(256, 3, padding='same')(block3_output)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.add([x, block3_output])             
    x = keras.layers.Conv2D(512, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    block4_output = keras.layers.MaxPooling2D(2)(x)      # 80,45,512
    
    x = keras.layers.Conv2D(512, 3, padding='same')(block4_output)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.add([x, block4_output])             
    x = keras.layers.Conv2D(1024, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    block5_output = keras.layers.MaxPooling2D(5)(x)      # 16,5,1024
    
    x = keras.layers.GlobalAveragePooling2D()(x) # 1,1,1024

    x = keras.layers.Dense(1024)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Dropout(0.35)(x)                     # Randomly changes 35% of elements into 0s so that the model can predict based on trends and patterns rather than memorisation
    x = keras.layers.Dense(2)(x)
    predictions = keras.layers.Softmax()(x)              

    model = keras.Model(inputs, predictions, name=model_name)
    model.compile( optimizer=tf.keras.optimizers.Adam(0.001),
                 loss=keras.losses.CategoricalCrossentropy(from_logits=False), 
                 metrics=['accuracy'] )
    return model
model_context = 'basic_model'
model = basic_classification_model(input_shape, model_name=model_context)
model.summary()
model_plot = tf.keras.utils.plot_model(model, show_shapes=True)
display(model_plot)
cat_folder = os.path.join( base_folder, 'cats' )
cat_files = list(os.listdir( cat_folder ))
dog_folder = os.path.join( base_folder, 'dogs' )
dog_files = list(os.listdir( dog_folder ))
from IPython.display import Image
Image(filename= os.path.join( dog_folder, dog_files[1] ) ) 
# Setting the pictures [] to be -1 will view the last picture.
bs = 1 #Batch Size

# An object that applies transformations to the images before they are consumed by the model
# These transformations include (1) preprocessing, like rescaling or normalization (2) data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,                  # divide each pixel value by 255. Each pixel is in the range 0-255, so after division it is in 0-1
        rotation_range=20,               # rotate the image between -20 to +20 degrees
        width_shift_range=0.2,           # translate the image left-right for 20% of the image's width
        height_shift_range=0.2,          # same, for up-down and height
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)
print('Making training data generator...')
train_gen = datagen.flow_from_directory(
        base_folder,
        target_size=input_shape[:2],
        batch_size=bs,
        subset='training')
print('Making validation data generator...')
val_gen = datagen.flow_from_directory(
        base_folder,
        target_size=input_shape[:2],
        batch_size=bs,
        subset='validation')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join( "/kaggle/working", '{}-best_val_loss.h5'.format(model_context) ),
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)

# If the validation loss doesn't improve for 20 epochs, stop training
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

# If the validation loss doesn't improve for 5 epochs, reduce the learning rate to 0.2 times it's previous value
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
# Start training the model
n_epochs=14
model.fit(train_gen,
          epochs=n_epochs,
          steps_per_epoch=train_gen.n // bs,
          validation_data=val_gen,
          validation_steps=val_gen.n // bs,
          callbacks=[model_checkpoint, earlystopping, reduce_lr])
#test_folder = "/kaggle/input/dogs-cats-images/dataset/test_set/"
#test_img_path = os.path.join( test_folder , "cat.4030.jpg")
test_img_path = "/kaggle/input/catdogtestpic/cattest.jpg"
# for dog: test_img_path = "/kaggle/input/catdogtestpic/dogtest.jpg"

from tensorflow.keras.preprocessing.image import load_img, img_to_array

def run_image_on_model(img_path, model, label_map):
    pil_img = load_img(test_img_path)
    pil_img = pil_img.resize( (720,1280) )
    np_img = img_to_array(pil_img)
    np_img = np_img / 255. # Normalize the image values the same way you did when you trained the model
  # We need to wrap this in an np.array with dimensions (b,H,W,C). Currently, the shape is only (H,W,C)
    np_img = np.array( [np_img] )
    pred = model.predict(np_img, batch_size=1)[0]
    pred_idx = np.argmax(pred)
    print("[cat confidence, dog confidence]")
    print(pred)
    return label_map[pred_idx]
Image(filename=test_img_path)
print(train_gen.class_indices)
label_map = {v:k for k,v in train_gen.class_indices.items()}
label_map
print( 'model prediction: {}'.format(run_image_on_model(test_img_path, model, label_map)) )