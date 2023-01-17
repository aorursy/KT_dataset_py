import pandas as pd

# testing Kaggle output folder (since sometimes it bugged and need to be restarted)

x = pd.DataFrame({'x','y'})
x.to_csv('tes.csv')
!pip install --upgrade efficientnet tensorflow_addons tensorflow
!pip install -q efficientnet
import efficientnet.tfkeras as efn
import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from kaggle_datasets import KaggleDatasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    

print(tf.__version__)
os.listdir('/kaggle/input/shopee-product-detection-student/')
train_path = "/kaggle/input/shopee-product-detection-student/train/train/train/"
test_path= "/kaggle/input/shopee-product-detection-student/test/test/test/"
broken_fnames = []
for label in os.listdir(train_path):
    label_path = train_path + label + '/'
    for filename in os.listdir(label_path):
        if len(filename) > 36:
            print(label_path + filename)
            broken_fnames.append(label_path + filename)
            #finding broken file name
print()
for filename in os.listdir(test_path):
    if len(filename) > 36:
        print(test_path + filename)
        broken_fnames.append(test_path + filename)
        
f = open('broken-file-names.txt', 'w')
#creates broken file texts.
f.write('\n'.join(broken_fnames))
f.close()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (400, 400)
BATCH_SIZE = 128
SEED = 48

def get_set():
    train_path = "/kaggle/input/shopee-product-detection-student/train/train/train/"
    test_path= "/kaggle/input/shopee-product-detection-student/test/test/"

    train_gen = ImageDataGenerator(rescale=1./255., 
                                    validation_split=0.25,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.1)
    train_set = train_gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, \
                                              batch_size=BATCH_SIZE, seed=SEED, \
                                              subset='training')
    val_set = train_gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, \
                                            batch_size=BATCH_SIZE, seed=SEED, \
                                            subset='validation')

    test_gen = ImageDataGenerator(rescale=1./255)
    test_set = train_gen.flow_from_directory(test_path, target_size=IMAGE_SIZE, \
                                             batch_size=BATCH_SIZE, seed=SEED, \
                                             shuffle=False, class_mode=None)
    
    return train_set, val_set, test_set

train_set, val_set, test_set = get_set()
from tensorflow.keras.applications.inception_v3 import InceptionV3

base = InceptionV3(input_shape = (400, 400, 3), 
                    include_top = False, 
                    weights ='imagenet')
base.trainable = False
model = tf.keras.Sequential([
        base,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(42, activation='softmax')
    ])
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False),
    loss = 'categorical_crossentropy',
    metrics=['acc']
    )
model.summary()
# Alternate model (using InceptionV3 until layer mixed10)
last_layer = pre_trained_model.get_layer('mixed10')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Adding dense layer
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (42, activation='softmax')(x)           

model1 = Model( pre_trained_model.input, x) 

model1.compile(optimizer ='adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['acc'])
model1.summary()
EPOCHS = 1

hist = model.fit(train_set, epochs=EPOCHS, 
                 validation_data=val_set, shuffle=True)

# Running model only for demonstration since the model are pretty large
# And it could crash the kaggle output if saved 

#model.save('model-InceptionV3-SHOPEE-1.hdf5')
def generate_prediction(model, save_name):
    subm = pd.read_csv('/kaggle/input/shopee-product-detection-student/test.csv')
    subm = subm.sort_values(by='filename')
    
    fnames = sorted(os.listdir('/kaggle/input/shopee-product-detection-student/test/test/test/'))
    unbroken_index = np.where(np.vectorize(len)(np.array(fnames)) == 36)[0]
    
    y_pred = model.predict(test_set)
    pred = y_pred.argmax(axis=1)
    pred = pred[unbroken_index]
    subm['category'] = pred
    
    #adding zero padding (from 1 to 01)
    subm['category'] = subm['category'].apply(lambda x: '{0:0>2}'.format(x)) 
    
    #saving the prediction into csv file
    subm.to_csv(save_name, index=False)
    return subm
subm = generate_prediction(model, 'kaggle_submission.csv')
subm
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (299, 299)
BATCH_SIZE = 128
SEED = 48

def get_set():
    train_path = "/kaggle/input/shopee-product-detection-student/train/train/train/"
    test_path= "/kaggle/input/shopee-product-detection-student/test/test/"

    train_gen = ImageDataGenerator(rescale=1./255., 
                                    validation_split=0.25,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.1)
    train_set = train_gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, \
                                              batch_size=BATCH_SIZE, seed=SEED, \
                                              subset='training')
    val_set = train_gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, \
                                            batch_size=BATCH_SIZE, seed=SEED, \
                                            subset='validation')

    test_gen = ImageDataGenerator(rescale=1./255)
    test_set = train_gen.flow_from_directory(test_path, target_size=IMAGE_SIZE, \
                                             batch_size=BATCH_SIZE, seed=SEED, \
                                             shuffle=False, class_mode=None)
    
    return train_set, val_set, test_set

train_set, val_set, test_set = get_set()
base_model = tf.keras.applications.Xception(input_shape=(299,299,3),weights="imagenet", include_top=False)
base_model.trainable = False
model = tf.keras.Sequential([
        base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1042, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(42, activation='softmax')
        ])
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False),
    loss = 'categorical_crossentropy',
    metrics=['acc']
    )
model.summary()


EPOCHS = 3

hist = model.fit(train_set, epochs=EPOCHS, 
                 validation_data=val_set, shuffle=True)
