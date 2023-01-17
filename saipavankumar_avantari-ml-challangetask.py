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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model, Sequential
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
import tensorflow as tf
import json
import os
image_size = (180,180)
batch_size = 32
image_path = "../input/animal-faces/afhq/" #path_to_animal-faces dataset
train_dir = image_path + "train"  #train_directory
validation_dir = image_path + "val" #validation_directory

train_image_generator = ImageDataGenerator( rescale=1./255, 
                                            rotation_range=40, 
                                            width_shift_range=0.2,
                                            height_shift_range=0.2, 
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True, 
                                            fill_mode='nearest')  #image_data_generator_for_train_data

train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size= image_size,
                                                           class_mode='categorical')  #applying_data_generator_on_train_data
    
total_train = train_data_gen.n #getting_total_number_of_train_images


validation_image_generator = ImageDataGenerator(rescale=1./255) #image_data_generator_for_val_data

val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                              batch_size=batch_size,
                                                              shuffle=False,
                                                              target_size=image_size,
                                                              class_mode='categorical') #applying_data_generator_on_val_data
    
total_val = val_data_gen.n #getting_total_number_of_val_images
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout

# build a sequential model
model = Sequential()
model.add(InputLayer(input_shape=(180, 180, 3)))

# 1st conv block
model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
# 2nd conv block
model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())
# 3rd conv block
model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(BatchNormalization())
# ANN block
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu',name='feature_generator'))
model.add(Dropout(0.25))
# output layer
model.add(Dense(units=3, activation='softmax'))

from tensorflow import keras
epochs = 30

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit_generator(
    train_data_gen, epochs=epochs, callbacks=callbacks, validation_data=val_data_gen,
)
from keras.models import Model
encoder = Model(inputs=model.input, outputs=model.get_layer('feature_generator').output)
images_path = '../input/animals-dataset/dataset/' #path to image dataset
from keras.preprocessing import image #function for extracting features of a single image using the trained model
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(180,180))
    img = image.img_to_array(img)
    img = img/255
    img = img.reshape(1,180,180,3)
    return encoder.predict(img)
def batch_extractor(path): #function for generating features for batches of iamges
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    
    result = {} #dictionary for storing image_id and the feature of patricular image
    for f in files:
        print ('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f)
    return result
    
features_images = batch_extractor(images_path) #here features of all the 5000 images are extracted and stored.
features_images
def query_image_features_extractor(image_path):#extracting the featutres of query images
    img = image.load_img(image_path, target_size=(180,180))
    img = image.img_to_array(img)
    img = img/255
    img = img.reshape(1,180,180,3)
    return encoder.predict(img)
    
query_image_features = query_image_features_extractor('../input/animals-dataset/dataset/2893.jpg')
query_image_features
import numpy as np
#calculating the eucledian distance
def euclidean_distance(a,b):
    distance = np.linalg.norm(a - b)
    return distance
    
def search(query_features,limit=10): #search function for calculating the distance between 
                                        #the features of all images to one query image features
    results = {}
    for key in features_images:
        features = features_images[key]
        d = euclidean_distance(features,query_features)
        results[key] = d
    results = sorted([(v, k) for (k, v) in results.items()])
    
    return results[:limit]
results_search = search(query_image_features) #returning the search function's results
results_search
#displaying the search results

results_final = []

for (score, resultID) in results_search:
    # load the result image and display it
    results_final.append(cv2.imread('../input/animals-dataset/dataset/'+ resultID))
for i in range(1,len(results_final)):
    plt.subplot(5,4,i)
    plt.imshow(results_final[i])
plt.show()
