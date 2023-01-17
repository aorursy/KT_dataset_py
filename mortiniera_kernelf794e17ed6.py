import tensorflow as tf

from glob import glob

from sklearn.model_selection import train_test_split

from matplotlib import image

from PIL import Image

import numpy as np

from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.preprocessing.image import load_img

import pandas as pd

tf.test.gpu_device_name()
#from google.colab import drive

#drive.mount('/content/gdrive')

chair = glob('../input/objects/chair/*')

glass = glob('../input/objects/glass/*')

water_bottle = glob('../input/objects/water_bottle/*')
size = 299 #Inception uses image sizes of 299 x 299



def add_to_set(subset) :

    images = []

    for filename in subset : 

        image = Image.open(filename)

        if (image.mode != 'RGB'):

            image=image.convert('RGB')

        img_resized = np.array(image.resize((size,size)))

        images.append(img_resized)    

    return images



chair = add_to_set(chair)

glass = add_to_set(glass)

water_bottle = add_to_set(water_bottle)



chair_train, chair_test = train_test_split(chair, test_size=0.2)

glass_train, glass_test = train_test_split(glass, test_size=0.2)

water_bottle_train, water_bottle_test = train_test_split(water_bottle, test_size=0.2)
def select_indices(d, indices) :

    return [d[i] for i in list(indices)]
#Visualize some images

chair_visu = select_indices(chair_train, np.random.choice(len(chair_train), 5))

glass_visu = select_indices(glass_train,  np.random.choice(len(glass_train), 5))

water_bottle_visu = select_indices(water_bottle_train, np.random.choice(len(water_bottle_train), 5))

data = np.concatenate((chair_visu, glass_visu, water_bottle_visu))

labels = 5 * ['Chair'] + 5 *['Glass'] + 5 *['Water Bottle']



N, R, C = 25, 5, 5

plt.figure(figsize=(12, 9))

for k, (im, label) in enumerate(zip(data, labels)):

    plt.subplot(R, C, k+1)

    plt.title(label)

    plt.imshow(np.asarray(im))

    plt.axis('off')
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

CLASSES = 3

    

    

# setup model

base_model = InceptionV3(weights='imagenet', include_top=False)



x = base_model.output

x = GlobalAveragePooling2D(name='avg_pool')(x)

x = Dropout(0.4)(x)

predictions = Dense(CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

   

# transfer learning

for layer in base_model.layers:

    layer.trainable = False

      

model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
X_train = chair_train + glass_train + water_bottle_train

y_train = [0]*len(chair_train) + [1]*len(glass_train) + [2]*len(water_bottle_train)

X_train, y_train = shuffle(np.asarray(X_train), y_train, random_state=42)

y_train = pd.Series(y_train).astype('category')

y_train = pd.get_dummies(y_train.reset_index(drop=True)).as_matrix()



X_test = chair_test + glass_test + water_bottle_test

y_test = [0]*len(chair_test) + [1]*len(glass_test) + [2]*len(water_bottle_test)

X_test, y_test = shuffle(np.asarray(X_test), y_test, random_state=42)

y_test = pd.Series(y_test).astype('category')

y_test = pd.get_dummies(y_test.reset_index(drop=True)).as_matrix()





train_datagen = ImageDataGenerator(rescale=1./255, 

                                   rotation_range=30, 

                                   # zoom_range = 0.3, 

                                   width_shift_range=0.2,

                                   height_shift_range=0.2, 

                                   horizontal_flip = 'true')

train_generator = train_datagen.flow(X_train, y_train, batch_size=100, shuffle=False, seed=10)
val_datagen = ImageDataGenerator(rescale = 1./255)

val_generator = train_datagen.flow(X_test, y_test, shuffle=False, batch_size=100, seed=10)
# Train the model

model.fit_generator(train_generator,

                      steps_per_epoch = 175,

                      validation_data = val_generator,

                      validation_steps = 44,

                      epochs = 5,

                      verbose = 2)