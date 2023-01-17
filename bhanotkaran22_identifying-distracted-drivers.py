import os

import tensorflow

os.environ['KERAS_BACKEND'] = 'tensorflow'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = INFO, WARNING, and ERROR messages are not printed



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.preprocessing.image import ImageDataGenerator
dataset = pd.read_csv('../input/driver_imgs_list.csv')

dataset.head(5)
import os

from IPython.display import display, Image

import matplotlib.image as mpimg



activity_map = {'c0': 'Safe driving', 

                'c1': 'Texting - right', 

                'c2': 'Talking on the phone - right', 

                'c3': 'Texting - left', 

                'c4': 'Talking on the phone - left', 

                'c5': 'Operating the radio', 

                'c6': 'Drinking', 

                'c7': 'Reaching behind', 

                'c8': 'Hair and makeup', 

                'c9': 'Talking to passenger'}



plt.figure(figsize = (12, 20))

image_count = 1

BASE_URL = '../input/train/'

for directory in os.listdir(BASE_URL):

    if directory[0] != '.':

        for i, file in enumerate(os.listdir(BASE_URL + directory)):

            if i == 1:

                break

            else:

                fig = plt.subplot(5, 2, image_count)

                image_count += 1

                image = mpimg.imread(BASE_URL + directory + '/' + file)

                plt.imshow(image)

                plt.title(activity_map[directory])
classifier = Sequential()

classifier.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', input_shape = (240, 240, 3), data_format = 'channels_last'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 1024, activation = 'relu'))

classifier.add(Dense(units = 256, activation = 'relu'))

classifier.add(Dense(units = 10, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.summary()
train_datagen = ImageDataGenerator(rescale = 1.0/255, 

                                   shear_range = 0.2, 

                                   zoom_range = 0.2, 

                                   horizontal_flip = True, 

                                   validation_split = 0.2)



training_set = train_datagen.flow_from_directory('../input/train', 

                                                 target_size = (240, 240), 

                                                 batch_size = 32,

                                                 subset = 'training')



validation_set = train_datagen.flow_from_directory('../input/train', 

                                                   target_size = (240, 240), 

                                                   batch_size = 32,

                                                   subset = 'validation')
classifier.fit_generator(training_set,

                         steps_per_epoch = 17943/32,

                         epochs = 10,

                         validation_data = validation_set,

                         validation_steps = 4481/32)
from PIL import Image



def get_data(image_path):

    img = Image.open(image_path)

    img = img.resize((240, 240), Image.ANTIALIAS) # resizes image in-place

    return np.asarray(img)/255
test_file = pd.read_csv('../input/sample_submission.csv')

test_file.head(5)
for i, file in enumerate(test_file['img']):

    image = get_data('../input/test/' + file)

    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))

    result = classifier.predict(image)

    test_file.iloc[i, 1:] = result[0]
test_file.to_csv('results.csv', index = False)