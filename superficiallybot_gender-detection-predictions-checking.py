import keras

from keras.models import *

from keras.layers import *

from keras.optimizers import SGD

import cv2

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

import os
def model1_smallervgg(size = 224):

    model = Sequential()

    model.add(Conv2D(32, (3,3), padding = 'same', input_shape = (size, size, 3) , activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D((3,3)))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(64, (3,3), padding = 'same', activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, (3,3), padding = 'same', activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(128, (3,3), padding = 'same', activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(128, (3,3), padding = 'same', activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(0.25))

    

    model.add(Flatten())

    model.add(Dense(1024, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    model.add(Dense(1, activation = 'sigmoid'))

    #opt = keras.optimizers.Adam(lr=lr, decay=lr/epochs)

    #model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    

    return model
model1 = model1_smallervgg()
model1.load_weights('../input/gender-detection-from-face-another-dataset/model1.h5')
model1.summary()
females = ['https://media-exp1.licdn.com/dms/image/C4D03AQFjvXb55AwrTA/profile-displayphoto-shrink_200_200/0?e=1592438400&v=beta&t=FjMIH5j4dSlFgDU9b0Ed810-yPR4WIvkAiW3HhFwrfY',

          'https://media-exp1.licdn.com/dms/image/C4D03AQECKT9M0okpKw/profile-displayphoto-shrink_200_200/0?e=1592438400&v=beta&t=1EjMz3PtI8gdBwQNWLAQPBkFNoJsWDHgiJ99pSVVUV8',

          'https://media-exp1.licdn.com/dms/image/C4D03AQEdnvvCRfSK4A/profile-displayphoto-shrink_200_200/0?e=1592438400&v=beta&t=7eyiTt8fUFkYJJMVb1Lu71mTdk8r5hnRMlog1e8XacE',

          'https://media-exp1.licdn.com/dms/image/C4D03AQFaf_NpuCGm_w/profile-displayphoto-shrink_200_200/0?e=1592438400&v=beta&t=-xWIpbuI7n4YRCyS1naUfe4_HvkWPZp07n1yOEAmZdY',

           'https://media-exp1.licdn.com/dms/image/C4D03AQHxkJVCPNS21A/profile-displayphoto-shrink_200_200/0?e=1592438400&v=beta&t=siM5a690BB5gtPeC3kjgnjxnfjeR-sEZrTEWbeUj8Pk'

          ]



males = ['https://media-exp1.licdn.com/dms/image/C4E03AQFOwIgCwqTKjQ/profile-displayphoto-shrink_200_200/0?e=1592438400&v=beta&t=RtMpQFCWzAeOsofTIVLjlVXA5S60hnZM4anCQ5gqY5g',

          'https://media-exp1.licdn.com/dms/image/C4E03AQHvhRIROINsgQ/profile-displayphoto-shrink_200_200/0?e=1592438400&v=beta&t=HjyaltGVVxN-T9WERA_UngeldQhbbfgPuqHfsmgZIrU',

        'https://media-exp1.licdn.com/dms/image/C4D03AQGHxRl5xyoMog/profile-displayphoto-shrink_200_200/0?e=1592438400&v=beta&t=bn6r8YCY-_YHSw3t9hmTAGlaQHSQlh436DPe-O4fceI',

         'https://media-exp1.licdn.com/dms/image/C5603AQENDOWr3WPJtQ/profile-displayphoto-shrink_200_200/0?e=1592438400&v=beta&t=2eaJwZmWGULN7VfWPF8KYcrXIHYMK33gp_AKJglvDIo',

         'https://media-exp1.licdn.com/dms/image/C5103AQGUmvpQ-mYMgg/profile-displayphoto-shrink_200_200/0?e=1592438400&v=beta&t=NJDTkBW570g5EFDOs-2fI9wuInCCL8XspO2IQinJxV8'

        ]
from PIL import Image

import requests

from io import BytesIO

#url = 'https://media-exp1.licdn.com/dms/image/C5103AQF10FzP5dk7yw/profile-displayphoto-shrink_200_200/0?e=1592438400&v=beta&t=RU5adt3zuPIMVGFm83XokMR5uRwwbPOdFnJgtHwnur8'

def image_loader_url(url, size = 100):

    response = requests.get(url)

    img = Image.open(BytesIO(response.content))

    #print(img,'\n\n\n')

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    #print(img.shape)

    img = cv2.resize(img, (size, size))

    #plt.imshow(img[:,:,::-1])

    img = img.astype('float32')

    img = img/255.0

    return img
lst = []

f = 0

m = 0

for i in range(10):

    if i % 2 == 0:

        lst.append(image_loader_url(females[f],224))

        f+= 1

    else:

        lst.append(image_loader_url(males[m], 224))

        m += 1

lst = np.array(lst)

        
lst.shape
females = [os.path.join('../input/linkedin-profile-pic-data/LinkedIn Test Images/Female',i) for i in os.listdir('../input/linkedin-profile-pic-data/LinkedIn Test Images/Female')]

females
males = [os.path.join('../input/linkedin-profile-pic-data/LinkedIn Test Images/Male',i) for i in os.listdir('../input/linkedin-profile-pic-data/LinkedIn Test Images/Male')]

males
lst = []

f = 0

m = 0

for i in range(10):

    if i % 2 == 0:

        lst.append(np.array(Image.open(females[f]).convert('RGB'))[:,:,::-1])

        f+= 1

    else:

        lst.append(np.array(Image.open(males[m]).convert('RGB'))[:,:,::-1])

        m += 1

lst = np.array(lst)
lst
lst.shape
print('Image #1')

plt.imshow(lst[0][:,:,::-1])
print('Image #2')

plt.imshow(lst[1][:,:,::-1])
print('Image #3')

plt.imshow(lst[2][:,:,::-1])
print('Image #4')

plt.imshow(lst[3][:,:,::-1])
print('Image #5')

plt.imshow(lst[4][:,:,::-1])
print('Image #6')

plt.imshow(lst[5][:,:,::-1])
print('Image #7')

plt.imshow(lst[6][:,:,::-1])
print('Image #8')

plt.imshow(lst[7][:,:,::-1])
print('Image #9')

plt.imshow(lst[8][:,:,::-1])
print('Image #10')

plt.imshow(lst[9][:,:,::-1])
actual = ['Female', 'Male','Female', 'Male','Female', 'Male','Female', 'Male','Female', 'Male']
preds = model1.predict_classes(lst)

result = []

for i, item in enumerate(preds):

    if item == 1:

        result.append('Male')

    else:

        result.append('Female')
result
score = 0

for i in zip(actual, result):

    if i[0] == i[1]:

        score +=1 



print('Accuracy :' , score/10)
import numpy as np

import matplotlib.pyplot as plt



w=10

h=10

fig=plt.figure(figsize=(20, 20))

columns = 4

rows = 5

j = 0

fig.suptitle('Result of the model prediction')



for i in range(1, columns*rows +1):

    

    img = Image.fromarray(lst[j][:,:,::-1])

    

    fig.add_subplot(rows, columns, i).set_title('Pred->' + result[j] +  ', ' + 'Actual ->' +  actual[j])

    plt.imshow(img)

    j+=1

    if j == 10:

        break

plt.show()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import cv2

import matplotlib.pyplot as plt

import os

import seaborn as sns

import umap

from PIL import Image

from scipy import misc

from os import listdir

from os.path import isfile, join

import numpy as np

from scipy import misc

from random import shuffle

from collections import Counter

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.utils.np_utils import to_categorical
model2 = tf.keras.Sequential()



# Must define the input shape in the first layer of the neural network

model2.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(128,128,3))) 

model2.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model2.add(tf.keras.layers.Dropout(0.3))



model2.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))

model2.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model2.add(tf.keras.layers.Dropout(0.3))



model2.add(tf.keras.layers.Flatten())

model2.add(tf.keras.layers.Dense(256, activation='relu'))

model2.add(tf.keras.layers.Dropout(0.5))

model2.add(tf.keras.layers.Dense(2, activation='sigmoid'))



# Take a look at the model summary

model2.summary()
model2.load_weights('../input/gender-group-classification-with-cnn-obtain-model/model2.h5')
"""lst = []

f = 0

m = 0

for i in range(10):

    if i % 2 == 0:

        lst.append(image_loader_url(females[f],32))

        f+= 1

    else:

        lst.append(image_loader_url(males[m], 32))

        m += 1

lst = np.array(lst)

        """
lst = []

f = 0

m = 0

for i in range(10):

    if i % 2 == 0:

        lst.append(np.array(Image.open(females[f]).resize((128,128)).convert('RGB'))[:,:,::-1])

        f+= 1

    else:

        lst.append(np.array(Image.open(males[m]).resize((128,128)).convert('RGB'))[:,:,::-1])

        m += 1

lst = np.array(lst)
preds = model2.predict_classes(lst)

result = []

for i, item in enumerate(preds):

    if item == 0:

        result.append('Male')

    else:

        result.append('Female')
result
score = 0

for i in zip(actual, result):

    if i[0] == i[1]:

        score +=1 



print('Accuracy :' , score/10)
import numpy as np

import matplotlib.pyplot as plt



w=10

h=10

fig=plt.figure(figsize=(20, 20))

columns = 4

rows = 5

j = 0

fig.suptitle('Result of the model prediction')



for i in range(1, columns*rows +1):

    

    img = Image.fromarray(lst[j][:,:,::-1])

    

    fig.add_subplot(rows, columns, i).set_title('Pred->' + result[j] +  ', ' + 'Actual ->' +  actual[j])

    plt.imshow(img)

    j+=1

    if j == 10:

        break

plt.show()