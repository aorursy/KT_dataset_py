# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from PIL import Image

import os

print(os.listdir("../input"))

from keras import backend as K

K.tensorflow_backend._get_available_gpus()

# Any results you write to the current directory are saved as output.
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   

ls=os.listdir('../input/dogs-vs-cats-redux-kernels-edition/train')[:5]
from keras.preprocessing import image
img=image.load_img('../input/dogs-vs-cats-redux-kernels-edition/train/dog.1501.jpg',target_size=(224, 224))

img
TRAIN_DIR='../input/dogs-vs-cats-redux-kernels-edition/train/'

TEST_DIR = '../input/dogs-vs-cats-redux-kernels-edition/test/'
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]
train_images = train_dogs[:9000] + train_cats[:9000]
import random
random.shuffle(train_images)
train_images[:5]
import cv2 
ROWS=244

COLS=244

CHANNELS=3
def read_image(file_path):

    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE

    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)





def prep_data(images):

    count = len(images)

    data = np.ndarray((count,  ROWS, COLS,CHANNELS), dtype=np.uint8)



    for i, image_file in enumerate(images):

        image = read_image(image_file)

        data[i] = image

        if i%1000 == 0: print('Processed {} of {}'.format(i, count))

    

    return data

test_images =  test_images[:25]
train = prep_data(train_images)

test = prep_data(test_images)
print("Train shape: {}".format(train.shape))

print("Test shape: {}".format(test.shape))
import seaborn as sns

import matplotlib.pyplot as plt
labels = []

for i in train_images:

    if 'dog.' in i:

        labels.append(1)

    else:

        labels.append(0)



sns.countplot(labels)
def show_cats_and_dogs(idx):

    cat = read_image(train_cats[idx])

    dog = read_image(train_dogs[idx])

    pair = np.concatenate((cat, dog), axis=1)

    plt.figure(figsize=(10,5))

    plt.imshow(pair)

    plt.show()

    

for idx in range(0,5):

    show_cats_and_dogs(idx)
from keras import backend as k
from keras.applications.vgg16 import VGG16

vgg16_weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

vgg16_model = VGG16(weights=vgg16_weights,input_shape=(244,244,3))
from keras.layers import Dense,GlobalAveragePooling1D,Dropout



from keras.models import Model
for i in vgg16_model.layers:

    i.trainable = False
x =  vgg16_model.output

x = Dense(256, activation='relu')(x)   

x = Dense(256, activation='relu')(x)   

x = Dense(1, activation='sigmoid')(x)  
model = Model(inputs=vgg16_model.input, outputs=x)
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.optimizers import Adam,SGD
model.compile(loss='categorical_hinge', optimizer=Adam(), metrics=['binary_crossentropy'])
epochs=10

batch_size=50

red =ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
history=model.fit(train,labels, batch_size=batch_size,epochs = epochs,verbose =1,validation_split=0.3,callbacks=[red,early_stopping])
plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()
predictions=model.predict(test)[:,0]

predictions
fig = plt.figure(figsize=(40, 40))

for i,j in enumerate(test):

    str1=''

    if predictions[i] >= 0.5: 

        str1=('I am {:.2%} sure this is a Dog'.format(predictions[i]))

    else: 

        str1=('I am {:.2%} sure this is a Cat'.format(1-predictions[i]))

    ax = fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])    

    plt.imshow(j)

    ax.set_title(str1)
model.save('../input/vgg_model.h5')
os.listdir('../input')
from keras.models  import load_model
new_model = load_model("../input/vgg_model.h5")
predictions=new_model.predict(test)[:,0]

predictions
fig = plt.figure(figsize=(40, 40))

for i,j in enumerate(test):

    str1=''

    if predictions[i] >= 0.5: 

        str1=('I am {:.2%} sure this is a Dog'.format(predictions[i]))

    else: 

        str1=('I am {:.2%} sure this is a Cat'.format(1-predictions[i]))

    ax = fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])    

    plt.imshow(j)

    ax.set_title(str1)