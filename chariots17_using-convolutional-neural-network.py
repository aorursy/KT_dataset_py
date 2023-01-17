# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample_submission = pd.read_csv("../input/plant-pathology-2020-fgvc7/sample_submission.csv")

test = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")

train = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
train.head()
x=train['image_id']
import cv2

from tqdm.auto import tqdm

from glob import glob

import time, gc

import cv2

from keras import backend as K

import matplotlib.image as mpimg

from keras import backend as K

import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.models import clone_model

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization,Activation

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

from matplotlib import pyplot as plt

import seaborn as sns
img_size=128
train_image=[]

for name in train['image_id']:

    path='/kaggle/input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'

    img=cv2.imread(path)

    image=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)

    train_image.append(image)
fig, ax = plt.subplots(1, 4, figsize=(15, 15))

for i in range(4):

    ax[i].set_axis_off()

    ax[i].imshow(train_image[i])
test.head()
test_image=[]

for name in test['image_id']:

    path='/kaggle/input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'

    img=cv2.imread(path)

    image=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)

    test_image.append(image)
fig, ax = plt.subplots(1, 4, figsize=(15, 15))

for i in range(4):

    ax[i].set_axis_off()

    ax[i].imshow(test_image[i])
#from keras.preprocessing.image import img_to_array

X_Train = np.ndarray(shape=(len(train_image), img_size, img_size, 3),dtype = np.float32)

i=0

for image in train_image:

    #X_Train[i]=img_to_array(image)

    X_Train[i]=train_image[i]

    i=i+1

X_Train=X_Train/255

print('Train Shape: {}'.format(X_Train.shape))
X_Test = np.ndarray(shape=(len(test_image), img_size, img_size, 3),dtype = np.float32)

i=0

for image in test_image:

    #X_Test[i]=img_to_array(image)

    X_Test[i]=test_image[i]

    i=i+1

    

X_Test=X_Test/255

print('Test Shape: {}'.format(X_Test.shape))
y = train.copy()

del y['image_id']

y.head()
y_train = np.array(y.values)

print(y_train.shape,y_train[0])
X_train, X_val, Y_train, Y_val = train_test_split(X_Train, y_train, test_size=0.2, random_state=42)
X_train.shape
X_val.shape
import keras

from keras.models import Sequential

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization
#模型的构建

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (128,128,3)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=0.15))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,3)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.4))

model.add(Dense(4, activation = "softmax"))
NUM_CLASSES = 4

EPOCHS = 50

BATCH_SIZE = 64
optmize = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)



model.compile(loss = keras.losses.categorical_crossentropy, optimizer = optmize,  metrics=['accuracy'])

History = model.fit(X_Train, y_train,

          batch_size = BATCH_SIZE,

          epochs = EPOCHS,

          verbose = 1)
model.evaluate(X_Train, y_train)
predict = model.predict(X_Test)

all_predict = np.ndarray(shape = (test.shape[0],4),dtype = np.float32)

for i in range(0,test.shape[0]):

    for j in range(0,4):

        if predict[i][j]==max(predict[i]):

            all_predict[i][j] = 1

        else:

            all_predict[i][j] = 0 
healthy = [y_test[0] for y_test in all_predict]

multiple_diseases = [y_test[1] for y_test in all_predict]

rust = [y_test[2] for y_test in all_predict]

scab = [y_test[3] for y_test in all_predict]
df = {'image_id':test.image_id,'healthy':healthy,'multiple_diseases':multiple_diseases,'rust':rust,'scab':scab}
data = pd.DataFrame(df)

data.tail()
data.to_csv('submission.csv',index = False)