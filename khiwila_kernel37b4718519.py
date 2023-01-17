# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train= pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")

test_0 = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
train.head()
train_0 = train.copy()

train_0 = train_0.drop(['healthy', 'multiple_diseases', 'rust','scab'], axis=1)

train_0.head()
labels_0 = train.copy()

labels_0 = labels_0.drop(['image_id','healthy'], axis=1)

labels_0 = np.array(labels_0.values)

labels_0.shape
test_0.head()
from matplotlib import pyplot as plt

import matplotlib.image as mpimg

img = []

img0=mpimg.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_631.jpg')

img1=mpimg.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_632.jpg')

img2=mpimg.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_633.jpg')

img3=mpimg.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_634.jpg')

img = (img0, img1, img2, img3)



fig, ax = plt.subplots(1, 4, figsize=(15, 15))

for i in range(4):

    ax[i].set_axis_off()

    ax[i].imshow(img[i])
import cv2





img_size = 128





train_images = []

for name in train_0['image_id'] :

    path = '/kaggle/input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'

    image = cv2.imread(path)

    image = cv2.resize(image,(img_size,img_size),interpolation=cv2.INTER_AREA)

    train_images.append(image)

test_images = []

for name in test_0['image_id'] :

    path = '/kaggle/input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'

    image = cv2.imread(path)

    image = cv2.resize(image,(img_size,img_size),interpolation=cv2.INTER_AREA)

    test_images.append(image)
X_train = np.ndarray(shape=(len(train_images),img_size,img_size,3),dtype=np.float32)

for i,image in enumerate(train_images) :

    X_train[i] = image/255

print('X_train.shape = ', X_train.shape) 



X_test = np.ndarray(shape=(len(test_images),img_size,img_size,3),dtype=np.float32)

for i,image in enumerate(test_images) :

    X_test[i] = image/255

print('X_test.shape = ', X_test.shape)
#from keras.models import Sequential

from keras.layers import Dense , Conv2D , Dropout , Flatten , Activation, MaxPooling2D , GlobalAveragePooling2D

from keras.optimizers import Adam , RMSprop 

from keras.regularizers import l2

from keras.layers.normalization import BatchNormalization

from keras.callbacks import ReduceLROnPlateau , EarlyStopping , ModelCheckpoint
import tensorflow as tf



from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[1], 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(3, activation='softmax')) 
model.summary()
from sklearn.model_selection import train_test_split

X_train_0, X_val_0, y_train_0, y_val_0 = train_test_split(X_train, labels_0, test_size=0.10, random_state=42)
model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])



history = model.fit(X_train_0, y_train_0, epochs=20,validation_data=(X_val_0, y_val_0))

plt.plot(history.history['accuracy'], label='train_accuracy')

plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

#plt.ylim([0.5, 1])

plt.legend(loc='lower right')
predict = model.predict(X_test)



prediction= np.zeros((predict.shape[0],predict.shape[1]))

for i in range(len(predict)):

    for j in range(3):

        if predict[i,j]>=0.5:

            prediction[i,j]=1

  
multiple_diseases = prediction[:,0]

rust = prediction[:,1]

scab = prediction[:,2]

health = multiple_diseases + rust + scab

healthy=np.ones(health.shape[0])

for i in range(3):#(len(health)):

    if health[i]==1:

        healthy[i]=0  
df = {'image_id':test_0.image_id,'healthy':healthy,'multiple_diseases':multiple_diseases,'rust':rust,'scab':scab}
submission = pd.DataFrame(df)

submission.tail()
submission.to_csv('submission.csv',index = False)