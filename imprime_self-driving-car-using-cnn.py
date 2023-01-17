import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential

from keras.optimizers import Adam

from keras.layers import Convolution2D,Dense,MaxPooling2D,Dropout,Flatten

import cv2

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

import pandas as pd

import random

import ntpath
datadir = '/kaggle/input/selfdriving-car-dataset/data-master/'
columns = ['center','left','right','steering','throttle','reverse','speed']
dataset = pd.read_csv(os.path.join(datadir,'/kaggle/input/selfdriving-car-dataset/data-master/driving_log.csv'),names=columns)
os.path.join(datadir,'/kaggle/input/selfdriving-car-dataset/data-master/driving_log.csv')
dataset.head()
def removePath(path):

  base,tail = ntpath.split(path)

  return tail
dataset['center'] = dataset['center'].apply(removePath)
dataset['left'] = dataset['left'].apply(removePath)
dataset['right'] = dataset['right'].apply(removePath)
dataset.head()
num_bins = 25
hist,bins = np.histogram(dataset['steering'],num_bins)
print(hist)

print(bins)
center = (bins[:-1]+bins[1:])*0.5
center
center1 = []

for i in range(0,len(bins)-1):

  x = (bins[i] + bins[i+1]) * 0.5

  center1.append(x)
center1
threshold = 500

plt.figure(figsize=(15,10))

plt.bar(center,hist,width=0.05)

plt.xticks(np.linspace(-1,1,25),rotation=90)

(x1,x2) = (np.min(dataset['steering']),np.max(dataset['steering']))

(y1,y2) = (threshold,threshold)

plt.title('Steering Angles')

plt.plot((x1,x2),(y1,y2))
remove_list = []

for i in range(num_bins):

  List = []

  for j in range(len(dataset['steering'])):

    if dataset['steering'][j] >= bins[i] and dataset['steering'][j] <= bins[i+1]:

      List.append(j)

  List = shuffle(List)

  List = List[threshold:]

  remove_list.extend(List)
len(dataset['steering']) 
len(remove_list)
dataset.drop(dataset.index[remove_list],inplace=True)
hist,_ = np.histogram(dataset['steering'],num_bins)
plt.bar(center,hist,width=0.05)

plt.xticks(np.linspace(-1,1,25),rotation=90)

(x1,x2) = (np.min(dataset['steering']),np.max(dataset['steering']))

(y1,y2) = (threshold,threshold)

plt.title('Steering Angles')

plt.plot((x1,x2),(y1,y2))
dataset.iloc[1]
datadir
def loadImageSteering(datadir,dataset):

  imagePath = []

  steeringPath = []

  for i in range(len(dataset)):

    center = dataset.iloc[i][0]

    steering = float(dataset.iloc[i][3])

    imagePath.append(os.path.join(datadir,center))

    steeringPath.append(steering)

  imagePath = np.asarray(imagePath)

  steeringPath = np.asarray(steeringPath)

  return imagePath,steeringPath
dataset.iloc[0][0]
imagePath,steeringPath = loadImageSteering(datadir+'/IMG',dataset)
imagePath[0]
len(steeringPath)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(imagePath,steeringPath,random_state=6,test_size=0.2)
len(x_train)
plt.figure(figsize=(15,10))

plt.hist(y_train,bins=num_bins,width=0.05)

plt.xticks(np.linspace(-1,1,25),rotation=45)

plt.title("Training Dataset")

plt.show()
plt.figure(figsize=(15,10))

plt.hist(y_test,bins=num_bins,width=0.05)

plt.xticks(np.linspace(-1,1,25),rotation=45)

plt.title("Testing Dataset")

plt.show()
def imagePreprocessing(img):

  img = mpimg.imread(img)

  img = img[60:135,:,:]

  img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)

  img = cv2.GaussianBlur(img,(3,3),0)

  img = cv2.resize(img,(200,66))

  img = img/255

  return img
image = imagePath[1]

image = mpimg.imread(image)

fig,axs = plt.subplots(1,2,figsize=(15,10))

fig.tight_layout()

axs[0].imshow(image)

axs[0].grid(False)

axs[0].set_title("Original Image")

axs[1].imshow(imagePreprocessing(imagePath[1]))

axs[1].grid(False)

axs[1].set_title("Precessed Image")

plt.show()
x_train = np.array(list(map(imagePreprocessing,x_train)))
x_test = np.array(list(map(imagePreprocessing,x_test)))
# from google.colab import files
#upload = files.upload()
# img = plt.imread('nvidia architectue.png')
# plt.figure(figsize=(15,50))

# plt.imshow(img)

# plt.title("NVIDIA Architecture")
def nvidiaModel():

  model = Sequential()

  model.add(Convolution2D(24,(5,5),strides=(2,2),input_shape=(66,200,3),activation="elu"))

  model.add(Convolution2D(36,(5,5),strides=(2,2),activation="elu"))

  model.add(Convolution2D(48,(5,5),strides=(2,2),activation="elu")) 

  model.add(Convolution2D(64,(3,3),activation="elu"))   

  model.add(Convolution2D(64,(3,3),activation="elu"))

  model.add(Dropout(0.5))

  

  model.add(Flatten())

  

  model.add(Dense(100,activation="elu"))

  model.add(Dropout(0.5))

  

  model.add(Dense(50,activation="elu"))

  model.add(Dropout(0.5))

  

  model.add(Dense(10,activation="elu"))

  model.add(Dropout(0.5))

  

  model.add(Dense(1))

  model.compile(optimizer=Adam(lr=1e-3),loss="mse")

  

  return model
model = nvidiaModel()
model.summary()
h = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=30,batch_size=100,shuffle=1,verbose=1)
plt.plot(h.history['loss'])

plt.plot(h.history['val_loss'])
model.save('car.h5')
# from google.colab import files
# files.download('car.h5')
type('car.h5')