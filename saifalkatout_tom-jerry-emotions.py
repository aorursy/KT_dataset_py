import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import cv2 as cv
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import cv2 
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
      #for filename in filenames:
       #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
traindf=pd.read_csv("/kaggle/input/detect-emotions-of-your-favorite-toons/96714c94-6-Dataset/Dataset/Train.csv")
print(traindf.shape)
testdf=pd.read_csv("/kaggle/input/detect-emotions-of-your-favorite-toons/96714c94-6-Dataset/Dataset/Test.csv")
print(testdf.shape)

traindf.head()
testdf.head()
print(traindf.shape)
traindf.loc[traindf['Frame_ID'] == 'frame0.jpg']['Emotion']
class_names =np.unique(traindf['Emotion'])
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
print(class_names_label)
"""
import cv2
vidcap = cv2.VideoCapture('detect-emotions-of-your-favorite-toons/96714c94-6-Dataset/Dataset/Test Tom and jerry.mp4')

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("/kaggle/input/detect-emotions-of-your-favorite-toons/frames/train_frames/"+str(count)+".jpg", image) # save frame as JPG file
    return hasFrames,image
train_images =[]
IMAGE_SIZE = (150,150)
sec = 1
frameRate = 1 #//it will capture image in each 0.5 second
count=1
success,image = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
    
"""
IMAGE_SIZE = (150, 150)
dataset = '/kaggle/input/detect-emotions-of-your-favorite-toons/frames/train_frames'
output = []
train_images = []
train_labels = []
for files in tqdm(os.listdir(dataset)):
    try:
        label = class_names_label[traindf.loc[traindf['Frame_ID'] == files]['Emotion'].values[0]]
    except:
        #do nothing
        a=1
    img_path=os.path.join(dataset, files)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE) 
    train_images.append(image)
    train_labels.append(label)
train_images = np.array(train_images, dtype = 'float32')/255
train_labels = np.array(train_labels, dtype = 'int32') 

print(train_images[1].shape)
plt.imshow(train_images[1])
plt.show()
IMAGE_SIZE = (150, 150)
dataset = '/kaggle/input/detect-emotions-of-your-favorite-toons/frames/test_frames'
output = []
test_images = []
for files in tqdm(os.listdir(dataset)):
    img_path=os.path.join(dataset, files)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE) 
    test_images.append(image)
test_images = np.array(test_images, dtype = 'float32')/255 
print(test_images[2].shape)
plt.imshow(test_images[2])
plt.show()

from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(train_images,train_labels,test_size=0.3)
input_shape = x_train.shape[1:]
# Normalize data.
x_train = x_train.astype('float32') 
y_train = y_train.astype('float32') 
print(input_shape)
print(x_train.shape)
print(x_val.shape)
classifier = Sequential()


classifier.add(Conv2D(32,(3,3), input_shape = (150,150, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.5))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 300, activation = 'relu'))
classifier.add(Dense(output_dim = 100, activation = 'relu'))
classifier.add(Dense(output_dim = 5, activation = 'softmax'))

# Compiling the CNN
from keras import optimizers


classifier.compile(optimizer = Adam(lr=0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
epochs=100
classifier.fit(x_train,
             y_train,
             batch_size=32,
             nb_epoch=epochs,
             verbose=1,
             validation_data=(x_val,y_val))
classifier.summary()

scores = classifier.evaluate(x_val, y_val, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])