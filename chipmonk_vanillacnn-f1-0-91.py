# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import numpy as np
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import sys
import tensorflow as tf
import keras
import glob
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import csv
import os
print(os.listdir("../input/plant-seedlings-classification/"))

# Any results you write to the current directory are saved as output.
plants=os.listdir("../input/plant-seedlings-classification/train/")
path='../input/plant-seedlings-classification/train/'
print(plants)
meanl=[]
meanw=[]
minl=[]
minw=[]
maxl=[]
maxw=[]
for k in plants:
    leng=[]
    wid=[]
    print(k)
    l=os.listdir(path+k)
    for j in l:
        im=cv2.imread(path+k+'/'+j)
        leng.append(im.shape[0])
        wid.append(im.shape[1])
    meanl.append(int(sum(leng)/len(leng)))
    meanw.append(int(sum(wid)/len(wid)))
    minl.append(int(min(leng)))
    minw.append(int(min(wid)))
    maxl.append(int(max(leng)))
    maxw.append(int(max(leng)))
print(meanl,meanw,maxl,maxw,minl,minw)
def load_train(path,plants,size):
    images=[]
    labels=[]
    ids=[]
    clas=[]
    for fold in plants:
        index=plants.index(fold)
        pat = os.path.join(path, fold)
        files = glob.glob(pat+'/*')
        for f1 in files:
            im=cv2.imread(f1)
            image = cv2.resize(im, (size, size), cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(plants))
            label[index] = 1.0
            labels.append(label)
            #filename = os.path.basename(f1)
            #ids.append(filename)
            #clas.append(fold)
    images = np.array(images)
    labels = np.array(labels)
        #ids = np.array(ids)
        #cls = np.array(cls)
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)
    return images,labels
plants=os.listdir("../input/plant-seedlings-classification/train/")
path='../input/plant-seedlings-classification/train/'
img_train,y_train=load_train(path,plants,100)
x_train,x_test,y_train,y_test=train_test_split(img_train,y_train,test_size=0.25,random_state=7)
print(x_train.shape,x_test.shape)
batch_size = 32
num_classes = 12
epochs = 30
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'plant_detection_trained_model.h5'
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(x_train)
model.fit_generator(datagen.flow(x_train, y_train,
                                    batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    workers=4,verbose=1)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

def load_test(path,size):
  files = sorted(glob.glob(path+'/*'))
  X_test=[]
  X_test_id=[]
  for f in files:
      filename = os.path.basename(f)
      img = cv2.imread(f)
      img = cv2.resize(img, (size, size), cv2.INTER_LINEAR)
      X_test.append(img)
      X_test_id.append(filename)

  X_test = np.array(X_test, dtype=np.uint8)
  X_test = X_test.astype('float32')
  X_test = X_test / 255

  return X_test, X_test_id
path='../input/plant-seedlings-classification/test/'
x_test,test_id=load_test(path,100)
# print int(model.predict_classes(x_test[793:],batch_size=2,verbose=1))
# print test_id[1]
plt.imshow(x_test[1])
plt.show()
result=list(model.predict_classes(x_test))
res=[plants[x] for x in result]
sample_sub = pd.read_csv('../input/plant-seedlings-classification/sample_submission.csv')
sample_sub.head()
def finalprediction():
    results = [[0,0]]*(x_test.shape[0])
    plants=os.listdir('../input/train/')
    result=list(model.predict_classes(x_test))
    res=[plants[x] for x in result]
    tid=list(test_id)
    results=[[x,y] for x,y in zip(tid,res)]
    results=np.array(results)
    title=[[0,0]]
    title[0][0]='file'
    title[0][1]='species'
    with open("../input/prediction.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(title)
        writer.writerows(results)
# finalprediction()