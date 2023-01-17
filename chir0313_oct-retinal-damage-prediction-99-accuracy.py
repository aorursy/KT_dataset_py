import tensorflow as tf
import numpy as np

import cv2

import os

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from keras.layers import Dropout,Flatten

from keras.layers.convolutional import Conv2D,MaxPooling2D

import pickle
path1 ="/kaggle/input/kermany2018/OCT2017 /train"

path2 ="/kaggle/input/kermany2018/OCT2017 /test"

path3 ="/kaggle/input/kermany2018/OCT2017 /val"

image_size=(128,128,3)

epochs = 10
myList = os.listdir(path1)

print("Total Number of Classes Detected :",len(myList))

print(myList)
noOfclasses= len(myList)
print(myList)
print("Importing Classes...")
images=[]

classNo=[]

CATEGORIES = ['NORMAL',"CNV","DME","DRUSEN"]



for x in myList:

  myPicList = os.listdir(path1+"/"+str(x))

  for y in myPicList:

    curImg = cv2.imread(path1+"/"+str(x)+"/"+y)

    curImg = cv2.resize(curImg,(image_size[0],image_size[1]))

    images.append(curImg)

    classNo.append(CATEGORIES.index(x)) #[0 1 2 3]

  print(x,end=" ")
x_test=[]

y_test=[]

CATEGORIES = ['NORMAL',"CNV","DME","DRUSEN"]

for x in myList:

  myPicList = os.listdir(path2+"/"+str(x))

  for y in myPicList:

    curImg = cv2.imread(path2+"/"+str(x)+"/"+y)

    curImg = cv2.resize(curImg,(image_size[0],image_size[1]))

    x_test.append(curImg)

    y_test.append(CATEGORIES.index(x))

  print(x,end=" ")
x_val=[]

y_val=[]

CATEGORIES = ['NORMAL',"CNV","DME","DRUSEN"]

for x in myList:

  myPicList = os.listdir(path3+"/"+str(x))

  for y in myPicList:

    curImg = cv2.imread(path3+"/"+str(x)+"/"+y)

    curImg = cv2.resize(curImg,(image_size[0],image_size[1]))

    x_val.append(curImg)

    y_val.append(CATEGORIES.index(x))

  print(x,end=" ")
print(len(images))

print(len(classNo))
x_train = np.array(images)

y_train = np.array(classNo)
x_test = np.array(x_test)

y_test = np.array(y_test)

x_val = np.array(x_val)

y_val = np.array(y_val)
print(x_train.shape)

print(x_test.shape)
del images

del classNo
print(x_train.shape)

print(x_test.shape)

print(x_val.shape)
numofSamples=[]

for x in range(0,noOfclasses):

  numofSamples.append(len(np.where(y_train==x)[0]))
print(numofSamples)
plt.figure(figsize=(10,5))

plt.bar(range(0,noOfclasses),numofSamples)

plt.title('No of Images for each Class')

plt.xlabel("Class ID")

plt.ylabel("No of Images")

plt.show()
from imblearn.under_sampling import RandomUnderSampler

import pandas as pd

import seaborn as sns



X_trainShape = x_train.shape[1]*x_train.shape[2]*x_train.shape[3] #49k



X_trainFlat = x_train.reshape(x_train.shape[0], X_trainShape)

Y_train = y_train





ros = RandomUnderSampler()

X_trainRos, Y_trainRos = ros.fit_sample(X_trainFlat, Y_train)





# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])[0010]

Y_trainRosHot = to_categorical(Y_trainRos, num_classes = 4)

# Make Data 2D again

for i in range(len(X_trainRos)):

    height, width, channels = image_size[0],image_size[1],3

    X_trainRosReshaped = X_trainRos.reshape(len(X_trainRos),height,width,channels)

# Plot Label Distribution

dfRos = pd.DataFrame()

dfRos["labels"]=Y_trainRos

labRos = dfRos['labels']

sns.countplot(labRos)
del x_train

del y_train
x_train = X_trainRosReshaped

print(x_train[0].shape)
print(x_test[0].shape)
#print(x_validation[0].shape)

x_validation = x_val

print(x_validation[0].shape)
print(x_train.shape)
X_train=x_train

print(X_train.shape)

X_test=x_test

X_validation = x_validation
del x_test

del x_train

del x_validation
y_train = to_categorical(Y_trainRos,noOfclasses)

y_test = to_categorical(y_test,noOfclasses)

y_validation = to_categorical(y_val,noOfclasses)
print(X_train.shape)

print(Y_trainRosHot.shape)

print(X_validation.shape)

print( y_validation.shape)
class CustomCallback(tf.keras.callbacks.Callback):

  def __init__(self,fraction):

    super(CustomCallback,self).__init__()

    self.fraction = fraction

    self.train_a = [];

    self.val_a =[];



    with open('log.txt','w') as f:

      f.write('Starting of logging..\n')



    self.fig = plt.figure(figsize=(4,3))

    self.ax = plt.subplot(1,1,1)

    plt.ion()



  def on_train_begin(self,logs=None):

    self.fig.show()

    self.fig.canvas.draw()

  

  def on_train_end(self,logs=None):

    with open('log.txt','a') as f:

      f.write('End of logging..\n')

  def on_epoch_begin(self,epoch,logs=None):

    lr= tf.keras.backend.get_value(self.model.optimizer.lr)

    lr *= self.fraction

    tf.keras.backend.set_value(self.model.optimizer.lr,lr)

    with open('log.txt','a') as f:

      f.write('At epoch {:02d}, learning rate changed to {:.4f}\n'.format(epoch,lr))

  def on_epoch_end(self,epoch,logs=None):

    val_acc = logs.get('val_accuracy')

    train_acc = logs.get('accuracy')

    self.train_a.append(train_acc)

    self.val_a.append(val_acc)

    with open('log.txt','a') as f:

        f.write('At epoch {:02d}, training accuracy: {:.3f}, validation accuracy: {:.3f}\n'.format(epoch,train_acc,val_acc))

    self.ax.clear()

    self.ax.plot(range(1+epoch),self.train_a,label="Training")

    self.ax.plot(range(1+epoch),self.val_a,label="Validation")

    self.ax.set_xlabel('Epochs')

    self.ax.set_ylabel('Accuracy')

    self.ax.legend()

    self.fig.canvas.draw()

    self.fig.show()
from keras.applications.vgg16 import VGG16

model = Sequential()

model.add(VGG16(include_top=False, input_shape=(image_size[0],image_size[1],image_size[2])))

model.add(Flatten())

model.add(Dense(noOfclasses,activation="softmax"))

model.summary()

model.compile(Adam(lr=0.0001),loss="categorical_crossentropy",metrics=['accuracy'])

history = model.fit(X_train,Y_trainRosHot,epochs=epochs,validation_data =(X_validation,y_validation) ,batch_size=32,

                    shuffle=True,

                    max_queue_size=20,

                    use_multiprocessing=True,

                    workers=1,

                   callbacks=[CustomCallback(fraction=0.9)])
plt.figure(1)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['training','validation'])

plt.title('Loss')

plt.xlabel('epoch')

plt.figure(2)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend(['training','validation'])

plt.title('Accuracy')

plt.xlabel('epoch')

plt.show()
score = model.evaluate(X_test,y_test,verbose=0)

print('Test Score = ',score[0])

print('Test Accuracy = ',score[1])
model.save('VGG16_model_trained'+str(epochs)+'.model')
from keras.applications.vgg19 import VGG19

model = Sequential()

model.add(VGG19(include_top=False, input_shape=(image_size[0],image_size[1],image_size[2])))

model.add(Flatten())

model.add(Dense(noOfclasses,activation="softmax"))

model.summary()

model.compile(Adam(lr=0.0001),loss="categorical_crossentropy",metrics=['accuracy'])



history = model.fit(X_train,Y_trainRosHot,epochs=epochs,validation_data =(X_validation,y_validation) ,batch_size=32,

                    shuffle=True,

                    max_queue_size=20,

                    use_multiprocessing=True,

                    workers=1,

                   callbacks=[CustomCallback(fraction=0.9)])
plt.figure(1)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['training','validation'])

plt.title('Loss')

plt.xlabel('epoch')

plt.figure(2)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend(['training','validation'])

plt.title('Accuracy')

plt.xlabel('epoch')

plt.show()
score = model.evaluate(X_test,y_test,verbose=0)

print('Test Score = ',score[0])

print('Test Accuracy = ',score[1])
model.save('VGG19_model_trained'+str(epochs)+'.model')
from keras.applications import InceptionV3

model = Sequential()

model.add(InceptionV3(include_top=False, input_shape=(image_size[0],image_size[1],image_size[2])))

model.add(Flatten())

model.add(Dense(noOfclasses,activation="softmax"))

model.summary()

model.compile(Adam(lr=0.0001),loss="categorical_crossentropy",metrics=['accuracy'])

history = model.fit(X_train,Y_trainRosHot,epochs=epochs,validation_data =(X_validation,y_validation) ,batch_size=32,

                    shuffle=True,

                    max_queue_size=20,

                    use_multiprocessing=True,

                    workers=1,

                   callbacks=[CustomCallback(fraction=0.9)])
plt.figure(1)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['training','validation'])

plt.title('Loss')

plt.xlabel('epoch')

plt.figure(2)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend(['training','validation'])

plt.title('Accuracy')

plt.xlabel('epoch')

plt.show()
score = model.evaluate(X_test,y_test,verbose=0)

print('Test Score = ',score[0])

print('Test Accuracy = ',score[1])
model.save('inceptionv3_model_trained'+str(epochs)+'.model')


from keras.applications import ResNet50

model = Sequential()

model.add(ResNet50(include_top=False, input_shape=(image_size[0],image_size[1],image_size[2])))

model.add(Flatten())

model.add(Dense(noOfclasses,activation="softmax"))

model.summary()

model.compile(Adam(lr=0.0001),loss="categorical_crossentropy",metrics=['accuracy'])

history = model.fit(X_train,Y_trainRosHot,epochs=epochs,validation_data =(X_validation,y_validation) ,batch_size=32,

                    shuffle=True,

                    max_queue_size=20,

                    use_multiprocessing=True,

                    workers=1,

                   callbacks=[CustomCallback(fraction=0.9)])
plt.figure(1)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['training','validation'])

plt.title('Loss')

plt.xlabel('epoch')

plt.figure(2)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend(['training','validation'])

plt.title('Accuracy')

plt.xlabel('epoch')

plt.show()
score = model.evaluate(X_test,y_test,verbose=0)

print('Test Score = ',score[0])

print('Test Accuracy = ',score[1])
model.save('resnet50_model_trained'+str(epochs)+'.model')
from keras.applications import MobileNetV2

model = Sequential()

model.add(MobileNetV2(include_top=False, input_shape=(image_size[0],image_size[1],image_size[2])))

model.add(Flatten())

model.add(Dense(noOfclasses,activation="softmax"))

model.summary()

model.compile(Adam(lr=0.0001),loss="categorical_crossentropy",metrics=['accuracy'])

history = model.fit(X_train,Y_trainRosHot,epochs=epochs,validation_data =(X_validation,y_validation) ,batch_size=32,

                    shuffle=True,

                    max_queue_size=20,

                    use_multiprocessing=True,

                    workers=1,

                   callbacks=[CustomCallback(fraction=0.9)])
plt.figure(1)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['training','validation'])

plt.title('Loss')

plt.xlabel('epoch')

plt.figure(2)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend(['training','validation'])

plt.title('Accuracy')

plt.xlabel('epoch')

plt.show()
score = model.evaluate(X_test,y_test,verbose=0)

print('Test Score = ',score[0])

print('Test Accuracy = ',score[1])
model.save('mobilenetv2_model_trained'+str(epochs)+'.model')
def model():

  filters=60

  sizeoffilter1 = (5,5)

  sizeoffilter2 = (4,4)

  sizeoffilter3 = (3,3)

  sizeofpool = (2,2)

  node=5000



  model = Sequential();

  model.add((Conv2D(filters,sizeoffilter1,input_shape=(image_size[0],image_size[1],image_size[2]),activation="relu")))



  model.add((Conv2D(filters,sizeoffilter1,activation="relu")))

  model.add((Conv2D(filters//2,sizeoffilter2,activation="relu")))

  model.add((Conv2D(filters//2,sizeoffilter2,activation="relu")))

  model.add(MaxPooling2D(pool_size=sizeofpool))

  model.add(Dropout(0.2))



  model.add((Conv2D(filters,sizeoffilter1,activation="relu")))

  model.add((Conv2D(filters//2,sizeoffilter2,activation="relu")))

  model.add((Conv2D(filters//2,sizeoffilter2,activation="relu")))

  model.add(MaxPooling2D(pool_size=sizeofpool))

  model.add(Dropout(0.2))



  model.add((Conv2D(filters,sizeoffilter2,activation="relu")))

  model.add(MaxPooling2D(pool_size=sizeofpool))

  model.add((Conv2D(filters//2,sizeoffilter3,activation="relu")))

  model.add((Conv2D(filters//2,sizeoffilter3,activation="relu")))

  model.add(MaxPooling2D(pool_size=sizeofpool))

  model.add(Dropout(0.2))



  model.add(Flatten())

  model.add(Dense(node,activation="relu"))

  model.add(Dropout(0.2))

  model.add(Dense(noOfclasses,activation="softmax"))



  model.compile(Adam(lr=0.001),loss="categorical_crossentropy",metrics=['accuracy'])



  return model

  

model=model()
model.summary()


history = model.fit(X_train,Y_trainRosHot,epochs=epochs,validation_data =(X_validation,y_validation) ,batch_size=32,

                    shuffle=True,

                    max_queue_size=20,

                    use_multiprocessing=True,

                    workers=1,

                   callbacks=[CustomCallback(fraction=0.9)])
plt.figure(1)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['training','validation'])

plt.title('Loss')

plt.xlabel('epoch')

plt.figure(2)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend(['training','validation'])

plt.title('Accuracy')

plt.xlabel('epoch')

plt.show()

score = model.evaluate(X_test,y_test,verbose=0)

print('Test Score = ',score[0])

print('Test Accuracy = ',score[1])
model.save('custommodel_trained'+str(epochs)+'.model')