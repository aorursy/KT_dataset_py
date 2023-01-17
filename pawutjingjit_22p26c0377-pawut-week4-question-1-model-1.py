import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
######################################## Model 1 ######################################################3
import cv2
from skimage import feature
from skimage import measure
os.listdir('/kaggle/input/thai-mnist-classification')
train_img_path = '/kaggle/input/thai-mnist-classification/train'
train_label_path = '/kaggle/input/thai-mnist-classification/mnist.train.map.csv'
test_img_path = '/kaggle/input/thai-mnist-classification/test'
pd.read_csv(train_label_path)
from os import listdir
from os.path import isfile, join
mypath =  '/kaggle/input/thai-mnist-classification/test'
test_label_path = [f for f in listdir(mypath) if isfile(join(mypath, f))]
test_label_path[:3]
class getdata():
    def __init__(self,data_path,label_path):
        self.dataPath = data_path
        self.labelPath = label_path
        self.label_df = pd.read_csv(label_path)
        self.dataFile = self.label_df['id'].values
        self.label = self.label_df['category'].values
        self.n_index = len(self.dataFile)
        
    
    def get1img(self,img_index,mode='rgb',label = False):
        img = cv2.imread( os.path.join(self.dataPath,self.label_df.iloc[img_index]['id']) )
        if mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == 'gray':
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if label:
            return img,self.label_df.iloc[img_index]['category']
        return img
class getTestdata():
    def __init__(self,data_path,label_path):
        self.dataPath = data_path
        self.labelPath = label_path
        self.label_df = pd.DataFrame({'id':label_path})
        self.dataFile =  self.label_df['id'].values
        #self.label = self.label_df['category'].values
        self.n_index = len(self.dataFile)
        
    
    def get1img(self,img_index,mode='rgb',label = False):
        img = cv2.imread( os.path.join(self.dataPath,self.label_df.iloc[img_index]['id']) )
        if mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == 'gray':
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if label:
            return img,self.label_df.iloc[img_index]['category']
        return img
gdt = getdata(train_img_path,train_label_path)
gdtest = getTestdata(test_img_path,test_label_path)
test_label_path
gdt.get1img(23,'gray')
plt.gray()
from skimage.morphology import convex_hull_image
from skimage.util import invert
temp_img = invert(gdt.get1img(232,'gray'))
fig, [ax1,ax2] = plt.subplots(1, 2)
ax1.imshow(temp_img)
cvh =  convex_hull_image(temp_img)
ax2.imshow(cvh)
def convex_crop(img,pad=20):
    convex = convex_hull_image(img)
    r,c = np.where(convex)
    while (min(r)-pad < 0) or (max(r)+pad > img.shape[0]) or (min(c)-pad < 0) or (max(c)+pad > img.shape[1]):
        pad = pad - 1
    return img[min(r)-pad:max(r)+pad,min(c)-pad:max(c)+pad]
crop_img = convex_crop(temp_img,pad=10)
plt.imshow(crop_img)
def convex_resize(img):
    img = invert(img)
    img = convex_crop(img,pad=20)
    img = cv2.resize(img,(32,32))
    return img
def thes_resize(img,thes=40):
    img = invert(img)
    img = convex_crop(img,pad=20)
    img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 300):
        img = cv2.resize(img,(300,300))
        img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 150):
        img = cv2.resize(img,(150,150))
        img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(80,80))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(50,50))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(32,32))
    img = ((img > thes)*255).astype(np.uint8)
    return img
temp_img = gdt.get1img(64,'gray')
fig, [ax1,ax2] = plt.subplots(1, 2,figsize=(10,7))
ax1.imshow(convex_resize(temp_img))
ax1.set_title('Without thresholding')
ax2.imshow(thes_resize(temp_img))
ax2.set_title('Thresholding')
fig, ax = plt.subplots(5, 5, figsize=(15,15))
for i in range(5):
    for j in range(5):
        img_index = np.random.randint(0,gdt.n_index)
        ax[i][j].imshow(thes_resize(gdt.get1img(img_index,'gray')))
        ax[i][j].set_title('Class: '+str(gdt.label[img_index]))
        ax[i][j].set_axis_off()
X = []
for i in range(gdt.n_index):
    X.append(thes_resize(gdt.get1img(i,'gray')))
    if (i+1) % 100 == 0:
        print(i)
X = np.array(X)
Xtest = []
for i in range(gdtest.n_index):
    Xtest.append(thes_resize(gdtest.get1img(i,'gray')))
    if (i+1) % 100 == 0:
        print(i)
Xtest = np.array(Xtest)
y = gdt.label
X = X.reshape((-1,32,32,1))
X.shape,y.shape
import tensorflow as tf
y_cat = tf.keras.utils.to_categorical(y)
y_cat.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.25, random_state=1234)
X_train = X_train / 255.
X_test = X_test / 255.
Xtest = Xtest.reshape((-1,32,32,1))
Xtest = Xtest/255.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(6, (5,5), input_shape=(32, 32, 1), activation='relu'))
model.add(tf.keras.layers.MaxPool2D()) 
model.add(tf.keras.layers.Conv2D(16, (5,5), activation='relu')) 
model.add(tf.keras.layers.MaxPool2D()) 
model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dense(120, activation='relu'))
model.add(tf.keras.layers.Dense(84, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
model.summary()
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0000001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10,verbose=1)
history = model.fit(X_train, y_train, batch_size=64,validation_data=(X_test,y_test), epochs=100, callbacks=[learning_rate_reduction,early_stop])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
from tensorflow.keras import layers, Model
from tensorflow.keras.layers  import GlobalAveragePooling2D,Conv2D,BatchNormalization,Dropout,Flatten,Dense
from tensorflow.keras import Sequential

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience = 3, verbose=1,factor=0.3, min_lr=0.0000003)
from tensorflow.keras.applications import InceptionResNetV2 , ResNet152V2 , MobileNetV2 , ResNet50V2,NASNetMobile,ResNet101V2,InceptionV3,MobileNet 
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.applications import Xception,VGG19
from tensorflow.keras import layers, Model
from tensorflow.keras.layers  import GlobalAveragePooling2D , MaxPooling2D
from keras import backend as K
from tensorflow.keras import optimizers
ires  = MobileNetV2(include_top=False, weights='imagenet'  )
for l in ires.layers[:]:
  l.trainable  = False

x_in = layers.Input(shape=(32, 32, 1))
x = layers.Conv2D(3,(3,3),padding='same')(x_in) 
x = ires(x)
x=layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(10 , activation='softmax')(x)
model_1 = Model(x_in, x)
model_1.summary()
model_1.compile(loss='categorical_crossentropy' ,optimizer = optimizers.Adam(lr=1e-2),metrics=['acc'])
# BUILD CONVOLUTIONAL NEURAL NETWORKS
nets = 15
model = [0] *nets
for j in range(nets):
    model[j] = Sequential()

    model[j].add(layers.Conv2D(32, kernel_size = 3, activation='relu', input_shape = (32, 32, 1)))
    model[j].add(layers.BatchNormalization())
    model[j].add(layers.Conv2D(32, kernel_size = 3, activation='relu'))
    model[j].add(layers.BatchNormalization())
    model[j].add(layers.Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(layers.BatchNormalization())
    model[j].add(layers.Dropout(0.4))

    model[j].add(layers.Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(layers.BatchNormalization())
    model[j].add(layers.Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(layers.BatchNormalization())
    model[j].add(layers.Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(layers.BatchNormalization())
    model[j].add(layers.Dropout(0.4))

    model[j].add(layers.Conv2D(128, kernel_size = 4, activation='relu'))
    model[j].add(layers.BatchNormalization())
    model[j].add(layers.Flatten())
    model[j].add(layers.Dropout(0.4))
    model[j].add(layers.Dense(10, activation='softmax'))

    # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model[0].fit(X_train, y_train, batch_size=64,validation_data=(X_test,y_test), epochs=15, callbacks=[learning_rate_reduction,early_stop])
history = model[1].fit(X_train, y_train, batch_size=64,validation_data=(X_test,y_test), epochs=15, callbacks=[learning_rate_reduction,early_stop])
history = model[2].fit(X_train, y_train, batch_size=64,validation_data=(X_test,y_test), epochs=15, callbacks=[learning_rate_reduction,early_stop])
history = model[3].fit(X_train, y_train, batch_size=64,validation_data=(X_test,y_test), epochs=15, callbacks=[learning_rate_reduction,early_stop])
history = model[4].fit(X_train, y_train, batch_size=64,validation_data=(X_test,y_test), epochs=15, callbacks=[learning_rate_reduction,early_stop])
Xtest.shape
X_train.shape
pred_0 = model[0].predict(Xtest)
pred_1 =model[1].predict(Xtest)
pred_2 = model[2].predict(Xtest)
pred_3 = model[3].predict(Xtest)
pred_4 = model[4].predict(Xtest)
pred_0
_pred_0 =  np.argmax(pred_0,axis = 1)
_pred_1 =  np.argmax(pred_1,axis = 1)
_pred_2 =  np.argmax(pred_2,axis = 1)
_pred_3 =  np.argmax(pred_3,axis = 1)
_pred_4 =  np.argmax(pred_4,axis = 1)
_pred_0[:10]
_pred_1[:10]
test_label_path
temp = pd.DataFrame({'a':_pred_0,'b':_pred_1,'c':_pred_2,'d':_pred_3 , 'e':_pred_4})
temp[60:90]
def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num 
ens = [most_frequent([i['a'],i['b'],i['c']  , i['d'],i['e']])   for index , i in temp.iterrows()]
ens
s  = pd.DataFrame({ 'id': test_label_path, 'category':ens})
s.to_csv('s.csv',index = False)