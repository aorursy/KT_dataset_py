import pandas as pd
import os
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
# KERAS AND SKLEARN MODULES


#from tf.keras.utils import np_utils
#from tf.keras.preprocessing.image import ImageDataGenerator
#from tf.keras.models import Sequential
#from tf.keras.layers import Dense
#from tf.keras.layers import Dropout
#from tf.keras.layers import Flatten
#from tf.keras.layers.convolutional import Conv2D
#from tf.keras.layers.convolutional import MaxPooling2D
#from tf.keras.layers import BatchNormalization
#from tf.keras.callbacks import ModelCheckpoint
print(tf.__version__)
path=os.chdir('/kaggle/input/plant-seedlings-classification/train/')
os.listdir()
category_flowernames={'Cleavers' : '0','Common wheat' : '1', 'Black-grass' : '2', 'Maize' : '3', 'Fat Hen' : '4', 'Common Chickweed' : '5', 'Loose Silky-bent' : '6', 'Sugar beet' : '7', 'Scentless Mayweed' : '8', 'Charlock' : '9', 'Small-flowered Cranesbill' : '10', 'Shepherds Purse' : '11'}
#category_flowernames={'Cleavers' : '0', 'Fat Hen' : '1', 'Shepherds Purse' : '2'}
imgs=[]
imgs_label=[]
y_test=[]
y_train=[]
for category,(dirpath,dirname,filelist) in enumerate(os.walk(top='/kaggle/input/plant-seedlings-classification/train/')):
  #for dirpath in name:
  if dirpath == path :
    continue;
    
  print(category_flowernames.get(dirpath.split('/')[-1]))
  print(dirpath)
  imgs.append([Image.open((os.path.join(dirpath,filename))).convert('RGBA') for filename in filelist if os.path.isfile(os.path.join(dirpath,filename))])
  imgs_label.append([category_flowernames.get(dirpath.split('/')[-1]) for num in range(0,len(filelist))])
  print(dirpath);
imgs=np.array(imgs)
imgs.shape
allimgs=([im for li in imgs for im in li])
alllabels=([la for li in imgs_label for la in li])
len(alllabels)
allimgs[1].resize((100,100))
allimgs_new=[]
alllabels_new=[]
allimgs_resize=[]
allimgs_new=allimgs
alllabels_new=alllabels
Scale=120
for im1 in range(0,len(alllabels_new)):
  allimgs_new[im1]=allimgs_new[im1].resize((Scale, Scale))
X_train,X_test,y_train,y_test=train_test_split(allimgs_new,alllabels_new,test_size=0.2,random_state=5)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True) # randomly flip images

for im1 in range(0,len(X_train)):
    X_train[im1]=np.asarray(X_train[im1])
for im1 in range(0,len(X_test)):
    X_test[im1]=np.asarray(X_test[im1])
X_test=np.asarray(X_test,dtype='float32')
X_train=np.asarray(X_train,dtype='float32')
datagen.fit(X_train)
X_train=X_train/255
from keras.utils import to_categorical
y_train=np.asarray(y_train,dtype='float32')
y_test=np.asarray(y_test,dtype='float32')
y_train=tf.keras.utils.to_categorical(y_train,num_classes=12)
y_test=tf.keras.utils.to_categorical(y_test,num_classes=12)
X_test=X_test/255
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='/kaggle/working/model-{epoch:03d}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
callbacks_list = [checkpoint]
model = tf.keras.Sequential()

#model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(None, None, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=(120, 120, 4), activation='relu',name="conv1"))
model.add(tf.keras.layers.BatchNormalization(axis=3))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization(axis=3))
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), activation='relu'))
model.add(tf.keras.layers.BatchNormalization(axis=3))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization(axis=3))
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(2, 2), activation='relu'))
model.add(tf.keras.layers.BatchNormalization(axis=3))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(2, 2), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization(axis=3))
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(12,activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#model.fit(X_train,y_train,epochs=30,batch_size=50,validation_data=(X_test,y_test),callbacks=callbacks_list)
model.fit(datagen.flow(X_train, y_train, batch_size=50), epochs=30,validation_data=(X_test,y_test),callbacks=callbacks_list)
model_loaded = tf.keras.models.load_model('/kaggle/working/model-017.h5')
data={'file': [],'species' : []}
sample_sub=pd.DataFrame(data=data)
imgs_test=[]
for category,(dirpath,dirname,filelist) in enumerate(os.walk('/kaggle/input/plant-seedlings-classification/test/')):
  #for dirpath in name:
  if dirpath == path :
    continue;
    
  print(category_flowernames.get(dirpath.split('/')[-1]))
  print(dirpath)
 # tmp_img=[Image.open((os.path.join(dirpath,filename))) for filename in filelist if os.path.isfile(os.path.join(dirpath,filename))]
  imgs_test.append([Image.open((os.path.join(dirpath,filename))).convert('RGBA') for filename in filelist if os.path.isfile(os.path.join(dirpath,filename))])
imgs_test
sample_sub['file']=filelist
allimgs_test=[]
allimgs_test=([im for li in imgs_test for im in li])
for im1 in range(0,len(allimgs_test)):
  allimgs_test[im1]=allimgs_test[im1].resize((Scale, Scale))
for im1 in range(0,len(allimgs_test)):
    allimgs_test[im1]=np.asarray(allimgs_test[im1])
allimgs_test=np.asarray(allimgs_test,dtype='float32')
allimgs_test=allimgs_test/255
model_loaded.predict_classes(allimgs_test)
predict_classes=model_loaded.predict_classes(allimgs_test)
sample_sub['species']=predict_classes
