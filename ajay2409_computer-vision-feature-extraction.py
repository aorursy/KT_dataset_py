# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

from keras.applications import VGG16

conv_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(150,150,3))
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
# location of dataset in 2 different folders
original_dataset_cat = '../input/originalcatvsdog/PetImages/Cat'
original_dataset_Dog = '../input/originalcatvsdog/PetImages/Dog'
import shutil

base_dir = '/kaggle/working/base_dir'
os.mkdir(base_dir)

test_dir = os.path.join(base_dir,'test')
os.mkdir(test_dir)

train_dir = os.path.join(base_dir,'train')
os.mkdir(train_dir)

valid_dir = os.path.join(base_dir,'valid')
os.mkdir(valid_dir)
test_dir_cat = os.path.join(test_dir,'cat')
os.mkdir(test_dir_cat)

test_dir_dog = os.path.join(test_dir,'dog')
os.mkdir(test_dir_dog)

train_dir_cat = os.path.join(train_dir,'cat')
os.mkdir(train_dir_cat)

train_dir_dog = os.path.join(train_dir,'dog')
os.mkdir(train_dir_dog)

valid_dir_cat = os.path.join(valid_dir,'cat')
os.mkdir(valid_dir_cat)

valid_dir_dog = os.path.join(valid_dir,'dog')
os.mkdir(valid_dir_dog)

fnames =['{}.jpg'.format(i) for i in range(1001)]
for fname in fnames:
    src = os.path.join(original_dataset_cat, fname)
    dst = os.path.join(train_dir_cat,fname)
    shutil.copyfile(src,dst)

fnames = ['{}.jpg'.format(i) for i in range(1001,1501)]
for fname in fnames:
    src = os.path.join(original_dataset_cat, fname)
    dst = os.path.join(valid_dir_cat,fname)
    shutil.copyfile(src,dst)
    
fnames = ['{}.jpg'.format(i) for i in range(1501,2001)]
for fname in fnames:
    src = os.path.join(original_dataset_cat, fname)
    dst = os.path.join(test_dir_cat,fname)
    shutil.copyfile(src,dst)  
    
fnames =['{}.jpg'.format(i) for i in range(1001)]
for fname in fnames:
    src = os.path.join(original_dataset_Dog, fname)
    dst = os.path.join(train_dir_dog,fname)
    shutil.copyfile(src,dst)

fnames = ['{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_Dog, fname)
    dst = os.path.join(valid_dir_dog,fname)
    shutil.copyfile(src,dst)
    
fnames = ['{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_Dog, fname)
    dst = os.path.join(test_dir_dog,fname)
    shutil.copyfile(src,dst)    
datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 20

def extract_features(directory,sample_count):
    features = np.zeros(shape = (sample_count,4,4,512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
    directory,
        target_size =(150,150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for input_batch,labels_batch in generator:
        features_batch = conv_base.predict(input_batch)
        features[ i* batch_size: (i+1) * batch_size] = features_batch
        labels[ i* batch_size :(i+1)* batch_size] = labels_batch
        i +=1
        if i * batch_size >= sample_count:
            break
    return features,labels        
    
train_features, train_labels = extract_features(train_dir, 2000)
os.remove('/kaggle/working/base_dir/train/cat/666.jpg')
train_features, train_labels = extract_features(train_dir, 2001)
test_features, test_labels = extract_features(test_dir, 1000)
valid_features, valid_labels = extract_features(valid_dir, 2000)
train_features = np.reshape(train_features,(2001,4*4*512))
valid_features = np.reshape(valid_features,(2000,4*4*512))
test_features = np.reshape(test_features,(1000,4*4*512))
# defining densly connevcted layer

from keras.layers import Dense,Dropout

from keras.models import Sequential
from keras import optimizers

model = Sequential()
model.add(Dense(256,activation='relu',input_dim= 4*4*512))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(
optimizer= optimizers.RMSprop(lr=1e-4),
    loss='binary_crossentropy',
    metrics =['acc'])

history =model.fit(train_features,train_labels,
                  epochs=30,
                  batch_size=20,
                  validation_data=(valid_features,valid_labels))



import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss =history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label= 'Validating Accuracy')
plt.legend()
plt.show()

plt.plot(epochs,loss,'ro',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Vaalidation Loss')
plt.legend()
plt.show()
from keras.models import Sequential

from keras.layers import Flatten,Dense

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen = ImageDataGenerator(
                rescale =1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                  train_dir,
                  target_size=(150,150),
                  batch_size=20,
                  class_mode='binary')

valid_generator = train_datagen.flow_from_directory(
                  valid_dir,
                  target_size=(150,150),
                  batch_size=20,
                  class_mode='binary')

model.compile(loss='binary_crossentropy',
             optimizer = optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])

history = model.fit_generator(
train_generator,
steps_per_epoch=100,
epochs =30,
validation_data = valid_generator,
validation_steps=50)
acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='validating accuracy')
plt.legend()
plt.show()

plt.plot(epochs,loss,'ro',label='Training loss')
plt.plot(epochs,val_loss,'r',label='validating loss')
plt.legend()
plt.show()
conv_base.trainable=True
set_trainable = False
for layer in conv_base.layers:
    if layer.name=='block_conv1':
        set_trainable=True
    if set_trainable:
        layer.trainale=True
    else:
        layer.trainable=False
model.compile(loss='binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-5),
             metrics =['acc'])
history = model.fit_generator( train_generator, 
                              steps_per_epoch=100,
                              epochs=100,
                              validation_data=valid_generator, 
                              validation_steps=50)


acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label=' validating accuracy')
plt.legend()
plt.show()

plt.plot(epochs,loss,'ro',label=' Training loss')
plt.plot(epochs,val_loss,'r',label=' validating loss')
plt.legend()
plt.show()
def smooth_curve(points, factor=0.8): 
    smoothed_points = [] 
    for point in points: 
        if smoothed_points: 
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor)) 
        else: smoothed_points.append(point)
    return smoothed_points

acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)
plt.plot(epochs,smooth_curve(acc),'bo',label='Smooth Training accuracy')
plt.plot(epochs,smooth_curve(val_acc),'b',label='Smooth validating accuracy')
plt.legend()
plt.show()

plt.plot(epochs,smooth_curve(loss),'ro',label='Smooth Training loss')
plt.plot(epochs,smooth_curve(val_loss),'r',label='Smooth validating loss')
plt.legend()
plt.show()
test_generator = test_datagen.flow_from_directory( 
    test_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
