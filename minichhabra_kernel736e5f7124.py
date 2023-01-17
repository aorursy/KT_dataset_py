#import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
data_dir = '/kaggle/input/Train/Train'
augs_gen = ImageDataGenerator(
    rescale=1./255,        
    horizontal_flip=True,
    height_shift_range=.2,
    #vertical_flip = True,
    validation_split = 0.2
)  

train_gen = augs_gen.flow_from_directory(
    data_dir,
    target_size = (224,224),
    batch_size=32,
    class_mode = 'categorical',
    shuffle=True,
)

val_gen = augs_gen.flow_from_directory(
    data_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    subset = 'validation'
)
import keras
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D,BatchNormalization,Dropout
model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
model.add(BatchNormalization())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(45, activation='softmax'))
# ResNet-50 model is already trained, should not be trained
model.layers[0].trainable = False

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
history=model.fit_generator(train_gen,epochs=1,steps_per_epoch=30,validation_data=val_gen,validation_steps=40)
import cv2,os

data_path='/kaggle/input/Test/Test/Test1'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories,labels))

data=[]
for category in categories:
    data.append(category)
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]



data.sort(key=natural_keys)
#print(data)
target=[]
import numpy as np
for d in data:
    folder_path=os.path.join(data_path,d)
    img=cv2.imread(folder_path)
    
    try:
        #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
        resized=cv2.resize(img,(224,224))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,224,224,3))
        #print(d)
        result=model.predict(reshaped)
        label=np.argmax(result,axis=1)[0]
        #print(label)
        target.append(label)

    except Exception as e:
        print('Exception:',e)
            #if any exception rasied, the exception will be printed here. And pass to the next image
import pandas as pd
results=pd.DataFrame({'image':data,'predictions':target})
print(results)
results.to_csv('caar.csv',index=False)