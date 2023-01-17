import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense


test_path='../input/melanoma'
melanoma_path='../input/melanoma/DermMel/train_sep/Melanoma'
notmelanoma_path='../input/melanoma/DermMel/train_sep/NotMelanoma'
len(os.listdir(melanoma_path))
len(os.listdir(notmelanoma_path))
melanoma_list=[ f'../input/melanoma/DermMel/train_sep/Melanoma/{i}' for i in os.listdir(melanoma_path)]
notmelanoma_list=[f'../input/melanoma/DermMel/train_sep/NotMelanoma/{i}' for i in os.listdir(notmelanoma_path)]
len(notmelanoma_list)
train_images=melanoma_list+notmelanoma_list
del melanoma_list
del notmelanoma_list
len(train_images)
random.shuffle(train_images)

def read_images(train_images):
    x=[]
    y=[]
    for i,image in enumerate(train_images):
            img=cv2.imread(image)
            img=cv2.resize(img,(150,150),interpolation=cv2.INTER_CUBIC)
            x.append(img)
            if image.split('/')[-2] =='NotMelanoma':
                y.append(1) #notmelanoma
            else:
                y.append(0) #melanoma
            
    return x,y,image
x,y,image=read_images(train_images)
x=np.array(x)
y=np.array(y)
y
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.20,random_state=2)
len(x_train)+len(x_val)
plt.figure(figsize=(20,10))
for i,image in enumerate(x_train[:5]):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.subplot(1,5,i+1)
    plt.imshow(image)
    
    
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',
             optimizer='RMSprop',
             metrics=['acc'])
def get_model():
    model=Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['acc'])
    return model
train_datagen= ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')

val_datagen=ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow(x_train,y_train,batch_size=40)
validation_generator=val_datagen.flow(x_val,y_val,batch_size=40)
train_datagen
ntrain=len(x_train)
nval=len(x_val)

history=model.fit_generator(train_generator,
                            epochs=50,
                            validation_data=validation_generator)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
model = get_model()
train_generator=train_datagen.flow(x_train,y_train,batch_size=64)
validation_generator=val_datagen.flow(x_val,y_val,batch_size=64)

history=model.fit_generator(train_generator,
                            epochs=50,
                            validation_data=validation_generator, shuffle=True)
train_generator=train_datagen.flow(x_train,y_train,batch_size=64,shuffle=True)
validation_generator=val_datagen.flow(x_val,y_val,batch_size=64,shuffle=True)
model = get_model()
history=model.fit_generator(train_generator,
                            epochs=50,
                            validation_data=validation_generator, shuffle=True)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
