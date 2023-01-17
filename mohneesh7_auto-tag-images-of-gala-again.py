import os
import pandas as pd
import numpy as np
import seaborn as sns
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization,Convolution2D,MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization 
from keras import optimizers
import tensorflow.keras
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping

train=pd.read_csv('/kaggle/input/classification-of-images/dataset/train.csv')
test=pd.read_csv('/kaggle/input/classification-of-images/dataset/test.csv')
train.head()

Class_map={'Food':0,'Attire':1,'Decorationandsignage':2,'misc':3}
inverse_map={0:'Food',1:'Attire',2:'Decorationandsignage',3:'misc'}
train['Class']=train['Class'].map(Class_map)
train['Class']
train_img=[]
train_label=[]
j=0
path='/kaggle/input/classification-of-images/dataset/Train Images'
for i in tqdm(train['Image']):
    final_path=os.path.join(path,i)
    img=cv2.imread(final_path)
    img=cv2.resize(img,(150,150))
    img=img.astype('float32')
    train_img.append(img)
    train_label.append(train['Class'][j])
    j=j+1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.3, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(train_img)


test_img=[]
path='/kaggle/input/classification-of-images/dataset/Test Images'
for i in tqdm(test['Image']):
    final_path=os.path.join(path,i)
    img=cv2.imread(final_path)
    img=cv2.resize(img,(150,150))
    img=img.astype('float32')
    test_img.append(img)
train_img=np.array(train_img)
test_img=np.array(test_img)
train_label=np.array(train_label)
print(train_img.shape)
print(test_img.shape)
print(train_label.shape)
from keras import backend as K
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
sgd = SGD(lr=0.0001,momentum=0.9)
base_model=VGG16(include_top=False, weights='imagenet',input_shape=(150,150,3), pooling='avg')
i=0
for layer in base_model.layers:
    layer.trainable = False
    i = i+1
model=Sequential()
model.add(base_model)
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(4,activation='softmax'))




reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)

callbacks = [reduce_learning_rate]
   


model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy',f1_m])
model.fit_generator(datagen.flow(train_img, to_categorical(train_label,4), batch_size=64),
                    epochs=30)
model.save('vgg16_model.h5')
model_3.save_weights('vgg16_weights_model3.h5')
base_model=VGG16(include_top=False, weights=None,input_shape=(150,150,3), pooling='avg')
model_4 = Sequential()
model_4.add(base_model)
model_4.add(Dense(128,activation='relu'))
model_4.add(Dropout(0.3))
model_4.add(Dense(4,activation='softmax'))

model_4.load_weights('vgg16_weights_model2.h5')
model_4.compile( optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy',f1_m])
model_4.fit_generator(datagen.flow(train_img, to_categorical(train_label,4), batch_size=32),
                    epochs=20)
base_model=Xception(include_top=False, weights='imagenet',input_shape=(150,150,3), pooling='avg')
for layer in base_model.layers:
    layer.trainable = False
model=Sequential()
model.add(base_model)
model.add(Dense(1024, activation='sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(4,activation='softmax'))


model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy',f1_m])
model.fit_generator(datagen.flow(train_img, to_categorical(train_label,4), batch_size=256),
                    epochs=50)
model.save_weights('Xception_weights_model.h5')
base_model=Xception(include_top=False, weights=None,input_shape=(150,150,3), pooling='avg')
model_4 = Sequential()
model_4.add(base_model)
model_4.add(Dense(1024,activation='sigmoid'))
model_4.add(Dropout(0.4))
model_4.add(Dense(4,activation='softmax'))

model_4.load_weights('Xception_weights_model.h5')
model_4.compile( optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy',f1_m])
model_4.fit_generator(datagen.flow(train_img, to_categorical(train_label,4), batch_size=32),
                    epochs=20)
base_model=VGG19(include_top=False, weights='imagenet',input_shape=(150,150,3), pooling='avg')
for layer in base_model.layers:
    layer.trainable = False

model=Sequential()
model.add(base_model)
model.add(Dense(1024, activation='sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(4,activation='softmax'))


model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy',f1_m])
model.fit_generator(datagen.flow(train_img, to_categorical(train_label,4), batch_size=256),
                    epochs=50)

model.save_weights('vgg19_weights_model.h5')
base_model=VGG19(include_top=False, weights=None,input_shape=(150,150,3), pooling='avg')
model_4 = Sequential()
model_4.add(base_model)
model_4.add(Dense(1024,activation='sigmoid'))
model_4.add(Dropout(0.4))
model_4.add(Dense(4,activation='softmax'))

model_4.load_weights('vgg19_weights_model_afterft.h5')
model_4.compile( optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy',f1_m])
model_4.fit_generator(datagen.flow(train_img, to_categorical(train_label,4), batch_size=32),
                    epochs=20)
base_model=ResNet101V2(include_top=False, weights='imagenet',input_shape=(150,150,3), pooling='avg')
for layer in base_model.layers:
    layer.trainable = False

model=Sequential()
model.add(base_model)
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(4,activation='softmax'))


model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy',f1_m])
model.fit_generator(datagen.flow(train_img, to_categorical(train_label,4), batch_size=256),
                    epochs=50)

model.save('resnet101_v2_model.h5')
# base_model=ResNet101(include_top=False, weights=None,input_shape=(150,150,3), pooling='avg')
# model_4 = Sequential()
# model_4.add(base_model)
# model_4.add(Dense(1024,activation='sigmoid'))
# model_4.add(Dropout(0.4))
# model_4.add(Dense(4,activation='softmax'))
dep = {'f1_m':f1_m}

model_4 = load_model('resnet101_model.h5',custom_objects=dep)
model_4.compile( optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy',f1_m])
model_4.fit_generator(datagen.flow(train_img, to_categorical(train_label,4), batch_size=32),
                    epochs=20)
base_model=MobileNet(include_top=False, weights='imagenet',input_shape=(150,150,3), pooling='avg')
for layer in base_model.layers:
    layer.trainable = False

model=Sequential()
model.add(base_model)
model.add(Dense(1024, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(4,activation='softmax'))
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)

callbacks = [reduce_learning_rate]

model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy',f1_m])
model.fit_generator(datagen.flow(train_img, to_categorical(train_label,4), batch_size=32),callbacks=callbacks,
                    epochs=20)

model.save_weights('mobilenet_weights_model.h5')
base_model=VGG19(include_top=False, weights=None,input_shape=(150,150,3), pooling='avg')
model_4 = Sequential()
model_4.add(base_model)
model_4.add(Dense(1024,activation='sigmoid'))
model_4.add(Dropout(0.4))
model_4.add(Dense(4,activation='softmax'))

model_4.load_weights('mobilenet_weights_model.h5')
model_4.compile( optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy',f1_m])
model_4.fit_generator(datagen.flow(train_img, to_categorical(train_label,4), batch_size=32),
                    epochs=20)
labels = model.predict(test_img)
print(labels[:4])
label = [np.argmax(i) for i in labels]
class_label = [inverse_map[x] for x in label]
print(class_label[:3])
submission = pd.DataFrame({ 'Image': test.Image, 'Class': class_label })
submission.head(10)
submission.to_csv('submission_mobilenet.csv', index=False)
