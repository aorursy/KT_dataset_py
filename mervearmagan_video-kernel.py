import numpy as np

import matplotlib.pyplot as plt 

import cv2       

import os,random                    

from sklearn.model_selection import train_test_split   

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam,RMSprop,SGD,Adadelta

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,CSVLogger

import keras,math,time

from keras.utils import to_categorical

from keras import regularizers

import tensorflow as tf

from keras import Model

from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from keras.layers import GlobalMaxPooling2D, SpatialDropout2D, Input,LeakyReLU,Conv2D, BatchNormalization, Activation,Dense,MaxPooling2D,Flatten,Dropout,GlobalAveragePooling2D,AveragePooling2D,MaxPool2D

from keras.regularizers import l2



from keras.models import Sequential

from keras.applications import MobileNetV2, VGG19, InceptionV3



IMG_WIDTH,IMG_HEIGTH=125,125

oran=0.1

num_classes=8

class_names_label = {'1.kiƒi': 0,'2.kiƒi' : 1 ,'3.kiƒi' : 2,'4.kiƒi' : 3,'5.kiƒi' : 4,'6.kiƒi' : 5,'7.kiƒi' : 6,'8.kiƒi' : 7}



def load_data(DATADIR):

    path = os.path.join(DATADIR)

    dataset = []

    for imge in os.listdir(DATADIR):        

        for img in os.listdir(os.path.join(DATADIR,imge)): 

            curr_label = class_names_label[imge] 

            cap = cv2.VideoCapture(os.path.join(DATADIR,imge,img))

            while(cap.isOpened()):

                ret, frame = cap.read()

                if (ret != True):

                    break

                grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                dataset.append([np.asarray(grayFrame),curr_label]) 

            cap.release()



   # random.shuffle(dataset)

    data = []

    labels = []

    for features, label in dataset:        

        data.append(features)

        labels.append(label)        

    data = np.array(data)

    labels = np.array(labels)

    train_data,test_data,train_labels,test_labels = train_test_split(data,labels,test_size=oran,random_state=42, 

                                                                     shuffle=True,stratify=labels)

    return np.array(train_data),np.array(test_data),np.array(train_labels),np.array(test_labels)



train_data,test_data,train_labels,test_labels= load_data("../input/hareketbir/1.hareket")



train_data = train_data.reshape(train_data.shape[0], IMG_WIDTH ,IMG_HEIGTH, 1)

test_data = test_data.reshape(test_data.shape[0], IMG_WIDTH ,IMG_HEIGTH, 1)



datagen_train=ImageDataGenerator(   

    rescale = 1./255.,

    validation_split = 0.25

    )

datagen_train.fit(train_data)

datagen_test = ImageDataGenerator(

    rescale = 1./255.

)



datagen_test.fit(test_data)



train_labels = to_categorical(train_labels, num_classes)

test_labels = to_categorical(test_labels,num_classes)

print("bitti")


weight_decay=0.0005

input_ = Input(shape=(125, 125, 1))

# Block 1

x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same',kernel_regularizer=l2(weight_decay))(input_)

x = BatchNormalization()(x)

x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)



# Block 2

x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same',kernel_regularizer=l2(weight_decay))(x)

x = BatchNormalization()(x)

x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)



# Block 3

x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same',kernel_regularizer=l2(weight_decay))(x)

x = BatchNormalization()(x)

x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)



# Block 4

x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',kernel_regularizer=l2(weight_decay))(x)

x = BatchNormalization()(x)

x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)

x = GlobalMaxPooling2D()(x)



x = Dense(200,kernel_regularizer=l2('0.001'))(x)

x = Dropout(0.1)(x)

x = BatchNormalization()(x)

x = Dense(100,kernel_regularizer=l2('0.001'))(x)

x = BatchNormalization()(x)

x = Activation('relu')(x)

x = Dense(8)(x)

x = Activation('softmax')(x)

model = Model(inputs = input_, outputs=x)
Batch_Size=16

Epoch=100

learning_rate=1e-3

  

model.compile(optimizer=Adam(lr=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])

early_stopper = EarlyStopping(monitor='val_accuracy',patience=10,restore_best_weights=True, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,patience=4, verbose=1, mode='auto')

callbacks_list = [reduce_lr, early_stopper]



train_generator=datagen_train.flow(train_data, train_labels, batch_size=Batch_Size,subset="training",seed=42,shuffle=True)

val_generator =datagen_train.flow(train_data, train_labels, batch_size=Batch_Size,subset="validation",seed=42,shuffle=True)

test_generator = datagen_test.flow(test_data,test_labels, batch_size=Batch_Size,seed=42,shuffle=False) 



STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

STEP_SIZE_TEST=int(np.ceil(test_generator.n//test_generator.batch_size)) 

history = model.fit_generator(train_generator, steps_per_epoch=STEP_SIZE_TRAIN

                              ,validation_steps = STEP_SIZE_VALID

                              ,epochs=Epoch,verbose = 1

                              ,validation_data=val_generator

                             # ,callbacks=callbacks_list

                             ) 

final_losstr, final_acctr = model.evaluate_generator(train_generator,verbose=0,steps=1)

print("\ntrain="+str(final_acctr))

final_loss,final_acc = model.evaluate_generator(test_generator,verbose=0,steps=1)

print("test="+str(final_acc))



from sklearn.metrics import accuracy_score,classification_report,confusion_matrix



Y_test_pred = model.predict(test_generator,verbose=0,steps=STEP_SIZE_TEST)

score=accuracy_score(np.argmax(test_labels, axis=1), np.argmax(Y_test_pred, axis=1))

print("Predict="+str(score))

print(classification_report(np.argmax(test_labels, axis=1), np.argmax(Y_test_pred, axis=1)))

mat = confusion_matrix(np.argmax(test_labels, axis=1), np.argmax(Y_test_pred, axis=1))

print(mat)



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], )

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()