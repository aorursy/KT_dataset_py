# This Python 3 environment comes with many helpful analytics libraries installed

# For example, here's several helpful packages to load in 



import numpy as np # linear algebraئ

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from random import randint

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

#print(os.listdir("../input/digit-rec-model"))

#print(os.listdir("../input/d-r-model"))

# Any results you write to the current directory are saved as output.
x=[]



x1=np.load("../input/eval-train1/imtrain.npy")

x2=np.load("../input/eval-train2/imtrain2.npy")



x.extend(x1)

x.extend(x2)

np.save("../input/imtrainall.npy", x)



y=[]



y1=np.load("../input/eval-train1/labeltrain.npy")

y2=np.load("../input/eval-train2/labeltrain2.npy")



y.extend(y1)

y.extend(y2)

np.save("../input/labeltrainall.npy", y)



X_train=np.load("../input/imtrainall.npy")

Y_train=np.load("../input/labeltrainall.npy")

X_val=np.load("../input/eval-dataset/imtesteval.npy")

Y_val=np.load("../input/eval-dataset/labeltesteval.npy")
X_train.shape
Y_train.shape
from keras.utils.np_utils import to_categorical



Y_train= to_categorical(Y_train)

Y_val= to_categorical(Y_val)
Y_train.shape

#Splitting the train_images into the Training set and validation set

#from sklearn.model_selection import train_test_split

#X_train1=train_images

#X_train, X_val, Y_train, Y_val= train_test_split(X_train1, Y_train1,

#               test_size=0.4, random_state=42,stratify=Y_train1)



print(X_train.shape)

print(Y_train.shape)

print(X_val.shape)

print(Y_val.shape)

X_train= X_train.astype('float32')/255

X_val=X_val.astype('float32')/255

X_train[4]
import keras

from keras.models import Sequential

from keras.models import load_model

from keras.layers import Dense, Dropout, Flatten,BatchNormalization

from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D

from keras import regularizers

from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
def lr_schedule(epoch):

    lrate = 0.001

    if epoch > 10:

        lrate = 0.0003

    if epoch > 20:

        lrate = 0.00003

    elif epoch > 30:

        lrate = 0.000003       

    return lrate
lr_scheduler=LearningRateScheduler(lr_schedule)

#we can reduce the LR by half if the accuracy is not improved after 3 epochs.using the following code

reduceOnPlateau = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=5, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001, mode='auto')



#Save the model after every decrease in val_loss 

checkpoint = ModelCheckpoint(filepath='bestmodeltrain2.hdf5', verbose=0,monitor='val_loss',save_best_only=True,save_weights_only=False)



#Stop training when a monitored quantity has stopped improving.

earlyStopping=EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model = Sequential()

model.add(Conv2D(32, (3, 3),padding='valid', activation='relu',input_shape=(128,128,3)))

model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3),padding='valid',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3),padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3),padding='same',activation='relu'))

model.add(BatchNormalization()) 

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.15))



model.add(Conv2D(128, (3, 3),padding='same',activation='relu')) 

model.add(BatchNormalization()) 

model.add(Conv2D(128, (3, 3),padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(256, (3, 3),padding='same',activation='relu')) 

model.add(BatchNormalization()) 

model.add(Conv2D(256, (3, 3),padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Flatten()) 

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25)) 



model.add(Dense(51, activation='softmax')) 

model.summary()



#model=load_model("E:/nadiaaaaaa/f/diploma/master 2018/project/missing persons images/New folder/bestmodel.hdf5")
datagen = ImageDataGenerator(

        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.01, # Randomly zoom image 

        width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)    # randomly flip images)  



datagen.fit(X_train)
callbacks_list = [lr_scheduler,checkpoint]

model.load_weights("bestmodeltrain2.hdf5")

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9)

rmsprp_opt = keras.optimizers.rmsprop(lr=0.00003 ,decay=1e-4)

adam=keras.optimizers.adam(lr=0.00003)

model.compile(loss='categorical_crossentropy',

              optimizer=adam,

              metrics=['accuracy'])

H1=model.fit_generator(datagen.flow(X_train, Y_train, batch_size=100),

                    steps_per_epoch=len(X_train)//10, epochs=5,

                    verbose=1,callbacks=callbacks_list,

                    validation_data=(X_val, Y_val))
model.load_weights("bestmodeltrain2.hdf5")

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9)

rmsprp_opt = keras.optimizers.rmsprop(lr=0.00003 ,decay=1e-4)

adam=keras.optimizers.adam(lr=0.00003)

model.compile(loss='categorical_crossentropy',

              optimizer=adam,

              metrics=['accuracy'])

H1=model.fit_generator(datagen.flow(X_train, Y_train, batch_size=100),

                    steps_per_epoch=len(X_train)//10, epochs=5,

                    verbose=1,callbacks=callbacks_list,

                    validation_data=(X_val, Y_val))
model.load_weights("bestmodeltrain2.hdf5")

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9)

rmsprp_opt = keras.optimizers.rmsprop(lr=0.00003 ,decay=1e-4)

adam=keras.optimizers.adam(lr=0.00003)

model.compile(loss='categorical_crossentropy',

              optimizer=adam,

              metrics=['accuracy'])

H1=model.fit_generator(datagen.flow(X_train, Y_train, batch_size=100),

                    steps_per_epoch=len(X_train)//10, epochs=5,

                    verbose=1,callbacks=callbacks_list,

                    validation_data=(X_val, Y_val))
model.load_weights("bestmodeltrain2.hdf5")

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9)

rmsprp_opt = keras.optimizers.rmsprop(lr=0.00003 ,decay=1e-4)

adam=keras.optimizers.adam(lr=0.00003)

model.compile(loss='categorical_crossentropy',

              optimizer=adam,

              metrics=['accuracy'])

H1=model.fit_generator(datagen.flow(X_train, Y_train, batch_size=100),

                    steps_per_epoch=len(X_train)//10, epochs=5,

                    verbose=1,callbacks=callbacks_list,

                    validation_data=(X_val, Y_val))
model.load_weights("bestmodeltrain2.hdf5")

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9)

rmsprp_opt = keras.optimizers.rmsprop(lr=0.00003 ,decay=1e-4)

adam=keras.optimizers.adam(lr=0.00003)

model.compile(loss='categorical_crossentropy',

              optimizer=adam,

              metrics=['accuracy'])

H1=model.fit_generator(datagen.flow(X_train, Y_train, batch_size=100),

                    steps_per_epoch=len(X_train)//10, epochs=5,

                    verbose=1,callbacks=callbacks_list,

                    validation_data=(X_val, Y_val))
model.load_weights("bestmodeltrain2.hdf5")

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9)

rmsprp_opt = keras.optimizers.rmsprop(lr=0.00003 ,decay=1e-4)

adam=keras.optimizers.adam(lr=0.00003)

model.compile(loss='categorical_crossentropy',

              optimizer=adam,

              metrics=['accuracy'])

H1=model.fit_generator(datagen.flow(X_train, Y_train, batch_size=100),

                    steps_per_epoch=len(X_train)//10, epochs=5,

                    verbose=1,callbacks=callbacks_list,

                    validation_data=(X_val, Y_val))
plt.figure(0)

plt.plot(H1.history['acc'],'r')

plt.plot(H1.history['val_acc'],'g')

plt.xticks(np.arange(0, 10, 1.0))

plt.rcParams['figure.figsize'] = (14, 8)

plt.xlabel("Num of Epochs")

plt.ylabel("Accuracy")

plt.title("Training Accuracy vs Validation Accuracy")

plt.legend(['train','validation'])

plt.show()

plt.figure(1)

plt.plot(H1.history['loss'],'r')

plt.plot(H1.history['val_loss'],'g')

plt.xticks(np.arange(0, 5, 1.0))

plt.rcParams['figure.figsize'] = (14, 8)

plt.xlabel("Num of Epochs")

plt.ylabel("Loss")

plt.title("Training Loss vs Validation Loss")

plt.legend(['train','validation'])

plt.show()

score = model.evaluate(X_val, Y_val, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
test_images=np.load("../input/train-dataset/imftest.npy")

Y_test=np.load("../input/train-dataset/labelftest.npy")
from sklearn.metrics import classification_report



preds = model.predict_classes(X_val)

y_lable = [y.argmax() for y in Y_val]

print(classification_report(y_lable,preds))

preds1 = model.predict_classes(X_train)

ytr_lable = [y.argmax() for y in Y_train]

print(classification_report(ytr_lable,preds1))
# predict results

Test_perdect = model.predict(test_images)



# select the indix with the maximum probability

Test_perdect = np.argmax(Test_perdect,axis = 1)



Test_perdect = pd.Series(Test_perdect,name="Label")

submission1 = pd.concat([pd.Series(range(1,28001),name = "ImageId"),Test_perdect],axis = 1)



submission1.to_csv("../input/submission1.csv",index=False)
model.save("../input/train_model.h5")

