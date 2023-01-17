# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.layers.normalization import BatchNormalization



np.random.seed(0)

sns.set(style='white', context='notebook', palette='deep')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
from sklearn.model_selection import train_test_split

tmp = train["label"]

X_train, X_test, y_train, y_test = train_test_split(train.drop(labels=["label"],axis=1),

                                                    tmp,

                                                    test_size = 0.01,

                                                    random_state = 0)
X_train /= 255.0

X_test /= 255.0

test /= 255.0

#train /= 255.0



X_train = X_train.values.reshape(-1,28,28,1)

#train = train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)



print(X_train.shape,X_test.shape,test.shape)
y_train = to_categorical(y_train, num_classes = 10)

y_test = to_categorical(y_test, num_classes = 10)

#tmp = to_categorical(tmp, num_classes = 10)



plt.imshow(X_train[0][:,:,0],cmap='gray')
epochs = 30

batchsize = 16

steps_per_epoch = X_train.shape[0] / batchsize



learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3,  

                                            factor=0.5, 

                                            min_lr=0.00001)



dataset = ImageDataGenerator()

dataset.fit(X_train)
# BUILD CONVOLUTIONAL NEURAL NETWORKS

# EXPERIMENT 1

# convolution layer num

'''nets = 3

model = [0] *nets



for j in range(3):

    model[j] = Sequential()

    model[j].add(Conv2D(12,kernel_size=5,padding='same',activation='relu',

            input_shape=(28,28,1)))

    model[j].add(MaxPool2D())

    if j>0:

        model[j].add(Conv2D(24,kernel_size=5,padding='same',activation='relu'))

        model[j].add(MaxPool2D())

    if j>1:

        model[j].add(Conv2D(48,kernel_size=5,padding='same',activation='relu'))

        model[j].add(MaxPool2D(padding='same'))

    model[j].add(Flatten())

    model[j].add(Dense(256, activation='relu'))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])'''

'''

for i in range(3):

    print('model:',i)

    

    model[i].fit_generator(dataset.flow(X_train,y_train,batch_size=batchsize),

                           epochs=epochs, validation_data=(X_test,y_test),steps_per_epoch=X_train.shape[0]//batchsize)

print('finish.1')

'''
# EXPERIMENT 2

# kernel num



'''nets = 3

model = [0] *nets



batchsize = 16

epochs = 30



for j in range(3):

    tmp = 2 ** j #1 2_ 4 

    model[j] = Sequential()

    model[j].add(Conv2D(8*tmp,kernel_size=5,padding='same',activation='relu',

            input_shape=(28,28,1)))

    model[j].add(MaxPool2D())

    

    model[j].add(Conv2D(16*tmp,kernel_size=5,padding='same',activation='relu'))

    model[j].add(MaxPool2D())

    

    model[j].add(Flatten())

    model[j].add(Dense(256, activation='relu'))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    



for i in range(4):

    print("-------------------------------------------------------------")

    print('model:',i+1)



    model[i].fit_generator(dataset.flow(X_train,y_train,batch_size=batchsize),

                           epochs=epochs, validation_data=(X_test,y_test),steps_per_epoch=X_train.shape[0]//batchsize)

print('finish.2')'''

# EXPERIMENT 3

# kernel size

# 5*5 --> 3*3 + 3*3



'''nets = 2

model = [0] *nets



for j in range(2):

    model[j] = Sequential()

    if j == 0:

        model[j].add(Conv2D(32,kernel_size=5,padding='same',activation='relu',

                input_shape=(28,28,1)))

        model[j].add(MaxPool2D())

    elif j == 1:

        model[j].add(Conv2D(32,kernel_size=3,padding='same',activation='relu',

                input_shape=(28,28,1)))

        model[j].add(MaxPool2D())

        model[j].add(Conv2D(32,kernel_size=3,padding='same',activation='relu',

                input_shape=(28,28,1)))

        model[j].add(MaxPool2D())

    

    model[j].add(Conv2D(64,kernel_size=5,padding='same',activation='relu'))

    model[j].add(MaxPool2D())

    

    model[j].add(Flatten())

    model[j].add(Dense(256, activation='relu'))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    

for i in range(2):

    print("-------------------------------------------------------------------")

    print('model:',i+1)



    #model[i].fit_generator(dataset.flow(X_train,y_train,batch_size=batchsize),

    #                       epochs=epochs, validation_data=(X_test,y_test),steps_per_epoch=X_train.shape[0]//batchsize)

    model[i].fit_generator(dataset.flow(X_train,y_train,batch_size=batchsize),

                           epochs=epochs, validation_data=(X_test,y_test),

                           steps_per_epoch=X_train.shape[0]//batchsize,

                           callbacks=[learning_rate_reduction])

print('finish.3')'''
'''model = [0]

model[0] = Sequential()



model[0].add(Conv2D(32,kernel_size=3,padding='same',activation='relu',

        input_shape=(28,28,1)))

model[0].add(MaxPool2D())

model[0].add(Conv2D(32,kernel_size=3,padding='same',activation='relu',

        input_shape=(28,28,1)))

model[0].add(MaxPool2D())



#model[0].add(Conv2D(36,kernel_size=5,padding='same',activation='relu'))

#model[0].add(MaxPool2D())

model[0].add(Conv2D(36,kernel_size=3,padding='same',activation='relu',

        input_shape=(28,28,1)))

model[0].add(MaxPool2D())

model[0].add(Conv2D(36,kernel_size=3,padding='same',activation='relu',

        input_shape=(28,28,1)))

model[0].add(MaxPool2D())



model[0].add(Flatten())

model[0].add(Dense(256, activation='relu'))

model[0].add(Dense(10, activation='softmax'))

model[0].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



for i in range(1):

    if i == 0:

        print("-------------------------------------------------------------------")

        print('model:',3)



        model[i].fit_generator(dataset.flow(X_train,y_train,batch_size=batchsize),

                               epochs=epochs, validation_data=(X_test,y_test),

                               steps_per_epoch=X_train.shape[0]//batchsize,

                               callbacks=[learning_rate_reduction])

print('finish.3')'''
# EXPERIMENT 4

# different neurons



'''nets = 4

model = [0]*nets

for j in range(4):

    model[j] = Sequential()

    

    # CONV LAYER

    model[j].add(Conv2D(32,kernel_size=3,padding='same',activation='relu',

            input_shape=(28,28,1)))

    model[j].add(MaxPool2D())

    model[j].add(Conv2D(32,kernel_size=3,padding='same',activation='relu',

            input_shape=(28,28,1)))

    model[j].add(MaxPool2D())

    

    model[j].add(Conv2D(64,kernel_size=5,padding='same',activation='relu'))

    model[j].add(MaxPool2D())

    

    model[j].add(Flatten())

    

    tmp = 2**(j+6)

    

    model[j].add(Dense(tmp, activation='relu'))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



for i in range(2):

    print("-------------------------------------------------------------------")

    print('model:',i+1)



    model[i+2].fit_generator(dataset.flow(X_train,y_train,batch_size=batchsize),

                           epochs=epochs, validation_data=(X_test,y_test),

                           steps_per_epoch=X_train.shape[0]//batchsize,

                           callbacks=[learning_rate_reduction])

print('finish.4')'''
# EXPERIMENT 5

# dropout



'''nets = 7

model = [0] *nets



for j in range(7):

    model[j] = Sequential()



    # CONV LAYER

    model[j].add(Conv2D(32,kernel_size=3,padding='same',activation='relu',

            input_shape=(28,28,1)))

    model[j].add(MaxPool2D())

    model[j].add(Conv2D(32,kernel_size=3,padding='same',activation='relu',

            input_shape=(28,28,1)))

    model[j].add(MaxPool2D())

    model[j].add(Dropout(j*0.1))

    

    model[j].add(Conv2D(64,kernel_size=5,activation='relu'))

    model[j].add(MaxPool2D())

    model[j].add(Dropout(j*0.1))

    

    # FC LAYER

    model[j].add(Flatten())

    model[j].add(Dense(256, activation='relu'))

    model[j].add(Dropout(j*0.1))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    

for i in range(7):

    if i == 6:

        print("-------------------------------------------------------------------")

        print('model:',i+1)



        model[i].fit_generator(dataset.flow(X_train,y_train,batch_size=batchsize),

                               epochs=epochs, validation_data=(X_test,y_test),

                               steps_per_epoch=X_train.shape[0]//batchsize,

                               callbacks=[learning_rate_reduction])

print('finish.5')'''
# EXPERIMENT 6 

# add batch normalization



'''nets = 1

model = [0] *nets



batchsize = 16



for j in range(1):

    model[j] = Sequential()



    # CONV LAYER

    model[j].add(Conv2D(32,kernel_size=3,padding='same',activation='relu',

            input_shape=(28,28,1)))

    model[j].add(MaxPool2D())

    model[j].add(Conv2D(32,kernel_size=3,padding='same',activation='relu',

            input_shape=(28,28,1)))

    model[j].add(MaxPool2D())

    model[j].add(BatchNormalization())

    model[j].add(Dropout(0.3))

    

    model[j].add(Conv2D(64,kernel_size=5,activation='relu'))

    model[j].add(MaxPool2D())

    model[j].add(BatchNormalization())

    model[j].add(Dropout(0.3))

    

    # FC LAYER

    model[j].add(Flatten())

    model[j].add(Dense(256, activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Dropout(0.3))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    

for i in range(1):

    print("-------------------------------------------------------------------")

    print('model:',i+1)



    model[i].fit_generator(dataset.flow(X_train,y_train,batch_size=batchsize),

                           epochs=epochs, validation_data=(X_test,y_test),

                           steps_per_epoch=X_train.shape[0]//batchsize,

                           callbacks=[learning_rate_reduction])

print('finish.6')'''
# EXPERIMENT 7

# add data augmentation

# basically the final version



'''nets = 1

model = [0] *nets



epochs = 30

batchsize = 16

'''

augmented_dataset = ImageDataGenerator(rotation_range = 10,

                                       zoom_range = 0.1,

                                       width_shift_range = 0.1,

                                       height_shift_range = 0.1,

                                       horizontal_flip = False,

                                       vertical_flip = False)



'''

for j in range(1):

    model[j] = Sequential()



    # CONV LAYER

    model[j].add(Conv2D(32,kernel_size=3,padding='same',activation='relu',

            input_shape=(28,28,1)))

    model[j].add(MaxPool2D())

    model[j].add(Conv2D(32,kernel_size=3,padding='same',activation='relu',

            input_shape=(28,28,1)))

    model[j].add(MaxPool2D())

    model[j].add(BatchNormalization())

    model[j].add(Dropout(0.3))

    

    model[j].add(Conv2D(64,kernel_size=5,activation='relu'))

    model[j].add(MaxPool2D())

    model[j].add(BatchNormalization())

    model[j].add(Dropout(0.3))

    

    # FC LAYER

    model[j].add(Flatten())

    model[j].add(Dense(256, activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Dropout(0.3))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    

for i in range(1):

    print("-------------------------------------------------------------------")

    print('model:',i+1)



    model[i].fit_generator(augmented_dataset.flow(X_train,y_train,batch_size=batchsize),

                           epochs=epochs, validation_data=(X_test,y_test),

                           steps_per_epoch=X_train.shape[0]//batchsize,

                           callbacks=[learning_rate_reduction])

print('finish.7')'''
# THE STRUCTURE 

model = Sequential()



# CONV LAYER

model.add(Conv2D(32,kernel_size=3,padding='same',activation='relu',

        input_shape=(28,28,1)))

model.add(Conv2D(32,kernel_size=3,padding='same',activation='relu',

        input_shape=(28,28,1)))

model.add(MaxPool2D())

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Conv2D(64,kernel_size=3,padding='same',activation='relu',

        input_shape=(28,28,1)))

model.add(Conv2D(64,kernel_size=3,padding='same',activation='relu',

        input_shape=(28,28,1)))

model.add(MaxPool2D())

model.add(BatchNormalization())

model.add(Dropout(0.3))



# FC LAYER

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



print("-------------------------------------------------------------------")

print('my model')



history = model.fit_generator(augmented_dataset.flow(X_train,y_train,batch_size=batchsize),

                              epochs=epochs, validation_data=(X_test,y_test),

                              steps_per_epoch=X_train.shape[0]//batchsize,

                              callbacks=[learning_rate_reduction])



print("OuO")



# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# MAKE SUBMISSION 

predictions = model.predict(test)

pre_res = np.argmax(predictions,axis = 1)

res = pd.Series(pre_res,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),res],axis = 1)



submission.to_csv("mnist_submission.csv",index=False)