# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import some library to use

from tensorflow import keras   # keras for DL

import matplotlib.pyplot as plt  # for data visualization



import sklearn

from sklearn.model_selection import train_test_split

# read data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# review the data

Y = train['label']

X = train.drop(axis=1,columns='label')

X = X.values.reshape(-1,28,28,1) # convert to ndarray

Y = Y.values

test = test.values.reshape(-1,28,28,1)

X.shape
#display some data

plt.figure(figsize=(10,15))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.imshow(X[i,:,:,0])

    plt.title(Y[i])

    plt.grid(False)

    

plt.show()

# normialize the data

# X= X/255

Y=keras.utils.to_categorical(Y,num_classes=10)# change label from value to one-hot vector

# Split data

# Y.shape



# X_train,X_val, Y_train, Y_val = train_test_split(X,Y,test_size = 0.2, random_state = 5)

# Y_train[:5]
def mlp_model():

    # Build model for the first time, use MLP model

    model = keras.Sequential()

    # Dense(1024) is a full connected layer with 1024 hidden units. If it is the first layers, it should use input_dim as a shape of input



    model.add(keras.layers.Flatten(input_shape=(28,28,1)))

    model.add(keras.layers.Dense(units=1024, activation='relu'))

    model.add(keras.layers.Dropout(rate=0.2))



    model.add(keras.layers.Dense(units=1024,activation='relu'))

    model.add(keras.layers.Dropout(0.2))

    # use a softmax layers to classification with 10 classs (0-9 digit)



    model.add(keras.layers.Dense(units=10,activation='softmax'))



    # compile the model with loss function, optimizer algorithm

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



    return model



def cnn_model():

    model= keras.Sequential()

#     model.add(keras.layers.InputLayer(input_shape=(28,28,1)))

    model.add(keras.layers.Conv2D(filters=16,kernel_size=(3,3),

                                  activation='relu',padding ='SAME',input_shape=(28,28,1)))

    model.add(keras.layers.Conv2D(filters=16,kernel_size=(3,3),

                                  activation='relu',padding='same'))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),padding='SAME'))

    model.add(keras.layers.Dropout(0.2))

    

    model.add(keras.layers.Conv2D(filters=32,kernel_size=(3,3),

                                  activation='relu',padding='same'))

    model.add(keras.layers.Conv2D(filters=32,kernel_size=(3,3),

                                  activation='relu',padding='same'))

    model.add(keras.layers.MaxPool2D(pool_size=(2,2),padding='SAME'))

    model.add(keras.layers.Dropout(0.2))

    

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(512,activation='relu'))

    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(units=10,activation='softmax'))

    

    model.compile(loss='categorical_crossentropy', optimizer='adam',

                  metrics=['accuracy'])

    return model
# model = mlp_model()

model_cnn = cnn_model()

# model_cnn.summary()

# train the model

# history_mlp = model.fit(X_train,Y_train,epochs=20,validation_data=(X_val,Y_val))





#data augmentation

# rescale data to [0,1]

# split data to validation

# rotation and zoom



datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2, rotation_range= 20, zoom_range=0.2)

batch_size = 32

train_generator = datagen.flow(X,Y,subset='training')

valid_generator = datagen.flow(X,Y,subset='validation')



# history_cnn =model_cnn.fit(X_train,Y_train,epochs=20,validation_data=(X_val,Y_val))

history_cnn = model_cnn.fit_generator(train_generator,epochs=20,steps_per_epoch=len(X)/batch_size, validation_data=valid_generator, validation_steps=len(X)/batch_size)



# model_mlp = mlp_model()

# history_mlp = model_mlp.fit(X_train,Y_train,epochs=20,validation_data=(X_val,Y_val))


# history_cnn = model_cnn.fit_generator(train_generator,epochs=20,steps_per_epoch=len(X)/batch_size, validation_data=valid_generator, validation_steps=len(X)/batch_size)
def plot_loss_accuracy(history):

    # plot loss and accuracy of model

    plt.figure(figsize=(10,10))

    # loss figure

    plt.subplot(2,1,1)

    plt.title("Loss")

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.legend(['train','validation'])

    plt.xlim(0,20)

    plt.ylim(0,0.5)



    # acc figure

    plt.subplot(2,1,2)

    plt.title("Accuracy")

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.legend(['train','validation'])

    plt.xlim(0,20)

    plt.ylim(0.80,1)

    plt.show()
plot_loss_accuracy(history_cnn)

# plot_loss_accuracy(history_mlp)
#display some image

# y_pre = model_cnn.predict(X_val)



# display some image in the data with predict and truth values

def display_result(images,image_row,image_column,class_pre,class_real):

# """images: list of image to display

#     image_row, image_column: quantity of image each row, column to display

#     class_pre: predict of class

#     class_real: the truth class of image

# """



    plt.figure(figsize=(15,15))

    for i in range(image_row * image_column):

        plt.subplot(image_row,image_column,i+1)        

        plt.imshow(images[i,:,:,0],cmap='gray')

        plt.xlabel("Predict:{} of {}".format(class_pre[i],class_real[i]))

    plt.show()

    

    

def visualize_result(x_pre,y_pre,y_real):

#     display the result of model predict the result

    class_pre = y_pre.argmax(axis =1)

    class_real = y_real.argmax(axis =1)

    idx_correct = np.unique(np.nonzero(class_pre == class_real)[0])

    idx_incorrect = np.unique(np.nonzero(class_pre != class_real)[0])

#     display correct result

    display_result(x_pre[idx_correct],5,5,class_pre[idx_correct],class_real[idx_correct])

    

#     display incorrect result

    display_result(x_pre[idx_incorrect],5,5,class_pre[idx_incorrect],class_real[idx_incorrect])



 # visualize_result(X_val,y_pre,Y_val)
# predict and submission

predictions = model_cnn.predict(test)

predictions = np.argmax(predictions,axis=1)

submissions = pd.DataFrame({'ImageID':list(range(1,len(predictions)+1)),'Label':predictions})

keras.utils.plot_model(model_cnn,to_file='model.png')

model_cnn.save("model.h5")



#submission

submissions.to_csv('submission.csv',index = False, header = True)
submissions.head()