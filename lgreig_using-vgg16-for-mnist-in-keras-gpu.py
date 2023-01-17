#Load the require libraries for processing



import tensorflow as tf

import numpy as np 



import pandas as pd 

from sklearn.model_selection import train_test_split





#Required libraries for keras model

from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import VGG16

from keras.models import Model

from keras.layers import Dense

from keras.layers import Flatten

from keras.models import Sequential

from keras.optimizers import SGD, Adam

import matplotlib.pyplot as plt

from keras import Input

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator
#Load the files 



train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

target = train['label']

train  = train.drop(['label'], axis=1)

#Convert the train and test to arrays and reshape and normalize

full = pd.concat([train, test])

full=full.to_numpy()

full=full.reshape(-1, 28, 28 )

#Get the dimensions

full.shape
#Plot an image

im=full[0]

import matplotlib.pyplot as plt

plt.imshow(im, cmap="gray")
#resize the numpy arrays for VGG16 - the CV alg requires a 32 x 32 array rather that 28 x 28.

full = np.pad(full, ((0,0), (2,2), (2,2)), mode='constant')

full = stacked_img = np.stack([full, full, full], axis=3)

full.shape
#Get back the train and test set

full=full.astype("float32")

full = full/255

train=full[:42000, :, :, :]

test=full[42000:, :, :, :]
 

#Create a train and validation set

X_train, X_val, Y_train, Y_val = train_test_split(train, target, test_size = 0.2, random_state = 1 )
#Retain the validation answers for measuring performance later

Y_val_access = Y_val
#Convert the target to a dummy variable 

Y_val=to_categorical(Y_val, num_classes=10)

Y_train=to_categorical(Y_train, num_classes=10)
### Model



def create_model():

    VGG = VGG16(

    input_shape=(32, 32, 3),

    weights='imagenet',

    include_top=False,

    )        

    

    model =tf.keras.Sequential([VGG,

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation='relu'),

        tf.keras.layers.Dense(10, activation='softmax')

    ])

    

    return model
#Create an image generator class for augmentation to improve generalisation

im = ImageDataGenerator(zoom_range=0.1, rotation_range=15, height_shift_range= 0.05, width_shift_range= 0.05)

flow=im.flow(X_train, Y_train, batch_size=32)
#Build model with standard Adam as optimizer 

mymod=create_model()

mymod.summary()

mymod.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#Fit the model. Each time we do 20 epochs and retain the prediction probabilities for validation and test set - we will average the predictions later.





predictions_val = np.zeros(shape=(5, X_val.shape[0], 10))

predictions_test = np.zeros(shape=(5,  test.shape[0], 10))



#Run 5 times



for i in range(5):

    print('training model ', i+1, '...')

    perf = mymod.fit_generator(flow, epochs=10, steps_per_epoch= X_train.shape[0]/32, validation_data=(X_val, Y_val), verbose=0)

    pred_val = mymod.predict(X_val)

    pred_test = mymod.predict(test)

    predictions_val[i, :, :] = pred_val

    predictions_test[i, :, :] = pred_test

    



#Average to ensemble the predictions

pvall = np.zeros(shape=( X_val.shape[0], 10))

ptall = np.zeros(shape=( test.shape[0], 10))



for i in range(10):

    pv1 = np.mean(predictions_val[:, :, i], axis=0)

    pt1 = np.mean(predictions_test[:, :, i], axis=0)

    pvall[:, i] = pv1

    ptall[:, i] = pt1



#Get hold out accuracy

preds_val= np.argmax(pvall,axis = 1)

preds_test = np.argmax(ptall, axis=1)

val_acc = np.mean(preds_val == Y_val_access)

print('Overall validation accuracy is', val_acc)

    
#Predict the test

preds= pd.Series(preds_test,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),preds],axis = 1)

submission.to_csv("MINST.csv",index=False)