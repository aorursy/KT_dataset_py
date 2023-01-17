# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split
#converting to one_hot encoding

def convert_to_oh(Y,n):

    m=Y.shape[0]

    Y_oh=np.zeros((m,n))

    for i in range(m):

        Y_oh[i][Y[i]]=1

    return Y_oh
#converting from one_hot encoding

def from_oh(Y):

    m=Y.shape[0]

    temp=np.array([0,1,2,3,4,5,6,7,8,9])

    Y=np.multiply(Y,temp)

    Y=np.sum(Y,axis=1).reshape((m,1))

    temp=np.arange(1,m+1,1).reshape((m,1))

    Y=np.append(temp,Y,axis=1)

    Y=Y.astype(int)

    return Y
#model

def my_net(input_shape=(28,28,1)):

    X_input=tf.keras.Input(input_shape)



    X=tf.keras.layers.ZeroPadding2D(padding=3)(X_input)



    X=tf.keras.layers.Conv2D(6,(5,5),strides=(1,1),name='conv1')(X)

    X=tf.keras.layers.BatchNormalization()(X)

    X=tf.keras.layers.Activation('relu')(X)



    X=tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=2,name="avg_poo11")(X)



    X=tf.keras.layers.Dropout(0.3)(X)

    X=tf.keras.layers.Conv2D(16,(5,5),strides=(1,1),name='conv2')(X)

    X=tf.keras.layers.BatchNormalization()(X)

    X=tf.keras.layers.Activation('relu')(X)



    X=tf.keras.layers.AveragePooling2D(pool_size=(3,3),strides=(2,2),name="avg_pool2")(X)

    '''

    X=tf.keras.layers.Dropout(0.3)(X)

    X=tf.keras.layers.Conv2D(16,(5,5),strides=(1,1),name='conv3')(X)

    X=tf.keras.layers.BatchNormalization()(X)

    X=tf.keras.layers.Activation('relu')(X)



    X=tf.keras.layers.AveragePooling2D(pool_size=(3,3),strides=1,name="avg_pool3")(X)

    '''

    X=tf.keras.layers.Flatten()(X)

    X=tf.keras.layers.Dropout(0.3)(X)

    X=tf.keras.layers.Dense(120,activation="relu",name='fc1')(X)

    X=tf.keras.layers.Dropout(0.3)(X)

    X=tf.keras.layers.Dense(80,activation="relu",name='fc2')(X)

    X=tf.keras.layers.Dense(10,activation="softmax",name='fc3')(X)



    model=tf.keras.Model(inputs=X_input,outputs=X,name='my_net')

    return model
#pre-processing

j=10



f1 = pd.read_csv('../input/digit-recognizer/train.csv', header = None)

f2 = pd.read_csv('../input/emnist/emnist-digits-train.csv', header = None)

f3 = pd.read_csv('../input/emnist/emnist-digits-test.csv', header = None)

f = pd.concat([f1, f2, f3], ignore_index = True)

print(f.head())

del f1

del f2

del f3



Y=np.array(f.iloc[1:,0], dtype=np.int32)

X=np.array(f.iloc[1:,1:], dtype=np.float)

m=X.shape[0]

del f



Y=Y.reshape((m,1))

Y_oh=convert_to_oh(Y,10)



X=X.reshape((m,28,28,1))

X=X/255



f2=pd.read_csv('../input/digit-recognizer/test.csv')

test_X=np.array(f2,dtype=np.float)

mt=test_X.shape[0]

test_X=test_X.reshape((mt,28,28,1))

test_X=test_X



print(Y[j])

print(Y_oh[j])

del Y

#print(X[j])

#print(test_X[j])

plt.figure(1)

plt.gray()

plt.imshow(X[j].reshape((28,28)))

plt.show()
#splitting the data

train_X, dev_X, train_Y, dev_Y=train_test_split(X, Y_oh, test_size=0.01, random_state=1)

print(train_X.shape)

print(train_Y.shape)

print(dev_X.shape)

print(dev_Y.shape)

del X

del Y_oh
#prepare the model

model=my_net(input_shape=(28,28,1))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
'''

datagen = tf.keras.preprocessing.image.ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(train_X)

'''
#train the model

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



model.fit(train_X,train_Y,epochs=10,batch_size=42, validation_data = (dev_X,dev_Y), callbacks=[learning_rate_reduction])

'''

epochs=6

batch_size=42

history = model.fit_generator(datagen.flow(train_X,train_Y, batch_size=batch_size),

                              epochs = epochs, validation_data = (dev_X,dev_Y),

                              verbose = 2, steps_per_epoch=train_X.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])

'''                              
#check on validation set

'''preds=model.evaluate(dev_X,dev_Y)

print('loss : ',(str(preds[0])))

print('accuracy : ',(str(preds[1])))'''
#predict on test set

predictions=model.predict(test_X)

#print(predictions)

Y_test=from_oh(predictions)

print(Y_test[:10])
#save model

model.save('dig_rec')
#save test result

pd.DataFrame(Y_test).to_csv("ans.csv",header=['ImageId','Label'],index=False)