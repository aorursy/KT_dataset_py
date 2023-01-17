# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
df.head()
df.label.value_counts()
X=df.drop('label',axis=1)

y=df.label



test = test.drop('id',axis=1)
X=X/255



test = test/255
X=X.values.reshape(-1,28,28,1)

test=test.values.reshape(-1,28,28,1)
len(X[0]),len(X[0][0])
y
from keras.utils.np_utils import to_categorical



y=to_categorical(y)

y
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.15)



X_train[0][:,:,0] # pixel values of first entry 
import matplotlib.pyplot as plt

import seaborn as sns





plt.imshow(X_train[0][:,:,0]) #Visualizing this matrix as picture



from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

        featurewise_center=False, 

        samplewise_center=False,  

        featurewise_std_normalization=False, 

        samplewise_std_normalization=False,  

        zca_whitening=False,  

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  

        vertical_flip=False) 





datagen.fit(X_train) # only augment training dataset 
from keras.models import Sequential



from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization





model = Sequential()



# add conv layers 





model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))





# normalization latyer 



model.add(BatchNormalization(momentum=.15))



#Pooling layer 



model.add(MaxPool2D(pool_size=(2,2)))



#dropout 



model.add(Dropout(0.25))



# same structure again with different parameters 



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))



model.add(BatchNormalization(momentum=0.15))



model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))



model.add(Dropout(0.25))



# again 



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



#flatteining our output to feed into a ANN 

model.add(Flatten())



model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.4))



#output layer that will give us the label 

model.add(Dense(10, activation = "softmax"))







model.summary()



from keras.optimizers import Adam #,RMSprop

optimizer=Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)





model.compile(optimizer=optimizer,loss=['categorical_crossentropy'],metrics=['accuracy'])







from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs=5 #change this to 30 if you need to get better score

batch_size=64



# Fit the model

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, 

                              validation_data = (X_test,y_test),

                              verbose = 2, 

                              steps_per_epoch=X_train.shape[0] // batch_size, 

                              callbacks=[learning_rate_reduction])

fig,ax=plt.subplots(2,1)

fig.set

x=range(1,1+epochs)

ax[0].plot(x,history.history['loss'],color='red')

ax[0].plot(x,history.history['val_loss'],color='blue')



ax[1].plot(x,history.history['accuracy'],color='red')

ax[1].plot(x,history.history['val_accuracy'],color='blue')

ax[0].legend(['trainng loss','validation loss'])

ax[1].legend(['trainng acc','validation acc'])

plt.xlabel('Number of epochs')

plt.ylabel('accuracy')
history.history['accuracy'][-1]
test=pd.read_csv('../input/Kannada-MNIST/test.csv')









test=test.drop('id',axis=1)

test=test/255

test=test.values.reshape(-1,28,28,1)



test
plt.imshow(test[0][:,:,0])
test[[0]][:,:,:].shape
test[0].shape
preds = model.predict(test)





preds[0]




preds=np.argmax(preds,axis=1)

preds
test
test[[55]][:,:,:].shape
preds = model.predict(test[[55]][:,:,:])

preds=np.argmax(preds,axis=1)

preds
model.save('my_model.h5')
