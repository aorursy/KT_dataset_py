from importlib import reload

#from __future__ import print_function

import keras

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense, Dropout,Flatten,Conv2D, MaxPooling2D

from keras.optimizers import RMSprop

from keras.utils.np_utils import to_categorical

import pandas as pd

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/train.csv')

train.head()

print('training data is (%d, %d).'% train.shape)
X_train_all = (train.iloc[:,1:].values).astype('float32')/255 # all pixel values, convert to value in [0,1]

y_train_all = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

y_train_all= to_categorical(y_train_all) # This convert y into onehot representation

X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.10, random_state=42)
X_train_img=np.reshape(X_train,(X_train.shape[0],28,28,1))

X_val_img = np.reshape(X_val,(X_val.shape[0],28,28,1))
plt.imshow(X_train_img[0,:,:,0],cmap = 'gray')

plt.show()
model = Sequential()

model.add(Conv2D(32, (3, 3),

                 activation='relu',

                 input_shape=(28,28,1),strides=(2,2)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten(name='flatten'))

model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer='RMSprop',

              metrics=['accuracy'])
batch_size = 64 

nb_epochs = 5 

history = model.fit(X_train_img, y_train,

                    batch_size=batch_size,

                    epochs=nb_epochs,

                    verbose=2, 

                    validation_data=(X_val_img, y_val),

                    initial_epoch=0)
model = Sequential()

model.add(Conv2D(64, (5, 5),

                 activation='relu',

                 input_shape=(28,28,1)))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))   

model.add(Flatten(name='flatten'))

model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',

              optimizer='RMSprop',

              metrics=['accuracy'])
batch_size = 64

nb_epochs = 2

history = model.fit(X_train_img, y_train,

                    batch_size=batch_size,

                    epochs=nb_epochs,

                    verbose=2, # verbose controls the infromation to be displayed. 0: no information displayed

                    validation_data=(X_val_img, y_val),

                    initial_epoch=0)

test = pd.read_csv('../input/test.csv')

X_test = (test.values).astype('float32')/255 # all pixel values

X_test_img = np.reshape(X_test,(X_test.shape[0],28,28,1))
pred_classes = model.predict_classes(X_test_img,verbose=0)
i=10

plt.imshow(X_test_img[i,:,:,0],cmap='gray')

plt.title('prediction:%d'%pred_classes[i])

plt.show()
def plot_difficult_samples(model,x,y, verbose=True):

    """

    model: trained model from keras

    x: size(n,h,w,c)

    y: is categorical, i.e. onehot, size(n,10)

    """ 

    #%%

    

    pred_classes = model.predict_classes(x,verbose= 0)

    y_val_classes = np.argmax(y, axis=1)

    er_id = np.nonzero(pred_classes!=y_val_classes)[0]

    #%%

    K = np.ceil(np.sqrt(len(er_id)))

    fig = plt.figure()

    print('There are %d wrongly predicted images out of %d validation samples'%(len(er_id),x.shape[0]))

    for i in range(len(er_id)):

        ax = fig.add_subplot(K,K,i+1)

        k = er_id[i]

        ax.imshow(x[er_id[i],:,:,0])

        ax.axis('off')

        if verbose:

            ax.set_title('%d as %d'%(y_val_classes[k],pred_classes[k]))
plot_difficult_samples(model,X_val_img,y_val,verbose=False)

plt.show()