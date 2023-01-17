# Importing Libraries for reading data 

import numpy as np 

import pandas as pd 



# For visuzlizing images

import matplotlib.pyplot as plt 

from skimage.io import imshow 



# Importing keras sequential model (See readme for details)

from keras.layers import Dense 

from keras.models import Sequential 

from keras.layers import Dense, Activation



from keras.models import Sequential

from keras.layers import Input

from keras.layers import Dense,Activation,Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPooling2D

from keras.utils import to_categorical

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping
# Loading data files for training



print ('Loading training data')

X_train_sat4 = '../input/X_test_sat4.csv'

y_train_sat4 = '../input/y_test_sat4.csv'



# Loading data to pandas dataframe

X_train = pd.read_csv(X_train_sat4)

Y_train = pd.read_csv(y_train_sat4)

X_train = X_train.as_matrix()

Y_train = Y_train.as_matrix()



print("Number of training examples are",X_train.shape[0])



#Reshaping the input to convert into a list for further processing

X_train_img = X_train.reshape([X_train.shape[0],28,28,4]).astype(float)
'''# Sequential model in Keras is a linear stack of layers. Sequential model could be created by passing a list of layer instances to the constructor.

model = Sequential()

model.add(Dense(4, input_dim=3136))

model.add(Activation('selu'))

'''
X_train = (X_train-X_train.mean())/X_train.std()
'''model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(X_train,Y_train,batch_size=32, epochs=5, verbose=1, validation_split=0.02)'''
model=Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 4)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0,25))



model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(X_train_img,Y_train,batch_size=32, epochs=5, verbose=1, validation_split=0.02)
print(history.history.keys())
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='lower right')

plt.show()


# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()