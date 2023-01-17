import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical



data_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

data_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
print(data_train.shape)

print(data_test.shape)
X = np.array(data_train.iloc[:, 1:])

# Convert the y_train values to be one-hot encoded for categorical analysis by Keras.

y = to_categorical(np.array(data_train.iloc[:, 0]))



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=101)



#Test data

X_test = np.array(data_test.iloc[:, 1:])

y_test = to_categorical(np.array(data_test.iloc[:, 0]))
X_train = X_train.reshape(-1, 28, 28, 1)

X_test = X_test.reshape(-1, 28, 28, 1)

X_val = X_val.reshape(-1, 28, 28, 1)



print(X_train.max())

print(X_test.max())

print(X_val.max())
X_train= X_train / 255

X_test= X_test /255

X_val= X_val / 255
print(X_train.max())

print(X_test.max())

print(X_val.max())

print(X_train[0])
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization
batch_size = 256

num_classes = 10

#image dimensions

img_rows, img_cols = 28, 28

input_shape = (img_rows,img_cols,1)

model = Sequential()



# CONVOLUTIONAL LAYER

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))

model.add(MaxPooling2D((2, 2))) # POOLING LAYER

model.add(Dropout(0.25)) #Dropping`25% of input units to 0 at each update during training time, to prevent overfitting.

#

#

#

model.add(Conv2D(64, (3, 3), activation='relu')) # CONVOLUTIONAL LAYER

model.add(MaxPooling2D(pool_size=(2, 2)))# POOLING LAYER 

model.add(Dropout(0.25))#Dropping`25% of input units

#

#

#



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.4)) #Dropping`40% of input units

#

#

#

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER

model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)

model.add(Dense(128, activation='relu'))

#Dropping`30% of input units

model.add(Dropout(0.3))

#

#

#

# Final Dense Layer of 10 Neurons(Can't change the number of Neurons) with a softmax activation

model.add(Dense(num_classes, activation='softmax'))



# Then compile the model

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()
epochs = 50

history = model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_val, y_val))
score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])

print('Test accuracy:', score[1])
import matplotlib.pyplot as plt

%matplotlib inline

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['loss','val_loss'])

plt.title('Loss')

plt.xlabel('epoch')
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend(['accuracy','val_accurac'])

plt.title('Accuracy')

plt.xlabel('epoch')
model.metrics_names
model.evaluate(X_test,y_test)
from sklearn.metrics import classification_report
predictions = model.predict_classes(X_test)
y_test.shape
y_test[0]
predictions[0]
y_test
#get the predictions for the test data

predicted_classes = model.predict_classes(X_test)



#get the indices to be plotted

y_true = data_test.iloc[:, 0]

correct = np.nonzero(predicted_classes==y_true)[0]

incorrect = np.nonzero(predicted_classes!=y_true)[0]





target_names = ["Class {}".format(i) for i in range(num_classes)]

print(classification_report(y_true, predicted_classes, target_names=target_names))

 
model.save('Fashion_mnist.h5')
from IPython.display import FileLink, FileLinks

FileLinks('.') #lists all downloadable files on server