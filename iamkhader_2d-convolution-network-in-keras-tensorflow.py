import tensorflow as tf

import numpy as np

from sklearn.cross_validation import train_test_split

from PIL import Image

from matplotlib.pyplot import imshow

from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential

from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D

from keras.utils import np_utils
data = np.genfromtxt('../input/train.csv', delimiter=',', skip_header = 1)

print(data.shape)
# train, test, validation split

Y = data[:,0]

X = data[:,1:]



# normalize independant variables

X = X/255



# reshaping to get image structure

X_reshaped = X.reshape((42000,28,28,1))

print(X_reshaped.shape)



# one hot label encoding

Y_encode = np.eye((np.unique(Y)).size)[Y.astype(int)]



x_train, x_test, y_train, y_test = train_test_split(X_reshaped, Y_encode, test_size = 0.3)

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size = 0.5)
conv_net = Sequential()



# convolution layer 1

conv_net.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))

conv_net.add(MaxPooling2D(pool_size=(3,3)))

conv_net.add(Dropout(0.5))



print(conv_net.output.shape)



# convolution layer 2

conv_net.add(Conv2D(64, (3, 3), activation='relu'))

conv_net.add(MaxPooling2D(pool_size=(3,3)))

conv_net.add(Dropout(0.5))



print(conv_net.output.shape)



# fully connected

conv_net.add(Flatten())

conv_net.add(Dense(128, activation='relu'))

conv_net.add(Dropout(0.5))

conv_net.add(Dense(10, activation='softmax'))



conv_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
conv_net.fit(x_train, y_train,batch_size=32, nb_epoch=10, verbose=0)
score = conv_net.evaluate(x_test, y_test, verbose=0)

print("%s: %.2f%%" % (conv_net.metrics_names[1], score[1]*100))



score = conv_net.evaluate(x_val, y_val, verbose=0)

print("%s: %.2f%%" % (conv_net.metrics_names[1], score[1]*100))
sub_data = np.genfromtxt('../input/test.csv', delimiter=',', skip_header = 1)

print(sub_data.shape)



sub_data = sub_data/255

sub_data_reshaped = sub_data.reshape((28000,28,28,1))



predictions = conv_net.predict(sub_data_reshaped, verbose=1)



final = np.argmax(predictions, axis=1)

print(final.shape)



np.savetxt(X = final, fname='predictions.csv',delimiter=',', newline='\n', header='Label')
