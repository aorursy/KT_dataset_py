
import tensorflow
import keras
import matplotlib.pyplot as plt
from keras.utils import np_utils
import numpy as np

from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
%matplotlib inline
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(x_train[i],cmap='gray')
    plt.title("class {}".format(y_train[i]))
    
#pre-processing input
cat=10
X_train=x_train.reshape(60000,784)
X_test=x_test.reshape(10000,784)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train/=255
X_test/=255
#categorical conversion
Y_train=np_utils.to_categorical(y_train,cat)
Y_test=np_utils.to_categorical(y_test,cat)

#creating model
model=Sequential()
model.add(Dense(600,activation='relu',input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(600,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
#compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#training model
model.fit(X_train,Y_train,batch_size=128,epochs=4,validation_data=(X_test,Y_test),verbose=1)
#Testing
predicted_classes=model.predict_classes(X_test)
#correct indices
correct=np.nonzero(predicted_classes == y_test)[0]
#incorrect
incorrect=np.nonzero(predicted_classes != y_test)[0]

#visualise result
plt.figure()
plt.tight_layout(pad=4)
for i,indice in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    
    plt.imshow(X_test[indice].reshape(28,28),cmap='gray',interpolation='none')
    plt.title("predicted {},class {}".format(predicted_classes[indice],y_test[indice]))
plt.figure()
plt.tight_layout(pad=4)
for i,indice in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    
    plt.imshow(X_test[indice].reshape(28,28),cmap='gray',interpolation='none')
    plt.title("predicted {},class {}".format(predicted_classes[indice],y_test[indice]))    
