from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np

# seed for reproducing same results
seed = 785
np.random.seed(seed)

# load dataset
dataset = np.loadtxt('../input/emnist-byclass-train.csv', delimiter=',')
# split into input and output variables
dataset_test = np.loadtxt('../input/emnist-byclass-test.csv', delimiter=',')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
%matplotlib inline
X = dataset[0:348966,1:785]/255
Y = dataset[0:348966,0]
X_test = dataset_test[:,1:785]/255
Y_test = dataset_test[:,0]
x_cv = dataset[348966:697932,1:785]/255
y_cv = dataset[348966:697932,0]
n=1
plt.imshow(X[n].reshape(28,28),cmap='Greys')
print(Y[n])
Y1= np_utils.to_categorical(Y)
y_test1= np_utils.to_categorical(Y_test)
y_cv1= np_utils.to_categorical(y_cv)
X.shape,X_test.shape,x_cv.shape,Y1.shape
X=X.reshape(348966,28,28,1)
X_test=X_test.reshape(116323,28,28,1)
x_cv=x_cv.reshape(348966,28,28,1)
X.shape,X_test.shape,x_cv.shape,Y1.shape
import keras.layers as l

model = Sequential()

model.add(l.InputLayer([784]))


# network body
model.add(l.Dense(500))
model.add(l.Activation('relu'))

model.add(l.Dense(500))
model.add(l.Activation('relu'))

model.add(l.Dense(128))
model.add(l.Activation('relu'))
# output layer: 10 neurons for each class with softmax
model.add(l.Dense(62 , activation='softmax'))

# categorical_crossentropy is your good old crossentropy
# but applied for one-hot-encoded vectors
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(X,Y1,validation_data=(x_cv, y_cv1), epochs=10,batch_size=1000)

print("\nLoss, Accuracy = ", model.evaluate(X_test, y_test1))
test_predictions = model.predict_proba(X_test).argmax(axis=-1)
test_answers = y_test1.argmax(axis=-1)
test_accuracy = np.mean(test_predictions==test_answers)
print("\nTest accuracy: {} %".format(test_accuracy*100))

test_predictions,test_answers
model.save("weightsnew.h5")
