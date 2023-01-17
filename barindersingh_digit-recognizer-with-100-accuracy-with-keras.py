import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from keras import models

from keras import layers

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

sample_submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

print(train.shape)

print(test.shape)
x_train = train.iloc[:,1:786]

y_train = train.iloc[:,0]

network = models.Sequential()

network.add(layers.Dense(784 , activation = 'relu' , input_shape = (28*28,)))

network.add(layers.Dense(784 , activation = 'relu' , input_shape = (28*28,)))

network.add(layers.Dense(10 , activation = 'softmax'))



network.compile(optimizer='adam' ,

                loss = 'categorical_crossentropy' ,

                metrics = ['accuracy'])

x_train = x_train.values.reshape((42000,28*28))   #42000 are images

x_train = x_train.astype('float32') /255

test = test.values.reshape((28000,28*28))     #28000 are images 

test = test.astype('float32') /255
y_train = to_categorical(y_train)

print(test.shape)

print(y_train.shape)

print(x_train.shape)

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=4)
network.fit(x_train , y_train , epochs = 100 , batch_size = 128)
test_loss , test_acc = network.evaluate(X_val , Y_val)

print("Accuracy:", test_acc *100,'%', 'Loss:', test_loss *100)
results = network.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

results
Final_results = sample_submission

Final_results['Label'] = results

Final_results.head()
Final_results.to_csv('Final_results_keras.csv', index=False)