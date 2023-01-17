import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import keras
from keras.datasets import mnist



(x_train,y_train), (x_test,y_test) = mnist.load_data()
plt.matshow(x_train[1])
y_train[1]
from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten



model = Sequential()



model.add(Flatten(input_shape=[28,28]))

model.add(Dense(200, activation = 'relu'))

model.add(Dense(10, activation = 'softmax'))



model.summary()

model.compile(loss = 'sparse_categorical_crossentropy',

              optimizer = 'adam',

              metrics = ['accuracy']

             )
model.fit(x_train,y_train,epochs=5)
plt.matshow(x_test[5])
yp = model.predict(x_test)

yp[5]
np.argmax(yp[5])
model.evaluate(x_test,y_test)