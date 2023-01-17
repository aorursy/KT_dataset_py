import numpy as np # linear algebra

import tensorflow as tf

import matplotlib.pyplot as plt



import time



def load_data(path):

    with np.load(path) as f:

        x_train, y_train = f['x_train'], f['y_train']

        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)



(x_train, y_train), (x_test, y_test) = load_data('../input/mnist-numpy/mnist.npz')



x_train, x_test = x_train/255, x_test/255
model = tf.keras.models.Sequential()



#this is not a conv model. plain ANN. 

#I tried adding one extra dense with 128 neurons and it showed improvement. I also increased the dropout a bit than usual.



model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dense(10, activation='softmax'))





model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])



fit_data = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
plt.plot(fit_data.history['loss'])

plt.plot(fit_data.history['val_loss'])
#generate some random index to test

test_indices = np.random.randint(low=1, high=1000, size=10)



for test_i in test_indices:

    print("Predicted",model.predict(x_test[test_i].reshape(1,28,28)).argmax(), " and Actual is: ", y_test[test_i])

    plt.imshow(x_test[test_i], cmap='gray')

    plt.show()

    time.sleep(2)

    