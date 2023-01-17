import numpy as np 



from keras.datasets import mnist

from keras.utils import np_utils

from keras.models import Sequential, load_model

from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten

from keras.optimizers import Adam







#X shape (60,000 28*28) Y shape (10,000)

(X_train, Y_train),(X_test, Y_test)=mnist.load_data()





# data pre-processing

X_train = X_train.reshape(-1, 1, 28, 28)

X_test = X_test.reshape(-1, 1, 28, 28)

Y_train = np_utils.to_categorical(Y_train, num_classes = 10)

Y_test = np_utils.to_categorical(Y_test, num_classes = 10)



# build neural net

model = Sequential()



model.add(Conv2D(input_shape=(1, 28, 28), kernel_size=(5, 5), filters=32, padding="same"))



model.add(Activation('relu'))



model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

model.add(Conv2D(kernel_size=(5, 5), filters=64, padding="same"))

model.add(Activation('relu'))



# model.add(MaxPooling2D(pool_size = (2, 2),border_mode = 'same'))



model.add(Flatten())



model.add(Dense(1024))

model.add(Activation('relu'))

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dense(10))

model.add(Activation('softmax'))



# optimizer

adam = Adam(lr = 1e-4)



model.compile(

    optimizer=adam,

    loss="categorical_crossentropy",

    metrics=['accuracy'],

)



model.fit(X_train, Y_train, epochs=3, batch_size=32)



loss, accuracy = model.evaluate(X_test, Y_test)



print("test loss:", loss)

print("test accuracy", accuracy)



model.save("mymodel.h5")
model = load_model("mymodel.h5")

loss, accuracy = model.evaluate(X_test, Y_test)



print("test loss:", loss)

print("test accuracy", accuracy)