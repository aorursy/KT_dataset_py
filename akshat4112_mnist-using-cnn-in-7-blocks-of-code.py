from keras.datasets import mnist

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Conv2D, MaxPooling2D, Flatten
(trainx,  trainy) , (testx, testy) = mnist.load_data()
trainx = trainx.reshape((60000, 28, 28, 1))

trainx = trainx.astype('float32') / 255



testx = testx.reshape((10000, 28, 28, 1))

testx = testx.astype('float32') / 255



trainy = to_categorical(trainy)

testy = to_categorical(testy)
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape= (28, 28, 1)))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer = 'rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])

model.summary()
history = model.fit(trainx, trainy, epochs=6, batch_size=64)
test_loss, test_accuracy = model.evaluate(testx, testy)
test_accuracy