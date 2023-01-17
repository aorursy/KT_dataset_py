import os

import numpy as np



from keras.models import Sequential, load_model

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D





import matplotlib.pyplot as plt

%matplotlib inline
train_path = '../input/fruits/fruits-360_dataset/fruits-360/Training'

test_path = '../input/fruits/fruits-360_dataset/fruits-360/Test'



train_labels = os.listdir(train_path)

test_labels = os.listdir(test_path)



train_generator = ImageDataGenerator(rescale=1./255)

test_generator = ImageDataGenerator(rescale=1./255)



train_data = train_generator.flow_from_directory(train_path, batch_size=32, classes=train_labels, target_size=(64,64))

test_data = test_generator.flow_from_directory(test_path, batch_size=32, classes=train_labels, target_size=(64,64))
cnn = Sequential()



cnn.add(Conv2D(16, (3, 3), input_shape = (64, 64, 3), padding = "same", activation = "relu"))

cnn.add(MaxPooling2D())



cnn.add(Conv2D(32, (3,3), padding='same', activation='relu'))

cnn.add(MaxPooling2D())



cnn.add(Conv2D(64, (3,3), padding='same', activation='relu'))

cnn.add(MaxPooling2D())



cnn.add(Flatten())



cnn.add(Dropout(0.25))



cnn.add(Dense(256, activation = "relu"))

cnn.add(Dense(len(train_labels), activation = "softmax"))
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn.summary()
history_cnn = cnn.fit(train_data, steps_per_epoch=1000, epochs=5, validation_steps=400, validation_data=test_data)
plt.plot(history_cnn.history['accuracy'])

plt.plot(history_cnn.history['val_accuracy'])
score = cnn.evaluate(test_data)
cnn.save("cnn_model.h5")
new_cnn = load_model("cnn_model.h5")