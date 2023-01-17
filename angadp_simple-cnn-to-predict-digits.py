import numpy

import pandas

import matplotlib.pyplot as plt



from keras.utils import to_categorical

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping
train_data = pandas.read_csv('../input/digit-recognizer/train.csv')

test_data = pandas.read_csv('../input/digit-recognizer/test.csv')
train_y = to_categorical(train_data["label"])

train_x = train_data.loc[:, train_data.columns != "label"]

train_x /= 256
train_x = train_x.values.reshape(-1, 28, 28, 1)
callback = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape=(28,28, 1)))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(256, activation='relu'))

model.add(Dense(10, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
datagen = ImageDataGenerator(

    rotation_range=10,

    zoom_range=0.1,

    width_shift_range=0.1,

    height_shift_range=0.1

)



datagen.fit(train_x)
history = model.fit(datagen.flow(train_x, train_y, batch_size=32), epochs=80, callbacks=[callback])
test_data /= 256

test_x = test_data.values.reshape(-1, 28, 28, 1)

y_pred = model.predict(test_x)

y_pred = numpy.argmax(y_pred, axis=1)

y_pred = pandas.Series(y_pred,name='Label')

submission = pandas.concat([pandas.Series(range(1, 28001), name='ImageId'), y_pred], axis=1)
submission.to_csv('my_submission.csv', index=False)