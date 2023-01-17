import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D, GaussianDropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
%matplotlib inline 
batch_size = 128
nb_classes = 10
nb_epoch = 40
img_rows, img_cols = 32, 32
img_channels = 3
classes = ['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
n = 101
# plt.imshow(X_train[n])
# plt.show()
# print("Номер класса:", y_train[n])
# print("Тип объекта:", classes[y_train[n][0]])
X_train.shape
# X_train = X_train.reshape((50000, 32, 32, 3))
X_train = X_train.astype('float32')
# X_test = X_test.reshape((10000, 32, 32, 3))
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = utils.to_categorical(y_train, nb_classes)
Y_test = utils.to_categorical(y_test, nb_classes)
%%time
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                        input_shape=(img_rows, img_cols, img_channels), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dropout(0.75))
model.add(Dense(nb_classes, activation='softmax'))
model.summary()

callbacks_list = [EarlyStopping(monitor='val_loss', patience=5),
                  ModelCheckpoint(filepath='my_model.h5',
                                  monitor='val_loss',
                                  save_best_only=True),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
                  ] 
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
%%time
history = model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              callbacks=callbacks_list,
              validation_split=0.1,
              verbose=1)
scores = model.evaluate(X_test, Y_test, verbose=1)
print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1] * 100, 4))
plt.plot(history.history['accuracy'], 
         label='Доля правильных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'], 
         label='Доля правильных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля правильных ответов')
plt.legend()
plt.show()
plt.plot(history.history['loss'], 
         label='Оценка потерь на обучающем наборе')
plt.plot(history.history['val_loss'], 
         label='Оценка потерь на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Оценка потерь')
plt.legend()
plt.show()
# model.save("cifar10_model.h5")
# files.download("cifar10_model.h5")
