from tensorflow.keras.datasets import cifar10

import matplotlib.pyplot as plt

%matplotlib inline 

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

classes=['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']
batch_size = 128

nb_classes = len(classes)

nb_epoch = 30

img_rows, img_cols, img_channels = X_train[0].shape
n = 101

plt.imshow(X_train[n])

plt.show()

print("Номер класса:", y_train[n])

print("Тип объекта:", classes[int(y_train[n])])
X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255
from tensorflow.keras import utils



Y_train = utils.to_categorical(y_train, nb_classes)

Y_test = utils.to_categorical(y_test, nb_classes)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D, GaussianDropout

from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
%%time

# Создаем последовательную модель

model = Sequential()



# Первый сверточный слой

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',

                        input_shape=(img_rows, img_cols, img_channels), activation='relu'))

# Второй сверточный слой

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

# # Первый слой подвыборки

model.add(MaxPooling2D(pool_size=(2, 2)))

# Первый слой нормализации данных

model.add(BatchNormalization())

# # Первый Слой регуляризации Dropout

model.add(Dropout(0.28))



# Третий сверточный слой

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

# Четвертый сверточный слой

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

# Второй слой подвыборки

model.add(MaxPooling2D(pool_size=(3, 3)))

# # Второй слой нормализации данных

model.add(BatchNormalization())

#  # Второй Слой регуляризации Dropout

model.add(Dropout(0.28))



# # Пятый сверточный слой

model.add(Conv2D(128, (4, 4), padding='same', activation='relu'))

# # Шестой сверточный слой

model.add(Conv2D(128, (4, 4), padding='same', activation='relu'))

# # Третий слой подвыборки

model.add(MaxPooling2D(pool_size=(2, 2)))

# # Третий слой нормализации данных

model.add(BatchNormalization())

# # Третий Слой регуляризации Dropout

model.add(Dropout(0.25))





# Слой преобразования данных из 2D представления в плоское

model.add(Flatten())

# Полносвязный слой для классификации

model.add(Dense(512, activation='relu'))

# # Четвертый слой нормализации данных

model.add(BatchNormalization()) 

# # Четвертый Слой регуляризации Dropout

model.add(Dropout(0.75))

# Выходной полносвязный слой

model.add(Dense(nb_classes, activation='softmax'))
model.summary()
callbacks_list = [EarlyStopping(monitor='val_loss', patience=7),

                  ModelCheckpoint(filepath='my_model.h5',

                                  monitor='val_loss',

                                  save_best_only=True),

                  ReduceLROnPlateau(monitor='val_loss', factor=0.15, patience=3)

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
# Оцениваем качество обучения модели на тестовых данных

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