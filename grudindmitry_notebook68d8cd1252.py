from tensorflow.keras.datasets import cifar10

import matplotlib.pyplot as plt

%matplotlib inline 

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

classes=['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']
import numpy as np

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D, GaussianDropout

from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras import utils

from tensorflow.keras.preprocessing import image

%matplotlib inline 
batch_size = 150

nb_classes = 10

nb_epoch = 40

img_rows, img_cols = 32, 32

img_channels = 3
print(X_train.shape)

print(X_test.shape)
X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255
Y_train = utils.to_categorical(y_train, nb_classes)

Y_test = utils.to_categorical(y_test, nb_classes)
%%time

# Создаем последовательную модель

model = Sequential()

# Первый сверточный слой

"""можно менять:

- количество сверточных слоев в блоке (блок - это свертка или несколько сверток, которые завершаются пулингом)

- количество сверточных блоков

- количество ядер свертки (filters)

- размер ядра свертки (kernel_size)

- MaxPooling2D на AveragePooling2D

- добавлять/убирать BatchNormalization. Его можно ставить после каждого слоя свертки, а можно в конце блока. Экспериментируйте

- добавлять/убирать Dropout. Его можно ставить после каждого блока свертки, а можно еще и после Dense слоя. Экспериментируйте

- менять в Dropout процент выключаемых нейронов

- менять Dropout на SpatialDropout2D, GaussianDropout

- менять количество слоев Dense

- менять количество нейронов в слое Dense"""

# padding='same' - не будет меняться размер картинки. padding='valid'

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',

                        input_shape=(img_rows, img_cols, img_channels), activation='relu'))

# Второй сверточный слой

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

# # Первый слой подвыборки

model.add(MaxPooling2D(pool_size=(2, 2)))

# # Первый слой нормализации данных

# model.add(BatchNormalization())

# # Первый Слой регуляризации Dropout

model.add(Dropout(0.25))



# Третий сверточный слой

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

# Четвертый сверточный слой

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

# Второй слой подвыборки

model.add(MaxPooling2D(pool_size=(2, 2)))

# # Второй слой нормализации данных

# model.add(BatchNormalization())

#  # Второй Слой регуляризации Dropout

# model.add(Dropout(0.25))



# # Пятый сверточный слой

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

# # Шестой сверточный слой

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

# # Третий слой подвыборки

model.add(MaxPooling2D(pool_size=(2, 2)))

# # Третий слой нормализации данных

# model.add(BatchNormalization())

# # Третий Слой регуляризации Dropout

model.add(Dropout(0.25))





# Слой преобразования данных из 2D представления в плоское

model.add(Flatten())

# Полносвязный слой для классификации

model.add(Dense(1050, activation='relu'))

# # Четвертый слой нормализации данных

# model.add(BatchNormalization()) 

# # Четвертый Слой регуляризации Dropout

model.add(Dropout(0.75))

# Выходной полносвязный слой

model.add(Dense(nb_classes, activation='softmax'))
model.summary()
# EarlyStopping - если patience эпох качество не растет или потери не убывают, то происходит останов обучения

# ModelCheckpoint - сохраняет в указанную директорию веса лучшей модели и в конце обучения возвращает их.

# ReduceLROnPlateau - уменьшает шаг обучения в factor раз после patience эпох без улучшения качества обучения

"""patience у ReduceLROnPlateau меньше, чем  patience у EarlyStopping

иначе не произойдет ни одного уменьшения шага обучения

можно следить за monitor=val_loss, а можно monitor=val_accuracy

меняйте factor в ReduceLROnPlateau - он значим"""

callbacks_list = [EarlyStopping(monitor='val_loss', patience=5),

                  ModelCheckpoint(filepath='my_model.h5',

                                  monitor='val_loss',

                                  save_best_only=True),

                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

                  ] 

# экспериментируйте с optimizer

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