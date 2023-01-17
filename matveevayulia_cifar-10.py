import numpy as np
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D, GaussianDropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
%matplotlib inline 
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
classes=['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']
# Размер мини-выборки
"""может повлиять на изменение качества обучения и скорость обучения"""
batch_size = 169
# Количество классов изображений
nb_classes = 10
# Количество эпох для обучения
"""можно не менять, если будете использовать callbacks"""
nb_epoch = 40
# Размер изображений
"""настроить под ваши изображения"""
img_rows, img_cols = 32, 32
# Количество каналов в изображении: RGB
img_channels = 3
n = 101
plt.imshow(X_train[n])
plt.show()
print("Номер класса:", y_train[n])
print("Тип объекта:", classes[y_train[n][0]])
X_train.shape
Y_train = utils.to_categorical(y_train, nb_classes)
Y_test = utils.to_categorical(y_test, nb_classes)
%%time
# Создаем последовательную модель
model = Sequential()
# Первый сверточный слой
# padding='same' - не будет меняться размер картинки. padding='valid'
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid',
                        input_shape=(img_rows, img_cols, img_channels), activation='elu'))
# Второй сверточный слой
model.add(Conv2D(32, (3, 3), activation='elu', padding='same'))
# # Первый слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# # Первый слой нормализации данных
model.add(BatchNormalization())
# # Первый Слой регуляризации Dropout
model.add(Dropout(0.2))

# Третий сверточный слой
model.add(Conv2D(64, (3, 3), padding='same', activation='elu'))
# Четвертый сверточный слой
model.add(Conv2D(64, (3, 3), padding='same', activation='elu'))
# Второй слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# # Второй слой нормализации данных
model.add(BatchNormalization())
#  # Второй Слой регуляризации Dropout
model.add(Dropout(0.3))

# # Пятый сверточный слой
model.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
# # Шестой сверточный слой
model.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
# # Третий слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# # Третий слой нормализации данных
model.add(BatchNormalization())
# # Третий Слой регуляризации Dropout
model.add(Dropout(0.4))


# Слой преобразования данных из 2D представления в плоское
model.add(Flatten())
# Полносвязный слой для классификации
model.add(Dense(512, activation='relu'))
# # Четвертый слой нормализации данных
# model.add(BatchNormalization()) 
# # Четвертый Слой регуляризации Dropout
model.add(Dropout(0.75))
# Выходной полносвязный слой
model.add(Dense(nb_classes, activation='softmax'))
model.summary()

callbacks_list = [EarlyStopping(monitor='val_loss', patience=4),
                  ModelCheckpoint(filepath='my_model.h5',
                                  monitor='val_accuracy',
                                  save_best_only=True),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2)
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