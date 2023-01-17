# последовательная модель (стек слоев)
from tensorflow.keras.models import Sequential
# полносвязный слой и слой выпрямляющий матрицу в вектор
from tensorflow.keras.layers import Dense, Flatten
# слой выключения нейронов и слой нормализации выходных данных (нормализует данные в пределах текущей выборки)
from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D, GaussianDropout
# слои свертки и подвыборки
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
# работа с обратной связью от обучающейся нейронной сети
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
# вспомогательные инструменты
from tensorflow.keras import utils
%matplotlib inline 
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
classes=['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']
# Размер мини-выборки
batch_size = 128*2
# Количество классов изображений
nb_classes = 10
# Количество эпох для обучения
nb_epoch = 30
# Размер изображений
img_rows, img_cols = 32, 32
# Количество каналов в изображении: RGB
img_channels = 3
n = 10
plt.imshow(X_train[n])
plt.show()
print("Номер класса:", y_train[n])
print("Тип объекта:", classes[y_train[n][0]])
X_train.shape
#Если у изображение отсутствует параметр глубина - необходимо его добавить с помощью закомментированного кода
#X_train = X_train.reshape((50000, 32, 32, 3))
X_train = X_train.astype('float32')
#X_test = X_test.reshape((10000, 32, 32, 3))
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = utils.to_categorical(y_train, nb_classes)
Y_test = utils.to_categorical(y_test, nb_classes)
%%time
# Создаем последовательную модель
model = Sequential()
# Первый сверточный слой
model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(img_rows, img_cols, img_channels), activation='relu'))
# Второй сверточный слой
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# # Первый слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# # Первый слой нормализации данных
model.add(BatchNormalization())
# # Первый Слой регуляризации Dropout
#model.add(Dropout(0.25))

# Третий сверточный слой
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# Четвертый сверточный слой
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# # Второй слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# # Второй слой нормализации данных
model.add(BatchNormalization())
# Второй Слой регуляризации Dropout
model.add(Dropout(0.75))

# Пятый сверточный слой
#model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# # Шестой сверточный слой
#model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# # Третий слой подвыборки
#model.add(MaxPooling2D(pool_size=(2, 2)))
# # Третий слой нормализации данных
# model.add(BatchNormalization())
# # Третий Слой регуляризации Dropout
# model.add(Dropout(0.25))


# Слой преобразования данных из 2D представления в плоское
model.add(Flatten())
# Полносвязный слой для классификации
model.add(Dense(512, activation='relu'))
# # Четвертый слой нормализации данных
# model.add(BatchNormalization())
# # Четвертый Слой регуляризации Dropout
# model.add(Dropout(0.75))
# Выходной полносвязный слой
model.add(Dense(nb_classes, activation='softmax'))
model.summary()
callbacks_list = [EarlyStopping(monitor='val_loss', patience=5),
                  ModelCheckpoint(filepath='my_model.h5',
                                  monitor='val_loss',
                                  save_best_only=True),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
                  ] 

model.compile(loss='categorical_crossentropy',
              optimizer='nadam',
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
index=4125
plt.imshow(X_test[index].reshape((32,32)))
plt.show()
x = X_test[index]
x = np.expand_dims(x, axis=0)
x.shape
prediction = model.predict(x)
print(prediction)
prediction = np.argmax(prediction)
print(classes[prediction])
print(classes[y_test[0]])
files.upload()
!ls
img_path = 't-shirt.jpg'
img = image.load_img(img_path, target_size=(img_rows, img_cols))
plt.imshow(img)
plt.show()
x.shape
prediction = model.predict(x)
prediction = np.argmax(prediction)
print(classes[prediction])
