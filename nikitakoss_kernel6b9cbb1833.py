
from keras.datasets import mnist #Загружаем базу mnist
from keras.datasets import cifar10 #Загружаем базу cifar10
from keras.datasets import cifar100 #Загружаем базу cifar100

from keras.models import Sequential
from keras.layers import Input,Dense, Activation, MaxPooling2D,Dropout, BatchNormalization,Average,Flatten, Maximum,Multiply,Conv2D, Concatenate, Minimum
import numpy as np
import matplotlib.pyplot as plt 
from keras import utils
from keras.models import Model
from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator # работа с изображениями
from keras.optimizers import Adam, Adadelta # оптимизаторы
from keras import utils #Используем дял to_categoricall
from keras.preprocessing import image #Для отрисовки изображений
#from google.colab import files #Для загрузки своей картинки
import numpy as np #Библиотека работы с массивами
import matplotlib.pyplot as plt #Для отрисовки графиков
from keras.callbacks import EarlyStopping
from PIL import Image #Для отрисовки изображений
import random #Для генерации случайных чисел 
import math # Для округления
import os #Для работы с файлами 
!ls /kaggle/input/

train_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/train" 
val_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/val"
test_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/test"
batch_size=50
img_width = 192
img_height =108

datagen = ImageDataGenerator(
    rescale=1. /255, #умножает значение на указанное число после преобразований
    #rotation_range = 10, #поворачивает изображение для генерации выборки
    #width_shift_range = 0.1, #двигает изображение поширине
    #height_shift_range = 0.1, #дdигает по высоте
    #zoom_range = 0.1, #зумируем
    #horizontal_flip = True,#горизонтальное зеркалирование
    #fill_mode='nearest', #Заполнение пикселей вне границ ввода
    validation_split=0.2, #Указываем разделение изображений на обучающую и тестовую выборку
    
)
#обучающая выборка
train_generator = datagen.flow_from_directory(
    train_path,#путь к выборке
    target_size = (img_height, img_width), #размер изображений
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True,
    subset = 'training',#набор для обучения
    color_mode ="grayscale"
)

#проверочная выборка
validation_generator = datagen.flow_from_directory(
    train_path,#путь к выборке
    target_size = (img_height, img_width), #размер изображений
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True,
    subset = 'validation', #набор для обучения
    color_mode ="grayscale"
)
fig, axs = plt.subplots(1,2,figsize=(15,5))
for i in range(2):
  xray_path = train_path +'/'+  os.listdir(train_path)[i] +'/'
  img_path  = xray_path + random.choice(os.listdir(xray_path))
  axs[i].imshow(image.load_img(img_path, target_size = (img_height, img_width)))
print()
plt.show()
#создаем модель
from keras.regularizers import l2
import numpy as np
np.random.seed(1000)



model = Sequential([
      Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(img_height,img_width,  1), kernel_constraint=maxnorm(3), kernel_initializer='normal'),
      Conv2D(32, (3, 3), padding='valid', activation='relu', kernel_constraint=maxnorm(3), kernel_initializer='normal'),
      BatchNormalization(),
      MaxPooling2D(pool_size=(2, 2)),
       
      Conv2D(64, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3), kernel_initializer='normal'), 
      Conv2D(64, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3), kernel_initializer='normal'),
      Conv2D(64, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3), kernel_initializer='normal'),  
              
      BatchNormalization(), 
      MaxPooling2D(pool_size=(2, 2)),
      Dropout(0.5),     
      Conv2D(128, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3), kernel_initializer='normal'),
      Conv2D(128, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3), kernel_initializer='normal'),
      Conv2D(128, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3), kernel_initializer='normal'),
      BatchNormalization(), 
      MaxPooling2D(pool_size=(2, 2)),
      Dropout(0.5),
    
      Conv2D(256, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3), kernel_initializer='normal'),
      Conv2D(256, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3), kernel_initializer='normal'), 
      Conv2D(256, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3), kernel_initializer='normal'),
      Conv2D(512, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3), kernel_initializer='normal'),
      BatchNormalization(),
      MaxPooling2D(pool_size=(2, 2)),   
      Dropout(0.5),
    
      Flatten(),      
      Dense(2048, activation='relu', kernel_constraint=maxnorm(3), kernel_initializer='normal'),    
      #Dropout(0.5), 
      Dense(1024, activation='relu', kernel_constraint=maxnorm(3), kernel_initializer='normal'),
      Dense(512, activation='relu', kernel_constraint=maxnorm(3), kernel_initializer='normal'),
      BatchNormalization(),
      Dense(len(train_generator.class_indices), activation='softmax')
      
      ])


model.summary()
import tensorflow as tf
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#109/109 [==============================] - 14s 132ms/step - loss: 0.1362 - accuracy: 0.9492 - val_loss: 0.8588 - val_accuracy: 0.7496
from keras.callbacks import EarlyStopping
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs=20,
    verbose=1,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)]
)

#Оображаем график точности обучения
plt.plot(history.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'], 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()
test_generator =datagen.flow_from_directory(
    directory=test_path,
    target_size=(img_height, img_width),
    color_mode="grayscale",
    batch_size=1,
    class_mode='categorical',
    shuffle=False
   
)
model.evaluate_generator(test_generator,steps= 624)
import glob
def load_image(img_path, show=True):

    img = image.load_img(img_path, target_size=( img_height,img_width), color_mode ="grayscale")
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    img2 = image.load_img(img_path)
    img_tensor2 = image.img_to_array(img2)                    # (height, width, channels)
    img_tensor2 = np.expand_dims(img_tensor2, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor2 /= 255.                                      # imshow expects values in the range [0, 1]
   
    
    
    if show:
        plt.imshow(img_tensor2[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor


def get_key(d, value):
    for k, v in d.items():
        if v == value:
            return k


for i in range(2):
    xray_path = test_path +'/'+  os.listdir(train_path)[i] +'/'
    img_path  = xray_path + random.choice(os.listdir(xray_path))
    #print(xray_path, img_path)
    new_image = load_image(img_path)
    pred = model.predict(new_image)
    print ("filename:", img_path)
    prediction = np.argmax(pred) # Получаем индекс самого большого элемента (это итоговая цифра, которую распознала сеть)
    print("Давгноз:",get_key(train_generator.class_indices,prediction) ) 




  