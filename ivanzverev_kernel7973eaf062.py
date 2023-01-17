import numpy as np

import pandas as pd

import cv2

from glob import glob

from matplotlib import pyplot as plt

import math

import os
scale = 70 # Масштаб

seed = 7 # Рандомизирующий множитель



path = '../input/plant-seedlings-classification/train/*/*.png' 

files = glob(path) # Путь к файлам



trainImg = []

trainLabel = [] # Массивы с данными для обучения

j = 1

num = len(files)



# Вычитывание данных из файлов

for img in files:

    print(str(j) + "/" + str(num), end="\r")

    trainImg.append(cv2.resize(cv2.imread(img), (scale, scale)))  # Получение изображения

    trainLabel.append(img.split('/')[-2])  # Получение названия растения

    j += 1



trainImg = np.asarray(trainImg)  # Набор изображений для обучения

trainLabel = pd.DataFrame(trainLabel)  # Набор названий для обучения
# Вывод примеров изображений

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(trainImg[i])
clearTrainImg = []

examples = []; getEx = True

for img in trainImg:

    # Размытие по Гауссу

    blurImg = cv2.GaussianBlur(img, (5, 5), 0)   

    

    # Конвертируем в HSV

    hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)  

    

    # Создаем маску для зеленого цвета

    lower_green = (25, 40, 50)

    upper_green = (75, 255, 255)

    mask = cv2.inRange(hsvImg, lower_green, upper_green)  

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    

    bMask = mask > 0  

    

    # Применяем маску

    clear = np.zeros_like(img, np.uint8)  # Создаем пустое изображение

    clear[bMask] = img[bMask]  # Применяем булеву маску к изображению

    

    clearTrainImg.append(clear)  # Результат - это то, что осталось после маски

    

    # Примеры

    if getEx:

        plt.subplot(2, 3, 1); plt.imshow(img)  # Оригинальное изображение

        plt.subplot(2, 3, 2); plt.imshow(blurImg)  # Размытое изображение

        plt.subplot(2, 3, 3); plt.imshow(hsvImg)  # HSV изображение

        plt.subplot(2, 3, 4); plt.imshow(mask)  # Маска

        plt.subplot(2, 3, 5); plt.imshow(bMask)  # Булева маска

        plt.subplot(2, 3, 6); plt.imshow(clear)  # Изображение без фона

        getEx = False



clearTrainImg = np.asarray(clearTrainImg) 

    
# Вывод примеров изображений

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(clearTrainImg[i])
clearTrainImg = clearTrainImg / 255 # Нормализация
# Обозначим классы как массив нулей и единиц

from sklearn import preprocessing

from keras.utils import np_utils

import matplotlib.pyplot as plt



labels = preprocessing.LabelEncoder()

labels.fit(trainLabel[0])

print('Classes'+str(labels.classes_))

encodeTrainLabels  = labels.transform(trainLabel[0])

clearTrainLabel = np_utils.to_categorical(encodeTrainLabels)

classes = clearTrainLabel.shape[1]

print(str(classes))

trainLabel[0].value_counts().plot(kind='pie') # Круговая диаграмма
# Разобьем выборку для обучения

from sklearn.model_selection import train_test_split



trainX, testX, trainY, testY = train_test_split(clearTrainImg, clearTrainLabel, 

                                                test_size=0.1, random_state=seed, 

                                                stratify = clearTrainLabel)


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

        rotation_range=180,  # randomly rotate images in the range

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally

        height_shift_range=0.1,  # randomly shift images vertically 

        horizontal_flip=True,  # randomly flip images horizontally

        vertical_flip=True  # randomly flip images vertically

    )  

datagen.fit(trainX)
# Будем использовать Keras Sequential

# Модель состоит из 6 слоев свертки с 64, 128 и 256 фильтрами и 3 полносвязных слоев

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers import BatchNormalization



np.random.seed(seed)



model = Sequential()



model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(scale, scale, 3), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



model.add(Flatten())



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(classes, activation='softmax'))



model.summary()



# Компиляция модели

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Обучим модель

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger



# Уменьшим скорость обучения

lrr = ReduceLROnPlateau(monitor='val_acc', 

                        patience=3, 

                        verbose=1, 

                        factor=0.4, 

                        min_lr=0.00001)



filepath="drive/DataScience/PlantReco/weights.best_{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoints = ModelCheckpoint(filepath, monitor='val_acc', 

                              verbose=1, save_best_only=True, mode='max')

filepath="drive/DataScience/PlantReco/weights.last_auto4.hdf5"

checkpoints_full = ModelCheckpoint(filepath, monitor='val_acc', 

                                 verbose=1, save_best_only=False, mode='max')



callbacks_list = [checkpoints, lrr, checkpoints_full]



#model.fit_generator(datagen.flow(trainX, trainY, batch_size=75), 

#                            epochs=35, validation_data=(testX, testY), 

#                            steps_per_epoch=trainX.shape[0], callbacks=callbacks_list)



# Загрузим модель

model.load_weights("../input/plantrecomodels/weights.best_17-0.96.hdf5")

dataset = np.load("../input/plantrecomodels/Data.npz")

data = dict(zip(("x_train","x_test","y_train", "y_test"), (dataset[k] for k in dataset)))

x_train = data['x_train']

x_test = data['x_test']

y_train = data['y_train']

y_test = data['y_test']



print(model.evaluate(trainX, trainY))  # Evaluate on train set

print(model.evaluate(testX, testY))  # Evaluate on test set
# Сформируем и выведем матрицу ошибок

from sklearn.metrics import confusion_matrix



y_pred = model.predict(x_test)

y_class = np.argmax(y_pred, axis = 1) 

y_check = np.argmax(y_test, axis = 1) 



cmatrix = confusion_matrix(y_check, y_class)

print(cmatrix)
# Загрузим тестовые изображения

path_to_test = '../input/plant-seedlings-classification/test/*.png'

pics = glob(path_to_test)



testimages = []

tests = []

count=1

num = len(pics)



for i in pics:

    print(str(count)+'/'+str(num),end='\r')

    tests.append(i.split('/')[-1])

    testimages.append(cv2.resize(cv2.imread(i),(scale,scale)))

    count = count + 1



testimages = np.asarray(testimages)
# Преобразуем тестовые изображения

newtestimages = []

sets = []

getEx = True

for i in testimages:

    blurr = cv2.GaussianBlur(i,(5,5),0)

    hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)

    

    lower = (25,40,50)

    upper = (75,255,255)

    mask = cv2.inRange(hsv,lower,upper)

    struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,struc)

    boolean = mask>0

    masking = np.zeros_like(i,np.uint8)

    masking[boolean] = i[boolean]

    newtestimages.append(masking)

    

    if getEx:

        plt.subplot(2,3,1);plt.imshow(i)

        plt.subplot(2,3,2);plt.imshow(blurr)

        plt.subplot(2,3,3);plt.imshow(hsv)

        plt.subplot(2,3,4);plt.imshow(mask)

        plt.subplot(2,3,5);plt.imshow(boolean)

        plt.subplot(2,3,6);plt.imshow(masking)

        plt.show()

        getEx=False



newtestimages = np.asarray(newtestimages)

# OTHER MASKED IMAGES

for i in range(6):

    plt.subplot(2,3,i+1)

    plt.imshow(newtestimages[i])
# Применим нашу модель для предсказания

newtestimages=newtestimages/255

prediction = model.predict(newtestimages)

# Сохраним результат в csv файл

pred = np.argmax(prediction,axis=1)

predStr = labels.classes_[pred]

result = {'file':tests,'species':predStr}

result = pd.DataFrame(result)

result.to_csv("Prediction.csv",index=False)