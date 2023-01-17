import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

from tensorflow.keras.preprocessing import image

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator


%matplotlib inline
train_dir = '/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/'
test_dir = '/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/'
path = train_dir + os.listdir(train_dir)[10]
img = mpimg.imread(path)
print(img.shape)
plt.imshow(img, cmap='gray')
path = train_dir + os.listdir(train_dir)[0]
img = mpimg.imread(path)
resized_img = cv2.resize(img, (30,30))
print(resized_img.shape)
for image in range(len(os.listdir(train_dir))):
    path = train_dir+os.listdir(train_dir)[image]
    img = mpimg.imread(path)
    resized_image = cv2.resize(img, (30,30))
    if 'NORMAL' in path:
        cv2.imwrite('Normal {}.jpeg'.format(image), resized_image)
    else:
        cv2.imwrite('Covid {}.jpeg'.format(image), resized_image)
os.mkdir('train')
os.mkdir('Normal')
os.mkdir('Covid')
os.getcwd()
d = os.listdir(os.getcwd())
for i in d:
    if os.path.isdir(i):
        print(i)
for file in os.listdir(os.getcwd()):
    if file.split(' ')[0] == 'Normal':
        dest = shutil.move(file, 'Normal')
    else:
        dest = shutil.move(file, 'Covid')
os.listdir(os.getcwd())
os.mkdir('train')
os.listdir(os.getcwd())
shutil.move('/kaggle/working/Normal','/kaggle/working/train')
shutil.move('/kaggle/working/Covid', '/kaggle/working/train')
os.chdir('/kaggle/working')
os.mkdir('test')
os.listdir(os.getcwd())
os.listdir('/kaggle/working/train/')
os.chdir('/kaggle/working/test')
os.mkdir('Covid')
os.mkdir('Normal')
for image in range(len(os.listdir(test_dir))):
    path = test_dir+os.listdir(test_dir)[image]
    img = mpimg.imread(path)
    resized_image = cv2.resize(image, (30,30))
    if 'NORMAL' in path:
        cv2.imwrite('Normal {}.jpeg'.format(image), resized_image)
    else:
        cv2.imwrite('Covid {}.jpeg'.format(image), resized_image)
for file in os.listdir(os.getcwd()):
    if file.split(' ')[0] == 'Normal':
        dest = shutil.move(file, 'Normal')
    else:
        dest = shutil.move(file, 'Covid')
os.listdir(os.getcwd())
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_data = '/kaggle/working/train'
test_data = '/kaggle/working/test'
train_dir_normal = '/kaggle/working/train/Normal/'
train_dir_covid = '/kaggle/working/train/Covid/'
test_dir_normal = '/kaggle/working/test/Normal'
test_dir_covid = '/kaggle/working/test/Covid'
training_data = train_datagen.flow_from_directory(directory=train_data,
                                                 target_size=(30,30),
                                                 batch_size=32,
                                                 class_mode='binary')
testing_data = test_datagen.flow_from_directory(directory=test_data,
                                               target_size = (30, 30),
                                               batch_size = 32,
                                               class_mode = 'binary')
print('Shape of Input image: {}'.format(training_data.image_shape))
print('Number of classes: {}'.format(len(set(training_data.classes))))
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = training_data.image_shape))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.3))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.15))

model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(rate = 0.15))

model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(rate = 0.1))

model.add(Dense(units = 32, activation = 'relu'))
model.add(Dropout(rate = 0.1))

model.add(Dense(units = len(set(training_data.classes)), activation = 'softmax'))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
fitted_model = model.fit_generator(training_data,
                   steps_per_epoch=150,
                   epochs=100,
                   validation_data=testing_data,
                   validation_steps=150)
accuracy = fitted_model.history['accuracy']
plt.plot(range(len(accuracy)), accuracy, 'bo', label = 'accuracy')
plt.legend()
model.save('corona.h5')
'''
test_image_path = '/kaggle/working/test/Covid/Covid 139.jpeg'
test_image = np.expand_dims(image.img_to_array(image.load_img(test_image_path, target_size=(30,30))), axis=0)
result = model.predict(x=test_image)

if result[0][0] == 1:
    prediction = 'Covid'
else:
    prediction = 'Normal'
print(prediction)
'''
os.chdir('/kaggle/working')
os.getcwd()
model.save('corona.h5')
os.listdir(os.getcwd())