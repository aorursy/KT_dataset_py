# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
cnt=0;

for files in {'COVID-19','NORMAL','Viral Pneumonia'}:

    for file in os.listdir(os.path.join('../input/covid19-radiography-database/COVID-19 Radiography Database',files)):

        cnt+=1

print('Total Images: {}'.format(cnt))
import matplotlib.pyplot as plt

import cv2



normal = cv2.imread('../input/covid19-radiography-database/COVID-19 Radiography Database/NORMAL/NORMAL (387).png')



covid = cv2.imread('../input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19/COVID-19(147).png')



pneumonia = cv2.imread('../input/covid19-radiography-database/COVID-19 Radiography Database/Viral Pneumonia/Viral Pneumonia (1130).png')
plt.imshow(normal)
plt.imshow(covid)
plt.imshow(pneumonia)
os.mkdir('../working/train')

os.mkdir('../working/test')
import shutil

def copytree(src, dst, symlinks=False, ignore=None):

    if not os.path.exists(dst):

        os.makedirs(dst)

    for item in os.listdir(src):

        s = os.path.join(src, item)

        d = os.path.join(dst, item)

        if os.path.isdir(s):

            copytree(s, d, symlinks, ignore)

        else:

            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:

                shutil.copy2(s, d)

                

copytree('../input/covid19-radiography-database/COVID-19 Radiography Database/', '../working/train')
cnt=0

os.listdir('../working/train')

for file in {'COVID-19', 'NORMAL', 'Viral Pneumonia'}:

    for files in os.listdir(os.path.join('../working/train/',file)):

        cnt+=1

cnt
os.remove('../working/train/COVID-19.metadata.xlsx')

os.remove('../working/train/README.md.txt')

os.remove('../working/train/Viral Pneumonia.matadata.xlsx')

os.remove('../working/train/NORMAL.metadata.xlsx')
# moving 20 percent data to test folder

src = '../working/train'

dst = '../working/test'

os.mkdir('../working/test/COVID-19')

os.mkdir('../working/test/Viral Pneumonia')

os.mkdir('../working/test/NORMAL')



for folders in os.listdir(src):

    num_files = len(os.listdir(os.path.join(src,folders)))

    cut_length = int(num_files*0.2)

    cnt=0

    for files in os.listdir(os.path.join(src, folders)):

        shutil.move(os.path.join(src, folders, files), os.path.join(dst, folders, files))

        if(cnt==cut_length):

            break

        cnt+=1
# shutil.rmtree('../working/test/COVID-19')

# shutil.rmtree('../working/test/NORMAL')

# shutil.rmtree('../working/test/Viral Pneumonia')

# len(os.listdir(os.path.join(dst,'Viral ')))
cnt=0

os.listdir('../working/test')

for file in {'COVID-19', 'NORMAL', 'Viral Pneumonia'}:

    for files in os.listdir(os.path.join('../working/train/',file)):

        if file == 'Viral Pneumonia':

            cnt+=1

cnt
from keras.preprocessing.image import ImageDataGenerator

def get_images(folder, augment=True, augment_size=200):

    X, Y, className = [], [], []

    k=0

    

    X_covid = []

    Y_covid = []

    ## reading images from the directory

    for classes in os.listdir(folder):

        className.append(classes)

        for images in os.listdir(os.path.join(folder, classes)):

            img = cv2.imread(os.path.join(folder, classes, images))

            img = cv2.resize(img, (224,224))

            img = img.astype('float16')

            img /= 255.0

            X.append(img)

            label = np.zeros(3)

            label[k]=1

            Y.append(label)

            if classes=='COVID-19':

                X_covid.append(img)

                Y_covid.append(label)

        k+=1

        

    X = np.array(X)

    X_covid = np.array(X_covid)

    Y = np.array(Y)

    Y_covid = np.array(Y_covid)

    train_size = X_covid.shape[0]

    print(X.shape, Y.shape, X_covid.shape)

    if augment:

        print(augment)

        indexes = np.random.permutation(len(X_covid))

        X_new = X_covid[indexes]

        Y_new = Y_covid[indexes]

#         train_perc = 0.10

#         train_count = int(train_perc * len(x_train))

#         X_new = X_new[:train_count, :]

#         Y_new = Y_new[:train_count, :]

#         print(X_new.shape, Y_new.shape)

        image_generator = ImageDataGenerator(

            rotation_range=10,

            zoom_range = 0.05, 

            width_shift_range=0.05,

            height_shift_range=0.05,

            horizontal_flip=False,

            vertical_flip=False, 

            data_format="channels_last")

        image_generator.fit(X_new, augment=True)

        print('fitted!')

        randidx = np.random.randint(train_size, size=augment_size)

        x_augmented = X_covid[randidx]

        y_augmented = Y_covid[randidx]

        x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),

                                    batch_size=augment_size, shuffle=False).next()[0]

        # append augment data to trainset

        X = np.concatenate((X, x_augmented))

        Y = np.concatenate((Y, y_augmented))

        print(X.shape, Y.shape)

    return X, Y
X_train, Y_train = get_images(src, augment_size=800)
X_test, Y_test = get_images(dst, augment=False)
X_train.shape, X_test.shape, X_test.shape, Y_test.shape
# train_generator = train_datagen.flow_from_directory('../working/train', 

#                                                    target_size=(224,224),

#                                                    batch_size=32,

#                                                    class_mode='categorical')
# train_generator.shape
# test_generator = test_datagen.flow_from_directory('../working/test',

#                                                  target_size=(224,224),

#                                                  batch_size=32,

#                                                  class_mode='categorical')
import tensorflow as tf

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

from keras.callbacks import ModelCheckpoint



model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 224x224 with 3 bytes color

    # This is the first convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(224, 224, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),





    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(3, activation='softmax')

])
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# early stopping

checkpoint = ModelCheckpoint('model_covid.h5', monitor='val_loss', save_best_only=True, verbose=1)



history = model.fit(x=X_train, y=Y_train, epochs = 40, validation_data = (X_test, Y_test), callbacks=[checkpoint])
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()



plt.show()
model.save("covid-19.h5")