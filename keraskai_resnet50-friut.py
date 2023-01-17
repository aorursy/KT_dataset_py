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
import os



image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")





def list_images(basePath, contains=None):

    # return the set of files that are valid

    return list_files(basePath, validExts=image_types, contains=contains)





def list_files(basePath, validExts=None, contains=None):

    # loop over the directory structure

    for (rootDir, dirNames, filenames) in os.walk(basePath):

        # loop over the filenames in the current directory

        for filename in filenames:

            # if the contains string is not none and the filename does not contain

            # the supplied string, then ignore the file

            if contains is not None and filename.find(contains) == -1:

                continue



            # determine the file extension of the current file

            ext = filename[filename.rfind("."):].lower()



            # check to see if the file is an image and should be processed

            if validExts is None or ext.endswith(validExts):

                # construct the path to the image and yield it

                imagePath = os.path.join(rootDir, filename)

                yield imagePath



import random





imagePaths = sorted(list(list_images("/kaggle/input/data19c3/data19c")))  #list 是将元祖转换成列表  sorted进行排序

random.seed(42) #随机种子进行排序

random.shuffle(imagePaths) 
import cv2





labels = []

data = []

for j,imagePath in enumerate(imagePaths): #获取图像

    image = cv2.imread(imagePath)



    image = cv2.resize(image, (224, 224))#设置图像的大小

    # cv2.imshow("ssdf",image)

    # cv2.waitKey(50)

    data.append(image) #extend 和 append 的区别  这里是每一个都加

    label = imagePath.split(os.path.sep)[-2]  #os.path.sep路径分割符号  #slit以分割符号进行分割 获取倒数第二个参数

    labels.append(label) #这里是获取列表

    

    aa = set(labels)

    num_classes  = len(aa)
data = np.array(data, dtype="float") / 255.0

labels = np.array(labels)
from sklearn.model_selection import train_test_split



(X_train, X_test, Y_train,Y_test) = train_test_split(data,

	labels, test_size=0.25, random_state=42)
batch_size = 2

nb_classes = 13

nb_epoch = 500
from keras.utils import np_utils















Y_train = np_utils.to_categorical(Y_train, nb_classes)

Y_test = np_utils.to_categorical(Y_test, nb_classes)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



# subtract mean and normalize

mean_image = np.mean(X_train, axis=0)

X_train -= mean_image

X_test -= mean_image

X_train /= 128.

X_test /= 128.

from keras.preprocessing.image import ImageDataGenerator





datagen = ImageDataGenerator(

     featurewise_center=True,

     featurewise_std_normalization=True,

     rotation_range=20,

     width_shift_range=0.2,

     height_shift_range=0.2,

     horizontal_flip=True

        )
from tensorflow.python.keras.applications import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization

from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

resnet_weights_path = '/kaggle/input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model = Sequential()



model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

model.add(Flatten())

model.add(BatchNormalization())

model.add(Dense(1024, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(13, activation='softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator 

    

datagen = ImageDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    zca_whitening=False,  # apply ZCA whitening

    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)

    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

    horizontal_flip=True,  # randomly flip images

    vertical_flip=False)  # randomly flip images

#from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping





#lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)





H = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),

                        steps_per_epoch=X_train.shape[0] // batch_size,

                        validation_data=(X_test, Y_test),

                        epochs=nb_epoch, verbose=1)
import matplotlib.pyplot as plt





N = np.arange(0,nb_epoch)

plt.style.use("ggplot")

plt.figure()

plt.plot(N, H.history["loss"], label="train_loss")

plt.plot(N, H.history["val_loss"], label="val_loss")

plt.plot(N, H.history["acc"], label="train_acc")

plt.plot(N, H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy (SmallVGGNet)")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()
