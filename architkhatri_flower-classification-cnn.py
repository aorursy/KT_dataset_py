# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

import cv2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print(os.listdir("../input/flowers/flowers"))
DAISY_DIR = '../input/flowers/flowers/daisy'
DANDELION_DIR = "../input/flowers/flowers/dandelion"
ROSE_DIR = "../input/flowers/flowers/rose"
SUNFLOWER_DIR = "../input/flowers/flowers/sunflower"
TULIP_DIR = "../input/flowers/flowers/tulip"
FLOWER_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
DIR = [DAISY_DIR, DANDELION_DIR, ROSE_DIR, SUNFLOWER_DIR, TULIP_DIR]
def create_training_data(flower_name, flower_dir,img_size):
    """
        flower_name = string
        flower_dir = input path
        img_size = tuple (containing size of resized image)
    """
    X = []
    y = []
    for num,i in enumerate(flower_dir):
        for j in tqdm(os.listdir(i)):
            path = os.path.join(i,j)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            try:
                img = cv2.resize(img, img_size)
            except:
                continue
                
            X.append(np.array(img))
            y.append(flower_name[num])
    
    return X,y
X,y = create_training_data(FLOWER_NAMES, DIR, (200,200))
def view_random_images(X, y, random_state = 42):
    np.random.seed(random_state)
    plt.figure(figsize = (6,12))    
    
    for i in range(10):
        plt.subplot(5,2,i+1)
        num = np.random.randint(len(X))
        plt.imshow(X[num])
        plt.title(y[num])
    
    plt.tight_layout()
view_random_images(X,y, random_state = 100)
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y,5)
X = (np.array(X))/255
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
del X
del y
np.random.seed(42)
cnn = Sequential()
cnn.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (200,200,3)))
cnn.add(MaxPooling2D(pool_size = (2,2)))
cnn.add(BatchNormalization(axis = 1))
cnn.add(Dropout(0.2))
cnn.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (200,200,3)))
cnn.add(MaxPooling2D(pool_size = (2,2)))
cnn.add(BatchNormalization(axis = 1))
cnn.add(Dropout(0.2))
cnn.add(Conv2D(filters = 64, kernel_size = (4,4), padding = 'same', activation = 'relu', input_shape = (200,200,3)))
cnn.add(MaxPooling2D(pool_size = (2,2)))
cnn.add(BatchNormalization(axis = 1))
cnn.add(Dropout(0.2))
cnn.add(Conv2D(filters = 96, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = (200,200,3)))
cnn.add(MaxPooling2D(pool_size = (2,2)))
cnn.add(BatchNormalization(axis = 1))
cnn.add(Flatten())
cnn.add(Dropout(0.15))
cnn.add(Dense(512, activation = 'relu'))
cnn.add(BatchNormalization())
cnn.add(Dense(512, activation = 'relu'))
cnn.add(BatchNormalization())
cnn.add(Dense(5, activation = 'sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
cnn.summary()
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.16, # Randomly zoom image 
        width_shift_range=0.32,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.32,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images


datagen.fit(X_train)
#train = cnn.fit_generator(datagen.flow(X_train,y_train, batch_size=100),
#                              epochs = 10, validation_data = (X_test,y_test),
#                              verbose = 1, steps_per_epoch = X_train.shape[0])

batch_size = 100
epochs = 24

train = cnn.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_test,y_test),
                              verbose = 1, steps_per_epoch=X_train.shape[0]//16)

cnn.save_weights('flower_classification_cnn_weights.h5')
plt.plot(train.history['loss'])
plt.plot(train.history['val_loss'])
plt.title('Loss Performance')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()
plt.plot(train.history['acc'])
plt.plot(train.history['val_acc'])
plt.title('Accuracy Performance')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()