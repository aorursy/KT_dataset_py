# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import keras
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import cv2
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_cat,list_dog, mode='fit',
                 cat_path='train',dog_path='train',
                 batch_size=32, dim=(256, 256), n_channels=3,
                 n_classes=1, random_state=2020, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.mode = mode
        self.cat_path = cat_path
        self.dog_path = dog_path
        self.list_cat = list_cat
        self.list_dog = list_dog
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state
        
        self.list_IDs=list_cat+list_dog
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        
        if self.mode == 'fit' or self.mode == 'val':
            X,y = self.__generate_Xy(list_IDs_batch)
            return X, y
        
        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)
    
    def __generate_Xy(self, list_IDs_batch):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 1))
        for i, ID in enumerate(list_IDs_batch):
            label=0    
            im_name = ID
            if im_name[0:3]=="cat":   #If it is a cat image  
                self.base_path=self.cat_path
                label=0
            elif im_name[0:3]=="dog":    #If it is a dog image
                self.base_path=self.dog_path
                label=1
            img_path = f"{self.base_path}/{im_name}"
            img = self.__load_rgb(img_path)
               
            if self.mode=='fit':
                img= self.vertical_flip(img)
                img= self.horizontal_flip(img)
                if np.random.rand() < 0.2:     
                    img= self.image_translation(img)
                if np.random.rand() < 0.2:
                    img= self.image_rotation(img)
                if np.random.rand() < 0.2:
                    img= self.image_contrast(img)
            img = img.astype(np.float32) / 255.
            X[i,] = img
            y[i,] = label
        return X.astype(np.float32),y.astype(np.float32)
    ###################################################3
    def vertical_flip(self,image, rate=0.5):
        if np.random.rand() < rate:
            image = image[::-1, :, :]
        return image


    def horizontal_flip(self,image, rate=0.5):
        if np.random.rand() < rate:
            image = image[:, ::-1, :]
        return image

    def image_contrast(self,img):
        params = np.random.randint(6, 17)*0.1
        alpha = params
        new_img = cv2.multiply(img, np.array([alpha]))                    # mul_img = img*alpha
        return new_img

    def image_rotation(self,img):
        params = np.random.randint(-20, 20)
        rows, cols, ch = img.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), params, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst
      
      
    def image_translation(self,img):
        params = np.random.randint(-50, 51)
        if not isinstance(params, list):
            params = [params, params]
        rows, cols, ch = img.shape

        M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    #######################################
    
    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img=cv2.resize(img, (self.dim[1],self.dim[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
train_cats="/kaggle/input/cat-and-dog/training_set/training_set/cats/"
train_dogs="/kaggle/input/cat-and-dog/training_set/training_set/dogs/"
test_cats="/kaggle/input/cat-and-dog/test_set/test_set/cats/"
test_dogs="/kaggle/input/cat-and-dog/test_set/test_set/dogs/"
cat_files=os.listdir(train_cats)
dog_files=os.listdir(train_dogs)

val_cat_files=os.listdir(test_cats)
val_dog_files=os.listdir(test_dogs)
# 900 train images
# 100 validation images

Batch_size=32
#train generator
train_gen=DataGenerator(cat_files[0:450],dog_files[0:450],cat_path=train_cats,dog_path=train_dogs,batch_size=Batch_size)  # Half dog Images and Cat Images

#validation data generator
val_gen=DataGenerator(val_cat_files[0:50],val_dog_files[0:50],cat_path=test_cats,dog_path=test_dogs,mode='val',batch_size=Batch_size)
resnet50=keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(256,256,3)
)

model = Sequential()
model.add(resnet50)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(1, activation='sigmoid'))
    
model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.0001),
        metrics=['accuracy']
    )
model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
#train generator sample
for i in range(5):
    plt.figure()
    plt.imshow(train_gen[0][0][i])
    plt.title("Label= "+str(train_gen[0][1][i]))
history = model.fit_generator(train_gen,
                              epochs = 30, validation_data = val_gen,
                              verbose = 1,callbacks=[learning_rate_reduction])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
