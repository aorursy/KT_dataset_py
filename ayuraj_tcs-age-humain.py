import numpy as np 

import pandas as pd

import os

import cv2

import random

import matplotlib.pyplot as plt



%matplotlib inline



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



from sklearn.preprocessing import MinMaxScaler

from sklearn.utils import shuffle



import keras.backend as K

from keras.models import Model

from keras.layers import Input

from keras.layers import Dense, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.optimizers import Adam

from keras.optimizers import SGD

from keras.applications import VGG16

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras.utils import Sequence
rootPath = "/kaggle/input/tcsage/age/"

trainPath = "/kaggle/input/tcsage/age/train/"

testPath = "/kaggle/input/tcsage/age/test/"

os.listdir(rootPath)
images = os.listdir(trainPath)

labels = pd.read_csv(rootPath+'age_dataset.csv')
labels.head()
labels['age'].hist(bins=5)
len(labels['age'].unique())
data = labels.values

data
len(data)
imageList = os.listdir(trainPath)
img = cv2.imread(trainPath+imageList[10])

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
dem = labels.loc[labels['face_id'] == imageList[1][:-5]].values

dem
dem[0][1]
imageList[1][:-4]
def prepareData(labels, trainPath):

    print("[INFO] Loading....")

    train = []

    trainLabel = []

    val = []

    valLabel = []

    count = 0

    

    sizes = []

    for image in os.listdir(trainPath):

        img = cv2.imread(trainPath+image)

        sizes.append(img.shape)

        

    H, W, C = np.mean(sizes, axis=0)

    H,W,C = int(H), int(W), int(C)

    

    images = os.listdir(trainPath)

    

    print("[INFO] Preparing Training Data")

    for id in range(180):

        if count%30==0:

            print("[INFO] {} images loaded".format(count))

        img = cv2.imread(trainPath+images[id])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img/255.0

        img = cv2.resize(img, (H,W))

        train.append(img)

        data = labels.loc[labels['face_id'] == images[id][:-5]].values

        trainLabel.append(to_categorical(data[0][1], num_classes=5))

        count+=1

        

    print("[INFO] Preparing validation Data")

    for id in range(180, len(images)):

        if count%5==0:

            print("[INFO] {} images loaded".format(count))

        img = cv2.imread(trainPath+images[id])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img/255.0

        img = cv2.resize(img, (H,W))

        val.append(img)

        data = labels.loc[labels['face_id'] == images[id][:-5]].values

        valLabel.append(to_categorical(data[0][1], num_classes=5))

        count+=1

        

    print("[INFO] Done")

    return np.array(train), np.array(trainLabel), np.array(val), np.array(valLabel)
X_train, y_train, X_val, y_val = prepareData(labels, trainPath)
print("Training Shape: ", X_train.shape)

print("Label Shape: ", y_train.shape)

print("Validation Shape: ", X_val.shape)

print("Validation Label Shape: ", y_val.shape)
K.clear_session()
# def model(input_shape):

#     inputs = Input(shape=input_shape)

#     x = Conv2D(10, kernel_size=(3,3), activation='relu', padding='same')(inputs)

#     x = Conv2D(10, kernel_size=(3,3), activation='relu', padding='same')(x)

#     x = MaxPooling2D(pool_size=(2,2))(x)

    

#     x = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(x)

#     x = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(x)

#     x = MaxPooling2D(pool_size=(2,2))(x)

    

#     x = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(x)

#     x = Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid')(x)

#     x = MaxPooling2D(pool_size=(2,2))(x)

    

#     x = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(x)

#     x = Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid')(x)

#     x = MaxPooling2D(pool_size=(2,2))(x)

    

#     x = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(x)

#     x = Conv2D(256, kernel_size=(3,3), activation='relu', padding='valid')(x)

#     x = GlobalAveragePooling2D()(x)

    

#     x = Dense(1024, activation='relu')(x)

#     x = Dense(256, activation='relu')(x)

#     x = Dense(64, activation='relu')(x)

#     age = Dense(5, activation='softmax')(x)

    

#     model = Model(inputs=inputs, outputs=age)

#     return model
def new_model(input_shape):

    inputs = Input(shape=input_shape)

    

    x = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(inputs)

    x = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(x)

    x = Dropout(0.25)(x)

    x = MaxPooling2D(pool_size=(2,2))(x)

    

    x = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(x)

    x = Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid')(x)

    x = Dropout(0.25)(x)

    x = MaxPooling2D(pool_size=(2,2))(x)

    

    x = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(x)

    x = Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid')(x)

    x = MaxPooling2D(pool_size=(2,2))(x)

    

    x = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(x)

    x = Conv2D(256, kernel_size=(3,3), activation='relu', padding='valid')(x)

    x = MaxPooling2D(pool_size=(2,2))(x)

        

    x = Flatten()(x)

    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)

    x = Dense(256, activation='relu')(x)

    x = Dense(64, activation='relu')(x)

    age = Dense(5, activation='softmax')(x)

    

    model = Model(inputs=inputs, outputs=age)

    return model
model = new_model(X_train.shape[1:])
model.summary()
# class DataGenerator(Sequence):

#     def __init__(self, face_ids, labels, imagePath, num_classes, batch_size = 16, shuffle=True, resize=False, resized_dim=(None,None)):

#         self.face_ids = face_ids

#         self.face_ages = labels

#         self.imagePath = imagePath

#         self.num_classes = num_classes

#         self.batch_size = batch_size

#         self.shuffle = shuffle

#         self.resize = resize

#         self.resized_dim = resized_dim

#         self.on_epoch_end()

        

#     def on_epoch_end(self):

#         ## update indexes after epoch end

#         self.indexes = np.arange(len(self.face_ids))

#         if self.shuffle is True:

#             np.random.shuffle(self.indexes)

        

# #         print("[INFO] on_epoch_end: ", self.indexes)

            

#     def __len__(self):

#         ## Number of data in batch per epoch

#         length = int(np.floor(len(self.face_ids)/self.batch_size))

# #         print("[INFO]__len__: ", length)

#         return length

    

#     def __getitem__(self, index):

#         ## Return X,y for each batch index

#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

# #         print("[INFO]__getitem__: ", indexes)

#         Data_temp = [self.face_ids[i] for i in indexes]

#         X, y = self._DataGeneration(Data_temp)

# #         print("X_train: ", X.shape)

# #         print("y_train: ", y.shape)

#         return (X,y)

    

#     def _DataGeneration(self, Data_temp):

#         X = []

#         y = []

# #         print("[INFO]_DataGeneration: ", Data_temp)

        

#         sizes = []

        

#         for image in Data_temp:

#             img = cv2.imread(self.imagePath+image+'.jpeg')

#             sizes.append(img.shape)

        

# #         print("[INFO]", sizes)

#         H, W, C = np.mean(sizes, axis=0)

#         H,W,C = int(H), int(W), int(C)

        

#         for image in Data_temp:

#             img = cv2.imread(self.imagePath+image+'.jpeg')

#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#             img = cv2.resize(img, (H,W))

#             img = img/255.0

#             X.append(img)

#             data = labels.loc[labels['face_id'] == image].values

#             y.append(to_categorical(data[0][1], num_classes=5))

            

#         return np.array(X),np.array(y)

# training_generator = DataGenerator(labels['face_id'].values, labels, trainPath, num_classes=5, shuffle=True)
batch_size_train = 16

batch_size_val = 4

epochs = 25

train_steps = 10000

val_steps = 100
gen = ImageDataGenerator(horizontal_flip=True, rotation_range=15, zoom_range=0.15, fill_mode='nearest')
trainGen = gen.flow(X_train, y_train, batch_size=batch_size_train)

valGen = gen.flow(X_val, y_val, batch_size=batch_size_val)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit_generator(trainGen, validation_data=valGen, epochs=epochs, 

                           steps_per_epoch=train_steps//batch_size_train,

                          validation_steps=val_steps//batch_size_val)
plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.grid(True)

plt.show()

# summarize history for loss

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.grid(True)

plt.show()
model.save('tcsage2.h5')