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
rootPath = "/kaggle/input/tcsgenderrace/gender_ethnicity/"

trainPath = "/kaggle/input/tcsgenderrace/gender_ethnicity/train/"

os.listdir(rootPath)
images = os.listdir(trainPath)

labels = pd.read_csv(rootPath+'gender_race_dataset.csv')
labels.head()
np.sort(labels['gender_race'].unique())
labels['gender_race'].hist(bins=12)
data = labels.values

len(data)
imageList = os.listdir(trainPath)
img = cv2.imread(trainPath+imageList[101])

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img);
def prepareData(labels, trainPath, num_classes):

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

    for id in range(len(images)):

        if count%50==0:

            print("[INFO] {} images loaded".format(count))

        img = cv2.imread(trainPath+images[id])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img/255.0

        img = cv2.resize(img, (H,W))

        train.append(img)

        data = labels.loc[labels['face_id'] == images[id][:-5]].values

        trainLabel.append(to_categorical(data[0][1], num_classes=num_classes))

        count+=1

        

    print("[INFO] Done")

    return np.array(train), np.array(trainLabel)
X_train, y_train = prepareData(labels, trainPath, 12)
print("Training Shape: ", X_train.shape)

print("Label Shape: ", y_train.shape)
K.clear_session()
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

    x = Dropout(0.25)(x)

    x = MaxPooling2D(pool_size=(2,2))(x)

    

    x = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(x)

    x = Conv2D(256, kernel_size=(3,3), activation='relu', padding='valid')(x)

    x = Dropout(0.25)(x)

    x = MaxPooling2D(pool_size=(2,2))(x)

        

    x = Flatten()(x)

    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)

    x = Dense(512, activation='relu')(x)

    x = Dense(256, activation='relu')(x)

    emotion = Dense(12, activation='softmax')(x)

    

    model = Model(inputs=inputs, outputs=emotion)

    return model
model = new_model(X_train.shape[1:])
model.summary()
batch_size_train = 32

epochs = 30

train_steps = 10000
gen = ImageDataGenerator(horizontal_flip=True, rotation_range=15, zoom_range=0.15, fill_mode='nearest')
trainGen = gen.flow(X_train, y_train, batch_size=batch_size_train)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit_generator(trainGen, epochs=epochs, 

                           steps_per_epoch=train_steps//batch_size_train)
model.save('tcsgenderrace3.h5')
model_yaml = model.to_yaml()

with open("model_gender_race3.yaml", "w") as yaml_file:

    yaml_file.write(model_yaml)
plt.plot(hist.history['acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.grid(True)

plt.show()

# summarize history for loss

plt.plot(hist.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.grid(True)

plt.show()