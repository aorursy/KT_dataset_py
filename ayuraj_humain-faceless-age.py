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

from keras.layers import Dense, Flatten, BatchNormalization, Dropout

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.optimizers import Adam

from keras.optimizers import SGD

from keras.applications import VGG16

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint
imagesPath = '/kaggle/input/utkface-images/utkfaceimages/UTKFaceImages/'

labelsPath = '/kaggle/input/utkface-images/'
files = os.listdir(labelsPath)

labels = pd.read_csv(labelsPath+files[0])
ages = labels['label'].unique()

plt.figure(figsize=(13,8))

labels['label'].hist(bins=len(ages));
plt.figure(figsize=(13,8))

labels['label'].hist(bins=[0, 5, 18, 24, 26, 27, 30, 34, 38, 46, 55, 65, len(ages)]);
labels = shuffle(labels)

labels = shuffle(labels)
images = os.listdir(imagesPath)
def groupAge(age):

#     [0, 5, 18, 24, 26, 27, 30, 34, 38, 46, 55, 65, len(ages)])

    if age>=0 and age<5:

        return 0

    elif age>=5 and age<18:

        return 1

    elif age>=18 and age<24:

        return 2

    elif age>=24 and age<26:

        return 3

    elif age>=26 and age<27:

        return 4

    elif age>=27 and age<30:

        return 5

    elif age>=30 and age<34:

        return 6

    elif age>=34 and age<38:

        return 7

    elif age>=38 and age<46:

        return 8

    elif age>=46 and age<55:

        return 9

    elif age>=55 and age<65:

        return 10

    else:

        return 11
data = labels.loc[labels['image_id'] == images[100][:-4]].values

data
# train:validation:test = 60:10:30 = 14225:948:8532

def train_val_test(labels):

    partitions = {'train': [],

                 'validation': [],

                 'test': []}

    labels_dict = {'train': [],

                 'validation': [],

                 'test': []}



    discarded_data = []



    random.seed(1)

    random.shuffle(images)



    print("[INFO] Preparing train data....")

    for ID in range(14225):

        try:

            data = labels.loc[labels['image_id'] == images[ID][:-4]].values

            labels_dict['train'].append(to_categorical(groupAge(data[0][1]), num_classes=12))

            partitions['train'].append(images[ID])

        except IndexError:

            print("[ERROR]", images[ID])

            discarded_data.append(images[ID])

    print("[INFO] Done")



    print("[INFO] Preparing validation data....")

    for ID in range(14225, 15173):

        try:

            data = labels.loc[labels['image_id'] == images[ID][:-4]].values

            labels_dict['validation'].append(to_categorical(groupAge(data[0][1]), num_classes=12))

            partitions['validation'].append(images[ID])

        except IndexError:

            print("[ERROR]", images[ID])

            discarded_data.append(images[ID])

    print("[INFO] Done")



    print("[INFO] Preparing test data....")

    for ID in range(15173, len(images)):

        try:

            data = labels.loc[labels['image_id'] == images[ID][:-4]].values

            labels_dict['test'].append(to_categorical(groupAge(data[0][1]), num_classes=12))

            partitions['test'].append(images[ID])

        except IndexError:

            print("[ERROR]", images[ID])

            discarded_data.append(images[ID])

    print("[INFO] Done")

    

    return partitions, labels_dict, discarded_data
partitions, labels_dict, discarded_data = train_val_test(labels)
def buildModel():

    inputs = Input(shape=(200,200,3))

    vgg16 = VGG16(weights='imagenet', include_top=False)(inputs)

    x = Flatten()(vgg16)

    x = BatchNormalization()(x)

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(512, activation='relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(12, activation='softmax', name='age')(x)



    model = Model(inputs=inputs, outputs=x)

    

    return model
model = buildModel()
model.summary()
def loadImages(images, imagesPath, discared_data):

    print("[INFO] Loading....")

    X = []

    count = 0

    for image in images:

        if image in discared_data:

            continue

        if count%1000==0:

            print("[INFO] {} images loaded".format(count))

        img = cv2.imread(imagesPath+image)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        X.append(img)

        count+=1

    print("[INFO] Done")

    return np.array(X)
print("[INFO] Training Data")

trainX = loadImages(partitions['train'], imagesPath, discarded_data)

print("[INFO] Validation Data")

validationX = loadImages(partitions['validation'], imagesPath, discarded_data)
print("[INFO] no. of Training Images: ", len(trainX))

print("[INFO] no. of Validation Images: ", len(validationX))
trainY = np.array(labels_dict['train'])

validationY = np.array(labels_dict['validation'])
epochs = 20

lr = 1e-3

batch_size = 16
datagen = ImageDataGenerator(rescale=1.0/255.0)
traingenerator = datagen.flow(trainX, trainY)
validationgenerator = datagen.flow(validationX, validationY)
earlyStopper = EarlyStopping(monitor='loss', patience=5)
checkpoint = ModelCheckpoint('{val_loss:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit_generator(traingenerator, validation_data=validationgenerator, epochs=epochs, 

                           steps_per_epoch=len(trainX)//batch_size, validation_steps=len(validationX)//batch_size, 

                           callbacks=[checkpoint, earlyStopper])
model.save_weights('age.hdf5')
model_yaml = model.to_yaml()

with open('model-age.yaml', 'w') as yaml_file:

    yaml_file.write(model_yaml)
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