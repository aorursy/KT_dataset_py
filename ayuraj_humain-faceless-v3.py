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

labels = pd.read_csv(labelsPath+files[2])
labels.head()
labels.groupby('label').count()
a = labels.groupby('label').count().values.flatten()

a
labels.hist(bins=10);
label_0 = labels.groupby('label').groups[0]

label_0 = labels.loc[label_0]

label_0 = label_0.sample(frac=0.42, random_state=99)



label_1 = labels.groupby('label').groups[1]

label_1 = labels.loc[label_1]



label_2 = labels.groupby('label').groups[2]

label_2 = labels.loc[label_2]



label_3 = labels.groupby('label').groups[3]

label_3 = labels.loc[label_3]



label_4 = labels.groupby('label').groups[4]

label_4 = labels.loc[label_4]



label_5 = labels.groupby('label').groups[5]

label_5 = labels.loc[label_5]

label_5 = label_5.sample(frac=0.48, random_state=99)



label_6 = labels.groupby('label').groups[6]

label_6 = labels.loc[label_6]



label_7 = labels.groupby('label').groups[7]

label_7 = labels.loc[label_7]



label_8 = labels.groupby('label').groups[8]

label_8 = labels.loc[label_8]



label_9 = labels.groupby('label').groups[9]

label_9 = labels.loc[label_9]



new_labels = pd.concat([label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9], ignore_index=True)



new_labels = shuffle(new_labels)
new_labels.groupby('label').count()
new_labels.hist(bins=10);
new_labels.describe()
images = os.listdir(imagesPath)
print("[INFO] Total number of images: ", len(images))

print("[INFO] Total number of labels: ", len(new_labels.values))

print("[INFO] Data Discarded: ", len(images)-len(new_labels.values))
data = labels.loc[labels['image_id'] == images[1000][:-4]].values
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

            labels_dict['train'].append(to_categorical(data[0][1], num_classes=10))

            partitions['train'].append(images[ID])

        except IndexError:

            print("[ERROR]", images[ID])

            discarded_data.append(images[ID])

    print("[INFO] Done")



    print("[INFO] Preparing validation data....")

    for ID in range(14225, 15173):

        try:

            data = labels.loc[labels['image_id'] == images[ID][:-4]].values

            labels_dict['validation'].append(to_categorical(data[0][1], num_classes=10))

            partitions['validation'].append(images[ID])

        except IndexError:

            print("[ERROR]", images[ID])

            discarded_data.append(images[ID])

    print("[INFO] Done")



    print("[INFO] Preparing test data....")

    for ID in range(15173, len(images)):

        try:

            data = labels.loc[labels['image_id'] == images[ID][:-4]].values

            labels_dict['test'].append(to_categorical(data[0][1], num_classes=10))

            partitions['test'].append(images[ID])

        except IndexError:

            print("[ERROR]", images[ID])

            discarded_data.append(images[ID])

    print("[INFO] Done")

    

    return partitions, labels_dict, discarded_data
partitions, labels_dict, discarded_data = train_val_test(new_labels)
print("[INFO] Training Data")

print("Size of train data: ", len(partitions['train']))

print("Size of age as label: ", len(labels_dict['train']))

print("\n")

print("[INFO] Validation Data")

print("Size of validation data: ", len(partitions['validation']))

print("Size of age as label: ", len(labels_dict['validation']))

print("\n")

print("[INFO] Test Data")

print("Size of test data: ", len(partitions['test']))

print("Size of age as label: ", len(labels_dict['test']))

print('\n')

print("Discarded data: ", len(discarded_data))
def buildModel():

    inputs = Input(shape=(200,200,3))

    vgg16 = VGG16(weights='imagenet', include_top=False)(inputs)

    x = Flatten()(vgg16)

    x = BatchNormalization()(x)

    x = Dense(1024, activation='relu')(x)

    x = Dense(512, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dense(128, activation='relu')(x)

    x = Dense(64, activation='relu')(x)

    x = Dense(10, activation='softmax', name='age')(x)



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
checkpoint = ModelCheckpoint('{val_loss:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='min')
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit_generator(traingenerator, validation_data=validationgenerator, epochs=epochs, 

                           steps_per_epoch=len(trainX)//batch_size, validation_steps=len(validationX)//batch_size, 

                           callbacks=[checkpoint, earlyStopper])
model.save_weights('gender-ethnicity.hdf5')
model_yaml = model.to_yaml()

with open('model-gender-ethnicity.yaml', 'w') as yaml_file:

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