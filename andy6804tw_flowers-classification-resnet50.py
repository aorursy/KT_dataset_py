import warnings

warnings.filterwarnings('ignore')



import tensorflow.keras as keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, load_model, Model

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.applications.xception import Xception



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import PIL.Image



daisy_path = "../input/flower/flower_classification/train/daisy/"

dandelion_path = "../input/flower/flower_classification/train/dandelion/"

rose_path = "../input/flower/flower_classification/train/rose/"

sunflower_path = "../input/flower/flower_classification/train/sunflower/"

tulip_path = "../input/flower/flower_classification/train/tulip/"

test_path="../input/flower/flower_classification/test/"

submission = pd.read_csv('../input/submission.csv')
from os import listdir

import cv2







img_data = []

labels = []



size = 224,224

def iter_images(images,directory,size,label):

    try:

        for i in range(len(images)):

            img = cv2.imread(directory + images[i])

            img = cv2.resize(img,size,PIL.Image.ANTIALIAS)

            img_data.append(img)

            labels.append(label)

    except:

        pass



iter_images(listdir(daisy_path),daisy_path,size,0)

iter_images(listdir(dandelion_path),dandelion_path,size,1)

iter_images(listdir(rose_path),rose_path,size,2)

iter_images(listdir(sunflower_path),sunflower_path,size,3)

iter_images(listdir(tulip_path),tulip_path,size,4)
len(img_data),len(labels)
test_data = []



size = 224,224

def test_images(images,directory,size):

    try:

        for i in range(len(images)):

            img = cv2.imread(directory + submission['id'][i]+".jpg")

            img = cv2.resize(img,size,PIL.Image.ANTIALIAS)

            test_data.append(img)

    except:

        pass





test_images(listdir(test_path),test_path,size)
len(test_data)
train_X = np.asarray(img_data)

train_Y = np.asarray(labels)



idx = np.arange(train_X.shape[0])

np.random.shuffle(idx)



train_X = train_X[idx]

train_Y = train_Y[idx]



testData=np.asarray(test_data)



print(train_X.shape)

print(train_Y.shape)
import numpy as np

data = np.asarray(img_data)

testData=np.asarray(test_data)



#div by 255

# data = data / 255.0

# testData=testData/255.0



labels = np.asarray(labels)
dict = {0:'daisy', 1:'dandelion', 2:'rose', 3:'sunflower', 4:'tulip'}

def plot_image(number):

    fig = plt.figure(figsize = (15,8))

    plt.imshow(testData[number])

    plt.title(dict[labels[number]])

plot_image(0)

labels[0]
from sklearn.model_selection import train_test_split



# Split the data

X_train, X_validation, Y_train, Y_validation = train_test_split(data, labels, test_size=0.10, shuffle= True)
print("Length of X_train:", len(X_train), "Length of Y_train:", len(Y_train))

print("Length of X_validation:",len(X_validation), "Length of Y_validation:", len(Y_validation))
from tensorflow.python.keras.models import Model

from tensorflow.python.keras.models import load_model

from tensorflow.python.keras.layers import Flatten, Dense, Dropout, BatchNormalization

from tensorflow.python.keras.applications.resnet50 import ResNet50

from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   channel_shift_range=10,

                                   horizontal_flip=True,

                                   fill_mode='nearest')
net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,

               input_shape=(224,224, 3))

x = net.output

x = Flatten()(x)



x = Dropout(0.5)(x)

x = Dense(200, activation='relu', name='dense1')(x)

x = BatchNormalization()(x)

x = Dense(200, activation='relu', name='dense2')(x)

x = BatchNormalization()(x)

output_layer = Dense(5, activation='softmax', name='softmax')(x)



net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:2]:

    layer.trainable = False

for layer in net_final.layers[2:]:

    layer.trainable = True



net_final.compile(optimizer=Adam(lr=5e-5, decay=0.005),

                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(net_final.summary())
History=net_final.fit_generator(train_datagen.flow(train_X, train_Y, batch_size=64), 

                        steps_per_epoch=len(train_X) / 64,

                        epochs=15)
plt.plot(History.history['loss'])

plt.plot(History.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
plt.plot(History.history['acc'])

plt.plot(History.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
pred =  np.argmax(net_final.predict(testData), axis=1)

newsSbmission=submission

newsSbmission["class"]=pred

newsSbmission.to_csv("submission.csv", index=False)
pred