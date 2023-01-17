import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')

from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical

import cv2 as cv

from keras.layers import Dense, Conv2D, BatchNormalization, Activation

from keras.layers import AveragePooling2D, Input, Flatten

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from keras.regularizers import l2

from keras import backend as K

from keras.models import Model

import cv2 

from tqdm import tqdm



import os
traindf=pd.read_csv("/kaggle/input/hackerearth-deep-learning-challenge-dance-forms/train.csv")

testdf=pd.read_csv("/kaggle/input/hackerearth-deep-learning-challenge-dance-forms/test.csv")
traindf.head()
testdf.head()
# to check any null values in the dataset

for col in traindf.columns:

    if traindf[col].isnull().values.any():

        print(f"Train Dataset Feature - {col} contains {traindf[col].isna().sum()*100/traindf[col].sum()}% of Null Values")

    

for col in testdf.columns:

    if testdf[col].isnull().values.any():

        print(f"Test Dataset Feature - {col} contains {testdf[col].isna().sum()*100/testdf[col].sum()}% of Null Values")
print(traindf["target"].value_counts())



sns.set_style('ticks')

fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(15, 4)

sns.countplot(traindf["target"],color='black')
train_path = '/kaggle/input/hackerearth-deep-learning-challenge-dance-forms/train/'



path = f'{train_path}'    

fig = plt.figure(figsize = (13, 8))

image = cv.imread(path + f'/362.jpg')

plt.imshow(image[:, :, ::-1])

plt.title("362.jpg")

plt.axis('off')

plt.show()
class_names =np.unique(traindf['target'])

class_names_label = {class_name:i for i, class_name in enumerate(class_names)}



nb_classes = len(class_names)

print(class_names)
IMAGE_SIZE = (150, 150)

dataset = train_path

output = []

train_images = []

train_labels = []

for files in tqdm(os.listdir(dataset)):

    try:

        label=class_names_label[traindf.loc[traindf['Image'] == files]['target'].values[0]]

    except:

        #do nothing

        a=1

    img_path=os.path.join(dataset, files)

    # Open and resize the img

    image = cv2.imread(img_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, IMAGE_SIZE) 

    # Append the image and its corresponding label to the output

    train_images.append(image)

    train_labels.append(label)
IMAGE_SIZE = (150, 150)

dataset =  '/kaggle/input/hackerearth-deep-learning-challenge-dance-forms/test/'



output = []

test_images = []

for files in tqdm(os.listdir(dataset)):

    img_path=os.path.join(dataset, files)

    # Open and resize the img

    image = cv2.imread(img_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, IMAGE_SIZE) 

    # Append the image and its corresponding label to the output

    test_images.append(image)
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap=plt.cm.binary)

    plt.xlabel(class_names[train_labels[i]])

plt.show()
train_images = np.array(train_images, dtype = 'float32')

test_images = np.array(test_images, dtype = 'float32')

train_labels = np.array(train_labels, dtype = 'int32')   
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val=train_test_split(train_images,train_labels,test_size=0.3)
# Normalize data.

x_train = x_train.astype('float32') / 255

x_test = test_images.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, len(class_names))

y_val = keras.utils.to_categorical(y_val, len(class_names))
print('x_train shape:', x_train.shape)

print('y_train shape:', y_train.shape)

print('x_val shape:', x_val.shape)

print('y_val shape:', y_val.shape)
datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)
datagen.fit(x_train)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Flatten, Dense,MaxPooling2D,Dropout

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

import tensorflow as tf



image_size = x_train.shape[1]

inputshape = (image_size, image_size, 3)

batchsize = 128

kernel_size = 3

pool_size = 3

filters = 64

dropout = 0.2

epochs=50



model = Sequential([

        Conv2D(filters=filters, input_shape=inputshape, kernel_size=kernel_size,activation='relu', name='conv_1'),

        MaxPooling2D(pool_size=(4, 4), name='pool_1'),

        Dropout(dropout),

        Conv2D(filters=64, kernel_size=kernel_size, activation='relu', name='conv_2'),

        MaxPooling2D(pool_size=(4, 4), name='pool_2'),

        Dropout(dropout),

        Flatten(name='flatten'),

        Dense(units=256, activation='relu', name='dense_1'),

        Dense(units=8, activation='softmax', name='dense_2')

    ])
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy','mae'])

early_stopping_callback = EarlyStopping(monitor='val_loss')

model_name="Conv2D_basic"

checkpoint_callback = ModelCheckpoint(model_name+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history=model.fit(datagen.flow(x_train, y_train, batch_size=batchsize), epochs=epochs, callbacks=[early_stopping_callback, checkpoint_callback],validation_data=(x_val, y_val))
df=pd.DataFrame(history.history)

df.head()
loss_plot=df.plot(y="loss",title="Loss vs Epoch")

loss_plot.set(xlabel="Epoch",ylabel="Loss")
loss,accuracy,mae=model.evaluate(x_val,y_val)

print(loss)

print(accuracy*100)

print(mae)
pred=model.predict(x_test)
prediction=np.argmax(pred, axis = 1)
prediction
data = pd.DataFrame(prediction,columns=["values"])


dow = {

    0:"bharatanatyam",

    1:"kathak",

    2:"kathakali",

    3:"kuchipudi", 

    4:"manipuri", 

    5:"mohiniyattam", 

    6:"odissi",

    7:"sattriya"

}



data["dow"] = data['values'].map(dow)
ImageName = np.array(testdf["Image"])

prediction = np.array(data["dow"])

submission_dataset = pd.DataFrame({'Image': ImageName, 'target': prediction}, columns=['Image', 'target'])

submission_dataset.to_csv('submission_Initial.csv', header=True, index=False) 