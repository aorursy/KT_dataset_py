



import numpy as np 

import pandas as pd 

import time, gc

import cv2

import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model,Sequential,Input

from keras.models import clone_model

from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout,BatchNormalization,Activation,GlobalAveragePooling2D, PReLU

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

from matplotlib import pyplot as plt

import seaborn as sns

from keras.applications import DenseNet121

import tensorflow as tf

from keras.layers import GaussianNoise



from keras.applications.densenet import DenseNet201
train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')

x = train['image_id']



img_size = 100
train_image=[]

for name in train['image_id']:

    path = '/kaggle/input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'

    img = cv2.imread(path)

    img = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_AREA)

    train_image.append(img)

   
fig, ax = plt.subplots(1,5, figsize = (15,15))



for i in range (5):

    ax[i].set_axis_off()

    ax[i].imshow(train_image[i])
test_image=[]

for name in test['image_id']:

    path = '/kaggle/input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'

    img = cv2.imread(path)

    img = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_AREA)

    test_image.append(img)

   
fig, ax = plt.subplots(1,5, figsize = (15,15))



for i in range (5):

    ax[i].set_axis_off()

    ax[i].imshow(test_image[i])
X_Train = np.ndarray(shape=(len(train_image), img_size, img_size, 3), dtype = np.float32)

i=0

for image in train_image:

    X_Train[i] = train_image[i]

    i = i+1

X_Train =  X_Train/255



print('Train Shape: {}'.format(X_Train.shape))
X_Test = np.ndarray(shape=(len(test_image), img_size, img_size, 3), dtype = np.float32)

i=0

for image in test_image:

    X_Test[i] = test_image[i]

    i = i+1

X_Test =  X_Test/255



print('Test Shape: {}'.format(X_Test.shape))
y =  train.copy()

del y['image_id']

y.head()
y_train = np.array(y.values)

print(y_train.shape, y_train[0])
X_train, X_val, Y_train, Y_val = train_test_split(X_Train, y_train, test_size = 0.2, random_state = 2)
def build_model():

    densenet = DenseNet121(weights='imagenet', include_top=False)



    input = Input(shape=(img_size, img_size, 3))

    lay = Conv2D(32, (3, 3), padding='same')(input)

    lay = Conv2D(6, (3, 3), padding='same')(input)

    

    lay = densenet(lay)

    

    lay = GlobalAveragePooling2D()(lay)

    lay = BatchNormalization()(lay)

    lay = Dropout(0.5)(lay)

    lay = Dense(256, activation='relu')(lay)

    lay = Dense(512, activation='relu')(lay)

    lay = Dense(256, activation='relu')(lay)

    lay = Dropout(0.4)(lay)

    

    lay = GaussianNoise(0.1)(lay)

    lay = Dense(64, activation='relu')(lay)

    lay = BatchNormalization()(lay)

    lay = Dropout(0.3)(lay)



    # multi output

    output = Dense(4,activation = 'softmax', name='root')(lay)

 



    # model

    model = Model(input,output)

    

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.01)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    

    return model
model = build_model()

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=4, verbose=1, min_lr=1e-4)

checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)

# Generates batches of image data with data augmentation

datagen = ImageDataGenerator(rotation_range=360, # Degree range for random rotations

                      

                        width_shift_range=0.4, # Range for random horizontal shifts

                        height_shift_range=0.4, # Range for random vertical shifts

                        zoom_range=0.4, # Range for random zoom

                        horizontal_flip=True, # Randomly flip inputs horizontally

                        vertical_flip=True) # Randomly flip inputs vertically



datagen.fit(X_train)

# Fits the model on batches with real-time data augmentation

hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),

               steps_per_epoch=X_train.shape[0] // 32,

               epochs=100,

               verbose=1,

               callbacks=[reduce_lr, checkpoint],

               validation_data=(X_val, Y_val))
predict = model.predict(X_Test)

all_predict = np.ndarray(shape = (test.shape[0],4),dtype = np.float32)

for i in range(0,test.shape[0]):

    for j in range(0,4):

        if predict[i][j]==max(predict[i]):

            all_predict[i][j] = 1

        else:

            all_predict[i][j] = 0 
healthy = [y_test[0] for y_test in all_predict]

multiple_diseases = [y_test[1] for y_test in all_predict]

rust = [y_test[2] for y_test in all_predict]

scab = [y_test[3] for y_test in all_predict]
df = {'image_id':test.image_id,'healthy':healthy,'multiple_diseases':multiple_diseases,'rust':rust,'scab':scab}
data = pd.DataFrame(df)

data.tail()
data.to_csv('submission.csv',index = False)