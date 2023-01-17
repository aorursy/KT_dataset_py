import pandas as pd

import keras

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input

from keras.callbacks import ReduceLROnPlateau

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import cv2

import albumentations as A

import numpy as np

import math

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# FOr parquet data 



BASE_URL = '/kaggle/input/bengaliaicv19feather/'

TRAIN = ['train_image_data_0.parquet',

         'train_image_data_1.parquet',

         'train_image_data_2.parquet',

         'train_image_data_3.parquet']



d1 = BASE_URL+'train_image_data_1.feather'

imageSize=64
# FOr parquet data 

# # # this take lots of server space so commented till testing

# dataList = []

# for i in TRAIN:

#     df = pd.read_parquet(BASE_URL+i)

#     dataList.append(df) 
# trainData = pd.concat(dataList)
# trainData = pd.read_parquet(BASE_URL+TRAIN[0])

trainData = pd.read_feather(d1)

train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

# del dataList
def resize(df, size=64, need_progress_bar=True):

    resized = {}

    resize_size=64

    if need_progress_bar:

         for i in range(df.shape[0]):

            image=df.loc[df.index[i]].values.reshape(137,236)

            augBright=A.RandomBrightnessContrast(p=1.0)

            image = augBright(image=image)['image']

            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]



            idx = 0 

            ls_xmin = []

            ls_ymin = []

            ls_xmax = []

            ls_ymax = []

            for cnt in contours:

                idx += 1

                x,y,w,h = cv2.boundingRect(cnt)

                ls_xmin.append(x)

                ls_ymin.append(y)

                ls_xmax.append(x + w)

                ls_ymax.append(y + h)

            xmin = min(ls_xmin)

            ymin = min(ls_ymin)

            xmax = max(ls_xmax)

            ymax = max(ls_ymax)



            roi = image[ymin:ymax,xmin:xmax]

            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)

            resized[df.index[i]] = resized_roi.reshape(-1)

    else:

        for i in range(df.shape[0]):

            image=df.loc[df.index[i]].values.reshape(137,236)

            augBright=A.RandomBrightnessContrast(p=1.0)

            image = augBright(image=image)['image']

            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]



            idx = 0 

            ls_xmin = []

            ls_ymin = []

            ls_xmax = []

            ls_ymax = []

            for cnt in contours:

                idx += 1

                x,y,w,h = cv2.boundingRect(cnt)

                ls_xmin.append(x)

                ls_ymin.append(y)

                ls_xmax.append(x + w)

                ls_ymax.append(y + h)

            xmin = min(ls_xmin)

            ymin = min(ls_ymin)

            xmax = max(ls_xmax)

            ymax = max(ls_ymax)



            roi = image[ymin:ymax,xmin:xmax]

            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)

            resized[df.index[i]] = resized_roi.reshape(-1)

    resized = pd.DataFrame(resized).T

    return resized
inputs = Input(shape = (imageSize, imageSize, 1))



model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(imageSize, imageSize, 1))(inputs)

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.9)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = Dropout(rate=0.3)(model)



model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.9)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.9)(model)

model = Dropout(rate=0.3)(model)



model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.9)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.9)(model)

model = Dropout(rate=0.3)(model)



model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.9)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.9)(model)

model = Dropout(rate=0.3)(model)



model = Flatten()(model)

model = Dense(1024, activation = "relu")(model)

model = Dropout(rate=0.3)(model)

dense = Dense(512, activation = "relu")(model)



head_root = Dense(168, activation = 'softmax')(dense)

head_vowel = Dense(11, activation = 'softmax')(dense)

head_consonant = Dense(7, activation = 'softmax')(dense)
model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])

model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
lrReductionRoot = ReduceLROnPlateau(monitor='dense_3_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)

lrReductionVowel = ReduceLROnPlateau(monitor='dense_4_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)

lrReductionConsonant = ReduceLROnPlateau(monitor='dense_5_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)

learningRate = [lrReductionRoot,lrReductionVowel,lrReductionConsonant]

class MultiOutputDataGenerator(keras.preprocessing.image.ImageDataGenerator):



    def flow(self,

             x,

             y=None,

             batch_size=32,

             shuffle=True,

             sample_weight=None,

             seed=None,

             save_to_dir=None,

             save_prefix='',

             save_format='png',

             subset=None):



        targets = None

        target_lengths = {}

        ordered_outputs = []

        for output, target in y.items():

            if targets is None:

                targets = target

            else:

                targets = np.concatenate((targets, target), axis=1)

            target_lengths[output] = target.shape[1]

            ordered_outputs.append(output)





        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,

                                         shuffle=shuffle):

            target_dict = {}

            i = 0

            for output in ordered_outputs:

                target_length = target_lengths[output]

                target_dict[output] = flowy[:, i: i + target_length]

                i += target_length



            yield flowx, target_dict
trainData = pd.merge(trainData, train, on='image_id').drop(['image_id'], axis=1).drop(['grapheme'], axis=1)
yCols = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']



xTrain = trainData.drop(yCols, axis=1)

xTrain = 255 - resize(xTrain) / 255

xTrain = xTrain.values.reshape(-1, imageSize, imageSize, 1)
yTrainRoot = pd.get_dummies(trainData['grapheme_root']).values

yTrainVowel = pd.get_dummies(trainData['vowel_diacritic']).values

yTrainConsonant = pd.get_dummies(trainData['consonant_diacritic']).values

yTrain = [yTrainRoot,yTrainVowel,yTrainConsonant]

del trainData
imgGen = MultiOutputDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.15, # Randomly zoom image 

        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

imgGen.fit(xTrain)
# history = model.fit_generator(imgGen.flow(xTrain, {'dense_3': yTrainRoot, 'dense_4':yTrainVowel, 'dense_5': yTrainConsonant}),epochs=30,steps_per_epoch=xTrain.shape[0],callbacks=learningRate)
history = model.fit(xTrain,yTrain,epochs=50, batch_size=36,validation_split = 0.1,callbacks=learningRate)
predsDict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}
components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target=[]

rowId=[]



for i in range(4):

    testData = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

    testData.set_index('image_id', inplace=True)



    xTest = resize(testData, need_progress_bar=False)/255

    xTest = xTest.values.reshape(-1, imageSize, imageSize, 1)

    

    preds = model.predict(xTest)



    for i, p in enumerate(predsDict):

        predsDict[p] = np.argmax(preds[i], axis=1)



    for k,id in enumerate(testData.index.values):  

        for i,comp in enumerate(components):

            idSample=id+'_'+comp

            rowId.append(idSample)

            target.append(predsDict[comp][k])





    del testData

    del xTest



sampleData = pd.DataFrame(

    {

        'row_id': rowId,

        'target':target

    },

    columns = ['row_id','target'] 

)

sampleData.to_csv('submission.csv',index=False)

sampleData.head()