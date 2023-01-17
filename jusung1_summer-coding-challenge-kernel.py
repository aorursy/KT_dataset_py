# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import

from PIL import Image, ImageDraw

import json



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.models import Sequential

from tensorflow.keras.metrics import top_k_categorical_accuracy

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



import matplotlib.pyplot as plt

%matplotlib inline
# 파일 경로와 label dictionary 설정
trainFiles = os.listdir("../input/train_simplified/")

commonDir = "../input/train_simplified/"

labelDict = {i: v[:-4].replace(" ", "_") for i, v in enumerate(trainFiles)}

labelDictInv = {v[:-4]: i for i, v in enumerate(trainFiles)}
#global variables
imgSize = 64

dataPerClass = 100

numClasses = 340
# DataFrame 에 저장된 이미지 데이터를 CNN 에 넣을수 있게 64x64 이미지 포맷으로 바꿔주는 helper functions
def convertTo2dImage(strokes):

    image = Image.new("P", (256,256), color=255)

    image_draw = ImageDraw.Draw(image)

    for stroke in strokes:

        for i in range(len(stroke[0])-1):

            image_draw.line([stroke[0][i],

                            stroke[1][i],

                            stroke[0][i+1],

                            stroke[1][i+1]],

                           fill=0, width=5)

    image = image.resize((imgSize,imgSize))

    return np.array(image)/255



def dfToImageArray(data, size=imgSize):

    x = np.zeros((len(data), size, size, 1))

    for i, strokes in enumerate(data):

        x[i, :, :, 0] = convertTo2dImage(strokes)

    

    return x
# Training 파일들을 읽어서 pandas DataFrame 으로 저장하기
colNames = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']

drawList = []

for file in trainFiles:

    data = pd.read_csv(commonDir+file, nrows=dataPerClass)

    data = data[data.recognized==True]

    drawList.append(data)

draw_df = pd.DataFrame(np.concatenate(drawList), columns=colNames)

#change str to list

draw_df['drawing'] = draw_df['drawing'].apply(json.loads)
# train = pd.read_csv(commonDir+trainFiles[0], nrows=500)

# label = np.full((train.shape[0],1),1)

# np.concatenate((train, label), axis=1)
# Use train_test_split method

# feature 와 label 만들기

featData = draw_df['drawing']

labels = draw_df['word'].replace(labelDictInv)



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(featData,labels,

                                                   test_size = 0.1,

                                                   random_state = 101)



X_train = dfToImageArray(X_train)

X_test = dfToImageArray(X_test)



y_train = keras.utils.to_categorical(y_train, num_classes=numClasses)

y_test = keras.utils.to_categorical(y_test, num_classes=numClasses)
# sample simplified image

plt.imshow(X_train[0,:,:,0])
print(X_train.shape, '\n',

      X_test.shape, '\n',

      y_train.shape, '\n',

      y_test.shape, '\n')
#Keras를 사용해서 Convolutional Neural Network 모델 만들기
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(imgSize, imgSize, 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(680, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(340, activation='softmax'))

model.summary()
def top_3_accuracy(x,y): 

    t3 = top_k_categorical_accuracy(x,y, 3)

    return t3



reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 

                                   verbose=1, mode='auto', min_delta=0.005, cooldown=5, min_lr=0.0001)

earlystop = EarlyStopping(monitor='val_top_3_accuracy', mode='max', patience=5) 

callbacks = [reduceLROnPlat, earlystop]



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy', top_3_accuracy])



model.fit(x=X_train, y=y_train,

          batch_size = 32,

          epochs = 10,

          validation_data = (X_test, y_test),

          callbacks = callbacks,

          verbose = 1)
ttvlist = []

reader = pd.read_csv('../input/test_simplified.csv', index_col=['key_id'],

    chunksize=2048)

drawList = []

for data in reader:

    data['drawing'] = data['drawing'].apply(json.loads)

    test = dfToImageArray(data.drawing.values)

    testPreds = model.predict(test, verbose=0)

    ttvs = np.argsort(-testPreds)[:, 0:3] #select top 3 categories

    ttvlist.append(ttvs)



ttvarray = np.concatenate(ttvlist)
preds_df = pd.DataFrame({'first': ttvarray[:,0], 'second': ttvarray[:,1], 'third': ttvarray[:,2]})

preds_df = preds_df.replace(labelDict)

preds_df['words'] = preds_df['first'] + " " + preds_df['second'] + " " + preds_df['third']



sub = pd.read_csv('../input/sample_submission.csv', index_col=['key_id'])

sub['word'] = preds_df.words.values

sub.to_csv('cnn_submission.csv')

sub.head()