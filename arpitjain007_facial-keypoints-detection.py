import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2

from tqdm import tqdm, tnrange, tqdm_notebook

import random

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D , Dense, MaxPool2D, Flatten, Dropout, Input, BatchNormalization

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split

%matplotlib inline
train_path = "../input/facial-keypoints-detection/training/training.csv"

test_path  = "../input/facial-keypoints-detection/test/test.csv"

IDLookUP   = "../input/facial-keypoints-detection/IdLookupTable.csv"

goggles    = "../input/imagesgoggles/images.jpg"
train = pd.read_csv(train_path)

test = pd.read_csv(test_path)

lookUp = pd.read_csv(IDLookUP)
print("The null values in Train Data:", train.isnull().values.sum()),

print("The null values in Train Data:", train.isnull().sum())
# Fill the null values with previous values

train.fillna(method='ffill',inplace=True)
print("The null values in Train Data:", train.isnull().values.sum())


'-------------------CONVERTS STR PIXELS TO INTEGER PIXELS-----------------------'



def strToImage(data):

    

    Images =[]

    for i in tnrange(0, len(data)):

        getList = data[i].split(' ')

        for j in range(0,len(getList)):

            getList[j] = int(getList[j])

        Images.append(getList)

    return Images





'-----------------RESHAPES image to (96x96)--------------------------------------'



def reimage(data):

    for i in tnrange(len(data)):

        data[i] = np.array(data[i]).reshape((96,96))

    return data



'------------------shows random 9 images-------------------------------------------'

def showImages(num, data):

    

    plt.figure(figsize=(15,10))

    for i in range(num):

        plt.subplot(3,3,i+1)

        r= random.randint(0,len(data))

        plt.imshow(data[r] , cmap='gray')

    plt.show()

    

'----------------------------------PLOTS KEYPOINTS---------------------------------'



def PlotKeypoints(num, data, points):

    

    plt.figure(figsize=(15,10))

    for i in range(num):

        plt.subplot(3,3,i+1)

        r = random.randint(1,len(data))

        plt.imshow(data[r], cmap='gray')

        plt.title('Image:{}'.format(r))

        for col in range(0,30,2):

            plt.plot(points.iloc[r][col] , points.iloc[r][col+1], 'ro')

    plt.show()

    

"---------------------------------FLIP DATA----------------------------------------------"

def flipimage(data, images):

    

    flip_indices = [

        (0, 2), (1, 3),

        (4, 8), (5, 9), (6, 10), (7, 11),

        (12, 16), (13, 17), (14, 18), (15, 19),

        (22, 24), (23, 25),

        ]

    image=[]

    flipdata = data.copy()

    for i in tnrange(0,len(data)):

        image.append(cv2.flip(images[i],1))

        for (a,b) in flip_indices:

            flipdata.iloc[i][a], flipdata.iloc[i][b] = (data.iloc[i][b], data.iloc[i][a])

            

    return flipdata, image





"---------------------------------Stacking------------------------------------------------------"



def stack(data1,data2, images1,images2):

    return np.vstack((data1,data2)), np.vstack((images1,images2))





"---------------------------------PLOT ROI-------------------------------------------------------"



def plotROI(num, points, images):

    

    p =['right_eyebrow_outer_end_x','right_eyebrow_outer_end_y',

        'left_eyebrow_outer_end_x','left_eyebrow_outer_end_y']

    plt.figure(figsize=(15,10))

    

    for i in range(num):

        plt.subplot(3,3,i+1)

        r = random.randint(1,1000)

        image = images[r]

        x1= int(points.iloc[r][p[0]])

        y1= int(points.iloc[r][p[1]])

        x2= int(points.iloc[r][p[2]])

        y2= int(points.iloc[r][p[3]])

        roi= image[y1:y1+y2-7 , x1:x1+x2-15]

        #roi = np.resize(roi , (25,65))

        plt.imshow(roi)
print("------------------------------")

print("-----convert train images-----")

print("------------------------------")

trainImages = strToImage(train['Image'])

print("------------------------------")

print("-----convert test images-----")

print("------------------------------")

testImages = strToImage(test['Image'])
print("------------------------------")

print("-----reshape train images-----")

print("------------------------------")

Images = reimage(trainImages)



print("------------------------------")

print("-----reshape test images-----")

print("------------------------------")

testImages = reimage(testImages)
trainX = np.array(trainImages).reshape((len(trainImages),96,96,1))

keypoints = train.drop(['Image'], axis=1)
flipPoints , augimage = flipimage(keypoints , trainImages)
y,X = stack(keypoints,flipPoints , trainImages, augimage)
df = pd.DataFrame(y , columns=list(keypoints.columns))
showImages(9,X)
PlotKeypoints(9, X, df)
def myModel():

    

    '''

    This model accepts an Image of size (96x96) 

    and it predicts keypoints on the face.

'''   

    inputs = Input(shape=(96,96,1), name='model_input')

    X = Conv2D(16,(2,2), padding='same',activation='relu')(inputs)

    X = MaxPool2D(pool_size=(2, 2),strides=(2,2))(X)

    X = Dropout(0.1)(X)

    X = BatchNormalization()(X)

    

    #

    X = Conv2D(32,(2,2),padding='same', activation='relu')(X)

    X = MaxPool2D(pool_size=(2, 2),strides=(2,2))(X)

    X = Dropout(0.2)(X)

    X = BatchNormalization()(X)

    

    '''

    X = Conv2D(64,(5,5),activation='relu')(X)

    X = MaxPool2D(pool_size=(2, 2),strides=(2,2))(X)

    X = Dropout(0.2)(X)

    X = BatchNormalization()(X)

    

    



    X = Conv2D(128,(3,3),activation='relu')(X)

    X = MaxPool2D(pool_size=(2, 2), strides=(2,2))(X)

    X = Dropout(0.4)(X)

    X = BatchNormalization()(X)

    

    

    X = Conv2D(128,(2,2), activation='relu')(X)

    X = MaxPool2D(pool_size=(2, 2))(X)

    X = Dropout(0.4)(X)

    X = BatchNormalization()(X)'''

    

    

    X = Flatten()(X)

#     X = Dense(64, activation='relu')(X)

#     X = Dropout(0.1)(X)

#     X = Dense(128, activation='relu')(X)

#     X = Dropout(0.1)(X)

    X = Dense(256 , activation='relu')(X)

    X = Dropout(0.1)(X)

    X = Dense(512 , activation='relu')(X)

    X = Dropout(0.1)(X)

    #X = Dense(64 , activation='relu')(X)

    classifier = Dense(30)(X)

    model = Model(inputs=inputs , outputs=classifier, name='model_output')



    return model



model = myModel()

#plot_model(model , 'keypoint_detection_model.jpg', show_shapes=True)
model.compile(optimizer=Adam(learning_rate=0.0001), 

                  loss='mse',

                  metrics=['accuracy'])
model.summary()
X_data = X.reshape(X.shape[0],96,96,1)

X_train, X_val, Y_train, Y_val = train_test_split(trainX, keypoints, test_size=0.3, random_state=42)
from time import time

t1 = time()

history= model.fit(X_train , Y_train , epochs=500,batch_size=128, validation_data=(X_val,Y_val)) 

print("total time:", time()-t1)
def plotgraph(history, RMSE=True):

    

    loss = history['loss']

    val_loss = history['val_loss']

    if RMSE:

        loss = np.sqrt(np.array(loss))

        val_loss = np.sqrt(np.array(val_loss))

    acc = history['acc']

    val_acc = history['val_acc']

    

    plt.figure(figsize=(10,15))

    plt.subplot(2,1,1)

    plt.plot(loss , linewidth=3 ,label='train loss')

    plt.plot(val_loss , linewidth=3, label='val loss')

    plt.xlabel('epochs')

    plt.ylabel('loss / val_loss')

    plt.legend()

    

    plt.subplot(2,1,2)

    plt.plot(acc , linewidth=3 ,label='train acc')

    plt.plot(val_acc , linewidth=3, label='val acc')

    plt.xlabel('epochs')

    plt.ylabel('Accuracy / Val_Accuracy')

    plt.legend()
plotgraph(history.history)
testX  = np.array(testImages).reshape((len(testImages),96,96,1))
predictions = model.predict(testX)
model.save("../working/model.h5")
test_df = pd.DataFrame(predictions , columns=list(keypoints.columns))
test_df.head()
test_df.to_csv('test.csv' , index=False , header=False)