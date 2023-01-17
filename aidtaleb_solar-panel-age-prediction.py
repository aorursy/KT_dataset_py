import zipfile 
from zipfile import ZipFile
from PIL import Image
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
import pandas as pd 
import numpy as np 
# Create a ZipFile Object and load sample.zip in it

zf = ZipFile('/content/drive//SolarPanelSoilingImageDataset.zip', 'r') 
c= zf.namelist()
zf.extractall('/kaggle/output')
import os
import glob
import cv2
import numpy as np
import tensorflow as tf

datax = []
name=[]
for i in glob.glob('/kaggle/input/solarpanelw/Solar_Panel_Soiling_Image_dataset/PanelImages/*.jpg', recursive=True):
    datax.append(cv2.imread(i))
    name.append(i)
    if len(name)==1500:
        break;
    


def solar_data(name):
    data=[]
    i=0 
    j=0
    data1=[]
    data2=[]
    data3=[]
    data4=[]
    data5=[]
    ageloss=[]
    irradiancelevel=[]
    for line in name: 
      data.append(line.split('s/s'))
      if len(data[i])==2 : 
        data1.append(data[i][1])
        data2.append(data1[j].split('.j'))
        data3.append(data2[j][0])
        j=j+1
      i=i+1
    for item in  data3 :
      data4.append(item.split('_'))

    for pp in  data4: 
      if len(pp)==14 : 
        data5.append(pp[3]+'/'+pp[2]+'/'+pp[9]+' '+pp[4]+':'+pp[6]+':'+pp[8])
        ageloss.append(pp[11])
        irradiancelevel.append(pp[13])
    
    df1=pd.DataFrame()
    df1['date']=data5
    df1['lossage']=ageloss
    df1['irradiancelevel']=irradiancelevel

    return df1
df1=solar_data(name)
month=['Jan','Feb','Mar', 'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
num=['1','2','3','4','5','6','7','8','9','10','11','12']
i=0
for mois in month : 
  df1['date']=df1['date'].str.replace(mois,num[i])
  i=i+1
df2=df1[['lossage','irradiancelevel']]
df2['lossage'] = df2['lossage'].astype(float)
df2['irradiancelevel'] = df2['irradiancelevel'].astype(float)


(trainAttrX, testAttrX) = train_test_split(df2,test_size=0.20, random_state=42)

(trainImages, testImages) = train_test_split( datax, test_size=0.20, random_state=42)

trainImagesX=np.array(trainImages)
testImagesX=np.array(testImages)
trainImagesX=trainImagesX/255
testImagesX=testImagesX/255
maxAgeloss = trainAttrX["lossage"].max()
trainY = trainAttrX["lossage"] / maxAgeloss
testY = testAttrX["lossage"] / maxAgeloss
trainX=trainAttrX["irradiancelevel"]
testX=testAttrX["irradiancelevel"]
trainX.shape[0]
def create_mlp(dim, regress=False):
 # define our MLP network
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))
    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))
    # return our model
    return model

def create_cnn(width, height, depth, filters=(16,32,64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1
        # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
    # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)
    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)
    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model
    
# create the MLP and CNN models
mlp = create_mlp(1, regress=False)
cnn = create_cnn(192, 192, 3, regress=False)
# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])
# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

model = Model(inputs=[mlp.input, cnn.input], outputs=x)

opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

print("[INFO] training model...")
model.fit(
     x=[trainX, trainImagesX], y=trainY,
    validation_data=([testX, testImagesX], testY),
     epochs=200, batch_size=8)

print("[INFO] predicting house prices...")
preds = model.predict([testX, testImagesX])
import locale
import os
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)     
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
diff
