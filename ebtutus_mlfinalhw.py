# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from IPython.display import Image
import matplotlib.pyplot as plt
import keras

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
photos = []
photosCNN = []
ratio = 0.10
imgWidth = round(3840*ratio)
imgHeight = round(5120*ratio)
for dirname, _, filenames in os.walk('/kaggle/input'):
    counter = 0;
    for filename in filenames:
        counter +=1
        if filename.endswith('.jpg'):
            print("\rLoading...[{:.0f}%]".format(counter/len(filenames)*100) , end='')
            img = cv2.imread(os.path.join(dirname, str(counter) + ".jpg"),cv2.IMREAD_GRAYSCALE)
            reIm = cv2.resize(img, dsize=(imgWidth,imgHeight), interpolation=cv2.INTER_CUBIC)
            img = cv2.imread(os.path.join(dirname, str(counter) + ".jpg"))
            reImCNN = cv2.resize(img, dsize=(imgWidth,imgHeight))
            if reIm is not None:
                photos.append(reIm)
                photosCNN.append(reImCNN)
        elif filename.endswith('.xlsx'):
            data = pd.read_excel(os.path.join(dirname, filename))
images = np.asarray(photos)
imagesCNN = np.asarray(photosCNN)      

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print("Sample image and it's data")
print(data.loc[19])
y = data.to_numpy()
plt.imshow(images[19])
print('Image array shape : ' + str(images.shape))
print('Results array shape: ' + str(y.shape))
# Normalization - convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)
imageDNN = images.reshape(images.shape[0],imgWidth*imgHeight)
index = np.random.choice(imageDNN.shape[0], imageDNN.shape[0], replace=False)
trainPercentage = 0.9
valPercentage = 0.15
trainCount = round(imageDNN.shape[0]*trainPercentage) 
valCount = round(imageDNN.shape[0]*valPercentage) 
testCount = imageDNN.shape[0] - trainCount - valCount
xTrain = imageDNN[index[0:trainCount-valCount]]
yTrain = y[index[0:trainCount-valCount]]
xVal = imageDNN[index[trainCount-valCount:trainCount]]
yVal = y[index[trainCount-valCount:trainCount]]
xTest = imageDNN[index[trainCount:imageDNN.shape[0]]]
yTest = y[index[trainCount:imageDNN.shape[0]]]
print('Number of train data : ' + str(xTrain.shape[0]))
print('Number of validation data : ' + str(xVal.shape[0]))
print('Number of test data: ' + str(yTest.shape[0]))
# Model Configuration
batchSize = round(xTrain.shape[0]/4)
lossFunction = "mse"
noEpochs = 150
verbosity = 1
inputDim = (imgWidth*imgHeight,)
outPutDim = 11

#Model Creation
inputs = keras.Input(shape=inputDim)
dense = keras.layers.Dense(256,input_shape = inputDim,kernel_initializer='normal',activation="relu")
x = dense(inputs)
x = keras.layers.Dense(128,kernel_initializer='normal',kernel_regularizer='l2',activation="relu")(x)
x = keras.layers.Dense(64,kernel_initializer='normal',kernel_regularizer='l2',activation="relu")(x)
x = keras.layers.Dense(32,kernel_initializer='normal',kernel_regularizer='l2',activation="relu")(x)
outputs = keras.layers.Dense(11)(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="DNN_model")

# Compile the model
model.compile(loss=lossFunction,
              optimizer="RMSprop",
              metrics=['accuracy'])
model.summary()

print("Starting training ")
# Fit data to model
historyobj = model.fit(xTrain, yTrain,
            batch_size=batchSize,
            validation_data=(xVal,yVal),
            epochs=noEpochs,
            verbose=verbosity)
print("Training finished \n")
# Generate Evaluation
eval = model.evaluate(xTest,yTest)
print("\nEnd  DNN hw \n")
yPredict = np.round_(model.predict(xTest))
yPredict[yPredict < 0] = 0
yPredict = yPredict.astype(int)
print(data.columns)
for i in range(yPredict.shape[0]):
    print('Predict Data'+str(yPredict[i]))
    print('True Data   '+str(yTest[i]) + "\n")

# Visualize history
# Plot history: Loss
plt.plot(historyobj.history['loss'],'r',linewidth=3.0)
plt.plot(historyobj.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.show()


print('Image array shape : ' + str(imagesCNN.shape))
print('Results array shape: ' + str(y.shape))
# Normalization - convert from [0:255] => [0.0:1.0]
imagesCNN = imagesCNN.astype('float32')
imagesCNN =imagesCNN / 255.0
#index already created DNN part other configuration are same suc as train percentage vs
trainCount = round(imagesCNN.shape[0]*trainPercentage) 
valCount = round(imagesCNN.shape[0]*valPercentage) 
testCount = imagesCNN.shape[0] - trainCount - valCount
xTrain = imagesCNN[index[0:trainCount-valCount]]
yTrain = y[index[0:trainCount-valCount]]
xVal = imagesCNN[index[trainCount-valCount:trainCount]]
yVal = y[index[trainCount-valCount:trainCount]]
xTest = imagesCNN[index[trainCount:imagesCNN.shape[0]]]
yTest = y[index[trainCount:imagesCNN.shape[0]]]
print('Number of train data : ' + str(xTrain.shape[0]))
print('Number of validation data : ' + str(xVal.shape[0]))
print('Number of test data: ' + str(yTest.shape[0]))
# Model Configuration
batchSize = round(xTrain.shape[0]/4)
lossFunction = "mse"
noEpochs = 100
verbosity = 1
inputDim = (imgHeight,imgWidth,3)
outPutDim = 11

#Model Creation
inputs = keras.Input(shape=inputDim)
convolution = keras.layers.Conv2D(32,3,3, input_shape=inputDim,kernel_initializer='normal', border_mode='same', activation='relu')
x = convolution(inputs)
x = keras.layers.Conv2D(32,3,3, activation='relu',kernel_initializer='normal', border_mode='same',kernel_regularizer='l2')(x)
x = keras.layers.MaxPooling2D(pool_size=(5,5))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(64, activation='relu',kernel_initializer='normal', kernel_regularizer='l2')(x)
outputs = keras.layers.Dense(11)(x)
model2 = keras.Model(inputs=inputs, outputs=outputs, name="CNN_model")


# Compile the model
model2.compile(loss=lossFunction,
              optimizer="RMSprop",
              metrics=['accuracy'])
model2.summary()
print("Starting training ")
# Fit data to model
historyobj = model2.fit(xTrain, yTrain,
            batch_size=batchSize,
            validation_data=(xVal,yVal),
            epochs=noEpochs,
            verbose=verbosity)
print("Training finished \n")
# Generate Evaluation
eval = model2.evaluate(xTest,yTest)
print("\nEnd  CNN hw \n")
yPredict = np.round_(model2.predict(xTest))
yPredict[yPredict < 0] = 0
yPredict = yPredict.astype(int)
print(data.columns)
for i in range(yPredict.shape[0]):
    print('Predict Data'+str(yPredict[i]))
    print('True Data   '+str(yTest[i]) + "\n")

# Visualize history
# Plot history: Loss
plt.plot(historyobj.history['loss'],'r',linewidth=3.0)
plt.plot(historyobj.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.show()