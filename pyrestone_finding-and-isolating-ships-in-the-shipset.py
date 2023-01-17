# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#####disabled this because my machine runs on windows

#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#read dataset. n

dataset=pd.read_json(open("input/shipsnet.json","r"))
x_set=dataset["data"]

y_set=dataset["labels"]

X=[]

Y=[]

for k in x_set.keys():

    X.append(x_set[k])

    Y.append(y_set[k])

X=np.array(X)

Y=np.array(Y)
#constants

data_size=2800

np.random.seed(42)

#shuffle dataset

p=np.random.permutation(data_size)

X=X[p]

Y=Y[p]

#reshape to proper dimensions and range

X=X.reshape(-1,3,80,80).transpose(0,2,3,1)/255.

Y=Y.reshape(-1)

#sutract mean and divide by std

mean=np.mean(X)

std=np.std(X)

X-=mean

X/=std

#do train/val/test split

train_limit=data_size*8//10

val_limit=data_size*9//10

train_X=X[:train_limit]

train_Y=Y[:train_limit]

val_X=X[train_limit:val_limit]

val_Y=Y[train_limit:val_limit]

test_X=X[val_limit:]

test_Y=Y[val_limit:]
print("Train set: {} items".format(len(train_Y)))

print("Val set: {} items".format(len(val_Y)))

print("Test set: {} items".format(len(test_Y)))

print("Train set positive prob: {:.2f}".format(np.sum(train_Y)/len(train_Y)))

print("Val set positive prob: {:.2f}".format(np.sum(val_Y)/len(val_Y)))
i=0

plt.imshow(val_X[i]*std+mean)

plt.show()

print("Label: {}".format(val_Y[i]))
import keras

from keras.models import Sequential

from keras.layers.core import Dense,Activation

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPool2D
#build simple convnet model

model=Sequential()

model.add(Conv2D(kernel_size=(5,5),activation="elu",padding="valid",filters=100,input_shape=(80,80,3)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(kernel_size=(5,5),activation="elu",padding="valid",filters=100))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(kernel_size=(5,5),activation="elu",padding="valid",filters=100))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(kernel_size=(5,5),activation="elu",padding="valid",filters=100))

model.add(keras.layers.core.Flatten())

model.add(Dense(100,activation="elu"))

model.add(Dense(100,activation="elu"))

model.add(Dense(1,activation="sigmoid"))
model.compile(keras.optimizers.RMSprop(lr=1e-4),"binary_crossentropy",metrics=["binary_accuracy"])
#train model

model.fit(train_X,train_Y,validation_data=(val_X,val_Y),epochs=30,batch_size=200)
#show results on the first few validation set entries

for i in range(10):

    plt.imshow(val_X[i]*std+mean)

    plt.show()

    print("Predicted: {:.2f}".format(model.predict(val_X[i:i+1])[0,0]))

    print("Actual   : {:.2f}".format(val_Y[i]))
#define methods for heatmapping the image:

#we occlude parts of the image with zeros(neutral since normalized) 

#and check the difference between the original prediction and 

#the occluded prediction for each pixel.

def occlude(img,x,y,size=8):

    im=img.copy()

    im=im.reshape(1,80,80,-1)

    im[:,max(0,x-size):min(x+size,80),max(0,y-size):min(y+size,80),:]=0

    return im

def occlude_all(img,size=15):

    return np.array([occlude(img,x,y,size) for x in range(80) for y in range(80)]).reshape(-1,80,80,3)

def get_heat(img,size=15):

    return ((model.predict(occlude_all(img,size=size)).reshape(80,80)))

#the resulting heatmap, when bounded by 0, 

#is a heatmap of what parts of the image are important for the classification.
#since the heatmap is now in range [0..1] 

#we can multiply the original image by this heatmap and 

#get these nice cutouts of important regions.

#(admittedly cherrypicked) examples of one non-ship, and one ship, with highlighted ship-like regions

for i in range(18,20):

    prediction=model.predict(val_X[i:i+1])[0,0]

    heat=prediction-get_heat(val_X[i],size=20)

    heat[heat<0.1]=0

    print("Predicted: {:.2f}".format(model.predict(val_X[i:i+1])[0,0]))

    print("Actual   : {:.2f}".format(val_Y[i]))

    plt.imshow(val_X[i]*std+mean)

    plt.show()

    heat_img=heat.reshape(80,80,1)*(val_X[i]*std+mean)

    plt.imshow(heat_img)

    plt.show()
#all (predicted) ships and their hotspots regions in the validation set:

for i in range(len(val_X)):

    prediction=model.predict(val_X[i:i+1])[0,0]

    if(prediction>0.5):

        heat=prediction-get_heat(val_X[i],size=20)

        heat[heat<0.1]=0

        print("Predicted: {:.2f}".format(model.predict(val_X[i:i+1])[0,0]))

        print("Actual   : {:.2f}".format(val_Y[i]))

        plt.imshow(val_X[i]*std+mean)

        plt.show()

        heat_img=heat.reshape(80,80,1)*(val_X[i]*std+mean)

        plt.imshow(heat_img)

        plt.show()
#finally, you should probably check the model accuracy on the test set.

metrics=model.evaluate(test_X,test_Y)

print(metrics)
#my final accuracy was around 97~99% so i'm satisfied with these results.