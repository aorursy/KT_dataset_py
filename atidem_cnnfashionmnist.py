# import some libs

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# file finder for kaggle 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# read train data path

train = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")

train.shape



# read test data path

test = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")

test.shape



# split label 

yTrain = train["label"]

yTest = test["label"]



# delete label column 

train.drop(columns="label",inplace=True)

test.drop(columns="label",inplace=True)



# preview

train.head()
# train values dispersion

yTrain.value_counts()
# convert to binary form

from keras.utils.np_utils import to_categorical

y_train = to_categorical(yTrain,num_classes=10)

y_train
# train validation split

from sklearn.model_selection import train_test_split

xTrain,xValid,yTrain,yValid = train_test_split(train,y_train,test_size=0.2)
# normalize and reshape fro mr.Keras

xTrain = xTrain / 255.0 

test = test / 255.0



xTrain = xTrain.values.reshape(-1,28,28,1)

xValid = xValid.values.reshape(-1,28,28,1)



plt.imshow(xTrain[1].reshape((28,28)),cmap="gray")
# create model

from sklearn.metrics import confusion_matrix

import itertools



from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau

from keras.metrics import accuracy

model = Sequential()



model.add(Conv2D(filters=32,kernel_size=(4,4),padding="Same",activation="relu",input_shape=(28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(filters=32,kernel_size=(3,3),padding="Same",activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(512,activation="relu"))

model.add(Dropout(0.1))

model.add(Dense(128,activation="relu"))

model.add(Dropout(0.1))

model.add(Dense(10,activation="softmax"))



optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.999)



model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])



epochs = 30

batch_size = 100
# fit model

history = model.fit(x=xTrain,y=yTrain,batch_size=batch_size,epochs=epochs,validation_data=(xValid,yValid))

epochies = history.epoch

hist = pd.DataFrame(epochies)
# loss graphic

plt.plot(history.history["loss"],label="loss")

plt.title("train loss")

plt.xlabel("num of epochs")

plt.ylabel("loss")

plt.legend()
# train data predict and visualize

Y_pred_train = model.predict(xTrain)

Y_pred_train_classes = np.argmax(Y_pred_train,axis=1)

Y_true_train = np.argmax(yTrain,axis=1)



confusion_mtx_trn = confusion_matrix(Y_true_train,Y_pred_train_classes)

f,ax = plt.subplots(figsize=(8,8))

sns.heatmap(confusion_mtx_trn,annot=True,linewidths=0.01,cmap="Greens",linecolor="gray",fmt=".1f",ax=ax)

plt.xlabel("predict")

plt.ylabel("true")

plt.show()

# test data predict and visualize

test = test.values.reshape(-1,28,28,1)

Y_pred = model.predict(test)

Y_pred_classes = np.argmax(Y_pred,axis=1)

Y_true = yTest



confusion_mtx = confusion_matrix(Y_true,Y_pred_classes)

f,ax = plt.subplots(figsize=(8,8))

sns.heatmap(confusion_mtx,annot=True,linewidths=0.01,cmap="Blues",linecolor="gray",fmt=".1f",ax=ax)

plt.xlabel("predict")

plt.ylabel("true")

plt.show()
# measure the model

from sklearn.metrics import accuracy_score

print("Train Accuracy Score "+str(accuracy_score(Y_true_train,Y_pred_train_classes)))

print("Test Accuracy Score "+str(accuracy_score(Y_true,Y_pred_classes)))

print("\n")

print("Train Acc")

for i in range(len(confusion_mtx_trn[0])):

    print(i,confusion_mtx_trn[i,i]/sum(confusion_mtx_trn[i]))

print("\n")    

print("Test Acc")

for i in range(len(confusion_mtx[0])):

    print(i,confusion_mtx[i,i]/sum(confusion_mtx[i]))
