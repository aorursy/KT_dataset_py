#importing Libraries :)



import numpy as np

import pandas as pd



from keras.models import Sequential

from keras.layers import Dense,Activation,Layer,Lambda



from sklearn.cross_validation import train_test_split
#load dataset



dataset=pd.read_csv("../input/data.csv")
dataset.head()
#No idea why Unnamed:32 column exists. And we also don't need id column.

dataset=dataset.drop(["id","Unnamed: 32"],axis=1)

dataset.head()
dataset.shape
#just checking if there is any null value..

pd.isnull(dataset).sum()
#mapping function to map different string objects to integer

def mapping(data,feature):

    featureMap=dict()

    count=0

    for i in sorted(data[feature].unique(),reverse=True):

        featureMap[i]=count

        count=count+1

    data[feature]=data[feature].map(featureMap)

    return data
dataset=mapping(dataset,feature="diagnosis")
dataset.sample(5)
#divide dataset into x(input) and y(output)

X=dataset.drop(["diagnosis"],axis=1)

y=dataset["diagnosis"]
#divide dataset into training set, cross validation set, and test set

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)
#hein?? what is all this??

def getModel(arr):

    model=Sequential()

    for i in range(len(arr)):

        if i!=0 and i!=len(arr)-1:

            if i==1:

                model.add(Dense(arr[i],input_dim=arr[0],kernel_initializer='normal', activation='relu'))

            else:

                model.add(Dense(arr[i],activation='relu'))

    model.add(Dense(arr[-1],kernel_initializer='normal',activation="sigmoid"))

    model.compile(loss="binary_crossentropy",optimizer='rmsprop',metrics=['accuracy'])

    return model
firstModel=getModel([30,50,1])
import keras

import matplotlib.pyplot as plt

from IPython.display import clear_output

class PlotLosses(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        

        self.fig = plt.figure()

        

        self.logs = []



    def on_epoch_end(self, epoch, logs={}):

        

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.i += 1

        

        clear_output(wait=True)

        plt.plot(self.x, self.losses, label="loss")

        plt.legend()

        plt.show();

        

plot_losses = PlotLosses()
firstModel.fit(np.array(trainX),np.array(trainY),epochs=40,callbacks=[plot_losses])
scores=firstModel.evaluate(np.array(valX),np.array(valY))

print("Loss:",scores[0])

print("Accuracy",scores[1]*100)
secondModel=getModel([30,100,1])

secondModel.fit(np.array(trainX),np.array(trainY),epochs=40,callbacks=[plot_losses])
scores2=secondModel.evaluate(np.array(valX),np.array(valY))
print(scores2)
thirdModel=getModel([30,50,70,40,1])
thirdModel.fit(np.array(trainX),np.array(trainY),epochs=100,callbacks=[plot_losses])
scores3=thirdModel.evaluate(np.array(valX),np.array(valY))
print(scores3)
predY=firstModel.predict(np.array(testX))

predY=np.round(predY).astype(int).reshape(1,-1)[0]
from sklearn.metrics import confusion_matrix

m=confusion_matrix(predY,testY)

tn, fn, fp, tp=confusion_matrix(predY,testY).ravel()

m=pd.crosstab(predY,testY)

print("Confusion matrix")

print(m)
sens=tp/(tp+fn)

spec=tn/(tn+fp)

print("Senstivity:",sens)

print("Specificity:",spec)