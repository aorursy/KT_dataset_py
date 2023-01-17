import numpy as np

import pandas as pd

import os

import pylab
print(os.listdir(os.getcwd()))
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import sys

modulenames = set(sys.modules)&set(globals())

print(modulenames)
plt.rcParams['figure.figsize'] = 10,10
%matplotlib inline
# Load the data

train_data = pd.read_json("../input/train.json")
train_data.head(10)
test_data = pd.read_json("../input/test.json")
test_data.head(10)
Xband1 = np.array([np.array(eachband).reshape(75,75) for eachband in train_data["band_1"]])

Xband2 = np.array([np.array(eachband).reshape(75,75) for eachband in train_data["band_2"]])

Xtrain = np.concatenate((Xband1[:,:,:,np.newaxis],Xband2[:,:,:,np.newaxis],((Xband1+Xband2)/2)[:,:,:,np.newaxis]),axis = -1)
# 1604 images

# Each image having 75 rows and each row has 75 parts and each part having 3 rgb channels

Xtrain.shape
import keras

from keras.models import Model

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Input,Flatten,Activation

from keras.layers import GlobalMaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import Concatenate

from keras import initializers

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping
# Define the model



model = Sequential()



model.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(75,75,3)))

model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.2))



model.add(Flatten())



model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.2))



model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(0.2))



model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',optimizer=Adam(lr = 0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0),metrics=['accuracy'])

model.summary()
def get_callbacks(filepath, patience=2):

    es = EarlyStopping('val_loss', patience=patience, mode="min")

    msave = ModelCheckpoint(filepath, save_best_only=True)

    return [es, msave]

file_path = ".model_weights.hdf5"

callbacks = get_callbacks(filepath=file_path, patience=5)
Ytrain = train_data["is_iceberg"]



XtrainCV,Xvalid,YtrainCV,Yvalid = train_test_split(Xtrain,Ytrain,random_state = 1,train_size = 0.75)
test1 = np.array([np.array(eachband).reshape(75,75) for eachband in test_data["band_1"]])

test2 = np.array([np.array(eachband).reshape(75,75) for eachband in test_data["band_2"]])
Xtest = np.concatenate((test1[:,:,:,np.newaxis],test2[:,:,:,np.newaxis],((test1+test2)/2)[:,:,:,np.newaxis]),axis = -1)

del test1,test2
Xtest.shape
def findmaxindices(model,number,xtest):

    test_proba = model.predict_proba(xtest)

    max_indices = np.argpartition(test_proba,-number,axis=0)[-number:]

    return max_indices



def addthistoxtrain(indices,xtrain,xtest):

    xtrain = np.append(xtrain,xtest[indices],axis=0)

    return xtrain
from sklearn.model_selection import train_test_split

import numpy as np



class Semisupervise_model(object):

    

    def __init__(self,model,xtest,xtrain,ytrain,percentage = 0.1,seed=69,loopstorun = 20):

        self.model = model

        self.percentage = percentage

        self.seed = seed

        self.xtest = xtest

        self.xtrain = xtrain

        self.ytrain = ytrain

        self.loopstorun = loopstorun

    

    def findmaxminindices(self,xtest,extra_number = None):

        

        # Based on percentage given to include we are adding that many population to current existing training set

        # Taking the length of training set and using percentage of that 

        

        number = extra_number if extra_number is not None else int(self.percentage*len(self.xtrain))

        

        # Predicting the probablities of unlabled data using the model that we have already fit

        testproba = self.model.predict_proba(xtest)

        

        # Find the max and min indices until the percentage we want

        max_indices = np.argpartition(testproba,-number,axis=0)[-number:]

        min_indices = np.argpartition(testproba,number,axis=0)[:number]

        

        print("Max indices shape is ",max_indices.shape)

        print("Mind indices shape is ",min_indices.shape)

        

        return max_indices,min_indices

    

    def get_y(self,X):

        # For a given X find the y's using the model 

        return self.model.predict_proba(X)

    

    def addthistotrain(self,indices,xtest,xtrain,ytrain):

        # getting all the y's

        ytest = self.get_y(xtest)

        ytest = np.reshape(ytest,(ytest.shape[0],))

        

        # adding the min and max indices populations to xtrain and ytrain there

        print("Ytest shape is ",ytest.shape)

        print("Xtrain shape is",xtrain.shape)

        print("Xtest shape is ",xtest.shape)

        print("Ytrain shape is ",ytrain.shape)

        print("Indices shape is ",indices.shape)

        print("xtest[indices] shape is",xtest[indices].shape)

        

        xtrain = np.concatenate((xtrain,xtest[indices]),axis=0)

        ytrain = np.concatenate((ytrain,ytest[indices]),axis=0)

        ytrain[ytrain>=0.5]=1

        ytrain[ytrain<0.5]=0

        

        print("################################")

        print("Ytest shape is ",ytest.shape)

        print("Xtrain shape is",xtrain.shape)

        print("Xtest shape is ",xtest.shape)

        print("Ytrain shape is ",ytrain.shape)

        print("Indices shape is ",indices.shape)

        print("xtest[indices] shape is",xtest[indices].shape)

        

        return xtrain,ytrain

    

    def removethisfromtest(self,indices,xtest):

        print("Now in removethisfromtest")

        print("indices shape is",indices.shape)

        print("xtest shape before removing",xtest.shape)

        indices = sorted(indices,reverse=True)

        xtest=np.delete(xtest,indices,axis=0)

        print("xtest shape after",xtest.shape)

        return xtest

        

        

    def fit_here(self,xtest=None,xtrain=None,ytrain=None,extra_number = None):

        xtest = xtest if xtest is not None else self.xtest

        xtrain = xtrain if xtrain is not None else self.xtrain

        ytrain = ytrain if ytrain is not None else self.ytrain

        extra_number = extra_number if extra_number is not None else self.percentage

        

        # find the max and min indices where the predicted probablities are higher

        maxindices,minindices = self.findmaxminindices(xtest,extra_number)

        

        # joinging both max and minindices in that given order

        indices = np.concatenate((maxindices,minindices),axis=0)

        

        # reshaping the indices so as y dimesion will remain same

        indices = np.reshape(indices,(indices.shape[0],))

        

        # Add the selected max and min population to the existing training set

        xtrain,ytrain = self.addthistotrain(indices,xtest,xtrain,ytrain)

        

        # Remove the selected max and min population to the existing testing set

        xtest = self.removethisfromtest(indices,xtest)

        

        # Split the modified training set to new sets of training and validation 

        xtrainnew,Xvalid,ytrainnew,Yvalid = train_test_split(xtrain,ytrain,random_state = 1,train_size = 0.75)

        

        # Fit the new traindata using the model we created (This model is given as input to our created instance)

        self.model.fit(xtrainnew,ytrainnew,batch_size=24,epochs=50,verbose=1,validation_data=(Xvalid,Yvalid))

        

        return self.model,xtrain,ytrain,xtest

    

    def loopthefit(self,xtest = None,xtrain=None,ytrain=None):

        xtrain = xtrain if xtrain is not None else self.xtrain

        xtest = xtest if xtest is not None else self.xtest

        ytrain = ytrain if ytrain is not None else self.ytrain

        

        xtrainlen = xtrain.shape[0]

        xtestlen = xtest.shape[0]

        count = 0

        while(xtestlen>self.percentage*xtrainlen and count<self.loopstorun):

            

            print("The count running now is %d" %count)

            extra_number = int(self.percentage*xtrainlen)

            

            if xtestlen > extra_number:

                

                ## Do the process here

                self.model,xtrain,ytrain,xtest = self.fit_here(xtest,xtrain,ytrain,extra_number)

                model.save_weights(os.getcwd()+'\Count_%d_modelweights.hdf5' %count)

                

                xtrainlen = xtrain.shape[0]

                xtestlen = xtest.shape[0]

                

            count = count+1

            

        return self.model,xtrain,ytrain,count

        
testrun = Semisupervise_model(model,Xtest,Xtrain,Ytrain,percentage=0.05,seed=69,loopstorun=6)
model,newxtrain,newytrain,countofsteps = testrun.loopthefit()
newxtrain.shape
newytrain.shape
probs = model.predict_proba(Xtest)
test_data['id'].shape
submit = pd.DataFrame()

submit["id"]=test_data['id']

submit['is_iceberg']=probs.reshape((probs.shape[0],))

submit.to_csv('sub.csv',index=False)
def testfunc(l,m,percentage,iterations):

    count = 0

    while(m>percentage*l and count < iterations):

        l = int(l+percentage*l)

        print("l is %f",l)

        if m>percentage*l:

            m = int(m-percentage*l)

            print("m is %f",m)

        count += 1

    return l,m,count