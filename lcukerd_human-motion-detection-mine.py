from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import csv
import numpy as np
import pandas as pd
def loadData(filename,isy):
    y = np.array([0,0,0,0,0,0])
    x = np.zeros(562)
    x = np.resize(1,562)
    with open(filename, newline='\n') as File:  
        reader = csv.reader(File,delimiter = ' ')
        for row in reader:
            if (isy):
                gd = [0,0,0,0,0,0]
                gd[int (row[0]) - 1] = 1
                y = np.vstack((y,np.array([gd])))
            else:
                x = np.vstack((x,np.array([row])))
        if (isy == False):
            x = np.delete(x,0,0)
            x = np.delete(x,0,1)
            return x.astype(float)
        else:
            y = np.delete(y,0,0)
            return y.astype(float)
X = loadData('../input/X_train.txt',False)
print (X.shape)
Y = loadData('../input/y_train.txt',True)
print (Y.shape)
X_test = loadData('../input/X_test.txt',False)
print (X_test.shape)
Y_test = loadData('../input/y_test.txt',True)
print (Y_test.shape)
def buildModel():
    model = Sequential()
    model.add(Dense(250,input_dim = 561,activation = 'tanh'))
    model.add(Dense(50,activation = 'tanh'))
    model.add(Dense(10,activation = 'tanh'))
    model.add(Dense(6,activation = 'softmax'))
    model.compile(loss = 'mse', optimizer = Adam(lr = 0.00001),metrics=['accuracy'])
    return model
print (X.shape)
for i in range(X.shape[1]):
    output = pd.Series(X[:,i])
    print (output.describe(), end = '\n \n')
model = buildModel()
model.fit(X,Y,epochs=50,validation_split=0.25)
pred = model.predict_classes(X_test)
check = np.argmax(Y_test,axis=1) == pred
print ('Accuracy : ',sum(check)*100/len(check),'%')
