import time 

#let's also import the abstract base class for our callback
from keras.callbacks import Callback

#defining the callback
class TimerCallback(Callback):
    
    def __init__(self, maxExecutionTime, byBatch = False, on_interrupt=None):
        
# Arguments:
#     maxExecutionTime (number): Time in minutes. The model will keep training 
#                                until shortly before this limit
#                                (If you need safety, provide a time with a certain tolerance)

#     byBatch (boolean)     : If True, will try to interrupt training at the end of each batch
#                             If False, will try to interrupt the model at the end of each epoch    
#                            (use `byBatch = True` only if each epoch is going to take hours)          

#     on_interrupt (method)          : called when training is interrupted
#         signature: func(model,elapsedTime), where...
#               model: the model being trained
#               elapsedTime: the time passed since the beginning until interruption   

        
        self.maxExecutionTime = maxExecutionTime * 60
        self.on_interrupt = on_interrupt
        
        #the same handler is used for checking each batch or each epoch
        if byBatch == True:
            #on_batch_end is called by keras every time a batch finishes
            self.on_batch_end = self.on_end_handler
        else:
            #on_epoch_end is called by keras every time an epoch finishes
            self.on_epoch_end = self.on_end_handler
    
    
    #Keras will call this when training begins
    def on_train_begin(self, logs):
        self.startTime = time.time()
        self.longestTime = 0            #time taken by the longest epoch or batch
        self.lastTime = self.startTime  #time when the last trained epoch or batch was finished
    
    
    #this is our custom handler that will be used in place of the keras methods:
        #`on_batch_end(batch,logs)` or `on_epoch_end(epoch,logs)`
    def on_end_handler(self, index, logs):
        
        currentTime      = time.time()                           
        self.elapsedTime = currentTime - self.startTime    #total time taken until now
        thisTime         = currentTime - self.lastTime     #time taken for the current epoch
                                                               #or batch to finish
        
        self.lastTime = currentTime
        
        #verifications will be made based on the longest epoch or batch
        if thisTime > self.longestTime:
            self.longestTime = thisTime
        
        
        #if the (assumed) time taken by the next epoch or batch is greater than the
            #remaining time, stop training
        remainingTime = self.maxExecutionTime - self.elapsedTime
        if remainingTime < self.longestTime:
            
            self.model.stop_training = True  #this tells Keras to not continue training
            print("\n\nTimerCallback: Finishing model training before it takes too much time. (Elapsed time: " + str(self.elapsedTime/60.) + " minutes )\n\n")
            
            #if we have passed the `on_interrupt` callback, call it here
            if self.on_interrupt is not None:
                self.on_interrupt(self.model, self.elapsedTime)
import pandas as pd
import numpy as np
from keras.utils import to_categorical

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#loading data - as for a demonstration of the callback,
#we won't worry about validation data in this kernel (but it would work the same way)
trainData = pd.read_csv('../input/train.csv')
y_train = to_categorical(np.array(trainData['label']))                   
x_train = np.array(trainData[list(trainData)[1:]]).reshape((-1,28,28,1)) 
    #labels as one-hot encoded vectors (ex: label 2 will become [0,0,1,0,0,0,0,0,0,0])
    #x shaped as images with one channel
    
    
#quick check
def quickCheck(x, y, predicted=None):
    #plotting images
    fig,ax = plt.subplots(nrows=1,ncols=10, figsize=(10,2))
    for i in range(10):
        ax[i].imshow(x[i].reshape((28,28)))
    plt.show()
    
    #printing labels
    y = np.argmax(y, axis=1) #converting from one-hot to numerical labels
    print("  " + "      ".join([str(i) for i in y[:10]]) + " <- labels")
    
    #printing predicted if passed
    if predicted is not None:
        predicted = np.argmax(predicted, axis=1)
        print("  " + "      ".join([str(i) for i in predicted[:10]]) + " <- predicted labels")

quickCheck(x_train,y_train)



    
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Model

#a simple convolutional model (not worried about it's capabilities)
def createModel():
    inputImage = Input((28,28,1))
    output = Conv2D(10, 3, activation='tanh')(inputImage)
    output = Conv2D(20, 3, activation='tanh')(output)
    output = MaxPooling2D((4,4))(output)
    output = Conv2D(10, 3, activation='tanh')(output)
    output = Flatten()(output)
    output = Dense(10, activation='sigmoid')(output)

    model = Model(inputImage, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

model = createModel()
model.fit(x_train, y_train, epochs = 1000000000, callbacks=[TimerCallback(5)])
#a function compatible with the on_interrupt handler
def saveWeights(model, elapsed):
    model.save_weights("model_weights.h5")

#fitting with the callback
callbacks = [TimerCallback(5, on_interrupt=saveWeights)]
model.fit(x_train,y_train, epochs = 100000000000, callbacks=callbacks)


#check that the weights were saved:
import os
os.listdir(".")
#although it uses the same creator function, it's a different model from the previous one
del(model)
model2 = createModel()

#load weights - this only works if the model has the same layer types and the same parameters
model2.load_weights('model_weights.h5') #

#evaluate model2
print("\n\nEvaluating model 2:")
loss, acc = model2.evaluate(x_train, y_train)
print('model 2 loss: ' + str(loss))
print('model 2 acc:  ' + str(acc))

#predicting and checking
predicts = model2.predict(x_train[25:35])
quickCheck(x_train[25:35],y_train[25:35], predicts)