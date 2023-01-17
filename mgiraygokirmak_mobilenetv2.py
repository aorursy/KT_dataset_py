import numpy as np

import cv2

import pandas as pd

import math

import gc
from keras.layers import Dense, GlobalAveragePooling2D, Dropout

from keras.optimizers import Adam

from keras.applications import MobileNetV2

from keras.models import Sequential

from keras.callbacks import Callback,ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
IMG_SIZE = 224
def resize(image_path, img_size = IMG_SIZE):

    img = cv2.imread(image_path)

    pad_diff = max(img.shape) - img.shape[0], max(img.shape) - img.shape[1]

    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)

    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)

    pad_width = ((t,b), (l,r), (0, 0))

    padded = np.pad(img, pad_width=pad_width, mode='constant')

    resized = cv2.resize(padded, (img_size,)*2).astype('uint8')

    return resized
train_diag = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

train_diag = train_diag.drop(train_diag[train_diag["diagnosis"]==0].sample(600).index)
%%time

x_train = []

for img_id in train_diag['id_code']:

    x_train.append(resize(f'../input/aptos2019-blindness-detection/train_images/{img_id}.png'))
y_train = pd.get_dummies(train_diag['diagnosis']).values

del train_diag

gc.collect()
# Ref: https://www.kaggle.com/danmoller/make-best-use-of-a-kernel-s-limited-uptime-keras

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
model = Sequential()

model.add(MobileNetV2(weights="../input/pretrained-models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",

                        include_top=False,

                        input_shape=(IMG_SIZE, IMG_SIZE, 3)))

model.add(GlobalAveragePooling2D())

model.add(Dropout(0.2))

model.add(Dense(5, activation='softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer=Adam(lr=2e-5,amsgrad=True),

              metrics=['accuracy'])
model.fit(np.stack(x_train), 

          y_train,

          epochs=10000000,

          batch_size=32,

          verbose=2,

          shuffle=True,

          callbacks=[TimerCallback(500)],

         validation_split=0.25)
del x_train,y_train

gc.collect()
test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
%%time

test_resized = []

for img_id in test['id_code']:

    test_resized.append(resize(f'../input/aptos2019-blindness-detection/test_images/{img_id}.png'))
preds = model.predict(np.stack(test_resized))
test['diagnosis'] = preds.argmax(axis=1)

test.to_csv('submission.csv',index=False)