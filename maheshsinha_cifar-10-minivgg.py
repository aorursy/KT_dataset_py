#importing liberaries:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint
import cv2
import matplotlib.pyplot as plt
import urllib
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
#load data:
((X_train, y_train), (X_test, y_test)) = cifar10.load_data()
#scale data into range [0,1]:
X_train = X_train.astype('float')/255.0
X_test = X_test.astype('float') /255.0

#encode labels from intiger to vectors:
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)
# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
#define Architecture:
class arch():
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (width, height, depth)
        ch_dim = -1

        model = Sequential()
        #1layer CONV => RELU => BN => CONV => RELU => BN
        model.add(Conv2D(32, (3,3), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=ch_dim))
        model.add(Conv2D(32, (3,3), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=ch_dim))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        #2layer CONV => RELU => BN => CONV => RELU => BN
        model.add(Conv2D(32, (3,3), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=ch_dim))
        model.add(Conv2D(32, (3,3), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=ch_dim))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        #3layer FC => RELU
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=ch_dim))
        model.add(Dropout(0.5))

        #4 SOFTMAX:
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        #store the output path for the figure, path for the JSON serialized file and the starting epoch:
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt
        
    def on_train_begin(self, logs={}):
        #initialize the history dictonary:
        self.H = {}
        
        #if the JSON history path exist, load the training history:
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())
                
                #check to see if starting Epoch (startAt) is supplied:
                if self.startAt > 0:
                    #loop pver the entries in the history log--
                    #-- and trim any entries that are past the starting epoch:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]
                        
                        
    def on_epoch_end(self, epoch, logs={}):
        #loop over the logs and update the loss, accuracy etc.--
        #for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l
            
        #check to see it the training history should be serialized to file:
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(str(self.H)))
            f.close()
            
        #Construct plot:
        #ensure atleast two epochs have been pased befor plotting:
        #(epoch starts at 0):
        if len(self.H['loss'])>1:
            #plot the training loss and accuracy:
            N = np.arange(0, len(self.H['loss']))
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(N, self.H['loss'], label='train_loss')
            plt.plot(N, self.H['val_loss'], label='val_loss')
            plt.plot(N, self.H['accuracy'], label='train_accuracy')
            plt.plot(N, self.H['val_accuracy'], label='val_accuracy')
            plt.title(f"Trainin Loss and Accuracy [Epoch {len(self.H['loss'])}]")
            plt.xlabel('Epoch')
            plt.ylabel('Loss/Accuracy')
            plt.legend()
            
            #save figure:
            plt.savefig(self.figPath)
            plt.close()

#Applying Monitoring traing model: 
#construct the set of callbacks:
figPath = f'/kaggle/working/{os.getpid()}.png'
jsonPath = f'/kaggle/working/{os.getpid()}.json'
callbacks = [TrainingMonitor(figPath=figPath, jsonPath=jsonPath)]

#Custom STEP WISE learning rate decay function:
def step_decay(epoch):
    #initialize the base initial learning rate, drop factor and drop every:
    initAlpa=0.01
    factor = 0.25
    dropEvery = 5
    
    #compute th learning rate for current epoch:
    alpha = initAlpa * (factor**np.floor((epoch+1)/dropEvery))
    
    #return learning rate:
    return alpha

#define the set of callbacks to be passed to the model during training:
#callbacks = [LearningRateScheduler(step_decay)]
#construct the callback to save the *best* model to disk based on validation loss:
fname = "/kaggle/working/weights-{epoch:03d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(fname, monitor='val_loss', mode=min, save_best_only=True, verbose=1)
callbacks = [checkpoint]
#Compiling network
print('Network compiling...')
sgd = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = arch.build(32, 32, 3, 10)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])

#train network:
print('Network Training...')
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=64,callbacks = callbacks, verbose=1)
#evaluation, selct random 10 images and predict result:
prediction = model.predict(X_test, batch_size=64)
print(classification_report(y_test.argmax(axis=1), prediction.argmax(axis=1), target_names=labelNames))
#reading image from web:
req = urllib.request.urlopen('https://images.unsplash.com/photo-1529074963764-98f45c47344b?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&w=1000&q=80')
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
image = cv2.imdecode(arr, -1)
#display image:
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#preprocessing
img = cv2.resize(image, (32, 32), 3).reshape((1,32,32,3)).astype('float')/255.0
p = labelNames[model.predict(img).argmax()]
cv2.putText(image, p, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# with standard DECAY:
#define the set of callbacks to be passed to the model during training:
#callbacks = [LearningRateScheduler(step_decay)]

#Applying Monitoring traing model: 
#construct the set of callbacks:
figPath = f'/kaggle/working/{os.getpid()}.png'
jsonPath = f'/kaggle/working/{os.getpid()}.json'
callbacks = [TrainingMonitor(figPath=figPath, jsonPath=jsonPath)]
#Compiling network
print('Network compiling...')
sgd = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = arch.build(32, 32, 3, 10)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])

#train network:
print('Network Training...')
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=64,callbacks = callbacks, verbose=1)
