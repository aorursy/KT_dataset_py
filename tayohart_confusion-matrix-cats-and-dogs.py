from tensorflow.python.keras.models import  Sequential

from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, GlobalMaxPooling2D, Dropout

from tensorflow.python.keras.regularizers import l2

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.losses import categorical_crossentropy

from tensorflow.python.keras.callbacks import LearningRateScheduler

from tensorflow.python.keras.applications import ResNet50

from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.optimizers import SGD

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/resnet50"))



# Any results you write to the current directory are saved as output.
data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = data_gen.flow_from_directory('../input/dogs-and-cats-fastai/dogscats/dogscats/train', batch_size=32, target_size=(299,299))

test_data = data_gen.flow_from_directory('../input/dogs-and-cats-fastai/dogscats/dogscats/valid', batch_size=32, target_size=(299,299))















#def linear_step_decay(epoch):

#    initial_lr = 0.1

#    k = 0.1

#    if initial_lr*(1 - epoch*k) <= 0:

#        lr = 0.01

#    else:

#        lr = initial_lr*(1 - epoch*k)

#    return lr



#lr = LearningRateScheduler(linear_step_decay)

    
#class LossHistory(Callback):

#    

#    def on_train_begin(self, logs={}):

#        self.losses = []

#        self.lr = []

#        

#    def on_epoch_end(self, batch, logs={}):

#        self.losses.append(logs.get('loss'))

#        self.lr.append(linear_step_decay(len(self.losses)))

        
#loss_history = LossHistory()

#callbacks_list = [lr, loss_history]
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(299,299,3)))

model.add(GlobalMaxPooling2D())

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dense(2, activation='softmax'))





model.summary()
model.compile(optimizer=SGD(1,0,0,False), loss=categorical_crossentropy, metrics=['accuracy'])
hist = model.fit_generator(train_data, epochs=5, validation_data=test_data)
import matplotlib.pyplot as plt
plt.plot(hist.history['acc'], label='accuracy')

plt.plot(hist.history['val_acc'], label='validation accuracy')

plt.plot(hist.history['loss'], label='loss')

plt.plot(hist.history['val_loss'], label='validation loss')

plt.legend()

plt.show()

plt.savefig('plt.png')

plt.savefig('plt.png')
import json

with open('file.json', 'w') as f:

    json.dump(hist.history, f)