# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import os

from keras.layers import Dense, Convolution2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout

from keras.models import Sequential

from nltk import word_tokenize

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dropout

from keras.layers import Flatten

from keras.constraints import maxnorm

from keras.optimizers import SGD , Adam

from keras.layers import Conv2D , BatchNormalization

from keras.layers import MaxPooling2D

from keras.utils import np_utils

from keras import backend as K

df = pd.read_csv('../input/fer2013.csv')

print(df.shape)
print(df.columns)
df.head()
pixels = df.loc[:,'pixels'].values



print(pixels.shape)

print(type(pixels))
px = []

for x in pixels : 

    x = word_tokenize(x)

    x = [float(t) for t in x]

    px.append(x)
x = np.array(px)

print(x.shape)
y = df.loc[:, 'emotion'].values

print(y.shape)

print(type(y))
for ix in range(5,10):

    plt.figure(ix)

    plt.imshow(x[ix].reshape((48, 48)), interpolation='none', cmap='gray')

plt.show()

x = x / 255

x_train = x[0:28710,:]

y_train = y[0:28710]

print (x_train.shape, y_train.shape)

x_val = x[28710:32300,:]

y_val = y[28710:32300]

print (x_val.shape, y_val.shape)

from keras.utils import np_utils



x_train = x_train.reshape((x_train.shape[0],48, 48,1 ))

x_crossval = x_val.reshape((x_val.shape[0],48, 48,1))

print (y.shape)

yy = np_utils.to_categorical(y, 7)

print (yy.shape)

y_train = yy[:28710]

y_crossval = yy[28710:32300]

print (x_crossval.shape, y_crossval.shape)



datagen = ImageDataGenerator(

        featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        zca_whitening=False,  

        rotation_range=10,  

        zoom_range = 0.0,  

        width_shift_range=0.1,  

        height_shift_range=0.1,  

        horizontal_flip=False, 

        vertical_flip=False)  



datagen.fit(x_train)

##https://github.com/bckenstler/CLR

import keras



class CyclicLR(keras.callbacks.Callback):

    """This callback implements a cyclical learning rate policy (CLR).

    The method cycles the learning rate between two boundaries with

    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).

    The amplitude of the cycle can be scaled on a per-iteration or 

    per-cycle basis.

    This class has three built-in policies, as put forth in the paper.

    "triangular":

        A basic triangular cycle w/ no amplitude scaling.

    "triangular2":

        A basic triangular cycle that scales initial amplitude by half each cycle.

    "exp_range":

        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 

        cycle iteration.

    For more detail, please see paper.

    

    # Example

        ```python

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., mode='triangular')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```

    

    Class also supports custom scaling functions:

        ```python

            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., scale_fn=clr_fn,

                                scale_mode='cycle')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```    

    # Arguments

        base_lr: initial learning rate which is the

            lower boundary in the cycle.

        max_lr: upper boundary in the cycle. Functionally,

            it defines the cycle amplitude (max_lr - base_lr).

            The lr at any cycle is the sum of base_lr

            and some scaling of the amplitude; therefore 

            max_lr may not actually be reached depending on

            scaling function.

        step_size: number of training iterations per

            half cycle. Authors suggest setting step_size

            2-8 x training iterations in epoch.

        mode: one of {triangular, triangular2, exp_range}.

            Default 'triangular'.

            Values correspond to policies detailed above.

            If scale_fn is not None, this argument is ignored.

        gamma: constant in 'exp_range' scaling function:

            gamma**(cycle iterations)

        scale_fn: Custom scaling policy defined by a single

            argument lambda function, where 

            0 <= scale_fn(x) <= 1 for all x >= 0.

            mode paramater is ignored 

        scale_mode: {'cycle', 'iterations'}.

            Defines whether scale_fn is evaluated on 

            cycle number or cycle iterations (training

            iterations since start of cycle). Default is 'cycle'.

    """



    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',

                 gamma=1., scale_fn=None, scale_mode='cycle'):

        super(CyclicLR, self).__init__()



        self.base_lr = base_lr

        self.max_lr = max_lr

        self.step_size = step_size

        self.mode = mode

        self.gamma = gamma

        if scale_fn == None:

            if self.mode == 'triangular':

                self.scale_fn = lambda x: 1.

                self.scale_mode = 'cycle'

            elif self.mode == 'triangular2':

                self.scale_fn = lambda x: 1/(2.**(x-1))

                self.scale_mode = 'cycle'

            elif self.mode == 'exp_range':

                self.scale_fn = lambda x: gamma**(x)

                self.scale_mode = 'iterations'

        else:

            self.scale_fn = scale_fn

            self.scale_mode = scale_mode

        self.clr_iterations = 0.

        self.trn_iterations = 0.

        self.history = {}



        self._reset()



    def _reset(self, new_base_lr=None, new_max_lr=None,

               new_step_size=None):

        """Resets cycle iterations.

        Optional boundary/step size adjustment.

        """

        if new_base_lr != None:

            self.base_lr = new_base_lr

        if new_max_lr != None:

            self.max_lr = new_max_lr

        if new_step_size != None:

            self.step_size = new_step_size

        self.clr_iterations = 0.

        

    def clr(self):

        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))

        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)

        if self.scale_mode == 'cycle':

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)

        else:

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

        

    def on_train_begin(self, logs={}):

        logs = logs or {}



        if self.clr_iterations == 0:

            K.set_value(self.model.optimizer.lr, self.base_lr)

        else:

            K.set_value(self.model.optimizer.lr, self.clr())        

            

    def on_batch_end(self, epoch, logs=None):

        

        logs = logs or {}

        self.trn_iterations += 1

        self.clr_iterations += 1



        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.trn_iterations)



        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        

        K.set_value(self.model.optimizer.lr, self.clr())
from keras.layers import Dropout

from keras.layers import Flatten

from keras.constraints import maxnorm

from keras.optimizers import Adam

from keras.layers import Conv2D , BatchNormalization

from keras.layers import MaxPooling2D

from keras.utils import np_utils

from keras import backend as K

from keras import layers

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras.models import Model

from keras.initializers import glorot_uniform



from keras.layers import SpatialDropout2D

from keras import regularizers





model=Sequential()



model.add(Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=(48,48,1)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(SpatialDropout2D(0.5))





model.add(Conv2D(128, (3, 3),activation='relu',padding='same'))

model.add(Conv2D(128, (3, 3),activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(SpatialDropout2D(0.5))



model.add(Conv2D(256, (3, 3),activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.09)))

model.add(Conv2D(256, (3, 3),activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.09)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(SpatialDropout2D(0.5))







model.add(Flatten())



model.add(Dense(256,activation = 'relu',kernel_regularizer=regularizers.l2(0.09)))

model.add(BatchNormalization())

model.add(Dropout(0.7))



model.add(Dense(256,activation = 'relu',kernel_regularizer=regularizers.l2(0.09)))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(7,activation = 'softmax'))



print(model.summary())
clr_triangular = CyclicLR(mode='triangular2',step_size = 200)

opt = Adam(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])



history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=64),

                    validation_data=(x_crossval, y_crossval),

                    steps_per_epoch=x.shape[0] // 64,

                    callbacks=[clr_triangular],

                    epochs = 1800)

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
