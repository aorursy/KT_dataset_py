import keras

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from keras import optimizers

from keras.callbacks import *

from keras.models import Model

from keras.layers import Input

from keras import backend as K

import matplotlib.pyplot as plt

from keras.regularizers import l2

from scipy import asarray as ar,exp

from scipy.optimize import curve_fit

from IPython.display import clear_output

from keras.layers.merge import concatenate

from keras.layers.convolutional import Conv2D

from sklearn.model_selection import train_test_split

from keras.layers.core import Dense, Dropout, Activation

from keras.backend.tensorflow_backend import set_session

from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D

%matplotlib inline
#To stop potential randomness

seed = 128

rng = np.random.RandomState(seed)



#set environment

config = tf.ConfigProto()

config.gpu_options.allow_growth = True #allows dynamic growth

config.gpu_options.visible_device_list = "0" #set GPU number

sess = tf.Session(config=config)

set_session(sess)
%%time

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]

target = train_df['target']
popt_set = []



def gaus(x,a,x0,sigma):

    return a*exp(-(x-x0)**2/(2*sigma**2))



for var in features:

    temp_df = pd.DataFrame(train_df[var])

    df = pd.DataFrame(temp_df[var].value_counts()).sort_index().reset_index()

    

    x = ar(range(len(df[var])))

    y = ar(df[var].values)

    

    n = len(x)                          #the number of data

    mean = sum(x*y)/sum(y)                   #note this correction

    sigma = np.sqrt(sum(y*(x-mean)**2)/sum(y))       #note this correction

    popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])

    popt_set.append(popt)

    

    df['fit'] = gaus(x,*popt)

    df['diff'] = df[var] - df['fit']

    df = df[['index','diff']]

    df = df.rename(index=str, columns={'index':var})

    temp_df = pd.merge(temp_df, df, on=var, how='left').fillna(0)

    train_df[var+"_diff"] = temp_df['diff']

    

    #For Test

    temp_test_df = pd.DataFrame(test_df[var])

    df2 = pd.DataFrame(temp_test_df[var].value_counts()).sort_index().reset_index()

    

    x2 = ar(range(len(df2[var])))

    df2['fit'] = gaus(x2, *popt)

    df2['diff'] = df2[var] - df2['fit']

    df2 = df2[['index','diff']]

    df2 = df2.rename(index=str, columns={'index':var})

    temp_test_df = pd.merge(temp_test_df, df, on=var, how='left').fillna(0)

    test_df[var+"_diff"] = temp_test_df['diff']

    

train_df.head()

    
new_features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
X_train = train_df[new_features]

X_test = test_df[new_features]

Y_train = train_df['target']



#Normalization did not particularly help and hence i left it out

#It would also end up reducing the var_diff variables to extrememly miniscule numbers

#X_train=(X_train-X_train.min())/(X_train.max()-X_train.min())

#X_test=(X_test-X_test.min())/(X_test.max()-X_test.min())



X_train = X_train.values

X_test = X_test.values

Y_train = Y_train.values



x_train_temp = []

for row in X_train:

    x_train_temp.append(np.reshape(row, (10, 10, 4)))

x_train_temp = np.array(x_train_temp)



x_test_temp = []

for row in X_test:

    x_test_temp.append(np.reshape(row, (10, 10, 4)))

x_test_temp = np.array(x_test_temp)



y_train = np.reshape(np.array(Y_train), (Y_train.shape[0], 1))



print(x_test_temp.shape)

print(x_test_temp.shape)

print(y_train.shape)
(x_train, x_val, Ytrain, Yval) = train_test_split(x_train_temp, y_train, test_size=0.3, random_state=120)

(x_train.shape, Ytrain.shape, x_val.shape, Yval.shape)
####### HYPERPARAMETERS

#L = 20   # L=16,12 with k=12 getting resource exhausted!!!???

#num_filters = 64

#dropout_rate =  0.2

#num_dense_block = 1



#this is a binary classification problem, hence single class output with sigmoid activation function in the last layer will do the task

#main

num_classes = 1 

dropout_rate = None

theta = 0.5

bottleneck = True

num_epochs = 50

batch_size = 16

validation_split = 0.3

shape = (10, 10, 4)
#conv block : BatchNorm, Act, Conv2D, BatchNorm, Act, Conv2D, Dropout

def convBlock(inputData, numFilters, kernelSize = (3, 3), strides = (1, 1), bottleneck = False, 

              dropoutRate = None, weightDecay=1e-4, actFunc='tanh'):    

    #no bottleneck: BN-A-Conv

    #bottleneck: BN-A-Conv1-BN-A-Conv

    x = BatchNormalization()(inputData)

    x = Activation(actFunc)(x)

    #x = Activation(actFunc)(inputData)

    if bottleneck:

        bottleneck_filters = 4*numFilters

        x = Conv2D(bottleneck_filters, kernel_size=(1, 1), padding='same', use_bias=False, kernel_regularizer=l2(weightDecay))(x)

        x = BatchNormalization()(x)

        x = Activation(actFunc)(x)

        

    x = Conv2D(numFilters, kernel_size=kernelSize, strides=strides, padding='same', use_bias=False, data_format='channels_last')(x)

    

    #add dropout if needed

    if dropoutRate:

        x = Dropout(dropoutRate)(x)

    return x
#dense block: #nLayers(convBlk, concat)

def denseBlock(x, numLayers, numFilters, growthRate, growNumFilters=True, kernelSize = (3, 3), strides = (1, 1), 

               bottleneck = True, dropoutRate = None, weightDecay=1e-4, actFunc='tanh'):

    num_filters = numFilters

    for i in range(numLayers):

        conv_blk = convBlock(x, numFilters=growthRate, kernelSize=kernelSize, strides=strides, bottleneck=bottleneck, dropoutRate=dropoutRate, weightDecay=weightDecay, actFunc=actFunc)

        x = concatenate([x, conv_blk], axis=-1)

        

        if growNumFilters:

            num_filters += growthRate

    return x, num_filters

#transition block: BatchNorm, Act, Conv2D, AvgPool

def transitionBlock(inputX, numFilters, theta=1.0, weightDecay=1e-4, actFunc='tanh'):

    num_filters = int(numFilters*theta)



    x = BatchNormalization()(inputX)

    x = Activation(actFunc)(x)

    #x = Activation(actFunc)(inputX)

    x = Conv2D(num_filters, kernel_size=(1, 1), padding='same', use_bias=False, kernel_regularizer=l2(weightDecay))(x)

    x = AveragePooling2D((2, 2), strides = (2, 2))(x)

    

    return x, num_filters
#create final densenet model

#DenseNet: Conv, MaxPool, #(dbs-1)(DenseBlk, TransBlk), DenseBlk, BatchNorm, Act, GAP, Dense

def denseNet(numClasses, inputData, numDenseBlks = 2, numFilters = 24, growthRate = 12,#12, 8 

             numLayersPerBlk = [8, 8], kernelSize = (2, 2), strides = (1, 1),

             bottleneck=True, dropoutRate=None, theta=1.0, weightDecay=1e-4, actFunc='relu'):

    num_filters = numFilters

    

    x = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same',

               strides=(2, 2), use_bias=False, kernel_regularizer=l2(weightDecay))(inputData)

    x = MaxPooling2D((3, 3), strides = (1, 1), padding='same')(x)

    for blk_num in range(numDenseBlks-1):

        x, num_filters = denseBlock(x, numLayersPerBlk[blk_num], num_filters, growthRate, kernelSize=kernelSize, strides=strides, dropoutRate=dropoutRate, weightDecay=weightDecay, actFunc=actFunc)

        x, num_filters = transitionBlock(x, num_filters, theta=theta, weightDecay=weightDecay, actFunc=actFunc)

    x, num_filters = denseBlock(x, numLayersPerBlk[-1], num_filters, growthRate, kernelSize=kernelSize, strides=strides, dropoutRate=dropoutRate, weightDecay=weightDecay, actFunc=actFunc)

    x = BatchNormalization()(x)

    x = Activation(actFunc)(x)

    

    x = GlobalAveragePooling2D()(x)

    

    x = Dense(numClasses, activation='sigmoid')(x)

    

    return x
input_data = Input(shape = shape)

output_data = denseNet(num_classes, input_data, bottleneck=bottleneck, dropoutRate=dropout_rate, theta=theta)

model = Model(inputs = input_data, outputs = output_data)

model.summary()
class PlotLearning(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        self.acc = []

        self.val_acc = []

        self.fig = plt.figure()

        

        self.logs = []



    def on_epoch_end(self, epoch, logs={}):

        

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.acc.append(logs.get('acc'))

        self.val_acc.append(logs.get('val_acc'))

        self.i += 1

        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        

        clear_output(wait=True)

        

        ax1.set_yscale('log')

        ax1.plot(self.x, self.losses, label="loss")

        ax1.plot(self.x, self.val_losses, label="val_loss")

        ax1.legend()

        

        ax2.plot(self.x, self.acc, label="accuracy")

        ax2.plot(self.x, self.val_acc, label="validation accuracy")

        ax2.legend()

        

        plt.show();

        

plot = PlotLearning()
class CyclicLR(Callback):

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



        
def load_best_model():

    tmp_file_name = os.listdir('.')

    acc_ind = {}

    acc_list = []

    for i, file in enumerate(tmp_file_name):

        if file[:5] == "weigh" :

            acc = int((file.split('.')[1]).split('-')[0])

            acc_list.append(acc)

            acc_ind[acc] = i

    max_acc = max(acc_list)

    print(tmp_file_name[acc_ind[max_acc]])

    return tmp_file_name[acc_ind[max_acc]]

clr = lambda x: 1/(5**(x*0.0001))

clr_triangular = CyclicLR( max_lr=0.004, scale_fn=clr, scale_mode='cycle')



#model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

nadam = optimizers.Nadam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

model.compile(optimizer=nadam, loss='binary_crossentropy', metrics=['accuracy'])



filepath="weights-improvement-{epoch:02d}-{val_acc:.5f}-{val_loss:.5f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model.fit(x_train, Ytrain, batch_size=8, epochs=10, callbacks=[clr_triangular, plot, checkpoint], validation_data=(x_val, Yval), shuffle=True)

!ls
model.load_weights(load_best_model())
#model.load_weights('weights-improvement-06-0.90785-0.25704.hdf5')

score = model.evaluate(x_val, Yval, batch_size=128)#np.zeros((20000, 10, 10, 4))

print(score)
final_res = model.predict(x_test_temp)
sub = pd.DataFrame({"ID_code": test_df.ID_code.values})

sub["target"] = final_res

sub.to_csv("submission1.csv", index=False)
!ls
#reduced learning rate and increased batch size

clr = lambda x: 1/(5**(x*0.0001))

clr_triangular = CyclicLR( max_lr=0.004, scale_fn=clr, scale_mode='cycle')



#model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

nadam = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

model.compile(optimizer=nadam, loss='binary_crossentropy', metrics=['accuracy'])



filepath="weights-improvement10-{epoch:02d}-{val_acc:.5f}-{val_loss:.5f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model.load_weights(load_best_model())

model.fit(x_train, Ytrain, batch_size=16, epochs=10, callbacks=[clr_triangular, plot, checkpoint], validation_data=(x_val, Yval), shuffle=True)

!ls
model.load_weights(load_best_model())
#model.load_weights('weights-improvement-06-0.90785-0.25704.hdf5')

score = model.evaluate(x_val, Yval, batch_size=128)#np.zeros((20000, 10, 10, 4))

print(score)
final_res = model.predict(x_test_temp)
sub = pd.DataFrame({"ID_code": test_df.ID_code.values})

sub["target"] = final_res

sub.to_csv("submission2.csv", index=False)
!ls
#reduced learning rate and increased batch size

clr = lambda x: 1/(5**(x*0.0001))

clr_triangular = CyclicLR( max_lr=0.004, scale_fn=clr, scale_mode='cycle')



#model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

nadam = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

model.compile(optimizer=nadam, loss='binary_crossentropy', metrics=['accuracy'])



filepath="weights-improvement30-{epoch:02d}-{val_acc:.5f}-{val_loss:.5f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model.load_weights(load_best_model())

model.fit(x_train, Ytrain, batch_size=128, epochs=10, callbacks=[clr_triangular, plot, checkpoint], validation_data=(x_val, Yval), shuffle=True)

!ls
model.load_weights(load_best_model())
#model.load_weights('weights-improvement-06-0.90785-0.25704.hdf5')

score = model.evaluate(x_val, Yval, batch_size=128)#np.zeros((20000, 10, 10, 4))

print(score)
final_res = model.predict(x_test_temp)
sub = pd.DataFrame({"ID_code": test_df.ID_code.values})

sub["target"] = final_res

sub.to_csv("submission3.csv", index=False)