# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from sklearn.model_selection import train_test_split

from keras.callbacks import Callback
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn.metrics import roc_auc_score, precision_recall_curve,average_precision_score
data = pd.read_csv("../input/creditcard.csv")

# Any results you write to the current directory are saved as output.
data.shape
data.head()
# drop "time"
data.drop(['Time'],axis=1,inplace=True)
# Split target var from data
target = data['Class']
data.drop(['Class'],axis=1,inplace=True)
#Split train and test set.
train_x,test_x,train_y,test_y=train_test_split(data,target,test_size=0.2,stratify=target)
for col in train_x.columns:
    avg = np.mean(train_x[col])
    std = np.std(train_x[col])
    train_x.loc[:,col] = (train_x[col]-avg)/std
    test_x.loc[:,col] = (test_x[col] - avg)/std
nn_params = {
            'batch_size':128,
            'nb_epoch':10,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'class_weight':{0:0.1,1:0.2},#{0:0.0396, 1:0.9604},
            'normalize':False,#Whether to notmalize
            'categorize_y':True
            }
def build_model(nn_input_dim_NN):
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(nn_input_dim_NN,)))
    model.add(Dense(input_dim=nn_input_dim_NN, output_dim=120, init='uniform'))
    model.add(LeakyReLU(alpha=.00001))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=120,output_dim=280, init='uniform'))
    model.add(LeakyReLU(alpha=.00001))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=280,output_dim=100, init='uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(input_dim=100,output_dim=2, init='uniform', activation='softmax'))    
    sgd = SGD(lr=0.015, decay=1e-6, momentum=0.9, nesterov=True)
    
#     model.compile(optimizer=sgd, loss='binary_crossentropy',class_mode='binary')   
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    return model
class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            #logging.info("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
            print ("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
class NNWrapper(object):
    def __init__(self,nn,batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
            validation_split=0., validation_data=None,shuffle=True,
            show_accuracy=False, class_weight=None, normalize=False, categorize_y=False):
        self.nn = nn
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.shuffle = shuffle
        self.class_weight = class_weight
        self.normalize = normalize
        self.categorize_y = categorize_y
        #set initial weights
        self.init_weight = self.nn.get_weights()    
    def train(self, X, y, validation_data=None):
        if self.normalize:
            self.mean = np.mean(X,axis=0)
            self.std = np.std(X,axis=0) + 1 #CAUSION!!!
            X = (X - self.mean)/self.std
        if self.categorize_y:
            #Converts a class vector (integers) to binary class matrix.
            y = np_utils.to_categorical(y)
        if validation_data != None:
            self.validation_data = validation_data
            if self.normalize:
                self.validation_data[0] = (validation_data[0] - self.mean)/self.std
            if self.categorize_y:
                self.validation_data[1] = np_utils.to_categorical(validation_data[1])        
        
        #set initial weights
#         self.nn.set_weights(self.init_weight)

        #set callbacks
        if validation_data is None:
            self.callbacks = [IntervalEvaluation(validation_data=(X, y), interval=10)]
        else:
            self.callbacks = [IntervalEvaluation(validation_data=(self.validation_data[0], 
                                                                  self.validation_data[1]), interval=10)]
        return self.nn.fit(X, y, batch_size=self.batch_size, nb_epoch=self.nb_epoch, verbose=self.verbose, callbacks=self.callbacks, 
                           validation_split=self.validation_split, validation_data=self.validation_data, shuffle=self.shuffle, class_weight=self.class_weight)
        
    def predict(self, X, batch_size=128, verbose=1):
        if self.normalize:
            X = (X - self.mean)/self.std
        return self.nn.predict_proba(X, batch_size=batch_size, verbose=verbose)[:,1]
structure = build_model(train_x.shape[1])
nn_model = NNWrapper(nn=structure,**nn_params)
nn_model.train(train_x,train_y)
pred = nn_model.predict(test_x)
precision, recall, _ = precision_recall_curve(test_y, pred)
average_precision_score(test_y,pred)
plt.plot(precision,recall, lw=2, color='red') 
plt.xlabel('precision')
plt.ylabel('recall')
