# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import datetime

import warnings

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# importing seaborn and matplotlabsib



%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt



warnings.filterwarnings("ignore")
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train.describe()
test.describe()
train.info()
test.info()
train['target'].head()
sns.distplot(train['target'])

plt.show()
train['target'].value_counts().plot(kind="bar")

plt.show()
# importing the sklearn package



import sklearn

from sklearn import preprocessing



scaler = preprocessing.StandardScaler()
training = train.drop(['ID_code', 'target'], axis=1)

target = train['target']
testing = test.drop('ID_code', axis=1)
training.head()
target.head()
testing.head()
# scaling the training data

training_scaled = scaler.fit_transform(training)



# scaling the testing data

testing_scaled = scaler.fit_transform(testing)
print("Shape of the training data: {}".format(training_scaled.shape))

print("Shape of the testing data: {}".format(testing_scaled.shape))
# calling the garbage collector

import gc



gc.collect()
# importing packages



import keras

from keras import layers

from keras import utils

from keras import models



from keras.layers.core import (Dense, Activation, Flatten, Dropout)

from keras.layers import BatchNormalization

from keras.models import (Sequential, Model)

from keras import optimizers
# importing the misc., properties



BATCH_SIZE = 64

NP_EPOCHS = 50

NP_CLASSES = 1

VERBOSE = 1

VALIDATION_SPLIT = 0.2



# taking the instance as RMSProp

optimizer = optimizers.RMSprop()
# creating the generator the generator code has been taken from https://www.kaggle.com/mathormad/knowledge-distillation-with-nn-rankgauss



def mixup_data(x, y, alpha=1.0):

    if alpha > 0:

        lam = np.random.beta(alpha, alpha)

    else:

        lam = 1



    sample_size = x.shape[0]

    index_array = np.arange(sample_size)

    np.random.shuffle(index_array)



    mixed_x = lam * x + (1 - lam) * x[index_array]

    mixed_y = (lam * y) + ((1 - lam) * y[index_array])

    return mixed_x, mixed_y





def make_batches(size, batch_size):

    nb_batch = int(np.ceil(size / float(batch_size)))

    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]





def batch_generator(X, y, batch_size=128, shuffle=True, mixup=False):

    y = np.array(y)

    sample_size = X.shape[0]

    indexed = np.arange(sample_size)



    while True:

        if shuffle:

            np.random.shuffle(indexed)

        batches = make_batches(sample_size, batch_size)

        for batch_index, (batch_start, batch_end) in enumerate(batches):

            batch_ids = indexed[batch_start:batch_end]

            x_batch = X[batch_ids]

            y_batch = y[batch_ids]



            if mixup:

                x_batch, y_batch = mixup_data(x_batch, y_batch, alpha=1.0)

            yield x_batch, y_batch
# creating the checkpoints



from keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau, EarlyStopping)

from keras.layers import LeakyReLU



NP_OUTPUT_FUNCTION = "sigmoid"

DROPOUT_FIRST = 0.25

DROPOUT_SECOND = 0.20

NP_INPUT_SHAPE = training_scaled.shape[1]



print("Input shape has been taken as: {}".format(NP_INPUT_SHAPE))
# building the model



model = Sequential()

model.add(Dense(256, input_shape=(NP_INPUT_SHAPE,)))

model.add(LeakyReLU(alpha=0.02))

model.add(Dropout(DROPOUT_FIRST))

model.add(BatchNormalization())



model.add(Dense(128))

model.add(LeakyReLU(alpha=0.02))

model.add(Dropout(DROPOUT_FIRST))

model.add(BatchNormalization())



model.add(Dense(64))

model.add(LeakyReLU(alpha=0.02))

model.add(Dropout(DROPOUT_SECOND))

model.add(BatchNormalization())



model.add(Dense(32))

model.add(LeakyReLU(alpha=0.02))

model.add(Dropout(DROPOUT_SECOND))

model.add(BatchNormalization())



model.add(Dense(16))

model.add(LeakyReLU(alpha=0.02))

model.add(Dropout(DROPOUT_SECOND))

model.add(BatchNormalization())



model.add(Dense(8))

model.add(LeakyReLU(alpha=0.02))

model.add(Dropout(DROPOUT_SECOND))

model.add(BatchNormalization())



model.add(Dense(1))

model.add(Activation(NP_OUTPUT_FUNCTION))
# printing the summary of the model to see the trainable and non trainable parameters



model.summary()




current_dt_time = datetime.datetime.now()

model_name = 'model_init' + '_' + str(current_dt_time).replace(' ', '').replace(':', '_') + '/'



if not os.path.exists(model_name):

    os.mkdir(model_name)

    

file_path = model_name + "model-{epoch:05d}-{loss:.5f}-{val_auc:.5f}-{val_loss:.5f}-{val_auc:.5f}.h5"
!ls -lrt
## creating call back methods



## Call back method to calculate ROC AUC - We can found optins from: https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras

## We can calculate the ROC AUC for mini batches - so we can calculate the ROC AUC score at the end of each epoch by using callbacks method

     

from keras.callbacks import Callback

from sklearn import metrics





class findROC(Callback):

    def __init__(self, etraining, evalidation):

        # for training

        self.x_train = etraining[0]

        self.y_train = etraining[1]



        # for validation

        self.x_val = evalidation[0]

        self.y_val = evalidation[1]



    def on_train_begin(self, logs=None):

        return



    def on_batch_end(self, batch, logs=None):

        return



    def on_epoch_begin(self, epoch, logs=None):

        return



    def on_epoch_end(self, epoch, logs=None):

        y_pred_training = self.model.predict(self.x_train)

        roc_score_training = metrics.roc_auc_score(self.y_train, y_pred_training)



        y_pred_validation = self.model.predict(self.x_val)

        roc_score_validation = metrics.roc_auc_score(self.y_val, y_pred_validation)



        print("Training RoC score found: {}, validation RoC score found: {}".format(roc_score_training,

                                                                                    roc_score_validation))

        return

    

    def on_batch_begin(self, batch, logs=None):

        return
# creating model checkpoint

checkpoint = ModelCheckpoint(filepath=file_path, 

                             monitor='val_loss', 

                             verbose=1, 

                             save_best_only=True, 

                             save_weights_only=False, 

                             mode='auto', 

                             period=1)



# early stopping

early = EarlyStopping(monitor='val_loss',

                      mode='auto',

                      patience=5,

                      verbose=1)





LR = ReduceLROnPlateau(monitor="val_loss",

                       factor=0.2,

                       patience=2,

                       min_lr=0.000001,

                       verbose=1,

                       cooldown=1)
# ROC & AUC metric to monitor

import tensorflow as tf

import keras.backend as B



def auc(y_true, y_pred):

    score = tf.metrics.auc(y_true, y_pred)[1]

    B.get_session().run(tf.local_variables_initializer())

    return score
# creating the optimizer



optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.0)
# compiling the model



model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', auc])
# splitting the data into train and test using sklearn library

from sklearn import model_selection



X_train, X_val, y_train, y_val = model_selection.train_test_split(training_scaled, target, test_size=0.20, random_state=123456789)
print("X training data shape: {}".format(X_train.shape))

print("y training data shape: {}".format(y_train.shape))
print("X validation data shape: {}".format(X_val.shape))

print("y validation data shape: {}".format(y_val.shape))
# calculating the number of training and validation steps per epoch



num_training_seq = len(X_train)

num_validation_seq = len(X_val)



print("# training sequences: {}".format(num_training_seq))

print("# validation sequences: {}".format(num_validation_seq))
if(num_training_seq % BATCH_SIZE) == 0:

    training_steps_per_epoch = int(num_training_seq / BATCH_SIZE)

else:

    training_steps_per_epoch = int(num_training_seq / BATCH_SIZE) + 1
if(num_validation_seq % BATCH_SIZE) == 0:

    validation_steps_per_epoch = int(num_validation_seq / BATCH_SIZE)

else:

    validation_steps_per_epoch = int(num_validation_seq / BATCH_SIZE) + 1
print("Number of training steps are required for epoch: {}".format(training_steps_per_epoch))

print("Number of validation steps are requried for epoch: {}".format(validation_steps_per_epoch))
# fitting the model



history = model.fit_generator(generator=batch_generator(X_train, y_train, BATCH_SIZE),

                             validation_data=batch_generator(X_val, y_val, BATCH_SIZE),

                             epochs=NP_EPOCHS,

                             verbose=VERBOSE,

                             steps_per_epoch=training_steps_per_epoch,

                             validation_steps=validation_steps_per_epoch,

                             class_weight=None,

                             initial_epoch=0,

                             # callbacks=[checkpoint, early, LR, findROC(etraining=(X_train, y_train), evalidation=(X_val, y_val))])

                             callbacks=[checkpoint, LR, findROC(etraining=(X_train, y_train), evalidation=(X_val, y_val))])