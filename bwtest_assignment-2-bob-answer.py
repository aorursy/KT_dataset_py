from __future__ import absolute_import

from __future__ import print_function

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import pandas as pd

import numpy as np

from keras.utils import np_utils 

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras import optimizers

from keras import losses

from keras.models import load_model

from keras import regularizers

import time

from keras import initializers
#Load the training dataset ~87K states

all_train = pd.read_csv("../input/applied-ai-assignment-2/Assignment_2_train.csv")

all_train.loc[(all_train.state == 4),'state']=0

all_train.loc[(all_train.state == 5),'state']=1
len(all_train)
all_train[1:5]
#Create a train/validation split

data_to_use = 1

train=all_train[:int(len(all_train)*data_to_use)]

split = .9



Train = train[:int(len(train)*split)]

Valid = train[int(len(train)*split):]





#Remove the first and last column from the data, as it is the board name and the label

X_train = Train.iloc[:, 1:-1].values

X_valid = Valid.iloc[:, 1:-1].values



#Remove everything except the last column from the data, as it is the label and put it in y

y_train = Train.iloc[:, -1:].values

y_valid = Valid.iloc[:, -1:].values

len(X_train)
len(y_valid)
sample_train = X_train[50].reshape(-1,6,7)[0]

sample_train
import matplotlib.pyplot as plt

#plot the first image in the dataset

plt.imshow(sample_train)
#set input to the shape of one X value

dimof_input = X_train.shape[1]



# Set y categorical

dimof_output = int(np.max(y_train)+1)

y_train = np_utils.to_categorical(y_train, dimof_output)

y_valid = np_utils.to_categorical(y_valid, dimof_output)

y_valid
def initialize_model(layer1=16,layer2=0,layer=0,layer3=0,layer4=0,dropout1=0,dropout2=0,dropout3=0,dropout4=0,

                    activation1='relu',activation2='relu',activation3='relu',activation4='relu',

                    Optimizer = optimizers.Adam(learning_rate=0.001),

                    lossfunct = losses.categorical_crossentropy):

    

    layer_1 = layer1

    layer_2 = layer2

    layer_3 = layer3

    layer_4 = layer4

    dropout_1 = dropout1

    dropout_2 = dropout2

    dropout_3 = dropout3

    dropout_4 = dropout4

    activation_1 = activation1

    activation_2 = activation2

    activation_3 = activation3

    activation_4 = activation4

    optimizer = Optimizer

    loss_function = lossfunct

    

    glorot = initializers.glorot_normal(seed=None)

    

    mlp_model = Sequential()

    

    mlp_model.add(Dense(layer_1, input_dim=dimof_input, kernel_initializer=glorot, activation=activation_1))

    if dropout_1 > 0:

        mlp_model.add(Dropout(dropout_1))

        

    if layer_2 > 0:

        mlp_model.add(Dense(layer_2, input_dim=dimof_input, kernel_initializer=glorot, activation=activation_2))

        if dropout_2 > 0:

            mlp_model.add(Dropout(dropout_2))

        

    if layer_3 > 0:

        mlp_model.add(Dense(layer_3, input_dim=dimof_input, kernel_initializer=glorot, activation=activation_3))

        if dropout_3 > 0:

            mlp_model.add(Dropout(dropout_3))

        

    if layer_4 > 0:

        mlp_model.add(Dense(layer_4, input_dim=dimof_input, kernel_initializer=glorot, activation=activation_4))

        if dropout_4 > 0:

            mlp_model.add(Dropout(dropout_4))



    mlp_model.add(Dense(dimof_output, kernel_initializer=glorot, activation='softmax')) #do not change

    mlp_model.compile(loss=loss_function,   # **** pick any suggested loss functions

                      optimizer=optimizer, # **** pick any suggested optimizers

                      metrics=['accuracy']) #do not change



    return(mlp_model)
def save_output(layer1,dropout1,activation1,layer2,dropout2,activation2,layer3,

                dropout3,activation3,layer4,dropout4,activation4,Optimizer,pat,lossfunct,cur_output,score,time):

    if len(cur_output) == 0:

        columns = ['Lay1','DO1', 'Act1','Lay2','DO2', 'Act2','Lay3','DO3', 'Act3','Lay4','DO4', 'Act4','Opt','Loss','Pat','Score','Time']

        cur_output = pd.DataFrame(columns=columns)

    cur_output.loc[len(cur_output)] = [layer1,dropout1,activation1,layer2,dropout2,activation2,layer3,

                        dropout3,activation3,layer4,dropout4,activation4,Optimizer,pat,lossfunct,score,time]

    return(cur_output)
saved_output = []




sgd = optimizers.SGD(learning_rate=0.01) #default lr = 0.01

adagrad = optimizers.Adagrad(learning_rate=0.01) #default lr = 0.01

adadelta = optimizers.Adadelta(learning_rate=1.0) #default lr = 1.0

adam = optimizers.Adam(learning_rate=0.001) #default lr = 0.001

adamax = optimizers.Adamax(learning_rate=0.002) #default lr = 0.002

nadam = optimizers.Nadam(learning_rate=0.002) #default lr = 0.002





# Suggested loss functions

cat_cross = losses.categorical_crossentropy

mse = losses.mean_squared_error

binary = losses.binary_crossentropy



pat = 1

lay1 = 4096

DO1 = 0.5

lay2 = 0

DO2 = 0

lay3 = 0

DO3 = 0

lay4 = 0

DO4 = 0

lossf = mse



'''

for pat in [25,50,100]:

    for DO1 in [0.0,0.25,0.5,0.75]:







for lay4 in [0]:

    for lay3 in [0]:

        for lay2 in [0]:

            for lay1 in [8192]: 

'''

for lay1 in [20]:



                my_model = initialize_model(layer1=lay1,

                                            dropout1=DO1,

                                            activation1='relu',

                                            layer2=lay2,

                                            dropout2=DO2,

                                            activation2='relu',

                                            layer3=lay3,

                                            dropout3=DO3,

                                            activation3='relu',

                                            layer4=lay4,

                                            dropout4=DO4,

                                            activation4='relu',

                                            Optimizer = adam,

                                            lossfunct = lossf)

                start = time.time()

            

                es = EarlyStopping(monitor='val_loss', #do not change

                                   mode='min',  #do not change

                                   verbose=1, # allows you to see more info per epoch

                                   patience=pat) # **** patience is how many validations to wait with nothing learned (patience * validation_freq)



                mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True) #do not change





                history = my_model.fit(x=X_train, 

                                        y=y_train, 

                                        batch_size=7900, # **** set from 1 to length of training data

                                        epochs=10000, #do not change

                                        verbose=0, # allows you to see more info per epoch

                                        callbacks=[es, mc],

                                        validation_data=(X_valid,y_valid), 

                                        validation_freq=1,

                                        shuffle=True)



                #load the best model

                saved_model = load_model('best_model.h5')



                # evaluate the model

                _, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)

                _, valid_acc = saved_model.evaluate(X_valid, y_valid, verbose=0)

                print('Train: %.3f, Valid: %.3f' % (train_acc, valid_acc))

                runtime = (time.time()-start)

                print(runtime)

                if lossf == mse:

                    Lossf = 'mse'

                if lossf == cat_cross:

                    Lossf = 'cat_cross'

                if lossf == binary:

                    Lossf = 'binary'

                saved_output = save_output(lay1,DO1,'relu',lay2,DO2,'relu',lay3,

                            DO3,'relu',lay4,DO4,'relu','adam',Lossf,pat,saved_output,valid_acc,runtime)
saved_output.sort_values('Score',ascending=False)
pat = 1

lay1 = 32

DO1 = 0.5

lay2 = 0

DO2 = 0

lay3 = 0

DO3 = 0

lay4 = 0

DO4 = 0

lossf = mse



my_model = initialize_model(layer1=lay1,

                            dropout1=DO1,

                            activation1='relu',

                            layer2=lay2,

                            dropout2=DO2,

                            activation2='relu',

                            layer3=lay3,

                            dropout3=DO3,

                            activation3='relu',

                            layer4=lay4,

                            dropout4=DO4,

                            activation4='relu',

                            Optimizer = adam,

                            lossfunct = mse)

start = time.time()



es = EarlyStopping(monitor='val_loss', #do not change

                   mode='min',  #do not change

                   verbose=1, # allows you to see more info per epoch

                   patience=1) # **** patience is how many validations to wait with nothing learned (patience * validation_freq)



mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True) #do not change





history = my_model.fit(x=X_train, 

                        y=y_train, 

                        batch_size=7900, # **** set from 1 to length of training data

                        epochs=10000, #do not change

                        verbose=0, # allows you to see more info per epoch

                        callbacks=[es, mc],

                        validation_data=(X_valid,y_valid), 

                        validation_freq=1,

                        shuffle=True)



#load the best model

saved_model = load_model('best_model.h5')



# evaluate the model

_, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)

_, valid_acc = saved_model.evaluate(X_valid, y_valid, verbose=0)

print('Train: %.3f, Valid: %.3f' % (train_acc, valid_acc))

runtime = (time.time()-start)

print(runtime)
# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'valid'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'valid'], loc='upper left')

plt.show()
# Test for standard deviation of a model



pat = 1

lay1 = 32

DO1 = 0.5

lay2 = 0

DO2 = 0

lay3 = 0

DO3 = 0

lay4 = 0

DO4 = 0

lossf = mse





valid_list = []

for i in range(10):

    my_model = initialize_model(layer1=lay1,

                            dropout1=DO1,

                            activation1='relu',

                            layer2=lay2,

                            dropout2=DO2,

                            activation2='relu',

                            layer3=lay3,

                            dropout3=DO3,

                            activation3='relu',

                            layer4=lay4,

                            dropout4=DO4,

                            activation4='relu',

                            Optimizer = adam,

                            lossfunct = mse)

   



    start = time.time()



    es = EarlyStopping(monitor='val_loss', #do not change

                       mode='min',  #do not change

                       verbose=1, # allows you to see more info per epoch

                       patience=pat) # **** patience is how many validations to wait with nothing learned (patience * validation_freq)



    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True) #do not change



    history = my_model.fit(x=X_train, 

                            y=y_train, 

                            batch_size=7900, # **** set from 1 to length of training data

                            epochs=10000, #do not change

                            verbose=0, # allows you to see more info per epoch

                            callbacks=[es, mc],

                            validation_data=(X_valid,y_valid), 

                            validation_freq=1,

                            shuffle=True)



    #load the best model

    saved_model = load_model('best_model.h5')



    # evaluate the model

    _, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)

    _, valid_acc = saved_model.evaluate(X_valid, y_valid, verbose=0)

    valid_list.append(valid_acc)

    print('Train: %.3f, Valid: %.3f' % (train_acc, valid_acc))

    print(time.time()-start)

print('Average Score- '+ str(np.mean(valid_list)))

print('Standard deviation - '+ str(np.std(valid_list)))
# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'valid'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'valid'], loc='upper left')

plt.show()
#Go here when your saved model is ready

test = pd.read_csv("../input/applied-ai-assignment-2/Assignment_2_test.csv")

test.loc[(test.state == 4),'state']=0

test.loc[(test.state == 5),'state']=1

X_test = test.iloc[:, 1:-1].values

y_test = test.iloc[:, -1:].values

#creates the final output 

list_of_boards = [i for i in list(test['file_names'])]

result = saved_model.predict(X_test)

test_results = []

for i in result:

    test_results.append(np.argmax(i))

#Creates a dataframe that can be saved as a csv for submission

submission_data = pd.DataFrame(

    {'BoardId': list_of_boards,

     'Label': test_results

    })
submission_data[1:9]
submission_data.to_csv('submission.csv', sep=',',index=False)
saved_output = pd.read_csv("../input/sampleoutput/output.csv")
saved_output