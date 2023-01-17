# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

from numpy import argmax

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re



# additional machine learning dependencies

import tensorflow as tf

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM

from keras import optimizers

from keras.optimizers import SGD

from keras.optimizers import Adam

from keras.constraints import maxnorm

from keras.utils import np_utils

from keras.utils import to_categorical



from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.metrics import roc_curve

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc

from sklearn.model_selection import RandomizedSearchCV

from keras.callbacks import ModelCheckpoint



from sklearn.ensemble import RandomForestClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/cleaned_data')

data = data.loc[:,'Grade':'Holds']

data['Holds'] = data['Holds'].str[1:]

data['Holds'] = data['Holds'].str.split(' ')

data['Grade'] = data['Grade'].str.replace(' ','')

data['Grade'].value_counts()
# eliminating grades we have little data on

data = data[data['Grade'] != '6A']

data = data[data['Grade'] != '6A+']

data = data[data['Grade'] != '8A+']

data = data[data['Grade'] != '8B']

data = data[data['Grade'] != '8B+']

data = data.reset_index()

del data['index']



data['Grade'].value_counts()
# Creating HOLD-TO-INDEX dictionary

letter_dic = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10}

number_dic = {'18':0,'17':1,'16':2,'15':3,'14':4,'13':5,'12':6,'11':7,'10':8,'9':9,'8':10,'7':11,'6':12,'5':13,'4':14,'3':15,'2':16,'1':17}

hold_to_position = {}



for letter in letter_dic.keys():

    for number in number_dic.keys():

        hold = str(letter) + str(number)

        hold_to_position[hold] = [letter_dic[letter],number_dic[number]]



data['Holds'] = np.array(data['Holds'].values)



hold_names = []

letters = ['A','B','C','D','E','F','G','H','I','J','K']

numbers = [18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]

for number in numbers:

    for letter in letters:

        hold = str(letter) + str(number)

        hold_names.append(hold)

        

['A18', 'B18', 'C18', 'D18', 'E18', 'F18', 'G18', 'H18', 'I18', 'J18', 'K18',

 'A17', 'B17', 'C17', 'D17', 'E17', 'F17', 'G17', 'H17', 'I17', 'J17', 'K17', 

 'A16', 'B16', 'C16', 'D16', 'E16', 'F16', 'G16', 'H16', 'I16', 'J16', 'K16', 

 'A15', 'B15', 'C15', 'D15', 'E15', 'F15', 'G15', 'H15', 'I15', 'J15', 'K15', 

 'A14', 'B14', 'C14', 'D14', 'E14', 'F14', 'G14', 'H14', 'I14', 'J14', 'K14', 

 'A13', 'B13', 'C13', 'D13', 'E13', 'F13', 'G13', 'H13', 'I13', 'J13', 'K13', 

 'A12', 'B12', 'C12', 'D12', 'E12', 'F12', 'G12', 'H12', 'I12', 'J12', 'K12', 

 'A11', 'B11', 'C11', 'D11', 'E11', 'F11', 'G11', 'H11', 'I11', 'J11', 'K11', 

 'A10', 'B10', 'C10', 'D10', 'E10', 'F10', 'G10', 'H10', 'I10', 'J10', 'K10', 

 'A9', 'B9', 'C9', 'D9', 'E9', 'F9', 'G9', 'H9', 'I9', 'J9', 'K9', 

 'A8', 'B8', 'C8', 'D8', 'E8', 'F8', 'G8', 'H8', 'I8', 'J8', 'K8', 

 'A7', 'B7', 'C7', 'D7', 'E7', 'F7', 'G7', 'H7', 'I7', 'J7', 'K7', 

 'A6', 'B6', 'C6', 'D6', 'E6', 'F6', 'G6', 'H6', 'I6', 'J6', 'K6', 

 'A5', 'B5', 'C5', 'D5', 'E5', 'F5', 'G5', 'H5', 'I5', 'J5', 'K5', 

 'A4', 'B4', 'C4', 'D4', 'E4', 'F4', 'G4', 'H4', 'I4', 'J4', 'K4', 

 'A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'H3', 'I3', 'J3', 'K3', 

 'A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2', 'J2', 'K2', 

 'A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'I1', 'J1', 'K1']





# encoding X (hold positions)

def flatten(input_data):

    empty_board = np.zeros([11,18])

    for hold in input_data:

        hold = hold.upper()

        loc = hold_to_position[hold]

        empty_board[loc[0],loc[1]] += 1.0

    return empty_board.flatten()



data['Inputs'] = data['Holds'].map(flatten)

X = np.vstack((row for row in data['Inputs']))

X = pd.DataFrame(X, columns=hold_names, dtype=float)

Y = data['Grade']





# encoding Y (grades)

encoder = preprocessing.LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)

Y = np_utils.to_categorical(encoded_Y)

for index, grade in data['Grade'].iteritems():

    data.loc[index,'Grade'] = encoded_Y[index]

    

# partitioning data into training and testing data (80/20)

x_train,x_test,y_train,y_test = train_test_split(

        X, Y, test_size=0.2)
x_train.shape
y_train.shape
input_size = X.shape[1]





def simple_nn():

    input_size = X.shape[1]

    

    model = Sequential()

    

    model.add(Dense(input_size,input_shape=(input_size,), kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(5)))

    model.add(Dropout(0.1))

    

    model.add(Dense(int(input_size/2),kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(5)))

    model.add(Dropout(0.1))

    

    model.add(Dense(50,kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(5)))

    model.add(Dropout(0.1))

    

    model.add(Dense(10, kernel_initializer='normal',activation='softmax'))

    model.add(Dropout(0.1))

    

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    

    return model





pipeline = KerasClassifier(simple_nn)

fitted_nn = pipeline.fit(x_train,y_train,epochs=25,batch_size=400,validation_split=0.2,verbose=1)





nn_test = pd.DataFrame(y_test)

nn_test = nn_test.idxmax(1)

nn_pred = pipeline.predict(x_test)

nn_df = pd.DataFrame({'Predicted':nn_pred,'True':nn_test})

score = accuracy_score(nn_df['True'],nn_df['Predicted'])

print('Accuracy Score: ',score)
# plotting accuracy metrics



# train/validation accuracy

plt.plot(fitted_nn.history['acc'])

plt.plot(fitted_nn.history['val_acc'])

plt.title('Simple NN Accuracy')

plt.ylabel('accuracy')

plt.xlabel('epochs')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



# heatmap of performance

nn_pivoted = nn_df.groupby(['Predicted','True']).size().unstack()

labels = ['6B+','6C','6C+','7A','7A+','7B','7B+','7C','7C+','8A']

sns.heatmap(nn_pivoted, annot=True, fmt="g", cmap='viridis', xticklabels=labels, yticklabels=labels)

plt.title('Simple NN Heatmap')

plt.show()



# loss

plt.plot(fitted_nn.history['loss'])

plt.plot(fitted_nn.history['val_loss'])

plt.title('Simple NN Loss')

plt.ylabel('loss')

plt.xlabel('epochs')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
def random_forest():

    rf = RandomForestClassifier(

            criterion='gini',

            n_estimators = 1,

            min_samples_split=2,

            min_samples_leaf=1,

            max_features='auto',

            max_depth=100,

            bootstrap=True,

            random_state=1)

    rf.fit(x_train,y_train)

    

    return rf



rf_pred = random_forest().predict(x_test)
# reversing the one-hot encoding to compare results

rf_pred = pd.DataFrame(rf_pred)

rf_pred = rf_pred.idxmax(1)

rf_df = pd.DataFrame({'Predicted':rf_pred,'True':nn_test})



# calculating accuracy score

score = accuracy_score(rf_df['True'],rf_df['Predicted'])

print('Accuracy Score: ',score)



# heatmap of performance

rf_pivoted = rf_df.groupby(['Predicted','True']).size().unstack()

labels = ['6B+','6C','6C+','7A','7A+','7B','7B+','7C','7C+','8A']

sns.heatmap(rf_pivoted, annot=True, fmt="g", cmap='viridis', xticklabels=labels, yticklabels=labels)

plt.title('Random Forest Heatmap')

plt.show()
# reshaping input data for RNN (LSTMs expect 3-D input)

x_train = np.array(x_train)

x_test = np.array(x_test)



x_train = np.reshape(x_train,(x_train.shape[0],1,x_train.shape[1]))

x_test = np.reshape(x_test,(x_test.shape[0],1,x_test.shape[1]))
# building recurrent neural network



def recurrent_nn(): 

    

    model = Sequential()

    

    model.add(LSTM(198, input_shape=(1,198), activation='relu', return_sequences=True))

    model.add(Dropout(0.1))

    

    model.add(LSTM(198, input_shape=(1,198), activation='relu', return_sequences=False))

    model.add(Dropout(0.1))

    

    #model.add(Dense(int(198/2),kernel_initializer='normal', activation = 'relu', kernel_constraint=maxnorm(3)))

    #model.add(Dropout(0.1))

    

    #model.add(Dense(50,kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(3)))

    #model.add(Dropout(0.1));

    

    model.add(Dense(50,kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(3)))

    model.add(Dropout(0.1))

    

    model.add(Dense(10, kernel_initializer='normal',activation='softmax'))

    model.add(Dropout(0.1))

    

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



    return model





# fitting model

pipeline = KerasClassifier(recurrent_nn)

fitted_rnn = pipeline.fit(x_train,y_train,epochs=50,batch_size=400,validation_split=0.2,verbose=1)

rnn_pred = pipeline.predict(x_test)



# reversing one-hot encoding for accuracy metrics by building pandas dataframe

rnn_test = pd.DataFrame(y_test)

rnn_test = rnn_test.idxmax(1)

rnn_df = pd.DataFrame({'Predicted':rnn_pred,'True':rnn_test})



# calculating validation accuracy score

score = accuracy_score(rnn_df['True'],rnn_df['Predicted'])

print('Accuracy Score: ',score)
# plotting train/validation accuracy

plt.plot(fitted_rnn.history['acc'])

plt.plot(fitted_rnn.history['val_acc'])

plt.title('RNN Accuracy')

plt.ylabel('accuracy')

plt.xlabel('epochs')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



# heatmap of performance

nn_pivoted = nn_df.groupby(['Predicted','True']).size().unstack()

labels = ['6B+','6C','6C+','7A','7A+','7B','7B+','7C','7C+','8A']

sns.heatmap(nn_pivoted, annot=True, fmt="g", cmap='viridis', xticklabels=labels, yticklabels=labels)

plt.title('RNN Heatmap')

plt.show()



# plotting loss

plt.plot(fitted_rnn.history['loss'])

plt.plot(fitted_rnn.history['val_loss'])

plt.title('RNN Loss')

plt.ylabel('loss')

plt.xlabel('epochs')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
# searching for noise in dataset (climbs with 5 or less holds)

data[data['Holds'].map(len)==4]['Grade'].value_counts()