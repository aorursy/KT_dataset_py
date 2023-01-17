# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import numpy as np
import sys
import sklearn
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from keras import backend as K

def load_X(X_signals_paths):
	"""
	Given attribute (train or test) of feature, read all 9 features into an
	np ndarray of shape [sample_sequence_idx, time_step, feature_num]
	    argument:   X_signals_paths str attribute of feature: 'train' or 'test'
	    return:     np ndarray, tensor of features
	"""
	X_signals = []

	for signal_type_path in X_signals_paths:
	    file = open(signal_type_path, 'rb')
	    # Read dataset from disk, dealing with text files' syntax
	    X_signals.append(
	        [np.array(serie, dtype=np.float32) for serie in [
	            str(row).replace('  ', ' ').replace('\\n','').replace('\\r','').replace('b','').replace('\'','').strip().split(' ') for row in file
	        ]]
	    )
	    file.close()

	return np.transpose(np.array(X_signals), (1, 2, 0))
# Load "y" (the neural network's training and testing outputs)


def one_hot(y):
    """convert label from dense to one hot
      argument:
        label: ndarray dense label ,shape: [sample_num,1]
      return:
        one_hot_label: ndarray  one hot, shape: [sample_num,n_class]
    """

    y = y.reshape(len(y))
    n_values = np.max(y) + 1
    return np.eye(n_values)[np.array(y, dtype=np.int32)]  # Returns FLOATS


def load_y(y_path):
    """
    Read Y file of values to be predicted
        argument: y_path str attibute of Y: 'train' or 'test'
        return: Y ndarray / tensor of the 6 one_hot labels of each sample
    """
    file = open(y_path, 'rb')
    # Read dataset from disk, dealing with text file's syntax
    
    y_ = np.array(
        [elem for elem in [
            str(row).replace('  ', ' ').replace('\\n','').replace('\\r','').replace('b','').replace('\'','').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return one_hot(y_ - 1)
	
###################################################################################
############ get X,Y train ,test###################################################
###################################################################################
def get_data(): 
	
	

	
	# Useful Constants

	# Those are separate normalised input features for the neural network
	INPUT_SIGNAL_TYPES = [
	    "body_acc_x_",
	    "body_acc_y_",
	    "body_acc_z_",
	    "body_gyro_x_",
	    "body_gyro_y_",
	    "body_gyro_z_",
	    "total_acc_x_",
	    "total_acc_y_",
	    "total_acc_z_"
	]

	# Output classes to learn how to classify
	LABELS = [
	    "WALKING",
	    "WALKING_UPSTAIRS",
	    "WALKING_DOWNSTAIRS",
	    "SITTING",
	    "STANDING",
	    "LAYING"
	]
	DATASET_PATH = "/kaggle/input/ucidataset/UCI HAR Dataset/"

	TRAIN = "train/"
	TEST = "test/"

	X_train_signals_paths = [
	    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
	]
	X_test_signals_paths = [
	    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
	]
	#dataset+DATA_PATH

	y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
	y_test_path = DATASET_PATH + TEST + "y_test.txt"

	x_train = load_X(X_train_signals_paths)
	x_test = load_X(X_test_signals_paths)
	print (x_train.shape)
	print (x_test.shape)

	y_train = load_y(y_train_path)
	y_test = load_y(y_test_path)
	print (y_train.shape)
	print (y_test.shape)

	y_true = np.argmax(y_test, axis=1)

	nb_classes = y_train.shape[1]
	input_shape=x_train.shape[1:]
	#output_directory='/home/sakkat/Desktop/Seminar/'
	
	return x_train, y_train, x_test, y_test


from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

x_train, y_train, x_test, y_test=get_data(); 
nb_classes = y_train.shape[1]
input_shape=x_train.shape[1:]
epoc = 100
from keras.layers import Flatten,Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

def build_simple(input_shape, nb_classes): 
    model = Sequential()
#     model.add(Reshape(input_shape[0]*input_shape[1],input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(576, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(nb_classes, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy',f1_m, precision_m, recall_m])
    return model 
modelsimple = build_simple(input_shape,nb_classes)


callback = ModelCheckpoint('/kaggle/working/SimpleNN{val_accuracy:.4f}.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

historysimple=modelsimple.fit(x_train, y_train, epochs=epoc,
                              verbose=1, validation_data=(x_test, y_test),shuffle=True,callbacks=[callback])
modelsimple.evaluate(x_test,y_test)
def build_CNN(input_shape, nb_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128*2, kernel_size=8, padding='same',kernel_regularizer=tf.keras.regularizers.l1(0.01),
                              activity_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
    conv1 = keras.layers.Dropout(rate=0.5)(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same',kernel_regularizer=tf.keras.regularizers.l1(0.01),
                              activity_regularizer=tf.keras.regularizers.l2(0.01))(conv1)
    conv2 = keras.layers.Dropout(rate=0.5)(conv2)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128*2, kernel_size=3,padding='same',kernel_regularizer=tf.keras.regularizers.l1(0.01),
                              activity_regularizer=tf.keras.regularizers.l2(0.01))(conv2)
    conv3 = keras.layers.Dropout(rate=0.5)(conv3)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
    metrics=['accuracy',f1_m, precision_m, recall_m])

    return model 
modelCNN=build_CNN(input_shape,nb_classes)
modelCNN.summary()
callback = tf.keras.callbacks.ModelCheckpoint('/kaggle/working/CNN{val_accuracy:.4f}.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

historyCNN=modelCNN.fit(x_train, y_train, epochs=epoc,
                              verbose=2, validation_data=(x_test, y_test),shuffle=True,callbacks=[callback])
modelCNN.evaluate(x_test,y_test,verbose=2)
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
from keras import optimizers
def build_RNN(input_shape,nb_classes):
    model = Sequential()
    model.add(SimpleRNN(100, input_shape = input_shape, return_sequences = True))
    model.add(Activation('relu'))

    model.add(SimpleRNN(100, return_sequences = False))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy',f1_m, precision_m, recall_m])
    
    return model

modelRNN = build_RNN(input_shape,nb_classes)
modelRNN.summary()
callback = tf.keras.callbacks.ModelCheckpoint('/kaggle/working/RNNSimple{val_accuracy:.4f}.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

historyRNN=modelRNN.fit(x_train, y_train, epochs=epoc,
                              verbose=1, validation_data=(x_test, y_test))
modelRNN.evaluate(x_test,y_test)
from keras.layers import LSTM, Masking,Activation 
def build_modelLSTM(input_shape, nb_classes):

    model = Sequential()
    model.add(LSTM(50, input_shape = input_shape, return_sequences = True))
    model.add(LSTM(50, return_sequences = False))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy',f1_m, precision_m, recall_m])
    return model    
modelLSTM = build_modelLSTM(input_shape,nb_classes)
modelLSTM.summary()

historyLSTM=modelLSTM.fit(x_train, y_train, epochs=epoc,
                              verbose=1, validation_data=(x_test, y_test))
modelLSTM.evaluate(x_test,y_test)
def build_modelRES(input_shape, nb_classes):
    n_feature_maps = 256

    input_layer = keras.layers.Input(input_shape)

    # BLOCK 1

    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same',kernel_regularizer=tf.keras.regularizers.l1(0.01),
                              activity_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
    conv_x = keras.layers.Dropout(rate=0.5)(conv_x)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same',kernel_regularizer=tf.keras.regularizers.l1(0.01),
                              activity_regularizer=tf.keras.regularizers.l2(0.01))(conv_x)
    conv_y = keras.layers.Dropout(rate=0.5)(conv_y)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same',kernel_regularizer=tf.keras.regularizers.l1(0.01),
                              activity_regularizer=tf.keras.regularizers.l2(0.01))(conv_y)
    conv_z = keras.layers.Dropout(rate=0.6)(conv_z)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=2, padding='same',kernel_regularizer=tf.keras.regularizers.l1(0.01),
                              activity_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)



    # BLOCK 2

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same',kernel_regularizer=tf.keras.regularizers.l1(0.01),
                              activity_regularizer=tf.keras.regularizers.l2(0.01))(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same',kernel_regularizer=tf.keras.regularizers.l1(0.01),
                              activity_regularizer=tf.keras.regularizers.l2(0.01))(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same',kernel_regularizer=tf.keras.regularizers.l1(0.01),
                              activity_regularizer=tf.keras.regularizers.l2(0.01))(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=2, padding='same',kernel_regularizer=tf.keras.regularizers.l1(0.01),
                              activity_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(output_block_1)
    

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)

    # FINAL

    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy',f1_m, precision_m, recall_m])
    return model 
modelRES =build_modelRES(input_shape,nb_classes)
modelRES.summary()
callback = tf.keras.callbacks.ModelCheckpoint('/kaggle/working/CNNRES{val_accuracy:.4f}.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
historyRES=modelRES.fit(x_train, y_train, epochs=epoc,
                              verbose=1, validation_data=(x_test, y_test),shuffle=True,callbacks=[callback])
modelRES.evaluate(x_test,y_test)
from keras.layers import Bidirectional
def build_biLSTM(input_shape,nb_classes): 
    model = Sequential()
    model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
              metrics=['accuracy',f1_m, precision_m, recall_m])
    
    return model 

modelbiLSTM = build_biLSTM(input_shape,nb_classes)
modelbiLSTM.summary()
callback = tf.keras.callbacks.ModelCheckpoint('/kaggle/working/BiLSTM{val_accuracy:.4f}.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
historybiLSTM=modelbiLSTM.fit(x_train, y_train, epochs=epoc,
                              verbose=1, validation_data=(x_test, y_test),callbacks=[callback])
modelbiLSTM.evaluate(x_test,y_test)
from keras.layers.merge import add
from keras.models import Model
from keras.layers import LSTM, Lambda,Input,Activation,BatchNormalization

def build_RESLSTM(input_shape,nb_classes): 
    input_layer = Input(shape=input_shape)
    #bock1
    reluin =Activation('relu')(input_layer)
    lstm1= LSTM(10,return_sequences=True) (reluin)
    relout =Activation('relu')(lstm1)
    lstm2= LSTM(10,return_sequences=True) (relout)
    relout2 =Activation('relu')(lstm2)
    #Residual connection
    res1=add([relout,relout2])
    
    outBlock1=BatchNormalization()(res1)
    
    #endblock1
    
        #bock2
    lstm1= LSTM(10,return_sequences=True) (outBlock1)
    relout =Activation('relu')(lstm1)
    lstm2= LSTM(10,return_sequences=False) (relout)
    relout2 =Activation('relu')(lstm2)
    #Residual connection
    res1=add([relout,relout2])
    
    BN=BatchNormalization()(res1)
    gap_layer = keras.layers.GlobalAveragePooling1D()(BN)
    
    #endblock1

    output_layer=Dense(nb_classes,activation='softmax')(gap_layer)
    model = Model(inputs=input_layer, outputs=output_layer)


    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
    
    return model 

modelRESLSTM = build_RESLSTM(input_shape,nb_classes)
modelRESLSTM.summary()
callback = tf.keras.callbacks.ModelCheckpoint('/kaggle/working/ResLSTM{val_accuracy:.4f}.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

historybiLSTM=modelRESLSTM.fit(x_train, y_train, epochs=epoc,
                              verbose=1, validation_data=(x_test, y_test),callbacks=[callback])
modelbiLSTM.evaluate(x_test,y_test)
from keras.layers.merge import add
from keras.layers import LSTM, Lambda,Input,Activation,BatchNormalization,Dense,GlobalAveragePooling1D
from keras.models import Model
from keras import optimizers

def build_RESbiLSTM(input_shape,nb_classes): 
    input_layer = Input(shape=input_shape)
    #bock1
    reluin =Activation('relu')(input_layer)
    lstm1= LSTM(10,return_sequences=True) (reluin)
    relout =Activation('relu')(lstm1)
    lstm2= LSTM(10,return_sequences=True) (relout)
    relout2 =Activation('relu')(lstm2)
    #Residual connection
    res1=add([relout,relout2])
    
    outBlock1=BatchNormalization()(res1)
    
    #endblock1
    
        #bock2
    lstm1= LSTM(10,return_sequences=True) (outBlock1)
    relout =Activation('relu')(lstm1)
    lstm2= LSTM(10,return_sequences=False) (relout)
    relout2 =Activation('relu')(lstm2)
    #Residual connection
    res1=add([relout,relout2])
    
    BN=BatchNormalization()(res1)
    gap_layer = GlobalAveragePooling1D()(BN)

    #endblock1

    output_layer=Dense(nb_classes)(gap_layer)
    model = Model(inputs=input_layer, outputs=output_layer)


    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(),
              metrics=['accuracy',f1_m, precision_m, recall_m])
    
    return model 

modelRESbiLSTM = build_RESbiLSTM(input_shape,nb_classes)
modelRESbiLSTM.summary()
callback = tf.keras.callbacks.ModelCheckpoint('/kaggle/working/ResBiLSTM{val_accuracy:.4f}.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

historyResBiLSTM=modelRESbiLSTM.fit(x_train, y_train, epochs=epoc,
                              verbose=1, validation_data=(x_test, y_test),callbacks=[callback])
modelRESbiLSTM.evaluate(x_test,y_test)
import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt 
%matplotlib inline
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import preprocessing
def load_yy(y_path):
    """
    Read Y file of values to be predicted
        argument: y_path str attibute of Y: 'train' or 'test'
        return: Y ndarray / tensor of the 6 one_hot labels of each sample
    """
    file = open(y_path, 'rb')
    # Read dataset from disk, dealing with text file's syntax
    
    y_ = np.array(
        [elem for elem in [
            str(row).replace('  ', ' ').replace('\\n','').replace('\\r','').replace('b','').replace('\'','').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()
    return y_
DATASET_PATH = "/kaggle/input/ucidataset/UCI HAR Dataset/"

TRAIN = "train/"
TEST = "test/"
y_test_path = DATASET_PATH + TEST + "y_test.txt"
y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
Y_test_ =load_yy(y_test_path)
Y_train_ =load_yy(y_train_path)
print(Y_test_.shape)
Y_train_=Y_train_.ravel()
Y_test_=Y_test_.ravel()
print(Y_train_.shape)

X=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
X_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
print(X.shape)

params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
svm_model = GridSearchCV(SVC(), params_grid, cv=5)
svm_model.fit(X_train_scaled, Y_train_,)
final_svm_model = svm_model.best_estimator_
Y_pred = final_svm_model.predict(X_test_scaled)