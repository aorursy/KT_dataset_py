# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import preprocessing

from sklearn.model_selection import StratifiedShuffleSplit,train_test_split



import keras

import tensorflow as tf

from keras.models import Sequential,load_model,Model

from keras.optimizers import *

from keras.utils import to_categorical

from keras.layers import *

from keras.callbacks import *

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers

sess = tf.Session()





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

K.tensorflow_backend._get_available_gpus()

K.set_session(sess)



# Any results you write to the current directory are saved as output.
xtrain = pd.read_csv('../input/X_train.csv')

ytrain = pd.read_csv('../input/y_train.csv')

test=pd.read_csv("../input/X_test.csv")
### feature extraction of orientation, angular_velocity, linear_acceleration, velocity_to_acceleration and velocity_linear_acceleration

def feature_extraction(raw_frame):

    raw_frame['orientation'] = raw_frame['orientation_X'] + raw_frame['orientation_Y'] + raw_frame['orientation_Z']+ raw_frame['orientation_W']

    raw_frame['angular_velocity'] = raw_frame['angular_velocity_X'] + raw_frame['angular_velocity_Y'] + raw_frame['angular_velocity_Z']

    raw_frame['linear_acceleration'] = raw_frame['linear_acceleration_X'] + raw_frame['linear_acceleration_Y'] + raw_frame['linear_acceleration_Y']

    raw_frame['velocity_to_acceleration'] = raw_frame['angular_velocity'] / raw_frame['linear_acceleration']

    raw_frame['velocity_linear_acceleration'] = raw_frame['linear_acceleration'] * raw_frame['angular_velocity']

    return raw_frame
xtrain = feature_extraction(xtrain)

test = feature_extraction(test)
### more feature extraction with mean, mode, std, variance, min, max and so on...



def feature_extraction_more(raw_frame):

    frame = pd.DataFrame([])

    for col in raw_frame.columns[3:]:

        frame[col + '_mean'] = raw_frame.groupby(['series_id'])[col].mean()

        frame[col + '_std'] = raw_frame.groupby(['series_id'])[col].std()

        frame[col + '_var'] = raw_frame.groupby(['series_id'])[col].var()

        frame[col + '_sem'] = raw_frame.groupby(['series_id'])[col].sem()

        frame[col + '_max'] = raw_frame.groupby(['series_id'])[col].max()

        frame[col + '_min'] = raw_frame.groupby(['series_id'])[col].min()

        frame[col + '_max_to_min'] = frame[col + '_max'] / frame[col + '_min']

        frame[col + '_max_minus_min'] = frame[col + '_max'] - frame[col + '_min']

        frame[col + '_std_to_var'] = frame[col + '_std'] * frame[col + '_var']

        frame[col + '_mean_abs_change'] = raw_frame.groupby('series_id')[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        frame[col + '_abs_max'] = raw_frame.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

    return frame
train_df = feature_extraction_more(xtrain)

test_df = feature_extraction_more(test)
print("train shape",train_df.shape)

print("test shape", test_df.shape)
scaler = preprocessing.StandardScaler()

# Apply transform to both the training set and the test set.

train_df = scaler.fit_transform(train_df)

test_df = scaler.fit_transform(test_df)
### lable encoding 

le = preprocessing.LabelEncoder()

le.fit(ytrain.surface)

ytrain['surface'] = le.transform(ytrain.surface)

train_label = to_categorical(ytrain['surface'])

train_label.shape
train_x,val_x,train_y,val_y = train_test_split(train_df, train_label, test_size = 0.10, random_state=14)

train_x.shape,val_x.shape,train_y.shape,val_y.shape
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1], 1))

test_df = np.reshape(test_df, (test_df.shape[0], test_df.shape[1],1))
train_x.shape,val_x.shape,test_df.shape
nb_features = train_df.shape[1]

nb_out = train_label.shape[1]

nb_features,nb_out
# https://www.kaggle.com/ist597/simple-keras-lstm-classifier-98-74

model = Sequential()

model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=((nb_features), 1)))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(nb_out, activation='softmax'))



model.summary()



model.compile(loss='categorical_crossentropy',

              optimizer=Adam(),

              metrics=['accuracy'])
history = model.fit(train_x, train_y,

                    batch_size=32,

                    epochs=10,

                    verbose=1,

                    validation_data=(val_x, val_y))
prediction = model.predict(test_df)

prediction=np.argmax(prediction, axis=1) 

submission = pd.read_csv("../input/sample_submission.csv")

submission['surface'] = le.inverse_transform(prediction)

submission.to_csv('lstm_38.csv', index=False)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))

val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1]))

test_df = np.reshape(test_df, (test_df.shape[0], test_df.shape[1]))

train_x.shape,val_x.shape,test_df.shape
## https://www.kaggle.com/kabure/titanic-eda-keras-nn-pipelines

## Creating the model

model = Sequential()



# Inputing the first layer with input dimensions

model.add(Dense(165, 

                activation='relu',  

                input_dim = nb_features,

                kernel_initializer='uniform'))



# Adding an Dropout layer to previne from overfitting

model.add(Dropout(0.50))



#adding second hidden layer 

model.add(Dense(60,

                kernel_initializer='uniform',

                activation='relu'))



# Adding another Dropout layer

model.add(Dropout(0.50))



# adding the output layer that is binary [0,1]

model.add(Dense(nb_out, activation='softmax'))



#Visualizing the model

model.summary()



sgd = SGD(lr = 0.01, momentum = 0.9)



# Compiling our model

model.compile(optimizer = sgd, 

                   loss = 'categorical_crossentropy', 

                   metrics = ['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

save_best = ModelCheckpoint('cnn.hdf', save_best_only=True, 

                               monitor='val_loss', mode='min')
history = model.fit(train_x, train_y,

                    batch_size=32,

                    epochs=50,

                    verbose=1,

                    validation_data=(val_x, val_y),callbacks=[early_stopping,save_best])
prediction = model.predict(test_df)

prediction=np.argmax(prediction, axis=1) 

submission = pd.read_csv("../input/sample_submission.csv")

submission['surface'] = le.inverse_transform(prediction)

submission.to_csv('cnn_74.csv', index=False)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1], 1))

test_df = np.reshape(test_df, (test_df.shape[0], test_df.shape[1],1))
## Creating the model

model = Sequential()



# Inputing the first layer with input dimensions

model.add(Dense(165,activation='relu',input_shape = (nb_features,1),kernel_initializer='uniform'))

model.add(MaxPooling1D(pool_size=2))

# Adding an Dropout layer to previne from overfitting

model.add(Dropout(0.50))

#adding second hidden layer 

model.add(Dense(128,kernel_initializer='uniform',activation='relu'))

# Adding another Dropout layer

model.add(Dropout(0.50))

model.add(GRU(64))

model.add(Dropout(0.50))

model.add(Dense(32,kernel_initializer='uniform',activation='relu'))

model.add(Dropout(0.50))

# adding the output layer that is binary [0,1]

model.add(Dense(nb_out, activation='softmax'))

#Visualizing the model

model.summary()

sgd = SGD(lr = 0.01, momentum = 0.9)

# Compiling our model

model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')

save_best = ModelCheckpoint('cnn.hdf', save_best_only=True,monitor='val_loss', mode='min')



history = model.fit(train_x, train_y,

                    batch_size=32,

                    epochs=10,

                    verbose=1,

                    validation_data=(val_x, val_y),callbacks=[early_stopping,save_best])
prediction = model.predict(test_df)

prediction=np.argmax(prediction, axis=1) 

submission = pd.read_csv("../input/sample_submission.csv")

submission['surface'] = le.inverse_transform(prediction)

submission.to_csv('gru_33.csv', index=False)