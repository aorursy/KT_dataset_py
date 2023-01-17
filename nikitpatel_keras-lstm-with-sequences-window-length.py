# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



### sklearn 

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, recall_score, precision_score

from sklearn.model_selection import StratifiedShuffleSplit,train_test_split

from sklearn import preprocessing

from keras.utils import to_categorical





### keras

import keras

import tensorflow as tf

from keras.models import Sequential,load_model

from keras.layers import Dense, Dropout, LSTM

from keras.callbacks import TensorBoard

from keras import backend as K



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_x = pd.read_csv("../input/X_train.csv")

train_y = pd.read_csv("../input/y_train.csv")

### column name and shape



print("train_x column name ---- \n",train_x.columns)

print("train_y column name ---- \n",train_y.columns)

print("train_x shape ---- \n",train_x.shape)

print("train_y shape ---- \n",train_y.shape)
### train_x head

train_x.head()
train_x.shape
### train_y head



train_y.head()
### check the traget variable

train_y.groupby('surface')['surface'].count()
train_df = train_x.merge(train_y, on=['series_id'], how='left')
train_df.head(5)
le = preprocessing.LabelEncoder()

le.fit(train_df.surface)
train_df['surface'] = le.transform(train_df.surface)
train_df.head()
train_df['cycle'] = train_df['measurement_number']
cols_normalize = train_df.columns.difference(['row_id','series_id','measurement_number','group_id','surface'])

min_max_scaler = preprocessing.MinMaxScaler()

norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 

                             columns=cols_normalize, 

                             index=train_df.index)

join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)

train_df = join_df.reindex(columns = train_df.columns)
train_df = train_df.drop(['row_id', 'group_id'], axis=1)

train_df.head()
##################################

# LSTM

##################################



# pick a large window size of 128 cycles

sequence_length = 128

#num_elements = train_df.shape[0]

sequence_cols = ['series_id', 'measurement_number', 'orientation_X',

       'orientation_Y', 'orientation_Z', 'orientation_W', 'angular_velocity_X',

       'angular_velocity_Y', 'angular_velocity_Z', 'linear_acceleration_X',

       'linear_acceleration_Y', 'linear_acceleration_Z','cycle']



# function to reshape features into (samples, time steps, features) 

def gen_sequence(id_df,num_elements,seq_cols):

    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing

    we need to drop those which are below the window-length. An alternative would be to pad sequences so that

    we can use shorter ones """

    # for one id I put all the rows in a single matrix

    data_matrix = id_df[seq_cols].values

    # Iterate over two lists in parallel.

    # For example id1 have 0 rows and sequence_length is equal to 128

    # so zip iterate over two following list of numbers (0,128)

    # 0 128 -> from row 0 to row 128

    # 128 256 -> from row 128 to row 256

    for start, stop in zip(range(0, num_elements + 128,128), range(128, num_elements + 128,128)):

        yield data_matrix[start:stop, :]
a = list(gen_sequence(train_df,train_df.shape[0],sequence_cols))

seq_array = np.array(a)

seq_array.shape
##################################

# LSTM

##################################



# pick a large window size of 128 cycles

sequence_length = 128

label_cols = ['surface']



# function to reshape features into (samples, time steps, features) 

def label_sequence(id_df,seq_cols):

    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing

    we need to drop those which are below the window-length. An alternative would be to pad sequences so that

    we can use shorter ones """

    # for one id I put all the rows in a single matrix

    data_matrix = id_df[seq_cols].values

    num_elements = data_matrix.shape[0]

    # Iterate over two lists in parallel.

    # For example id1 have 0 rows and sequence_length is equal to 128

    # so zip iterate over two following list of numbers (0,128)

    # 0 128 -> from row 0 to row 128

    # 128 256 -> from row 128 to row 256

    for start in range(1, num_elements,128):

        yield data_matrix[start, :]
b = list(label_sequence(train_df,label_cols))

label_array = np.array(b)

label_array = to_categorical(label_array)

label_array.shape
# unique, counts = np.unique(label_array, return_counts=True)

# print (np.asarray((unique, counts)).T)
nb_features = seq_array.shape[2]

nb_out = label_array.shape[1]

nb_features,nb_out
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42) # Want a balanced split for all the classes

for train_index, test_index in sss.split(seq_array, label_array):

    print("Using {} for training and {} for validation".format(len(train_index), len(test_index)))

    x_train, x_valid = seq_array[train_index], seq_array[test_index]

    y_train, y_valid = label_array[train_index], label_array[test_index]
# Next, we build a deep network. 

# The first layer is an LSTM layer with 100 units followed by another LSTM layer with 50 units. 

# Dropout is also applied after each LSTM layer to control overfitting. 

# Final layer is a Dense output layer with single unit and sigmoid activation since this is a binary classification problem.

# build the network



model = Sequential()



model.add(LSTM(

         input_shape=(sequence_length, nb_features),

         units=100,

         return_sequences=True))

model.add(Dropout(0.2))



model.add(LSTM(

          units=50,

          return_sequences=False))

model.add(Dropout(0.2))



model.add(Dense(units=nb_out, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



print(model.summary())
# fit the network

model_path = '../input/binary_model.h5'

history = model.fit(x_train, y_train, epochs=10, validation_split=0.33, verbose=1,batch_size=200)



# list all data in history

print(history.history.keys())
test = pd.read_csv("../input/X_test.csv")

test.columns
test['cycle'] = test['measurement_number']
cols_normalize = test.columns.difference(['row_id','series_id','measurement_number'])

min_max_scaler = preprocessing.MinMaxScaler()

norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(test[cols_normalize]), 

                             columns=cols_normalize, 

                             index=test.index)

join_df = test[test.columns.difference(cols_normalize)].join(norm_train_df)

test = join_df.reindex(columns = train_df.columns)
a = list(gen_sequence(test,test.shape[0],sequence_cols))

test_array = np.array(a)

test_array.shape
prediction = model.predict(test_array)

prediction=np.argmax(prediction, axis=1) 
submission = pd.read_csv("../input/sample_submission.csv")

submission['surface'] = le.inverse_transform(prediction)
submission.to_csv('lstm.csv', index=False)