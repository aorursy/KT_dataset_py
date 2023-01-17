import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import warnings

warnings.filterwarnings('ignore')

import keras

import keras.backend as K

from keras.layers.core import Activation

from keras.models import Sequential,load_model

from keras.layers import Dense, Dropout, LSTM

from keras.optimizers import Adam

from keras.models import load_model

from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#load files

train = pd.read_csv("/kaggle/input/train_label.csv")

features = pd.read_excel("/kaggle/input/feature.xlsx")

test = pd.read_csv("/kaggle/input/test_label_sample.csv")
train['date']= pd.to_datetime(train['date'],format='%d/%m/%Y') 

train.shape
test['date']= pd.to_datetime(test['date'],format='%d/%m/%Y') 

test.shape
new_header = features.iloc[0] #grab the first row for the header

features = features[1:] #take the data less the header row

features.columns = new_header #set the header row as the df header

features['date'] = features['date'].dt.date
error_id_count=features.iloc[:,1:28]

error_id_count['date']= pd.to_datetime(error_id_count['date'],format='%Y-%m-%d')

error_id_count.shape
df_train= pd.merge(error_id_count, train, on='date')

df_train.columns = ['date', 'e1', 'e2', 'e3','e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14',

                     'e15', 'e16', 'e17', 'e18', 'e19', 'e20', 'e21','e22','e23','e24','e25','e26','label']

df_train.head()
df_test = pd.merge(error_id_count, test, on='date')

df_test.columns = ['date', 'e1', 'e2', 'e3','e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14',

                     'e15', 'e16', 'e17', 'e18', 'e19', 'e20', 'e21','e22','e23','e24','e25','e26','label']

df_test.head()
features_col_name=[ 'e1', 'e2', 'e3','e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14',

                     'e15', 'e16', 'e17', 'e18', 'e19', 'e20', 'e21','e22','e23','e24','e25','e26']

target_col_name='label'
from sklearn.preprocessing import MinMaxScaler

sc=MinMaxScaler()

df_train[features_col_name]=sc.fit_transform(df_train[features_col_name])

df_test[features_col_name]=sc.transform(df_test[features_col_name])
def gen_sequence(id_df, seq_length, seq_cols):

    df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)

    id_df=df_zeros.append(id_df,ignore_index=True)

    data_array = id_df[seq_cols].values

    num_elements = data_array.shape[0]

    lstm_array=[]

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):

        lstm_array.append(data_array[start:stop, :])

    return np.array(lstm_array)
def gen_label(id_df, seq_length, seq_cols,label):

    df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)

    id_df=df_zeros.append(id_df,ignore_index=True)

    data_array = id_df[seq_cols].values

    num_elements = data_array.shape[0]

    y_label=[]

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):

        y_label.append(id_df[label][stop])

    return np.array(y_label)
# timestamp or window size

seq_length=50

seq_cols=features_col_name
X_train=gen_sequence(df_train, seq_length, seq_cols)

print(X_train.shape)

# generate y_train

y_train=gen_label(df_train, 50, seq_cols,'label')

print(y_train.shape)
X_test=gen_sequence(df_test, seq_length, seq_cols)

print(X_test.shape)

# generate y_train

y_test=gen_label(df_test, 50, seq_cols,'label')

print(y_test.shape)
nb_features =X_train.shape[2]

timestamp=seq_length



model = Sequential()



model.add(LSTM(

         input_shape=(timestamp, nb_features),

         units=100,

         return_sequences=True))

model.add(Dropout(0.2))



model.add(LSTM(

          units=50,

          return_sequences=False))

model.add(Dropout(0.2))



model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
model.fit(X_train, y_train, epochs=10, batch_size=200, validation_split=0.05, verbose=1,

          callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
# training metrics

scores = model.evaluate(X_train, y_train, verbose=1, batch_size=200)

print('Accurracy: {}'.format(scores[1]))
y_pred=model.predict_proba(X_test)