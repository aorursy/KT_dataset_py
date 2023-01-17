import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from keras.models import Sequential

from keras.layers import Dense

from keras import layers

from keras.utils import np_utils

from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split

print(os.listdir("../input"))
p1 = pd.read_csv('../input/WRKY_info_table_positive.txt', sep='\t')

n1 = pd.read_csv('../input/WRKY_info_table_negative_one.txt', sep='\t')

n2 = pd.read_csv('../input/WRKY_info_table_negative_two.txt', sep='\t')

n3 = pd.read_csv('../input/WRKY_info_table_negative_three.txt', sep='\t')
# 看一下資料集的形狀

print(p1.shape)

p1.head()
# 各特徵中相異值的個數

print('TF_ID\t', len(set(p1['TF_ID'])))

print('Pseq_ID\t', len(set(p1['Pseq_ID'])))

print('Pseq\t', len(set(p1['Pseq'])))

print('DBD_seq\t', len(set(p1['DBD_seq'])))

print('matrix_ID\t', len(set(p1['matrix_ID'])))
# 查看一下出現最多次的前幾筆matrix_ID與個數

p1['matrix_ID'].value_counts().head()
# 找出對應到同樣DNA序列的蛋白質序列與DBD_seq

p1.loc[p1['matrix_ID'] == 'TFmatrixID_0449'].head()
# 找出最長的蛋白質序列(Pseq)長度與DBD_sequence長度

max_len = 0

for i in p1['Pseq']:

    if len(i) > max_len:

        max_len = len(i)  

print('Pseq max length:', max_len)

max_len = 0

for i in p1['DBD_seq']:

    if len(i) > max_len:

        max_len = len(i)  

print('DBD_seq max length:', max_len)
# 找出同樣一串DBD_seq所屬的Pseq和它對應到的matrix_ID

one_dbd = p1.loc[p1['matrix_ID'] == 'TFmatrixID_0449'].loc[1:2]['DBD_seq'].values[0]

df = p1.loc[p1['DBD_seq'] == one_dbd]

df.head(10)
# Convert the sequences and IDs to integers

def create_to_dict(all_seq):

    pid = 0

    p_dict = {}

    for seq in all_seq:

        if seq in p_dict:

            pass

        else:

            p_dict[seq] = pid

            pid += 1

    print(len(p_dict))

    return p_dict



p_dict = create_to_dict(p1['Pseq_ID'])

dbd_dict = create_to_dict(p1['DBD_seq'])

dna_dict = create_to_dict(p1['matrix_ID'])
pseq_code = []

for Pseq_ID in p1['Pseq_ID']:

    pseq_code.append(p_dict[Pseq_ID])

dbd_code = []

for DBD_seq in p1['DBD_seq']:

    dbd_code.append(dbd_dict[DBD_seq])

dna_code = []

for matrix_ID in p1['matrix_ID']:

    dna_code.append(dna_dict[matrix_ID])

p1['pseq_code'] = pseq_code

p1['dbd_code'] = dbd_code

p1['dna_code'] = dna_code

#p1.head()

X = p1[['pseq_code', 'dbd_code']]

y = p1['dna_code']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

y_train_onehot = np_utils.to_categorical(y_train)

y_test_onehot = np_utils.to_categorical(y_test)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)

print(y_train_onehot.shape)

print(y_test_onehot.shape)
model = Sequential()

model.add(Dense(units=5, input_dim=2))

for i in range(3):

    model.add(Dense(units=5, activation='relu'))

model.add(Dense(units=107, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train_onehot, epochs=10, batch_size=5, validation_split=0.2)
y_pred = model.predict_classes(X_test)

print(set(y_pred))
max_len = 0

for i in p1['Pseq']:

    if len(i) > max_len:

        max_len = len(i)  

print('Pseq max length:', max_len)

max_len = 0

for i in p1['DBD_seq']:

    if len(i) > max_len:

        max_len = len(i)  

print('DBD_seq max length:', max_len)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
def create_input(df):

    seq_list = []

    for index, row in df.iterrows():

        temp = []

        for i in row['Pseq']:

            temp.append([ord(i)])

        temp.append([ord(' ')])

        for i in row['DBD_seq']:

            temp.append([ord(i)])

        for i in range(1970 - len(temp)):

            temp.append([ord(' ')])

        scaler.fit(temp)

        temp = scaler.transform(temp)

        seq_list.append(temp)

    return seq_list

x_lstm = np.asarray(create_input(p1))
y_lstm = np.asarray(y)

y_lstm = np_utils.to_categorical(y_lstm)

y_lstm = y_lstm.reshape(y_lstm.shape[0], 1, y_lstm.shape[1])

lstm_xtrain, lstm_xtest, lstm_ytrain, lstm_ytest = train_test_split(x_lstm, y_lstm, test_size=0.33, random_state=42)

print(lstm_xtrain.shape)

print(lstm_xtest.shape)

print(lstm_ytrain.shape)

print(lstm_ytest.shape)
def buildModel(shape):

    model = Sequential()

    model.add(layers.LSTM(20, input_shape=(shape[1], shape[2])))

    model.add(layers.RepeatVector(1))

    model.add(layers.LSTM(20, return_sequences=True))

    model.add(layers.LSTM(20, return_sequences=True))

    model.add(layers.TimeDistributed(layers.Dense(107, activation='softmax')))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    model.summary()

    return model
print('Build model...')

lstm_model = buildModel(lstm_xtrain.shape)
lstm_model.fit(lstm_xtrain, lstm_ytrain, batch_size=128, epochs=1, validation_split=0.2)
tryx = lstm_xtest[100].reshape(1, lstm_xtest.shape[1], 1)
y_pred = lstm_model.predict_classes(lstm_xtest[:10])

print(set(y_pred))