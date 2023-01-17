import numpy as np

import pandas as pd

import time
data = {

    'tra': pd.read_csv('../input/air_visit_data.csv'),

    'tes': pd.read_csv('../input/sample_submission.csv')

    }
max_dates = data['tra'].groupby('air_store_id')['visit_date'].max().reset_index()

max_dates.columns = ['air_store_id','anchor_date']

max_dates['anchor_date'] = pd.to_datetime(max_dates['anchor_date'], format='%Y-%m-%d')
# find max date in the training data

# can use to calculate how many days away each record is

dt = pd.to_datetime(data['tra']['visit_date'].max())
data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'], format='%Y-%m-%d')

data['tra'] = data['tra'].sort_values(['air_store_id','visit_date'], ascending=[True,True])



data['tra']['ind'] = data['tra'].groupby('air_store_id')['visitors'].shift(1)

data['tra'] = data['tra'].fillna(0)



data['tra'] = pd.merge(data['tra'], max_dates, on='air_store_id', how='inner')

data['tra']['time_diff'] = ((data['tra']['anchor_date'] - data['tra']['visit_date'])/(np.timedelta64(1, 'D'))).astype(int) + 1
x1 = data['tra'].groupby('air_store_id')['visitors'].apply(list)

x2 = data['tra'].groupby('air_store_id')['ind'].apply(list)

x3 = data['tra'].groupby('air_store_id')['time_diff'].apply(list)



store_group = pd.concat([x2,x3,x1], axis=1).reset_index()

store_group.columns = ['air_store_id','ind','time','dep']



df = store_group.drop('air_store_id', axis=1)



# Prepare LSTM training, split up records into training and test data:

train_size = int(len(df) * 0.7)

test_size = len(df) - train_size



train = df[:train_size].values

test  = df[train_size:].values



# Split into input and outputs

train_X, train_y = train[:,:-1], train[:,-1]

test_X, test_y = test[:, :-1], test[:, -1]



# LSTM requires 3D data sets: [samples, timesteps, features]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
from keras.models import Model

from keras.layers import Dense, Dropout, Input, TimeDistributed

from keras.layers.convolutional import Conv1D

from keras.layers.recurrent import LSTM

from keras.layers.merge import concatenate
def generate_arrays(X, Y, train=True):

    while 1:

        for i,j in enumerate(Y):

            ind = np.asarray([np.column_stack((a[0], a[1])) for a in X[i]])

            if train == False:

                yield (ind)

            dep = np.array([np.array(a) for a in j]).reshape(1, len(j), 1)

            yield (ind, dep)
# define input

visible = Input(shape=(None, 2))

# feature extraction

# we use padding of same so that the length of the output matches the dependent variable sequence

conv1 = Conv1D(8, kernel_size=2, activation='relu', padding="same")(visible)

conv2 = Conv1D(8, kernel_size=3, activation='relu', padding="same")(visible)

conv3 = Conv1D(8, kernel_size=5, activation='relu', padding="same")(visible)

conv4 = Conv1D(8, kernel_size=7, activation='relu', padding="same")(visible)



conv5 = Conv1D(16, kernel_size=2, activation='relu', padding="same")(visible)

conv6 = Conv1D(16, kernel_size=3, activation='relu', padding="same")(visible)

conv7 = Conv1D(16, kernel_size=5, activation='relu', padding="same")(visible)

conv8 = Conv1D(16, kernel_size=7, activation='relu', padding="same")(visible)



# merge convs

merge1 = concatenate([conv1,conv2,conv3,conv4])

merge2 = concatenate([conv5,conv6,conv7,conv8])



# LSTMs

lstm11 = LSTM(200, return_sequences=True)(merge1)

do1 = TimeDistributed(Dropout(0.5))(lstm11)

lstm12 = LSTM(200, return_sequences=True)(do1)



lstm21 = LSTM(200, return_sequences=True)(merge2)

do2 = TimeDistributed(Dropout(0.5))(merge2)

lstm22 = LSTM(200, return_sequences=True)(do2)



# concat lstm output

merge3 = concatenate([lstm12, lstm22])

do3 = TimeDistributed(Dropout(0.2))(merge3)



# Dense layer

dense1 = TimeDistributed(Dense(100))(do3)

dense2 = TimeDistributed(Dense(200))(dense1)

# output

output = TimeDistributed(Dense(1))(dense2)

model1 = Model(inputs=visible, outputs=output)

print(model1.summary())
model1.compile(loss='mse', optimizer='adam')
model1.fit_generator(generate_arrays(train_X, train_y, True), steps_per_epoch=580, epochs=3)
model1.evaluate_generator(generate_arrays(train_X, train_y, True), steps=580)
gen=generate_arrays(train_X, train_y, True)

steps=580

preds=[]

for i in range(steps):

    preds.append(model1.predict_on_batch(next(gen)[0]))
def rmsle(p,a):

    l = len(p)

    x = 0

    for i in range(0,len(p)):

        x = x + ((np.log(p[i] + 1) - np.log(a[i] + 1))**2)

    return(np.sqrt(x/l))
rmsle_series = []

for s in range(0, len(preds)):

    rmsle_series.append(rmsle(preds[s][0],train_y[s]))

    

np.mean(rmsle_series)
test_gen=generate_arrays(test_X, test_y, True)

steps=249

test_preds=[]

for i in range(steps):

    test_preds.append(model1.predict_on_batch(next(test_gen)[0]))
test_rmsle_series = []

for s in range(0, len(test_preds)):

    test_rmsle_series.append(rmsle(test_preds[s][0],test_y[s]))

    

np.mean(test_rmsle_series)
test_df = pd.read_csv("../input/sample_submission.csv")
test_df.head()
test_df['air_store_id'] = test_df['id'].apply(lambda x: x[:-11])

test_df['pred_date'] = test_df['id'].apply(lambda x: x[-10:])
test_df.head()
test_df['pred_date'] = pd.to_datetime(test_df['pred_date'], format='%Y-%m-%d')
test_seq = test_df.groupby('air_store_id')['pred_date'].min().reset_index()
test_seq.head()
test_data = pd.merge(data['tra'][['air_store_id','visit_date','visitors']], test_seq, on='air_store_id', how='inner')
test_data.head()
test_data['time_diff'] = ((test_data['pred_date'] - test_data['visit_date'])/(np.timedelta64(1,'D'))).astype(int)
test_data.head()
x1_test = test_data.groupby(['air_store_id','pred_date'])['visitors'].apply(list)

x2_test = test_data.groupby(['air_store_id','pred_date'])['time_diff'].apply(list)



store_group_test = pd.concat([x1_test,x2_test], axis=1).reset_index()

store_group_test.columns = ['air_store_id','pred_date','ind','time']
store_group_test.head()
store_group_test1 = store_group_test.drop(['air_store_id','pred_date'], axis=1)
store_group_test1.head()
lstm_test = store_group_test1.values

lstm_test = lstm_test.reshape((lstm_test.shape[0], 1, lstm_test.shape[1]))
lstm_test.shape
lstm_test_gen=generate_arrays(lstm_test, test_y, True)

steps=821

lstm_test_preds=[]

for i in range(steps):

    lstm_test_preds.append(model1.predict_on_batch(next(lstm_test_gen)[0]))
actual_preds = [x[0][-1:] for x in lstm_test_preds]
store_group_test1['visitors'] = actual_preds
store_group_test1['visitors'] = store_group_test1['visitors'].apply(lambda x: x[0][0])
store_group_test1 = store_group_test1[['visitors']]
store_group_test = pd.concat([store_group_test,store_group_test1], axis=1).drop(['ind','time'], axis=1)
store_group_test.head()
final_df = pd.merge(test_df[['air_store_id','pred_date']], store_group_test, on=['air_store_id','pred_date'], how='inner')
final_df.head()
final_df['id'] = final_df['air_store_id'] + '_' + final_df['pred_date'].astype(str)
pd.set_option('display.max_rows', 821)

final_df[['id','visitors']]
test_df1.to_csv("lstm_preds.csv",header=True,index=False)
test_data[test_data['air_store_id'] == 'air_fd6aac1043520e83']