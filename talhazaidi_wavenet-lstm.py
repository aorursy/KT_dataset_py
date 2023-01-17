# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR

from sklearn.metrics import mean_absolute_error
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.float32, 'time_to_failure': np.float32}).values
train [0:150000, 0 ] .mean(axis=0)
# pandas doesn't show us all the decimals

pd.options.display.precision = 15
rows = 150_000

segments = int(np.floor(train.shape[0] / rows))

print('train.shape',train.shape)

segments

X_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['ave', 'std', 'max', 'min'])

y_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])

y_train.shape







segments *rows+rows
# Create a training file with simple derived features

 

# total train.shape (629145480, 2)



rows = 150_000

segments = int(np.floor(train.shape[0] / rows))                         # 4194



X_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['ave', 'std', 'max', 'min'])             # 4194,4

y_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])                      # 4194,1



for segment in tqdm(range(segments)):                          # for loop from 1 to 4194 segment value

    seg = train.iloc[segment*rows:segment*rows+rows]           # seg.shape (150000,2)  i-e first iteration (0-150000) then (150000-300000) then (300000-4500000) ...... (629100000-629250000)

    x = seg['acoustic_data'].values                            # x.shape (150000,)

    y = seg['time_to_failure'].values[-1]                      # i-e last time_to_failure value of every 150000 length segment. 

    

    y_train.loc[segment, 'time_to_failure'] = y       # 4194,1      ie go on from (1,1) to (4194,1)

    

    X_train.loc[segment, 'ave'] = x.mean()         # 4194,4          ie go on from (1,4) to (4194,4)

    X_train.loc[segment, 'std'] = x.std()          # 4194,4          ie go on from (1,4) to (4194,4)

    X_train.loc[segment, 'max'] = x.max()          # 4194,4          ie go on from (1,4) to (4194,4)

    X_train.loc[segment, 'min'] = x.min()          # 4194,4          ie go on from (1,4) to (4194,4)
def extract_features(z):

     return np.c_[z.mean(axis=0), 

                  z.std(axis=0),

                  z.max(axis=0),

                  z.min(axis=0),

                #  np.transpose(np.percentile(np.abs(z), q=[0, 50, 75, 100], axis=0)) .reshape(1,4)

                 ]
def create_X(x, last_index=None, n_steps=150, step_length=1000):

    if last_index == None:

        last_index=len(x)

       

    assert last_index - n_steps * step_length >= 0



    # Reshaping and approximate standardization with mean 5 and std 3.

    # ORIGINAL: I changed this becuase I got an No OpKernel was registered to support Op 'CudnnRNN' error

    #temp = (x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1) - 5 ) / 3

    # MY CHANGE: This doesn't fix things, I get the same errors

    temp = (x[(int(last_index) - n_steps * step_length):int(last_index)].reshape(n_steps,step_length ).astype(np.float32) - 5 ) / 3

    

    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 

    # of the last 10 observations. 

    return np.c_[extract_features(temp),

                 extract_features(temp [ -n_steps // 10:]),

                 extract_features(temp [ -n_steps // 100:])]



# Query "create_X" to figure out the number of features

n_features = create_X(train [0:150000,0]).shape[1]

print("Our RNN is based on %i features"% n_features)     # 18 features each row of segment ie 150x18 features of 150000 chunk input



n_steps=150

step_length=1000

maxsize=train .shape[0]

seg = int(np.floor(maxsize / (n_steps*step_length))) 

batch_size = seg-1   # (4193,) 

xx=350





##############################################################################################

rows_initialize = np.zeros((seg), dtype=float)

print(rows_initialize.shape)



for seg1 in tqdm(range(1,seg)) :      # for loop from 1 to 4194 segment value

    rows_initialize [seg1] = seg1 * (n_steps*step_length) 



rows=np.delete(rows_initialize,0)    # (4193,)



print(rows.shape)



########################################################################################

batch_size=batch_size-xx    # training data

#batch_size=xx              # validation data

split_point=xx

second_earthquake = rows[xx]







##########################################################################################



if batch_size < 1000  :    # validation set 

               rows_1 = rows[:split_point+1]    #  0:350 

        

if batch_size > 1000 :   # training set

               rows_1 = rows[split_point+1 :]    # (351,) ie 351:4193    

            



       

    # Initialize feature matrices and targets

samples_tr= np.zeros((rows_1.shape[0], step_length, n_features), dtype=float)   # (16, 150, 18)  for validation (350, 1, 24)  for training ( 3843, 1, 24) 

targets_tr = np.zeros(rows_1.shape[0], )    # (16,)  for validation (350)    for training ( 3843)

        

for j, row in enumerate(rows_1):             # 16 for validation (350)    for training ( 3843)

    samples_tr[j] = create_X(train[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)

    targets_tr[j] = train[int(row - 1), 1]         

    

    

################################################################################################



print('samples_tr shape', samples_tr.shape)

print('targets_tr shape', targets_tr.shape)





#batch_size=batch_size-xx    # training data

batch_size=xx              # validation data

split_point=xx

second_earthquake = rows[xx]



##########################################################################################



if batch_size < 1000  :    # validation set 

               rows_1 = rows[:split_point+1]    #  0:350 

        

if batch_size > 1000 :   # training set

               rows_1 = rows[split_point+1 :]    # (351,) ie 351:4193    

            

       

    # Initialize feature matrices and targets

samples_vd= np.zeros((rows_1.shape[0], step_length, n_features), dtype=float)   #   for validation (350, 1, 24) 

targets_vd = np.zeros(rows_1.shape[0], )    #  for validation (350)    

        

for j, row in enumerate(rows_1):             # 16 for validation (350)    for training ( 3843)

    samples_vd[j] = create_X(train[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)

    targets_vd[j] = train[int(row - 1), 1]  

    

    

print('samples_vd shape', samples_vd.shape)

print('targets_vd shape',targets_vd.shape)  

#print('rows_1 shape',rows_1.shape[0])

    
# Query "create_X" to figure out the number of features

n_features = create_X(train [0:150000,0]).shape[1]

print("Our RNN is based on %i features"% n_features)     # 18 features each row of segment ie 150x18 features of 150000 chunk input



n_steps=150000

step_length=1



# initialization for rows

maxsize=train .shape[0]

seg = int(np.floor(maxsize / (n_steps*step_length)))    #4194

segment = (n_steps*step_length)* seg    # (150000* 4194) = 629100000

rows = np.zeros(seg, )    # (4194,)          

          

for seg1 in tqdm(range(1,seg)) :      # for loop from 1 to 4194 segment value

    rows [seg1] = seg1 * (n_steps*step_length) 



rows=np.delete(rows,0)    # (4193,) 



# The generator endlessly selects "batch_size" ending positions of sub-time series. For each ending position,

# the "time_to_failure" serves as target, while the features are created by the function "create_X".

def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150000, split_point=350, step_length=1):    

    if max_index is None:

        max_index = len(data) - 1     # data = float_data  len= 629145480-1

     

   # while True:

       # # Pick indices of ending positions

 #####       rows_1 = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)    # produce 16 random number bw min index and max index

              

        if batch_size < 1000  :    # validation set 

               rows_1 = rows[:split_point+1]    #  0:350 

        

        if batch_size > 1000 :   # training set

               rows_1 = rows[split_point+1 :]    # (351,) ie 351:4193    

            



       

    # Initialize feature matrices and targets

        samples = np.zeros((batch_size, step_length, n_features), dtype=float)   # (16, 150, 18)  for validation (350, 1, 24)  for training ( 3843, 1, 24) 

        targets = np.zeros(batch_size, )    # (16,)  for validation (350)    for training ( 3843)

        

        for j, row in enumerate(rows_1):             # 16 for validation (350)    for training ( 3843)

            samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)

            targets[j] = data[int(row - 1), 1]

        

    return samples, targets

        

       

batch_size = seg-1   # (4193,) 

###### batch_size = 32



# Position of second (of 16) earthquake. Used to have a clean split

# between train and validation

xx=350

second_earthquake = rows[xx]



#float_data[second_earthquake, 1] # time failure corresponding to this second earthquake indices



# Initialize generators

#train_gen = generator(float_data, batch_size=batch_size) # Use this for better score



train_gen = generator(train , batch_size=batch_size-xx, split_point=xx, min_index=second_earthquake + 1)     # batch_size-xx =  3843

#valid_gen = generator(train , batch_size=xx, max_index=second_earthquake)                    # xx =  350



######train_gen = generator(float_data, batch_size=batch_size, split_point=xx, min_index=second_earthquake + 1) 

######valid_gen = generator(float_data, batch_size=batch_size, max_index=second_earthquake)





#print(samples.shape)

#print(targets.shape) 

from keras.models import Sequential

from keras.layers import Dense, CuDNNGRU, SimpleRNN, LSTM

from keras.optimizers import adam

from keras.callbacks import ModelCheckpoint



cb = [ModelCheckpoint("model.hdf5", save_best_only=True, period=3)]



model = Sequential()

model.add(LSTM(48,dropout=0.2, recurrent_dropout=0.2, input_shape=(None, n_features), return_sequences=True))

model.add(LSTM(20,dropout=0.2, recurrent_dropout=0.2, return_sequences=False))

####model.add(CuDNNGRU(48, input_shape=(None, n_features)))

model.add(Dense(100, activation='relu'))

model.add(Dense(1, activation='linear'))



model.summary()
import keras

from keras.optimizers import RMSprop

opt = keras.optimizers.adam(lr=.01)



model.compile(loss="mae",

              optimizer=opt, metrics=['mean_absolute_error'])

             # metrics=['accuracy'])





batch_size = 128 # mini-batch with 32 examples

epochs = 10

history = model.fit(

    samples_tr, targets_tr,

    batch_size=batch_size,

    epochs=epochs,

    verbose=1,

    validation_data=(samples_vd  ,targets_vd ))
model.compile(optimizer=adam(lr=0.0005), loss="mae", metrics=['accuracy'])



history = model.fit_generator(train_gen,

                              steps_per_epoch=1000,

                              epochs=30,

                              verbose=1,

                              callbacks=cb,

                              validation_data=valid_gen,

                              validation_steps=351)
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})
for i, seg_id in enumerate(tqdm(submission.index)):

  #  print(i)

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    x = seg['acoustic_data'].values

    submission.time_to_failure[i] = model.predict(np.expand_dims(create_X(x), 0))



submission.head()
submission.to_csv('submission.csv')
x.mean()  
X_train
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_train_scaled
y_train.values
y_train.values.flatten()
from sklearn.ensemble import RandomForestRegressor



## Train the  Model

model = RandomForestRegressor(n_estimators=200)

model.fit(X_train_scaled, y_train.values.flatten())      # .fit used for training

y_pred = model.predict(X_train_scaled)
# number support vector regressor



# svm = NuSVR()

# svm.fit(X_train_scaled, y_train.values.flatten())

# y_pred = svm.predict(X_train_scaled)
y_pred.shape
plt.figure(figsize=(6, 6))

plt.scatter(y_train.values.flatten(), y_pred)

plt.xlim(0, 20)

plt.ylim(0, 20)

plt.xlabel('actual', fontsize=12)

plt.ylabel('predicted', fontsize=12)

plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

plt.show()
score = mean_absolute_error(y_train.values.flatten(), y_pred)

print(f'Score: {score:0.3f}')
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)

for seg_id in X_test.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x = seg['acoustic_data'].values

    

    X_test.loc[seg_id, 'ave'] = x.mean()

    X_test.loc[seg_id, 'std'] = x.std()

    X_test.loc[seg_id, 'max'] = x.max()

    X_test.loc[seg_id, 'min'] = x.min()
X_test_scaled = scaler.transform(X_test)

submission['time_to_failure'] = svm.predict(X_test_scaled)

submission.to_csv('submission.csv')