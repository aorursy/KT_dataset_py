import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from tqdm import tqdm

import os
test_path='../input/test/'

# Files in test folder

len(os.listdir(test_path))
# Load the training set

train_path='../input/train.csv'

train=pd.read_csv(train_path,dtype={'acoustic_data':np.int16,'time_to_failure':np.float32})
train.head(10)
ad_sample=train.acoustic_data.values[::100]

ttf_sample=train.time_to_failure.values[::100]
fig,ax1=plt.subplots(figsize=(12,8))

plt.title("Acoustic data and time to failure")

plt.plot(ad_sample,color='green')

plt.ylabel('Acoustic data',color='green')

plt.legend(['acoustic data'],loc=(0.01,0.95))

ax2=ax1.twinx()

plt.plot(ttf_sample,color='blue')

plt.ylabel('Time to Failure',color='blue')

plt.legend(['time to failure'],loc=(0.01,0.9))

plt.grid(True)



del ad_sample

del ttf_sample
ad_sample=train.acoustic_data.values[:6000000]

ttf_sample=train.time_to_failure.values[:6000000]

fig,ax1=plt.subplots(figsize=(12,8))

plt.title("Acoustic data and time to failure")

plt.plot(ad_sample,color='green')

plt.ylabel('Acoustic data',color='green')

plt.legend(['acoustic data'],loc=(0.01,0.95))

ax2=ax1.twinx()

plt.plot(ttf_sample,color='blue')

plt.ylabel('Time to Failure',color='blue')

plt.legend(['time to failure'],loc=(0.01,0.9))

plt.grid(True)



del ad_sample

del ttf_sample
rows = 150000

segments = int(np.floor(train.shape[0] / rows))



X_train = pd.DataFrame(index = range(segments),dtype = np.float32,columns = ['mean','std','99quat','50quat','25quat','1quat'])

y_train = pd.DataFrame(index = range(segments),dtype = np.float32,columns = ['time_to_failure'])
for segment in tqdm(range(segments)):

    x = train.iloc[segment*rows:segment*rows+rows]

    y = x['time_to_failure'].values[-1]

    x = x['acoustic_data'].values

    X_train.loc[segment,'mean'] = np.mean(x)

    X_train.loc[segment,'std']  = np.std(x)

    X_train.loc[segment,'99quat'] = np.quantile(x,0.99)

    X_train.loc[segment,'50quat'] = np.quantile(x,0.5)

    X_train.loc[segment,'25quat'] = np.quantile(x,0.25)

    X_train.loc[segment,'1quat'] =  np.quantile(x,0.01)

    y_train.loc[segment,'time_to_failure'] = y

    
from keras.layers import Dense

from keras.models import Sequential 

from sklearn.preprocessing import StandardScaler

import gc

gc.collect()
model = Sequential()

model.add(Dense(32,input_shape = (6,),activation = 'relu'))

model.add(Dense(32,activation = 'relu'))

model.add(Dense(32,activation = 'relu'))

model.add(Dense(1))

model.compile(loss = 'mae',optimizer = 'adam',metrics=['accuracy'])
scaler = StandardScaler()

X_scaler = scaler.fit_transform(X_train)

y_train=y_train.values.flatten()
history=model.fit(X_scaler,y_train,epochs = 100)
plt.plot(history.history['loss'])
sub_data = pd.read_csv('../input/sample_submission.csv',index_col = 'seg_id')

X_test = pd.DataFrame(columns = X_train.columns,dtype = np.float32,index = sub_data.index)



for seq in tqdm(X_test.index):

    test_data = pd.read_csv('../input/test/'+seq+'.csv')

    x = test_data['acoustic_data'].values

    X_test.loc[seq,'mean'] = np.mean(x)

    X_test.loc[seq,'std']  = np.std(x)

    X_test.loc[seq,'99quat'] = np.quantile(x,0.99)

    X_test.loc[seq,'50quat'] = np.quantile(x,0.5)

    X_test.loc[seq,'25quat'] = np.quantile(x,0.25)

    X_test.loc[seq,'1quat'] =  np.quantile(x,0.01)

    
X_test_scaler = scaler.transform(X_test)

pred = model.predict(X_test_scaler)
sub_data.to_csv('sub_earthquake.csv',index = False)