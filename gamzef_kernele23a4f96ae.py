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
import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
#train.csv'deki datayı okur, ilk data tipi acoustic_data, integer, ikincisi time to failure float. 
train_data = pd.read_csv('/kaggle/input/LANL-Earthquake-Prediction/train.csv', nrows=6000000, dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float32})
pd.options.display.precision = 15
train_data.head()
#visualize of 1% of dataset (her 100 elemanda bir veri alıyor.)
train_ad_sample_df = train_data['acoustic_data'].values[::100]
train_ttf_sample_df = train_data['time_to_failure'].values[::100]

def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title = "Acoustic data and time to failure: 1% sampled data"):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.title(title)
    plt.plot(train_ad_sample_df, color='r')
    ax1.set_ylabel('acoustic data', color='r')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))
    ax2 = ax1.twinx()
    plt.plot(train_ttf_sample_df, color='b')
    ax2.set_ylabel('time to failure', color='b')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)

plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)
del train_ad_sample_df
del train_ttf_sample_df

#There is a point before the actual earthquake 
#where there's a spike in acoustic activity seismographic activity.
#import
train_data = pd.read_csv('/kaggle/input/LANL-Earthquake-Prediction/train.csv',dtype = {'acoustic_data':np.int16,'time_to_failure':np.float32})
rows = 150_000
segments = int(np.floor(train_data.shape[0] / rows))  #630 milyon / 150000 = 4194

X_train = pd.DataFrame(index = range(segments),dtype = np.float32,columns = ['mean','std','99quat','50quat','25quat','1quat'])
y_train = pd.DataFrame(index = range(segments),dtype = np.float32,columns = ['time_to_failure'])
for segment in tqdm(range(segments)):
    x = train_data[segment*rows:segment*rows+rows]
    y = x['time_to_failure'].values[-1]
    x = x['acoustic_data'].values
    X_train.loc[segment,'mean'] = np.mean(x)
    X_train.loc[segment,'std']  = np.std(x)
    X_train.loc[segment,'99quat'] = np.quantile(x,0.99)
    X_train.loc[segment,'50quat'] = np.quantile(x,0.5)
    X_train.loc[segment,'25quat'] = np.quantile(x,0.25)
    X_train.loc[segment,'1quat'] =  np.quantile(x,0.01)
    y_train.loc[segment,'time_to_failure'] = y

X_train.head()
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X_train)
from keras.layers.core import Dropout

model = Sequential()
model.add(Dense(256, activation="relu", input_shape=(6,)))
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(96, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1, activation="linear"))

optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss='mae')
history = model.fit(X_scaler,y_train.values.flatten(),epochs = 500, batch_size=32)
plt.plot(history.history['loss'])

#https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
model.summary()     #256(neurons in the first layer) * 6(input) + 256 (bias)  for layer1
                    #256 * 128 + 128(bias) for layer2
                    #96 * 128 + 96 for layer 3
                    #1 * 96 + 1 for layer 4
sub_data = pd.read_csv('/kaggle/input/LANL-Earthquake-Prediction/sample_submission.csv',index_col = 'seg_id')
X_test = pd.DataFrame(columns = X_train.columns,dtype = np.float32,index = sub_data.index)
for seq in tqdm(X_test.index):
    test_data = pd.read_csv('/kaggle/input/LANL-Earthquake-Prediction/test/'+seq+'.csv')     #2624 .csv file and each of them has 150000 segments(row)
    x = test_data['acoustic_data'].values
    X_test.loc[seq,'mean'] = np.mean(x)
    X_test.loc[seq,'std']  = np.std(x)
    X_test.loc[seq,'99quat'] = np.quantile(x,0.99)
    X_test.loc[seq,'50quat'] = np.quantile(x,0.5)
    X_test.loc[seq,'25quat'] = np.quantile(x,0.25)
    X_test.loc[seq,'1quat'] =  np.quantile(x,0.01)

X_test_scaler = scaler.transform(X_test)
pred = model.predict(X_test_scaler)
sub_data['time_to_failure'] = pred
sub_data['seg_id'] = sub_data.index
sub_data.head()
sub_data.to_csv('sub_earthquake.csv',index = False)
sub_data.time_to_failure.describe()
ss = pd.read_csv('sub_earthquake.csv', nrows=1000, dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float32})
ss.plot(kind='hist',color='blue', bins= 100, figsize=(15, 5), alpha=0.5)
plt.plot(ss['time_to_failure'].values[::100])
train_data['time_to_failure'].plot(kind='hist',color='green', bins= 100, figsize=(15, 5), alpha=0.5)
plt.plot(train_data['time_to_failure'].values[::100])
