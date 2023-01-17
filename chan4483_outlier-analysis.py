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
df= pd.read_csv("/kaggle/input/s-and-p-index-data-part-2/SP_500_Index_Data.csv", parse_dates=['date'])
df.head()
import tensorflow as tf
pd.options.mode.chained_assignment = None
import seaborn as sns
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

%matplotlib inline

sns.set(style='whitegrid', palette='muted')
rcParams['figure.figsize'] = 14, 8
np.random.seed(1)
tf.random.set_seed(1)

print('Tensorflow version:', tf.__version__)
fig=go.Figure()
fig.add_trace(go.Scatter(x=df.date,y=df.close, mode= 'lines', name= 'close'))
#fig.update_layout(showLegend=True)
fig.show()
train_size= int(len(df)*0.8)
test_size= len(df)- train_size
train, test= df.iloc[0:train_size], df.iloc[train_size:len(df)]
print( 'train size = ', train_size, '\ntest size = ', test_size)
print( 'train shape:',train.shape,'\ntest shape:', test.shape)
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
scaler= scaler.fit(train[['close']])

train['close']= scaler.transform(train[['close']])
test['close']= scaler.transform(test[['close']])
def create_sequences(X,y, time_steps=1):
    Xs,ys=[],[]
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs), np.array(ys)
    
time_steps= 30
X_train, y_train= create_sequences(train[['close']],train.close,time_steps)
X_test, y_test= create_sequences(test[['close']],train.close,time_steps)
X_test.shape
X_train.shape
timesteps= X_train.shape[1]
num_features= X_train.shape[2]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

model = Sequential([
    LSTM(128,input_shape=(timesteps, num_features)),
    Dropout(0.2),
    RepeatVector(timesteps),
    LSTM(128, return_sequences= True),
    Dropout(0.2),
    TimeDistributed(Dense(num_features))
    
])

model.compile(loss='mae', optimizer= 'adam')
model.summary()
es= tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 3, mode='min')
history=model.fit(
X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks= [es],
    shuffle= False
    
)
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'], label= 'Validation Loss')
plt.legend()
X_train_pred= model.predict(X_train)

X_train_pred

train_mae_loss= pd.DataFrame(np.mean(np.abs(X_train_pred - X_train),axis=1),columns= ['Error'])
train_mae_loss
model.evaluate(X_test, y_test)
sns.distplot(train_mae_loss, bins= 20, kde= True)
threshold= 0.65
X_test_pred= model.predict(X_test)
test_mae_loss= np.mean(np.abs(X_test_pred - X_test),axis=1)
test_score_df= pd.DataFrame(test[time_steps:])
test_score_df['loss']= test_mae_loss
test_score_df['threshold']= threshold
test_score_df['anomaly']= test_score_df.loss-test_score_df.threshold>0
test_score_df['close']= test[timesteps:].close
test_score_df.head()
fig=go.Figure()
fig.add_trace(go.Scatter(x=test[time_steps:].date,y=test_score_df.loss, mode= 'lines', name= 'test Loss'))
fig.add_trace(go.Scatter(x=test[time_steps:].date,y=test_score_df.threshold, mode= 'markers', name= 'test threshold'))
fig.show()
anomalies=test_score_df[test_score_df.anomaly==True]
anomalies.tail()
fig=go.Figure()
fig.add_trace(go.Scatter(x=test[time_steps:].date,y=scaler.inverse_transform(test[time_steps:].close), mode= 'lines', name= 'Close Price'))
fig.add_trace(go.Scatter(x=anomalies.date,y=scaler.inverse_transform(anomalies.close), mode= 'markers', name= 'Anomalies'))
fig.show()
