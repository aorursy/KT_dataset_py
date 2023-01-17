import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns 

np.random.seed(42)

from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.callbacks import TensorBoard
#read data
dtrain = pd.read_csv('../input/train.csv')
dtest = pd.read_csv('../input/test.csv')
train_len = len(dtrain)
test_len = len(dtest)


#process all data
df = pd.concat([dtrain,dtest],axis=0)
df.head()
df.isnull().sum()
#fill in missing data markdown and weekly_sales
df.fillna(0,inplace=True)
df.isnull().sum()
#check data type
df.dtypes

# define dummies for type
df['Type'] = 'Type_' + df['Type'].astype(str)

# Get dummies
dummies = pd.get_dummies(df['Type'])

# Add dummies to dataframe
df = pd.concat([df,dummies],axis=1)

# Remove original Type column
del df['Type']
df.dtypes
#define dummies for Store and Dept
#df['Dept'] = 'Dept_' + df['Dept'].astype(str)
#df['Store'] = 'Store_' + df['Store'].astype(str)
# Get dummies
#dum_dept = pd.get_dummies(df['Dept'])
#dum_store = pd.get_dummies(df['Store'])
# Add dummies to dataframe
#df = pd.concat([df,dum_store,dum_dept],axis=1)

# Remove original store and dept column
# del df['Store']
# del df['Dept']
#df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#df['Week_Number'] = df['Date'].dt.strftime('%U')
#week_number = pd.get_dummies(df['Week_Number'])
#df= pd.concat([df,week_number], axis=1)
del df['Date']
del df['MarkDown1']
del df['MarkDown2']
del df['MarkDown3']
del df['MarkDown4']
del df['MarkDown5']
df.dtypes

df.shape
#seperate in train and test data
dtrain = df.iloc[:train_len]
dtest = df.iloc[train_len:]
X_test = dtest.loc[:, dtest.columns != 'Weekly_Sales'].values
Y_test = dtest['Weekly_Sales'].values
X_train = dtrain.loc[:, dtrain.columns != 'Weekly_Sales'].values
Y_train = dtrain['Weekly_Sales'].values
X_train.shape
df['diff']=df.groupby(['Store','Dept'])['Weekly_Sales'].shift(1)
df['diff1']=df.groupby(['Store','Dept'])['Weekly_Sales'].shift(0)
df['difference']=df['diff']-df['diff1']
df1=df['difference']
df1=df1.fillna(0).values
df1
Y_diff = np.diff(Y_train)
plt.plot(df['difference'])
from statsmodels.tsa.arima_model import ARIMA

#Y_diff=Y_diff.as_matrix()
model = ARIMA(df1,order=(1,1,0))
x = model.fit(disp=0)
x.summary()
import sklearn.metrics
testfile = pd.read_csv('../input/test.csv')
testfile
Y_pred = x.predict()
Ymae  = sklearn.metrics.mean_absolute_error(df1[-139119:], Y_pred[-139119:])
Ymae

submission = pd.DataFrame({'id':testfile['Store'].map(str) + '_' + testfile['Dept'].map(str) + '_' + testfile['Date'].map(str),
                          'Weekly_Sales':Y_pred[-139119:].flatten()})

submission.to_csv('submission.csv',index=False)
from keras.optimizers import adam
#make the model
model = Sequential()

#add first layer

model.add(Dense(64,activation = 'tanh', input_dim = 9))

# Input layer normalization
#model.add(BatchNormalization())

model.add(Dense(32,activation = 'tanh'))
#add output layer
model.add(Dense(1,activation = 'sigmoid'))
# Setup adam optimizer
adam_optimizer=adam(lr=0.01,
                beta_1=0.9, 
                beta_2=0.999, 
                epsilon=1e-08)

model.compile(optimizer = adam_optimizer, loss = 'mae')

#set up tenserboard
tensorboard  = TensorBoard(log_dir = './logs/' + 'tanh one hidden layer')

#train model
NN_tanh = model.fit(X_train, Y_train, epochs = 5, batch_size = X_train.shape[0], 
                    callbacks = [tensorboard], verbose = 0)

#NN_tanh = model.fit(X,y,batch_size=2048,epochs=5)

plt.style.use('fivethirtyeight')
plt.plot(NN_tanh.history['loss'], label = '2 tanh hidden layer')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
NN_tanh.history['loss']
#make the model
model = Sequential()

#add first layer

model.add(Dense(64,activation = 'relu', input_dim = 193))
# Add dropout layer
model.add(Dropout(rate=0.5))

model.add(Dense(32,activation = 'relu'))
# Add dropout layer
model.add(Dropout(rate=0.5))

#add output layer
model.add(Dense(1,activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'mae')

#set up tenserboard
tensorboard  = TensorBoard(log_dir = './logs/' + 'relu hidden layer')

#train model
NN_relu = model.fit(X_train, Y_train, epochs = 5, batch_size = 2048,#X_train.shape[0], 
                    callbacks = [tensorboard], verbose = 0)
plt.style.use('fivethirtyeight')
plt.plot(NN_relu.history['loss'], label = '2 relu hidden layer')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
Y_train.plot()
plot.show()
plt.plot(Y_train)
from statsmodels.tsa.arima_model import ARIMA
