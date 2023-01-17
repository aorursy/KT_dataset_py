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
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import matplotlib
import datetime as dt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
jan_to_jun_2009 = pd.read_csv("../input/thesis/jan_to_jun_2009.csv",index_col=0)
jul_to_dec_2009 = pd.read_csv("../input/thesis/jul_to_dec_2009.csv",index_col=0)
jan_to_jun_2010 = pd.read_csv("../input/thesis/jan_to_aug_2010.csv",index_col=0)
jul_to_dec_2010 = pd.read_csv("../input/thesis/sep_to_dec_2010.csv",index_col=0)
jan_to_jun_2011 = pd.read_csv("../input/thesis/jan_to_jun_2011.csv",index_col=0)
jul_to_dec_2011 = pd.read_csv("../input/thesis/jul_to_dec_2011.csv",index_col=0)
jan_to_jun_2012 = pd.read_csv("../input/thesis/jan_to_jun_2012.csv",index_col=0) 
jul_to_dec_2012 = pd.read_csv("../input/thesis/jul_to_dec_2012.csv",index_col=0)
jan_to_jun_2013 = pd.read_csv("../input/thesis/jan_to_jun_2013.csv",index_col=0)
jul_to_dec_2013 = pd.read_csv("../input/thesis/jul_to_dec_2013.csv",index_col=0)
jan_to_jun_2014 = pd.read_csv("../input/thesis/jan_to_jun_2014.csv",index_col=0)
jul_to_dec_2014 = pd.read_csv("../input/thesis/jul_to_dec_2014.csv",index_col=0)
jan_to_jun_2015 = pd.read_csv("../input/thesis/jan_to_jun_2015.csv",index_col=0)
jul_to_dec_2015 = pd.read_csv("../input/thesis/jul_to_dec_2015.csv",index_col=0)
jan_to_jul_2016 = pd.read_csv("../input/thesis/jan_to_jul_2016.csv",index_col=0)

df = pd.concat([jan_to_jun_2009,jul_to_dec_2009,jan_to_jun_2010,jul_to_dec_2010,jan_to_jun_2011,jul_to_dec_2011,jan_to_jun_2012,jul_to_dec_2012,jan_to_jun_2013,jul_to_dec_2013,jan_to_jun_2014,jul_to_dec_2014,jan_to_jun_2015,jul_to_dec_2015,jan_to_jul_2016],axis=0)

df = df.loc[df['RFDE_INSTR_TYPE'] == 'REG_DL_INSTR_EQ']
df = df.rename(columns={'VALUE (in Rs)': 'Sale'})
df['TR_DATE'] = df['TR_DATE'].astype('datetime64[D]')
df1 = pd.DataFrame()
df2 = pd.DataFrame()
df1['Date'] = df['TR_DATE']
df1['Sale'] = df['Sale']
df2['Date'] = df['TR_DATE']
df2['Inflation-Rate'] = df['Inflation-Rate']
df2['BSE_Close'] = df['BSE_Close']
df2['FDI-Inward'] = df['FDI-Inward']
df2['IIP'] = df['IIP']
df2['unemployment-rate'] = df['unemployment-rate']
df2['forex'] = df['foreign-exchange']
df2['GDP-Growth'] = df['GDP-Growth-Rate']
df2['FDI-Growth'] = df['FDI-Growth-Rate']
df2['twitter'] = df['twitter-sentiment']

df1 = df1.groupby(['Date']).sum()

df1 = df1.reset_index(level='Date')
df2 = df2.groupby(['Date'], as_index=False).mean()


df1['BSE_Close'] = df1['Date'].map(df2.set_index('Date')['BSE_Close'])
df1['FDI-Inward'] = df1['Date'].map(df2.set_index('Date')['FDI-Inward'])

df1['IIP'] = df1['Date'].map(df2.set_index('Date')['IIP'])
df1['forex'] = df1['Date'].map(df2.set_index('Date')['forex'])
df1['twitter'] = df1['Date'].map(df2.set_index('Date')['twitter'])
df1['U-R'] = df1['Date'].map(df2.set_index('Date')['unemployment-rate'])
df1['Inflation-Rate'] = df1['Date'].map(df2.set_index('Date')['Inflation-Rate'])
df1['GDP-Growth'] = df1['Date'].map(df2.set_index('Date')['GDP-Growth'])
df1['FDI-Growth'] = df1['Date'].map(df2.set_index('Date')['FDI-Growth'])

test = df1
test['U-R'] = test['U-R'].replace(to_replace=0, method='ffill')
test['FDI-Inward'] = test['FDI-Inward'].fillna(method='ffill')

test['twitter'] = test['twitter'].replace(to_replace=-3.000000, method='ffill')


abc = pd.DataFrame(data=test.values,columns=test.columns)

del test['Date']
del test['Sale']
data = []
data = test


data['GDP-Growth'] = data['GDP-Growth'].div(100)
data['Inflation-Rate'] = data['Inflation-Rate'].div(100)
data['U-R'] = data['U-R'].div(100)

data_UR = data['U-R'].to_numpy()
data_UR = data_UR.reshape(len(data_UR),1)
data_Inflation_Rate = data['Inflation-Rate'].to_numpy()
data_Inflation_Rate = data_Inflation_Rate.reshape(len(data_Inflation_Rate),1)
data_GDP_Growth = data['GDP-Growth'].to_numpy()
data_GDP_Growth = data_GDP_Growth.reshape(len(data_GDP_Growth),1)
data_FDI_Growth = data['FDI-Growth'].to_numpy()
data_FDI_Growth = data_FDI_Growth.reshape(len(data_FDI_Growth),1)

data_forex = data.forex.values
data_forex = data_forex.reshape(len(data_forex),1)
data_IIP = data.IIP.values
data_IIP = data_IIP.reshape(len(data_forex),1)
data_FDI_Inward = data['FDI-Inward'].values
data_FDI_Inward = data_FDI_Inward.reshape(len(data_forex),1)
data_BSE_Close = data.BSE_Close.values
data_BSE_Close = data_BSE_Close.reshape(len(data_BSE_Close),1)
data_twitter = data['twitter'].to_numpy()
data_twitter = data_twitter.reshape(len(data_twitter),1)


scaler1 = MinMaxScaler(feature_range=(0, 1))
data_BSE_Close_normalize = scaler1.fit_transform(data_BSE_Close)
scaler2 = MinMaxScaler(feature_range=(0, 1))
data_FDI_Inward_normalize = scaler2.fit_transform(data_FDI_Inward)
scaler3 = MinMaxScaler(feature_range=(0, 1))
data_IIP_normalize = scaler3.fit_transform(data_IIP)
scaler4 = MinMaxScaler(feature_range=(0, 1))
data_forex_normalize = scaler4.fit_transform(data_forex)
scaler5 = MinMaxScaler(feature_range=(0,1))
data_twitter_normalize = scaler5.fit_transform(data_twitter)

data_normalize = np.concatenate((data_BSE_Close_normalize,data_FDI_Inward_normalize,data_IIP_normalize,data_forex_normalize,data_twitter_normalize,data_UR,data_Inflation_Rate,data_GDP_Growth,data_FDI_Growth),axis=1)





from keras import optimizers
from matplotlib import pyplot
from keras.layers import Dropout
import tensorflow as tf
# lstm autoencoder recreate sequence
from numpy import array
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
from keras.layers import LeakyReLU
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import Dense
from keras.layers.recurrent import LSTM
batch = 4
lr = 0.0001
X_train = data_normalize.reshape((data_normalize.shape[0],1, data_normalize.shape[1]))

visible1 = Input(shape=(1,9))


hidden1 = LSTM(50,activation='relu',input_shape=(1,9),return_sequences=True)(visible1)

#dropout1 = Dropout(0.20)

hidden2 = LSTM(1,activation='relu')(hidden1)

repeatvector = RepeatVector(1, name="repeater")(hidden2)

hidden3 = LSTM(50, activation='relu', return_sequences=True)(repeatvector)

#dropout2 = Dropout(0.20)

output = TimeDistributed(Dense(9))(hidden3)

model_autoencoder = Model(inputs=visible1,outputs=output)

adam = optimizers.Adam(lr)

model_autoencoder.compile(loss='mae', optimizer=adam)


print(model_autoencoder.summary())

model_autoencoder.fit(X_train, X_train, epochs=200,batch_size=batch,verbose=1)
encoder = Model(inputs=visible1, outputs=[hidden2])
train_encoded = encoder.predict(X_train)

yhat = model_autoencoder.predict(X_train)


yhat = yhat.reshape(2052,9)
array1 = yhat[:,:1]
array2 = yhat[:,1:2]
array3 = yhat[:,2:3]
array4 = yhat[:,3:4]
array5 = yhat[:,4:5]

array1 = scaler1.inverse_transform(array1)
array2 = scaler2.inverse_transform(array2)
array3 = scaler3.inverse_transform(array3)
array4 = scaler4.inverse_transform(array4)
array5 = scaler4.inverse_transform(array5)

data_encoder = np.concatenate((array1,array2,array3,array4,array5,yhat[:,5:],),axis=1)

x = pd.DataFrame(data=data_encoder)
x


data
from keras.utils import plot_model
plot_model(model_autoencoder, show_shapes=True, to_file='lstm_encoder.png')
df_test_forecast = pd.DataFrame()
df_test_forecast['Sale'] = abc['Sale']

train_encoded_feature = pd.DataFrame(data=train_encoded)

df_test_forecast['encoded-feature'] = train_encoded_feature[0]



from numpy import array
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    #print(len(sequences))
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
		# gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix -1,:], sequences[end_ix -1,0]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

array_test_forecast = df_test_forecast.iloc[:2046,0:].to_numpy()
xforecast = df_test_forecast.iloc[2046:2051,0:].to_numpy()

scaler10 = MinMaxScaler(feature_range=(0, 1))
scaler10 = scaler10.fit(array_test_forecast[:,:1])
normalized_values = scaler10.transform(array_test_forecast[:,:1])

scaler11 = MinMaxScaler(feature_range=(0, 1))
scaler11 = scaler11.fit(xforecast[:,:1])
normalized_values_x_forcaste = scaler11.transform(xforecast[:,:1])

normalized_values_forecast = np.concatenate((normalized_values,array_test_forecast[:,1:2]),axis=1)
x_forecaste_normalize = np.concatenate((normalized_values_x_forcaste,xforecast[:,1:2]),axis=1)

xForTesting, yForTesting = split_sequences(normalized_values_forecast,6)

xt = xForTesting[:1500,:,:]
yt = yForTesting[:1500]
xv = xForTesting[1500:,:,:]
yv=yForTesting[1500:]

n_features = xForTesting.shape[2]


model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(5, n_features)))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(LSTM(100, activation='relu'))
model.add(Dense(1))
adam = optimizers.Adam(lr)
model.compile(loss='mae', optimizer=adam)
# fit model
history = model.fit(xt, yt,validation_data=(xv, yv),epochs=300,batch_size = 14,verbose=1,shuffle=False)
# demonstrate prediction


pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()



x_forecaste_normalize = x_forecaste_normalize.reshape((1, len(x_forecaste_normalize), 2))
yhat = model.predict(x_forecaste_normalize, verbose=0)

z = np.zeros((1,1), dtype=int)

yhat = np.append(yhat,z,axis=1)

yhat = scaler11.inverse_transform(yhat[:,0:1])


yhat
df_test_forecast
plot_model(model, show_shapes=True, to_file='lstm_encoder.png')
