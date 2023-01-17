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



#ALL Files are concatenated together 



df = pd.concat([jan_to_jun_2009,jul_to_dec_2009,jan_to_jun_2010,jul_to_dec_2010,jan_to_jun_2011,jul_to_dec_2011,jan_to_jun_2012,jul_to_dec_2012,jan_to_jun_2013,jul_to_dec_2013,jan_to_jun_2014,jul_to_dec_2014,jan_to_jun_2015,jul_to_dec_2015,jan_to_jul_2016],axis=0)



#Instrument type Equity is selected.



df = df.loc[df['RFDE_INSTR_TYPE'] == 'REG_DL_INSTR_EQ']



#Renaming of the column 



df = df.rename(columns={'VALUE (in Rs)': 'Sale'})



#Converting the TR_DATE columns which denotes the transaction date into date time formate.



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



#Data is day wise distributed. Thus summing together to get the total sum of Equity instrument brought per day 



df1 = df1.groupby(['Date']).sum()



df1 = df1.reset_index(level='Date')



#Getting the exact value of different macro-economic variables per day. 



df2 = df2.groupby(['Date'], as_index=False).mean()



#Formulating the dataset with columns Date, Sale, and macro-economic variables. 



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



#For the year 2009 there are 3 dates for which twitter sentiment is missing. They are replaced by the previous values.



test['twitter'] = test['twitter'].replace(to_replace=-3.000000, method='ffill')





abc = pd.DataFrame(data=test.values,columns=test.columns)



#The column of Stock Sale is deleted and Column of Date is deleted, before passing it to the autoencoder 



del test['Date']

del test['Sale']

data = []

data = test





#GDP-Growth, Inflation-Rate, Unemployment-Rate are percentage values given in whole number formate.    



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





# BSE_Close, FDI_inward, IIP, foreign-exchange are normalized using minmax scaler 



scaler1 = MinMaxScaler(feature_range=(0, 1))

data_BSE_Close_normalize = scaler1.fit_transform(data_BSE_Close)

scaler2 = MinMaxScaler(feature_range=(0, 1))

data_FDI_Inward_normalize = scaler2.fit_transform(data_FDI_Inward)

scaler3 = MinMaxScaler(feature_range=(0, 1))

data_IIP_normalize = scaler3.fit_transform(data_IIP)

scaler4 = MinMaxScaler(feature_range=(0, 1))

data_forex_normalize = scaler4.fit_transform(data_forex)

#scaler5 = MinMaxScaler(feature_range=(0,1))

#data_twitter_normalize = scaler5.fit_transform(data_twitter)



#All of the normalized data plus percentage valued features are concatenated together 



data_normalize = np.concatenate((data_BSE_Close_normalize,data_FDI_Inward_normalize,data_IIP_normalize,data_forex_normalize,data_twitter,data_UR,data_Inflation_Rate,data_GDP_Growth,data_FDI_Growth),axis=1)



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

from keras.layers import Input

from keras.layers import concatenate

from keras.layers import Dense

from keras.layers.recurrent import LSTM

batch = 19

lr = 0.0001

X_train = data_normalize.reshape((data_normalize.shape[0],1,data_normalize.shape[1]))

X_train.shape


#Defining the input shape for the model  



visible1 = Input(shape=(1,9))


#Layer 1



hidden1,state_h_1,state_c_1= LSTM(50,activation='tanh',input_shape=(1,9),return_sequences=True,return_state= True)(visible1)





hidden_state_1 = Model(inputs = visible1,outputs=[hidden1,state_h_1,state_c_1])





#Layer 2



hidden2,state_h_2,state_c_2 = LSTM(1,activation='tanh',return_state=True)(hidden1)





hidden_state_2 = Model(inputs = visible1,outputs=[hidden2,state_h_2,state_c_2])



#Layer 3



repeatvector = RepeatVector(1, name="repeater")(hidden2)

repearvector_state = Model(inputs=visible1,output=repeatvector)



#Layer 4



hidden3,state_h_3,state_c_3 = LSTM(50,activation='tanh', return_sequences=True,return_state=True)(repeatvector)



hidden_state_3 = Model(inputs = visible1,outputs=[hidden3,state_h_3,state_c_3])







#Layer 5



output = Dense(9)(hidden3)



model = Model(inputs=visible1,outputs=output)



adam = optimizers.Adam(lr)



model.compile(loss='mse', optimizer=adam)





print(model.summary())
#FITTING THE MODEL



model.fit(X_train, X_train, epochs=200,batch_size=batch,verbose=1)
#LAYER 1 OUTPUT FROM THE LAYER ,HIDDEN STATE, AND THE CELL STATE ARE OUTPUTED  



Hidden_State_1_Output = hidden_state_1.predict(X_train)

Hidden_State_1_Output[2].shape 

hidden_state_1_outut_return_sequence = Hidden_State_1_Output[0]

hidden_state_1_outut_return_sequence.shape
#LAYER 2 OUTPUT FROM THE LAYER ,HIDDEN STATE, AND THE CELL STATE ARE OUTPUTED 



Hidden_State_2_Output = hidden_state_2.predict(X_train)

hidden_state_2_outut_return_sequence = Hidden_State_2_Output[0]

hidden_state_2_outut_return_sequence.shape
#LAYER 3 OUTPUT FROM THE LAYER ,HIDDEN STATE, AND THE CELL STATE ARE OUTPUTED 



repeat_vector_output = repearvector_state.predict(X_train)

repeat_vector_output.shape
#LAYER 4 OUTPUT FROM THE LAYER ,HIDDEN STATE, AND THE CELL STATE ARE OUTPUTED 



#Reconstruction Layer 



hidden_state_3_outut_return_sequence = hidden_state_3.predict(X_train)

hidden_state_3_outut_return_sequence[0].shape
from keras.utils import plot_model

plot_model(model, show_shapes=True, to_file='lstm_autoencoder.png')
#encoder = Model(inputs=visible1, outputs=[hidden2])

#train_encoded = encoder.predict(X_train)

yhat = model.predict(X_train)

yhat.shape
#Predicted Yhat reshape into original 2D array 



yhat = yhat.reshape(2052,9)



#Storing individual columns to be rescaled back to original values 



#BSE_Close 

array1 = yhat[:,:1]



#FDI-Inward

array2 = yhat[:,1:2]



#IIP

array3 = yhat[:,2:3]



#Foreign-Exchange 

array4 = yhat[:,3:4]



#Twitter Sentiment Score 

#array5 = yhat[:,4:5]



array1 = scaler1.inverse_transform(array1)

array2 = scaler2.inverse_transform(array2)

array3 = scaler3.inverse_transform(array3)

array4 = scaler4.inverse_transform(array4)

#array5 = scaler5.inverse_transform(array5)





#Concatenated all the features together 



data_autoencoder = np.concatenate((array1,array2,array3,array4,yhat[:,4:],),axis=1)

#Data after Autoencoder Operation 



DataAfterAutoencoder = pd.DataFrame(data=data_autoencoder)

DataAfterAutoencoder
#Original Data 



data