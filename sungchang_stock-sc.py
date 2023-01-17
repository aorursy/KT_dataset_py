# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15,5)

import seaborn as sns

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, TimeDistributed, LSTM, ConvLSTM2D

from keras.layers.convolutional import Conv2D, MaxPooling2D, MaxPooling1D

from keras.datasets import mnist

from keras.utils import np_utils

from keras import regularizers

import sys

import warnings



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# importing data

FDM = pd.read_csv('/kaggle/input/nyse/fundamentals.csv')

PRS = pd.read_csv('/kaggle/input/nyse/prices.csv')

SCR = pd.read_csv('/kaggle/input/nyse/securities.csv')

PSA = pd.read_csv('/kaggle/input/nyse/prices-split-adjusted.csv')
FDM.columns
FDM = FDM.drop(['For Year'], axis = 1)

FDM = FDM.drop(['Earnings Per Share'], axis = 1)

FDM = FDM.drop(['Estimated Shares Outstanding'], axis = 1)

print(FDM.isnull().sum())

print(FDM.shape)
#correlation_matrix = FDM.corr()

#fig = plt.figure(figsize=(12,9))

#sns.heatmap(correlation_matrix,vmax=0.3, vmin=-0.3,linewidths=1)

#plt.show()
PRS
#FDM.head()

PRS_12 = PRS[PRS['date'] == '2012-12-31']

PRS_12 = PRS_12.drop(['open', 'low', 'high', 'volume'], axis = 1)

PRS_13 = PRS[PRS['date'] == '2013-12-31']

PRS_13 = PRS_13.drop(['open', 'low', 'high', 'volume'], axis = 1)

PRS_14 = PRS[PRS['date'] == '2014-12-31']

PRS_14 = PRS_14.drop(['open', 'low', 'high', 'volume'], axis = 1)

PRS_15 = PRS[PRS['date'] == '2015-12-31']

PRS_15 = PRS_15.drop(['open', 'low', 'high', 'volume'], axis = 1)

PRS_16 = PRS[PRS['date'] == '2016-12-30']

PRS_16 = PRS_16.drop(['open', 'low', 'high', 'volume'], axis = 1)

PRS_16['date'] = '2016-12-31'

PRS_last = pd.concat((PRS_12, PRS_13, PRS_14, PRS_15, PRS_16, ), axis = 0)
#https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

#SCR.head()
PRS_X = PRS_12['symbol'].values

all_L = PRS_16['symbol'].values



#NewStock = np.setdiff1d(all_L, PRS_X)

a = np.setdiff1d(PRS_X, all_L)
#plt.plot(PRS[PRS['symbol'] == 'TRIP']['close'])

PRS[PRS['symbol'] == 'TRIP'].head(10)
#FDM.head()

#FDM_all = FDM[FDM['Period Ending'] != '2016-12-31']

#FDM_liv = FDM[FDM['Period Ending'] == '2016-12-31']
FDM_all = FDM

FDM_all.rename(columns = {'Period Ending' : 'date'}, inplace = True)

FDM_all.rename(columns = {'Ticker Symbol' : 'symbol'}, inplace = True)
#FDM_all = FDM_all[FDM_all['symbol'] == 'AAL']

FDM_Merge = FDM_all.merge(PRS_last)

FDM_Merge = FDM_Merge.fillna(0)

FDM_Merge.head(4)
Symbol = FDM_Merge['symbol']

df_list = Symbol.unique()
def df_Normal(df, df_list):

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()

    newData = np.zeros((1, df.shape[1]-3))

    for i in range(0, len(df_list)):

        x_df = df[df['symbol'] == df_list[i]]

        x = x_df.iloc[:, 3:].values

        x_ = sc.fit_transform(x)

        newData = np.concatenate((newData, x_), axis = 0)

    newData = np.delete(newData, (0), axis=0)

    newData_DF = pd.DataFrame(newData)

    newData_DF.columns = df.columns[3:]

    return newData_DF

FDM_Norm = df_Normal(FDM_Merge, df_list)
FDM_Norm
#FDM_Norm.corr().sort_values(by=['close'], axis=0, ascending=False).head(20)

#FDM_Norm.corr().sort_values(by=['close'], axis=0, ascending=True).head(20)
#+

#Net Income, Earnings Before Tax Operating Income, Gross Profit, Retained Earnings, Net Cash Flow-Operating, Total Assets, Total Liabilities, Profit Margin

#당기 순이익, 세전 영업 이익, 순이익, 이익 잉여금, 순 현금 흐름 운영, 총자산, 부채, 이윤

#-

#Accounts Receivable, Capital Expenditures, Effect of Exchange Rate

#채권, 자본 지출, 환율의 영향
FDM_Norm_mod = FDM_Norm[['Net Income', 'Earnings Before Tax', 'Operating Income', 'Gross Profit', 'Retained Earnings', 'Net Cash Flow-Operating', 'Total Assets', 'Total Liabilities', 'Profit Margin', 'Accounts Receivable', 'Capital Expenditures', 'Effect of Exchange Rate', 'close']]
correlation_matrix = FDM_Norm_mod.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(correlation_matrix,vmax=0.4, vmin=-0.4,linewidths=1, annot=True)

plt.show()
#sns.pairplot(FDM_Norm_mod,kind="reg")

#plt.show()
SCR
SCR_sym = SCR[['Ticker symbol']]

SCR_sym.rename(columns = {'Ticker symbol' : 'symbol'}, inplace = True)

SCR_sym
SCR_new = SCR['GICS Sub Industry']

#SCR_new = SCR['GICS Sector']

SCR_new

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

OH = OneHotEncoder()

LE = LabelEncoder()

SCR_new = LE.fit_transform(SCR_new)

SCR_new = SCR_new.reshape(len(SCR_new), 1)

OH.fit(SCR_new)

SCR_new = OH.transform(SCR_new).toarray()

SCR_new = pd.DataFrame(SCR_new)



SCR_new.columns = LE.classes_

SCR_new = pd.concat((SCR_sym, SCR_new), axis=1)

SCR_new
FDM_Close = FDM_Merge[['symbol', 'date', 'close']]

FDM_Close = FDM_Close.merge(SCR_new)

FDM_Close
print(FDM_Close.corr().sort_values(by=['close'], axis=0, ascending=False).iloc[0:8, 0:1])

print(FDM_Close.corr().sort_values(by=['close'], axis=0, ascending=True).iloc[0:8, 0:1])
FDM_Close_mod = FDM_Close[['Internet & Direct Marketing Retail', 'Restaurants', 'Biotechnology', 'Life Sciences Tools & Services', 'Banks', 'MultiUtilities', 'Integrated Telecommunications Services', 'Electric Utilities', 'Oil & Gas Exploration & Production', 'close']]
P_Stock_1 = FDM_Close[FDM_Close[['Internet & Direct Marketing Retail', 'Restaurants', 'Biotechnology', 'Life Sciences Tools & Services']].sum(axis=1) == 1]['symbol'].unique()

M_Stock_1 = FDM_Close[FDM_Close[['Banks', 'MultiUtilities', 'Integrated Telecommunications Services', 'Electric Utilities', 'Oil & Gas Exploration & Production', 'Life Sciences Tools & Services']].sum(axis=1) == 1]['symbol'].unique()

print(P_Stock_1)

print(M_Stock_1)
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

N1 = P_Stock_1[:]

warnings.filterwarnings(action='ignore')

for i in range(0, len(N1)):

    new_gs = PRS[PRS['symbol'] == N1[i]][['date', 'close']]

    new_gs.time = pd.to_datetime(new_gs['date'])

    new_gs.set_index(['date'],inplace=True)

    new_gs['close'] = new_gs[['close']].rolling(window=20).mean()

    new_gs['close'] = new_gs['close'].fillna(0)

    #new_gs['close'] = sc.fit_transform(new_gs[['close']])

    plt.plot(new_gs.time, new_gs['close'], label = N1[i])

    plt.legend()

    plt.grid(True)

    if i % 4 == 3:

        plt.show()

warnings.filterwarnings(action='default')
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

N1 = M_Stock_1[0:]

warnings.filterwarnings(action='ignore')

for i in range(0, len(N1)):

    new_gs = PRS[PRS['symbol'] == N1[i]][['date', 'close']]

    new_gs.time = pd.to_datetime(new_gs['date'])

    new_gs.set_index(['date'],inplace=True)

    new_gs['close'] = new_gs[['close']].rolling(window=20).mean()

    new_gs['close'] = new_gs['close'].fillna(0)

    #new_gs['close'] = sc.fit_transform(new_gs[['close']])

    plt.plot(new_gs.time, new_gs['close'], label = N1[i])

    plt.legend()

    plt.grid(True)

    if i % 3 == 2:

        plt.show()
PRS_list = PRS['symbol'].unique()

#print(PRS_list)

PRS_list = PRS_list[30:50]

print(PRS_list)

#PRS_list = ['CL']
seq_length = 31

seq_length_div = int(seq_length / 2)

data_dim = 5



def build_dataset(time_series, seq_length):

    

    dataX = []

    dataY = []

    for i in range(0, (len(time_series) - seq_length)):

        _x = time_series[i:i + seq_length-1, :data_dim]

        _y = time_series[i + seq_length, [-1]]

        

        from sklearn.preprocessing import MinMaxScaler

        sc = MinMaxScaler()

        _x = sc.fit_transform(_x)



        dataX.append(_x)

        dataY.append(_y)

        

    return np.array(dataX), np.array(dataY)
def build_Stockset(Stocklist, seq_length, data_dim):

    trainX = np.zeros((1,(seq_length-1)*data_dim))

    trainY = np.zeros((1,1))

    for i in range(0, len(Stocklist)):

        PRS_rand = Stocklist[i]

        PRS_symbol_close = PRS[PRS['symbol'] == PRS_rand]

        PRS_symbol_close['close+1'] = PRS_symbol_close['close'].shift(-5)

        PRS_symbol_close = PRS_symbol_close.iloc[:-1]

        PRS_symbol_close['Fluctuation'] = (PRS_symbol_close['close+1']/PRS_symbol_close['close'] -1)*100

        ## lamda   elseif 하는법 = True if() false,  (false에서 다시 True if false)에서 다시 True if false

        PRS_symbol_close['target'] = PRS_symbol_close['Fluctuation'].apply(lambda x : 2 if x > 5 else (0 if x < -5 else 1)) 

        PRS_symbol_close = PRS_symbol_close.drop(['close+1', 'symbol', 'date'], axis = 1)

        Case0 = PRS_symbol_close.iloc[:-5].values



        Case0X, Case0Y = build_dataset(Case0, seq_length)

        Case0X = np.reshape(Case0X, [Case0X.shape[0], -1])

        Case0X.resize(Case0X.shape[0], data_dim*(seq_length-1))

        

        trainX = np.concatenate([trainX, Case0X], axis=0)

        trainY = np.concatenate([trainY, Case0Y], axis=0)

    return trainX, trainY

trainX, trainY = build_Stockset(PRS_list, seq_length, data_dim)

trainX = np.delete(trainX, (0), axis=0)

trainY = np.delete(trainY, (0), axis=0)

trainY = np_utils.to_categorical(trainY)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(trainX,trainY,test_size=0.3, random_state = 42)
import keras

# NN의 정의

# SGD 보다 아담이 잘됨

NNinput = X_train.shape[1]

act = 'relu'

opt = 'Adam'

los = 'categorical_crossentropy'



model = Sequential()

model.add(Dense(300, activation = act, input_shape = [NNinput,]))

model.add(Dense(400, activation = act))

model.add(Dense(200, activation = act))

model.add(Dense(3, activation = 'softmax'))

model.compile(optimizer = opt, loss = los, metrics = ['accuracy'])

#model.summary()
batch_size = 486

epoch = 100

history = model.fit(X_train, y_train, epochs = epoch, batch_size = batch_size, verbose = 1, validation_data = [X_test, y_test])
a = (y_test[:, :1] == 1).sum()

b = (y_test[:, :2] == 1).sum()

c = (y_test[:, :3] == 1).sum()

print(a, b, c)
(a + c) / b
########### test

PRS_list = PRS['symbol'].unique()

PRS_list = PRS_list[206:207]

print(PRS_list)
testX, testY = build_Stockset(PRS_list, seq_length, data_dim)

testX = np.delete(testX, (0), axis=0)

testY1 = np.delete(testY, (0), axis=0)

testY1 = np_utils.to_categorical(testY1)
pred = model.predict(testX)

#pred[50:100]
pred1 = pred[:, 0]

pred1 = np.where(pred1 > 0.4, pred1, 0)

pred1 = pred1.reshape(len(pred1),1)

pred2 = pred[:, 2]

pred2 = np.where(pred2 > 0.4, pred2, 0)

pred2 = pred2.reshape(len(pred2),1)

testY = np.argmax(testY1,axis=1)

testY = testY.reshape(len(testY),1)
Scale = 30

Shift = 45

L1 = 1500

L2 = 1700

PRS_rand = PRS_list[0]

test_PRS = PRS[PRS['symbol'] == PRS_rand][['date', 'close']].iloc[30:].values

test_PRS = test_PRS[L1:L2]

pred1_ = pred1[L1:L2]*Scale + Shift

pred2_ = pred2[L1:L2]*Scale + Shift + 10

testY_ = testY[L1:L2]*Scale/4 + Shift+20



test_PRS = np.concatenate((test_PRS, pred1_, pred2_, testY_), axis=1)



test_PRS = pd.DataFrame(test_PRS)

test_PRS.columns = ['date', 'close', 'Sell', 'Buy', 'test']



test_PRS.time = pd.to_datetime(test_PRS['date'])

test_PRS.set_index(['date'],inplace=True)

plt.plot(test_PRS.time, test_PRS['close'],'o-', label = "close")

plt.plot(test_PRS.time, test_PRS['Sell'],'-', label = "Sell")

plt.plot(test_PRS.time, test_PRS['Buy'],'-', label = "Buy")

plt.plot(test_PRS.time, test_PRS['test'],'-', label = "Real")

plt.legend()

#plt.plot(testY*Scale)

#plt.plot(pred1[L1:L2]*Scale)

#plt.plot(pred2[L1:L2]*Scale + Shift)



## 팔면 40봉동안 사지않고  다시 매수봉이나오면 사면됨.