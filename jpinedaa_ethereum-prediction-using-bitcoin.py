# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
crypto_markets = pd.read_csv("../input/crypto-markets.csv")
crypto_markets.head()

bitcoin = crypto_markets[crypto_markets['slug']=='bitcoin']

ethereum = crypto_markets[crypto_markets['slug']=='ethereum']

bitcoin.head()

bitcoin_close=(bitcoin['close']-bitcoin['close'].mean())/(bitcoin['close'].max()-bitcoin['close'].min())
ethereum_close=(ethereum['close']-ethereum['close'].mean())/(ethereum['close'].max()-ethereum['close'].min())

bitcoin[['close','volume','market']] = (bitcoin[['close','volume','market']]-bitcoin[['close','volume','market']].mean())/(bitcoin[['close','volume','market']].max()-bitcoin[['close','volume','market']].min())
ethereum[['close','volume','market']] = (ethereum[['close','volume','market']]-ethereum[['close','volume','market']].mean())/(ethereum[['close','volume','market']].max()-ethereum[['close','volume','market']].min())

plt.plot(bitcoin['date'],bitcoin_close,ethereum['date'],ethereum_close)
plt.xlabel('Date')
plt.ylabel('Normalized Closing Price')
plt.show

print(bitcoin.shape,ethereum.shape)

bitcoin_data = bitcoin[['date','close','volume','market']]
bitcoin_data = bitcoin_data[-1035:-1]
bitcoin_data= bitcoin_data.set_index('date')
bitcoin_data.columns = ['bclose', 'bvolume', 'bmarket']
bitcoin_data.loc[:,'bprev day close'] = bitcoin_data['bclose'].shift()
bitcoin_data.loc[:,'bprev day diff'] = bitcoin_data['bprev day close'].diff()
bitcoin_data.loc[:,'bprev day volume'] = bitcoin_data['bvolume'].shift()

bitcoin_data = bitcoin_data.dropna()
bitcoin_data.head()
ethereum_data = ethereum[['date','close','volume','market']] 
ethereum_data = ethereum_data.set_index('date')
ethereum_data.columns = ['eclose', 'evolume', 'emarket']
ethereum_data.loc[:,'eprev day close'] = ethereum_data['eclose'].shift()
ethereum_data.loc[:,'eprev day diff'] = ethereum_data['eprev day close'].diff()
ethereum_data.loc[:,'eprev day volume'] = ethereum_data['evolume'].shift()

ethereum_data = ethereum_data.dropna()
ethereum_data.head()
datas = [bitcoin_data,ethereum_data]
data = pd.concat(datas,axis= 1)
data = data.dropna()
print(data.shape)
data.head()
output_data =ethereum_data[['eclose']]
output_data.loc[:,'next_close']=output_data['eclose'].shift(-1)
output_data=output_data.dropna()
output_data.head()

x_train, x_test, y_train, y_test = train_test_split(data, output_data['next_close'], test_size=0.15)

print(x_train.shape,y_train.shape)
cnn  = MLPRegressor(solver = 'lbfgs')
cnn.fit(x_train,y_train)

cnn.score(x_test,y_test)


data.loc[:,'bprev day close -1'] = data['bprev day close'].shift()
data.loc[:,'eprev day close -1'] = data['eprev day close'].shift()
data.loc[:,'bprev day volume -1'] = data['bprev day volume'].shift()
data.loc[:,'eprev day volume -1'] = data['eprev day volume'].shift()

data.loc[:,'bprev day close -2'] = data['bprev day close -1'].shift()
data.loc[:,'eprev day close -2'] = data['eprev day close -1'].shift()
data.loc[:,'bprev day volume -2'] = data['bprev day volume -1'].shift()
data.loc[:,'eprev day volume -2'] = data['eprev day volume -1'].shift()

data.loc[:,'bprev day diff']=data['bprev day diff'].shift(2)
data.loc[:,'eprev day diff']=data['eprev day diff'].shift(2)

data = data.dropna()
data.head()
output_data = output_data[2:1032]
x_train, x_test, y_train, y_test = train_test_split(data, output_data['next_close'], test_size=0.15)

print(x_train.shape,y_train.shape)
cnn  = MLPRegressor(hidden_layer_sizes = (1000,)*10, solver = 'lbfgs',max_iter=100000)
cnn.fit(x_train,y_train)

cnn.score(x_test,y_test)
output_data.loc[:,'next_close diff'] = output_data.loc[:,'next_close'].diff()
output_data.iloc[0,2] = output_data.iloc[0,1] - output_data.iloc[0,0]

output_data.loc[:,'Price Movement'] = np.where(output_data.loc[:,'next_close diff']<0,'down','up')

output_data.head()
print(data.shape,output_data.shape)
x_train2, x_test2, y_train2, y_test2 = train_test_split(data, output_data['Price Movement'], test_size=0.15)

CNNclass = MLPClassifier(solver = 'lbfgs')
CNNclass.fit(x_train2,y_train2)

CNNclass.score(x_test2,y_test2)