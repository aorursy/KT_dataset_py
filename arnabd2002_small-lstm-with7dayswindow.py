# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from keras.utils import to_categorical
from sklearn.model_selection import TimeSeriesSplit
import datetime
# Any results you write to the current directory are saved as output.
stockFileNames=os.listdir('../input/Data/Stocks/')
stockSymbol=[]
col_names=['Date','Open','High','Low','Close','Volume','c']
dataList=list()
for f in stockFileNames:
    symbol=[str.upper(f.split('.')[0])]
    fileName='../input/Data/Stocks/'+f
    fp=open(fileName,'r')
    if len(fp.read())>0:
        data=np.array(pd.read_csv(fileName,header=None))
        for d in data[1:]:
            dataList.append(d)
            stockSymbol.append(symbol)
        
    fp.close()
print('DONE!!')
np.shape(dataList),np.shape(stockSymbol)
df1=pd.DataFrame(dataList,columns=col_names)
df2=pd.DataFrame(stockSymbol,columns=['SYMBOL'])
AllSymbolList=np.unique(df2)
df=pd.concat([df1,df2],axis=1)
df.head(5)
df['Date'][:10]
allCloseDF=df[['Date','Close','SYMBOL']]
allCloseDF.head(5)
def getCloseDataFromSymbols(df,symbol_list):
    return df[df['SYMBOL'].isin(symbol_list)][['Date','Close']]

resultDF=getCloseDataFromSymbols(allCloseDF,['AMZN','FB','GOOGL','NFLX'])
len(resultDF)
def getPastSequenceData(df,window):
    #df=np.array(df)
    X=[]
    y=[]
    for i in range(1,len(df)-window,window):
        #print(df[i-1:i+window-1],df[i+window-1])
        date=df['Date'].iloc[i]
        date=date.replace('-','')
        #print(date)
        date=np.array(date)
        vals=np.array(df['Close'].iloc[i-1:i+window-1].values)
        temp=np.hstack([date,vals])
        X.append(temp)
        y.append(df['Close'].iloc[i+window-1])
    return X,y
X,y=getPastSequenceData(resultDF,7)
X=np.array(X)
print(X.shape)
X=X.reshape(X.shape[0],X.shape[1],1)
y=np.array(y)
np.shape(X),np.shape(y)
X[0],y[0]
tsSplit=TimeSeriesSplit(n_splits=5)
for train_index,test_index in tsSplit.split(X):
    X_train, X_test = X[:len(train_index)], X[len(train_index): (len(train_index)+len(test_index))]
    y_train, y_test = y[:len(train_index)], y[len(train_index): (len(train_index)+len(test_index))]
np.shape(X_train),np.shape(y_train),np.shape(X_test),np.shape(y_test)
def createLSTM():
    model=Sequential()
    model.add(LSTM(200,input_shape=(8,1),return_sequences=True))
    model.add(LSTM(200))
    model.add(Dense(200))
    model.add(Dropout(0.20))
    model.add(Dense(50))
    model.add(Dropout(0.20))
    model.add(Dense(1))
    return model
model=createLSTM()
model.summary()
model.compile(loss='mae',optimizer='adam',metrics=['acc'])
model.fit(X_train,y_train,epochs=300,verbose=1)
model.evaluate(X_test,y_test)
test=allCloseDF[allCloseDF['SYMBOL']=='IBM']
idxTest=np.random.randint(len(test))
testVals=test.iloc[idxTest:idxTest+7][['Date','Close']]
testVals['Date']
idx=np.random.randint(len(X_test))
test=X_test[idx]
print(test)
test=np.reshape(test,(1,8,1))
preds=model.predict(test)
print('PRED:',preds[0],' ACT:',y_test[idx])
preds=model.predict(X_test)
plt.scatter(y_test,preds,c=['r','b'])
plt.xlabel('ACTUAL')
plt.ylabel('PRED')
plt.show()
!mkdir models
model.save('./models/lstmSmall.mdl')