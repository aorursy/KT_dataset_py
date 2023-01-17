# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 
import math
from keras import optimizers
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers import LSTM, Dropout, BatchNormalization
from sklearn.metrics import mean_squared_error

import warnings 
warnings.filterwarnings("ignore")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.DataFrame()

f = pd.read_csv("../input/cryptocurrency-historical-data/bitcoin.csv", delimiter=';')
data = data.append(f)
data.head()
dataOpen = data.loc[:,["Open"]].values
dataVolume= data.loc[:,["Volume"]].values/100000  #Volume değerleri çok yüksek olduğu için tüm değerli kapsayacak bir küçültme işlemi yapıyoruz
print(dataOpen)
print(dataVolume)
dataOpen=np.flip(dataOpen,0)
dataVolume=np.flip(dataVolume,0)
dataOpen
X_data = []
y_data = []
timesteps = 50
for i in range(timesteps, 1670):
    X_data.append(dataOpen[i-timesteps:i, 0])
    X_data.append(dataVolume[i-timesteps:i, 0]) 
    y_data.append(dataOpen[i, 0])
X_data, y_data = np.array(X_data), np.array(y_data)
X_data
X_data = X_data.flatten()
X_data = np.reshape(X_data, (1620, 1,100))
X_data.shape
Xtest=X_data[1500:]
ytest=y_data[1500:]
X_train=X_data[:1500]
y_train=y_data[:1500]



model = Sequential()
model.add(LSTM(2400,activation='relu',return_sequences=True, input_shape=(1,100))) # 10 lstm neuron(block)
model.add(Dense(2400))
model.add(LSTM(2400,activation='relu',return_sequences=True))
model.add(Dense(2400))
model.add(Dropout(0.2))
model.add(Dense(2400))
model.add(LSTM(2400,activation='relu',return_sequences=True))
model.add(Dense(2400))
model.add(Dropout(0.2))
model.add(LSTM(2400,activation='relu',return_sequences=True))
model.add(Dense(1400))
model.add(Dropout(0.2))
model.add(LSTM(2400,activation='relu',return_sequences=True))
model.add(Dense(1400))
model.add(LSTM(2400,activation='relu',return_sequences=True))
model.add(Dense(1400))
model.add(Dropout(0.2))
model.add(LSTM(2400,activation='relu',return_sequences=True))
model.add(Dense(1400))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='Adamax')
model.fit(X_train, y_train, epochs=200, batch_size=1)
model.save("./tahminModel")
trainPredict = model.predict(X_train)


trainScore = math.sqrt(mean_squared_error(y_train, trainPredict[:,0]))
    
print('Train Score: %.2f RMSE' % (trainScore))
print(math.sqrt(mean_squared_error(y_train, trainPredict[:,0])))
plt.plot(y_train,label="real")
plt.plot(trainPredict[:,0],label="pre")
plt.show()
testPredict = model.predict(Xtest)


testScore = math.sqrt(mean_squared_error(ytest, testPredict[:,0]))
    
print('Test Score: %.2f RMSE' % (testScore))
print(math.sqrt(mean_squared_error(ytest, testPredict[:,0])))
plt.plot(ytest,label="real")
plt.plot(testPredict[:,0],label="pre")
plt.show()
dogru =0
yanlis =0

for i in range(1,120):

    if testPredict[i]> testPredict[i-1] and ytest[i]>ytest[i-1]:
        dogru+=1
    elif testPredict[i]< testPredict[i-1] and ytest[i]<ytest[i-1]:
        dogru+=1
    else:
        yanlis+=1
print("Model ",dogru," kez fiyatlarda artışı ve azalmayı doğru bilmiş, ",yanlis," kez ise fiyat artış ve azalışlarında tahminlerinde hata yapmıştır.")
print("Model %",(61/119)*100," oranla günlük fiyat değişimini doğru tahmin etmiştir.")