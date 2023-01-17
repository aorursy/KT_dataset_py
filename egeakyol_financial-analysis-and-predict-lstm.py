# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
aselsan = pd.read_csv("../input/4-turkeys-biggest-companyin-stock-market/ASELS.IS (1).csv")
aselsan.head()
migros = pd.read_csv("../input/4-turkeys-biggest-companyin-stock-market/MGROS.IS.csv")
migros["stock_name"]="migros"
migros_copy = migros.copy()
#only two row null. I want to fill because of the graph and predictions.
#simply we take average nearest rows
print("migros stock null values:\n", migros[migros.isnull().any(axis=1)])

thy = pd.read_csv("../input/4-turkeys-biggest-companyin-stock-market/THYAO.IS.csv")
thy["stock_name"] = "thy"
thy_copy = thy.copy()
print("thy stock null values:\n", thy[thy.isnull().any(axis=1)])

aselsan = pd.read_csv("../input/4-turkeys-biggest-companyin-stock-market/ASELS.IS (1).csv")
aselsan["stock_name"] = "aselsan"
aselsan_copy = aselsan.copy()
print("aselsan stock null values:\n", aselsan[aselsan.isnull().any(axis=1)])

garan = pd.read_csv("../input/4-turkeys-biggest-companyin-stock-market/GARAN.IS.csv")
garan["stock_name"]= "garan"
garan_copy = garan.copy()
print("garan stock null values:\n", garan[garan.isnull().any(axis=1)])

datalist=[migros, thy, aselsan, garan]
all_data = pd.concat(datalist,axis=1, ignore_index=True)
def fillmissing_value(stocks,copy):
    #number of total rows which has null value
    nullrownumber = stocks.shape[0] - stocks.dropna().shape[0]
    
    for i in range(nullrownumber):
        null_data = stocks[stocks.isnull().any(axis=1)]
        getindex=null_data.index
        stocks.drop(["Date", "stock_name"], axis=1, inplace = True)
        
        ortalama1 = stocks.iloc[getindex[0]-1]
        ortalama2 = stocks.iloc[getindex[0]+1]
        ortalama = stocks.iloc[getindex[0]]
        ortalama = (ortalama1+ortalama2)/2
        ortalama[:4]=ortalama[:4].astype("float16")
        stocks.iloc[getindex[0]]=stocks.iloc[getindex[0]].fillna(ortalama)
        stocks["Date"] = copy["Date"]
        stocks["stock_name"] = copy["stock_name"]
        stocks["Volume"]=stocks["Volume"].astype("float")
        
datalist=[migros, thy,garan, aselsan]
dataliststr=["migros", "thy","garan", "aselsan"]
datalist_copy=[migros_copy, thy_copy,garan_copy, aselsan_copy]

for i in range(len(datalist)):
    fillmissing_value(datalist[i],datalist_copy[i])
    
for i in range(len(datalist)):
    null_data = datalist[i].isnull().any().sum()
    print("{} dataset has: {} null value".format(dataliststr[i] ,null_data))

print(aselsan.describe())
print(aselsan.info())
print("{} means weekdays in a one year.".format(aselsan.shape[0]))
plt.style.use("seaborn")
plt.figure(figsize=(14,8))
plt.plot(migros["Close"], color="purple", linewidth=3)
plt.title("Migros Stock Price", fontsize=40)
months=["November","December","January","February","March","April","May","June","July","August","September","October"]
plt.xticks(np.arange(0,252,21)+13,labels=months,rotation=12)  # Set text labels and properties.
plt.xlim(left=0, right=252) #252 equals number of rows
plt.ylim(top=45) 
plt.ylabel("Turkish lira", fontsize=18)
plt.show()

plt.style.use("seaborn")
fig, axs = plt.subplots(2, 2,figsize=(12,12))

axs[0, 0].plot(migros["Close"], color="Red", linewidth=2)
axs[0, 0].set_title("Migros Stock Price")
axs[0, 0].set_xticks(np.arange(0,252,21)+13)
axs[0, 0].set_xticklabels([i for i in months],fontsize=6)
axs[0, 0].set_xlim(left=0, right=252)

axs[0, 1].plot(thy["Close"], color="red", linewidth=2)
axs[0, 1].set_title("THY Stock Price")
axs[0, 1].set_xticks(np.arange(0,252,21)+13)
axs[0, 1].set_xticklabels([i for i in months],fontsize=6)
axs[0, 1].set_xlim(left=0, right=252)


axs[1, 0].plot(garan["Close"], color="red", linewidth=2)
axs[1, 0].set_title("Garanti Bank Stock Price")
axs[1, 0].set_xticks(np.arange(0,252,21)+13)
axs[1, 0].set_xticklabels([i for i in months],fontsize=6)
axs[1, 0].set_xlim(left=0, right=252)


axs[1, 1].plot(aselsan["Close"], color="red", linewidth=2)
axs[1, 1].set_title("Aselsan Stock Price")
axs[1, 1].set_xticks(np.arange(0,252,21)+13)
axs[1, 1].set_xticklabels([i for i in months],fontsize=6)
axs[1, 1].set_xlim(left=0, right=252)
datalist=[migros, thy,garan, aselsan]
for company in datalist:
    first_price = round(company["Close"].iloc[0],2)
    last_price =  round(company["Close"].iloc[-1],2)
    print("*" * 25)
    print("first Close price: {} and last Close price: {}".format(first_price,last_price))
    annual_return = ((last_price-first_price) * 100 ) / (first_price)
    print("{} annual return is: %{:.2f}".format(company["stock_name"][0], annual_return))
migros["Volume"] = migros["Volume"].astype({"Volume":"float"})
#(x-Xmin)/(Xmax- Xmin)
column = migros["Volume"]
max_value = column.max()
min_value = column.min()

migros["Volume"]=((migros["Volume"]-min_value)/(max_value -min_value))*15
#I use min-max scaler than I multiple 15 to make the graph clearer
plt.figure(figsize=(12,12))
plt.style.use("ggplot")
plt.title("Migros Stock Price(TRY)", color ="red", fontsize=20)
plt.bar(migros["Date"], migros["Volume"]*3,  color="orange")
plt.plot(migros["Close"], color="black", label="Close Price")
plt.xticks(np.arange(0,252,21)+13,labels=months,rotation=12)
plt.yticks(np.arange(0,50,5))
plt.xlim(left=0, right=252) #252 equals number of rows
plt.ylabel("Volume(10 million Turkish lira)")
plt.xlabel("2019-2020 years")

x = 101
y = 5
plt.scatter(x, y, color='red',linewidths=3, label="first corona case in Turkey")
plt.legend(prop={'size': 15})
plt.show()
describe = migros_copy["Volume"].describe()
print(describe)
print("*"*20)
print("Mean Volume of stock; 3,813,146 million TRY")
moving_average_day = [10, 20, 50]
datalist=[migros, thy,garan, aselsan]
for ma in moving_average_day:
    for company in datalist:
        column_name = f"MA for {ma} days"
        company[column_name] = company['Adj Close'].rolling(ma).mean()


plt.style.use("dark_background")
plt.figure(figsize=(16,10))
plt.plot(migros["Close"], color="purple", label="Close Price")
plt.plot(migros["MA for 10 days"], color="green", label="MA 10")
plt.plot(migros["MA for 20 days"], color="yellow", label="MA 20")
plt.plot(migros["MA for 50 days"], color="red", label="MA 50")
plt.title("Migros Stock Price", fontsize=40)
months=["November","December","January","February","March","April","May","June","July","August","September","October"]
plt.xticks(np.arange(0,252,21)+13,labels=months,rotation=12)  # Set text labels and properties.
plt.xlim(left=0, right=252) #252 equals number of rows
plt.ylim(top=45) 
plt.ylabel("Turkish lira", fontsize=30)
plt.legend(prop={'size': 20})
plt.style.use("seaborn")
fig, axs = plt.subplots(2, 2,figsize=(12,12))

axs[0, 0].plot(migros["Close"], color="black", linewidth=2, label="Close Price")
axs[0, 0].plot(migros["MA for 10 days"], color="green", label="MA 10")
axs[0, 0].plot(migros["MA for 20 days"], color="yellow", label="MA 20")
axs[0, 0].plot(migros["MA for 50 days"], color="red", label="MA 50")
axs[0, 0].set_title("Migros Stock Price")
axs[0, 0].set_xticks(np.arange(0,252,21)+13)
axs[0, 0].set_xticklabels([i for i in months],fontsize=6)
axs[0, 0].set_xlim(left=0, right=252)
axs[0, 0].legend(prop={'size': 10})

axs[0, 1].plot(thy["Close"], color="black", linewidth=2)
axs[0, 1].plot(thy["MA for 10 days"], color="green", label="MA 10")
axs[0, 1].plot(thy["MA for 20 days"], color="yellow", label="MA 20")
axs[0, 1].plot(thy["MA for 50 days"], color="red", label="MA 50")
axs[0, 1].set_title("THY Stock Price")
axs[0, 1].set_xticks(np.arange(0,252,21)+13)
axs[0, 1].set_xticklabels([i for i in months],fontsize=6)
axs[0, 1].set_xlim(left=0, right=252)
axs[0, 1].legend(prop={'size': 10})


axs[1, 0].plot(garan["Close"], color="black", linewidth=2)
axs[1, 0].plot(garan["MA for 10 days"], color="green", label="MA 10")
axs[1, 0].plot(garan["MA for 20 days"], color="yellow", label="MA 20")
axs[1, 0].plot(garan["MA for 50 days"], color="red", label="MA 50")
axs[1, 0].set_title("Garanti Bank Stock Price")
axs[1, 0].set_xticks(np.arange(0,252,21)+13)
axs[1, 0].set_xticklabels([i for i in months],fontsize=6)
axs[1, 0].set_xlim(left=0, right=252)
axs[1, 0].legend(prop={'size': 10})


axs[1, 1].plot(aselsan["Close"], color="black", linewidth=2)
axs[1, 1].plot(aselsan["MA for 10 days"], color="green", label="MA 10")
axs[1, 1].plot(aselsan["MA for 20 days"], color="yellow", label="MA 20")
axs[1, 1].plot(aselsan["MA for 50 days"], color="red", label="MA 50")
axs[1, 1].set_title("Aselsan Stock Price")
axs[1, 1].set_xticks(np.arange(0,252,21)+13)
axs[1, 1].set_xticklabels([i for i in months],fontsize=6)
axs[1, 1].set_xlim(left=0, right=252)
axs[1, 1].legend(prop={'size': 10})
plt.style.use("dark_background")
plt.figure(figsize=(14,10))
plt.plot(migros["MA for 20 days"], color="blue", linewidth=3)

#üst band
üstband = migros['Close'].rolling(20).std()
plt.plot((migros["MA for 20 days"]+üstband*2), color="red",linewidth=1.5)

#alt band
altband = migros['Close'].rolling(20).std()
plt.plot((migros["MA for 20 days"]-altband*2), color="red",linewidth=1.5)
plt.title("Migros Stock Price \n with bollinger bands", fontsize=40)
months=["November","December","January","February","March","April","May","June","July","August","September","October"]
plt.xticks(np.arange(0,252,21)+13,labels=months,rotation=12)  # Set text labels and properties.
plt.xlim(left=0, right=252) #252 equals number of rows
plt.ylim(top=50) 
plt.ylabel("Turkish lira", fontsize=20)
Ema12 = migros["Close"].rolling(12).mean()
Ema26 = migros["Close"].rolling(26).mean()
MACD = Ema12 - Ema26
migros["MACD"] = MACD
MacdSignal = migros["MACD"].rolling(9).mean()
migros["MacdSignal"] = MacdSignal

plt.style.use("ggplot")
plt.figure(figsize=(18,10))
plt.plot(migros["Close"], color="black", linewidth=2, label="Close Price")
plt.plot(migros["MACD"], color="c", linewidth=1.5, label = "MACD")
plt.plot(migros["MacdSignal"], color="red", linewidth=1.5, label = "MACD Signal")
plt.title("Migros with MACD", fontsize=40,color="black")
months=["November","December","January","February","March","April","May","June","July","August","September","October"]
plt.xticks(np.arange(0,252,21)+13,labels=months,rotation=12)  # Set text labels and properties.
plt.yticks(np.arange(0,50 ,5))
plt.xlim(left=0, right=252) #252 equals number of rows
plt.legend(prop={'size': 20})
plt.ylabel("Migros stock Price (TRY)", fontsize=25)
plt.show()
garan["volalite(kuruş)"] = garan["Close"]-garan["Open"]
garan["volalite"]= 100*garan["volalite(kuruş)"]/garan["Open"]
garan["volalite"].describe()
usdtry = pd.read_csv("../input/4-turkeys-biggest-companyin-stock-market/USDTRYX.csv")

plt.figure(figsize=(14,10))
plt.style.use("Solarize_Light2")
plt.plot(usdtry["Close"], color="red")
plt.title("USD - TRY Rate", fontsize=20, color="red")
months=["November","December","January","February","March","April","May","June","July","August","September","October"]
plt.xticks(np.arange(0,252,21)+13,labels=months,rotation=12)  
plt.xlim(left=0, right=252) #252 equals number of rows
plt.grid(True)
plt.ylim(top=8)

x = 101
y = 6.5
plt.scatter(x, y, color='black',linewidths=8, label="first corona case in Turkey", alpha=0.8)
plt.legend(prop={'size': 15}, borderpad=1, shadow=True, facecolor="red")
plt.show()
#print(usdtry["Close"])

annualy_difference_usd = usdtry["Close"][252] - usdtry["Close"][0]
print("annualy difference (USD) {}".format(annualy_difference_usd))
annualy_difference_rate = (100 * annualy_difference_usd)/usdtry["Close"][0] 
print("TRY value  %{:.2f} dropped in a year".format(annualy_difference_rate))
usdtry.rename(columns={"Close":"usdclose"}, inplace =True)
garan = pd.concat([garan,usdtry["usdclose"]],axis=1)
garan["usdvalue"] = garan["Close"]/garan["usdclose"]


plt.style.use("ggplot")
plt.figure(figsize=(16,10))

plt.plot(garan["Close"], color="black", linewidth=3, label="Try value")
plt.plot(garan["usdvalue"], color="red", linewidth=1.5, label="Usd value")
plt.plot(garan["usdclose"], color="blue", linewidth=1.5, label="Usd/Try")

plt.title("Garanti Bank TRY-USD", fontsize=40,color="black")
months=["November","December","January","February","March","April","May","June","July","August","September","October"]
plt.xticks(np.arange(0,252,21)+13,labels=months,rotation=12)  # Set text labels and properties.
plt.yticks(np.arange(0, 14, 1))
plt.xlim(left=0, right=252) #252 equals number of rows
plt.legend(prop={'size': 20}, facecolor="orange")
plt.show()

data = migros.filter(['Close'])
data_mig = data.values.astype("float32")
print(data_mig.shape)

#scaler for keras model
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_mig_price = scaler.fit_transform(np.array(data_mig).reshape(-1,1))
print(data_mig_price.shape)

train_size = int(len(data_mig_price) * 0.8)  
test_size = int(len(data_mig_price) * 0.2)

train_data = data_mig_price[0:train_size,:]
test_data =data_mig_price[train_size-20:len(data_mig_price),:]

print("train data len: {}, test data len : {}".format(len(train_data), len(test_data)))
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 20
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
print("X_train shape:",X_train.shape)
print("y_train shape:",y_train.shape)
print("X_test shape:",X_test.shape)
print("ytest shape:",ytest.shape)

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model = Sequential()

model.add(LSTM(64,input_dim=1,return_sequences=True))


model.add(LSTM(128,return_sequences=False))
model.add(Dropout(0.3))

model.add(Dense(64))
model.add(Dropout(0.3))

model.add(Dense(1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mean_squared_error', optimizer='Adam')
print ('compilation time : ', time.time() - start)

model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=25,
    validation_split=0.05)

model.summary()
model.fit(X_train,y_train,validation_data=(X_test,ytest),
          epochs=100,batch_size=64)

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

import math
from sklearn.metrics import mean_squared_error

#rmse : root mean square error
train_data_rmse = math.sqrt(mean_squared_error(y_train,train_predict))
print("train data rmse: {}".format(train_data_rmse))
test_data_rmse= math.sqrt(mean_squared_error(ytest,test_predict))
print("test data rmse: {}".format(test_data_rmse))
plt.figure(figsize=(16,8))
step_size=20

train = np.empty_like(data_mig_price)
train[:, :] = np.nan
train[step_size:len(train_predict)+step_size, :] = train_predict

test = np.empty_like(data_mig_price)
test[:, :] = np.nan
test[len(train_predict)+(step_size)+1:len(data_mig_price)-1, :] = test_predict

plt.plot(migros["Close"])
plt.plot(train, color ="purple", linestyle="dashed", linewidth=3)
plt.plot(test, color="black", linestyle="dotted",linewidth=3)
plt.title('For Migros Stock price predict')
plt.xlabel('october 2019----october 2020', fontsize=18)
plt.ylabel('Close Price(MİGROS) TRY', fontsize=18)
plt.xticks(np.arange(0,252,21)+13,labels=months,rotation=12)  
plt.xlim(left=0, right=252) #252 equals number of rows
plt.legend(['Train/Migros', 'Validation data', 'Predictions'], prop={'size': 20})
plt.show()
