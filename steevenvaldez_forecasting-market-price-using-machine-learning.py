import math

import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, LSTM

import datetime as dt

import matplotlib.pyplot as plt

import matplotlib.dates as mdates 

import matplotlib.ticker as tkr

plt.style.use('fivethirtyeight')
# Данние Microsoft

msft=pd.read_csv("/kaggle/input/tradingdata/MSFT.csv")

msft
# Данние Twitter

twtr=pd.read_csv("/kaggle/input/tradingdata/TWTR.csv")

twtr
# Данние Facebook

fb=pd.read_csv("/kaggle/input/tradingdata/FB.csv")

fb
# Данние EUR/USD

eurusd=pd.read_csv("/kaggle/input/tradingdata/EURUSDX.csv")

eurusd
# Данние Gold

gold=pd.read_csv("/kaggle/input/tradingdata/GCF.csv")

gold
# Данние Bitcoin

bitcoin=pd.read_csv("/kaggle/input/tradingdata/BTCUSDX.csv")

bitcoin
# Формат даты определяется для оси X графиков.

def xfmt(x,pos=None):

    x = mdates.num2date(x)

    label = x.strftime('%m/%Y')

    label = label.lstrip('0')

    return label
name_simbol_price_1 = "$"

name_instrument_1 = "Microsoft"

name_simbol_price_2 = "$"

name_instrument_2 = "Twitter"

name_simbol_price_3 = "$"

name_instrument_3 = "Facebook"

name_simbol_price_4 = "$"

name_instrument_4 = "EUR/USD"

name_simbol_price_5 = "$"

name_instrument_5 = "Gold"

name_simbol_price_6 = "$"

name_instrument_6 = "Bitcoin"
def buildGraphFromDataSet(data, instrument):

    xtempdates = [dt.datetime.strptime(i,'%Y-%m-%d') for i in data['Date']]

    plt.figure(figsize=(16,8))

    plt.title('Исторические цены - ' + instrument)

    plt.plot(xtempdates,data['Close'])

    plt.xlabel('Дата',fontsize=18)

    plt.ylabel('Цена в USD ($)',fontsize=18)

    plt.setp(plt.gca().xaxis.get_majorticklabels(),rotation=90)

    plt.gca().xaxis.set_major_formatter(tkr.FuncFormatter(xfmt))

    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=300))

    plt.show()
buildGraphFromDataSet(msft, name_instrument_1)
buildGraphFromDataSet(twtr, name_instrument_2)
buildGraphFromDataSet(fb, name_instrument_3)
buildGraphFromDataSet(eurusd, name_instrument_4)
buildGraphFromDataSet(gold, name_instrument_5)
buildGraphFromDataSet(bitcoin, name_instrument_6)
def generateTrainingAndPrediction(data):

    #Создание новых dataframes, используя только столбцы «Close» и «Date».

    closePriceData = data.filter(['Close'])

    dateData = data.filter(['Date'])



    #Преобразование dataframe в массив numpy

    dataset = closePriceData.values



    #Вычисление и получение количества рядов для обучения модели на

    training_data_len = math.ceil( len(dataset) * .8)



    #Масштабирование всех данных до значений от 0 до 1 

    scaler = MinMaxScaler(feature_range=(0, 1)) 

    scaled_data = scaler.fit_transform(dataset)



    #Создать масштабированный набор обучающих данных

    train_data = scaled_data[0:training_data_len, : ]



    #Деление данных на наборы x_train и y_train

    x_train=[]

    y_train=[]

    for i in range(60,len(train_data)):

        x_train.append(train_data[i-60:i,0])

        y_train.append(train_data[i,0])



    #Преобразование x_train и y_train в массив numpy

    x_train, y_train = np.array(x_train), np.array(y_train)



    #Преобразование данных в форму, принятую LSTM

    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



    #Построение модели нейронной сети долгой краткосрочной памяти

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))

    model.add(LSTM(units=50, return_sequences=False))

    model.add(Dense(units=25))

    model.add(Dense(units=1))



    #Компиляция и обучение модели

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, batch_size=1, epochs=1)



    #Тестирование набора данных

    test_data = scaled_data[training_data_len - 60: , : ]



    #Создание набора данных x_test и y_test

    x_test = []

    y_test =  dataset[training_data_len : , : ] 



    #Получение данных из набора данных для тестирования.

    for i in range(60,len(test_data)):

        x_test.append(test_data[i-60:i,0])

    

    #Преобразование x_test в массив numpy 

    x_test = np.array(x_test)



    #Преобразование данных в форму, принятую LSTM

    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))



    #Получение прогнозируемых цен на модели

    predictions = model.predict(x_test) 

    predictions = scaler.inverse_transform(predictions)



    #Вычисление и получение значения RMSE

    rmse=np.sqrt(np.mean(((predictions- y_test)**2)))



    # Создание данные для построение графики



    train = closePriceData[:training_data_len]

    train = train.assign(Date = dateData[:training_data_len])

    train = train[["Date", "Close"]]



    valid = closePriceData[training_data_len:]

    valid = valid.assign(Prediction = predictions)

    valid = valid.assign(Date = dateData[training_data_len:])

    valid = valid[["Date", "Close", "Prediction"]]

    

    result = [train, valid, scaler, model]

    

    return result
def buildPredictionGraph(train, valid, instrument):

    #Построение графики



    xtempdatestrain = [dt.datetime.strptime(i,'%Y-%m-%d') for i in train['Date']]

    xtempdatesvalid = [dt.datetime.strptime(i,'%Y-%m-%d') for i in valid['Date']]



    plt.figure(figsize=(16,8))

    plt.title('Тестирование цены - ' + instrument)

    plt.xlabel('Дата',fontsize=18)

    plt.ylabel('Цена в USD ($)',fontsize=18)

    plt.plot(xtempdatestrain,train['Close'])

    plt.plot(xtempdatesvalid,valid[['Close', 'Prediction']])

    plt.legend(['Train', 'Valid', 'Prediction'], loc='lower right')

    plt.setp(plt.gca().xaxis.get_majorticklabels(),rotation=90)

    plt.gca().xaxis.set_major_formatter(tkr.FuncFormatter(xfmt))

    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=300))

    plt.show()
msft_model = generateTrainingAndPrediction(msft)

buildPredictionGraph(msft_model[0], msft_model[1], name_instrument_1)
#Прогнозы

msft_model[1]
twtr_model = generateTrainingAndPrediction(twtr)

buildPredictionGraph(twtr_model[0], twtr_model[1], name_instrument_2)
#Прогнозы

twtr_model[1]
fb_model = generateTrainingAndPrediction(fb)

buildPredictionGraph(fb_model[0], fb_model[1], name_instrument_3)
#Прогнозы

fb_model[1]
eurusd_model = generateTrainingAndPrediction(eurusd)

buildPredictionGraph(eurusd_model[0], eurusd_model[1], name_instrument_4)
#Прогнозы

eurusd_model[1]
gold_model = generateTrainingAndPrediction(gold)

buildPredictionGraph(gold_model[0], gold_model[1], name_instrument_5)
#Прогнозы

gold_model[1]
bitcoin_model = generateTrainingAndPrediction(bitcoin)

buildPredictionGraph(bitcoin_model[0], bitcoin_model[1], name_instrument_6)
#Прогнозы

bitcoin_model[1]
def getPredictionNextDayAfterLastOne(data, scaler, model, instrument, simbolPrice):

    finalmessage1 = "Прогнозируемая цена"

    finalmessage2 = "на следующую дату после"

    finalmessage3 = "составляет"

    

    lastDate = data.filter(['Date'])

    finalDateMessage = dt.datetime.strptime(lastDate['Date'].iloc[-1], '%Y-%m-%d')



    #Создание нового dataframe

    new_df = data.filter(['Close'])



    #Получение цены закрытия за последние 60 дней

    last_60_days = new_df[-60:].values



    #Масштабирование всех данных до значений от 0 до 1

    last_60_days_scaled = scaler.transform(last_60_days)



    #Создать пустой массив

    X_test = []



    #Append the past 60 days

    X_test.append(last_60_days_scaled)



    #Convert the X_test data set to a numpy array

    X_test = np.array(X_test)



    #Reshape the data

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



    #Получение прогнозируемую масштабированной цены

    pred_price = model.predict(X_test)



    #Отмена масштабирования 

    pred_price = scaler.inverse_transform(pred_price)

    print(finalmessage1 + " " + instrument + " " + finalmessage2 + " " + str(dt.date.strftime(finalDateMessage, "%d/%m/%Y")) + " " + finalmessage3 + " " + simbolPrice + str(pred_price[0][0]))
getPredictionNextDayAfterLastOne(msft, msft_model[2], msft_model[3], name_instrument_1, name_simbol_price_1)
getPredictionNextDayAfterLastOne(twtr, twtr_model[2], twtr_model[3], name_instrument_2, name_simbol_price_2)
getPredictionNextDayAfterLastOne(fb, fb_model[2], fb_model[3], name_instrument_3, name_simbol_price_3)
getPredictionNextDayAfterLastOne(eurusd, eurusd_model[2], eurusd_model[3], name_instrument_4, name_simbol_price_4)
getPredictionNextDayAfterLastOne(gold, gold_model[2], gold_model[3], name_instrument_5, name_simbol_price_5)
getPredictionNextDayAfterLastOne(bitcoin, bitcoin_model[2], bitcoin_model[3], name_instrument_6, name_simbol_price_6)