# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import datetime as dt

import matplotlib.pyplot as plt

from fbprophet import Prophet



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
temp_path = "../input/temperature-timeseries-for-some-brazilian-cities/station_fortaleza.csv"

temp_data = pd.read_csv(temp_path)

#temp_data.head()

#temp_data.head(10)

plt.figure(figsize=(14,6))



temp_data = temp_data.replace(to_replace = 999.90, value = np.nan)

temp_data.head()



plt.title("10 Years of Temperatures in Fortaleza - CE")

plt.ylabel("Temperature(Cº)")

before_2000 = temp_data[temp_data.iloc[:,0].astype(int) <= 2009].index

temp_data = temp_data.loc[:, :"DEC"] 

temp_data = temp_data.drop(before_2000, axis=0)

sns.set_style("whitegrid")



temp_data.set_index("YEAR", inplace=True)

temp_data = temp_data.transpose()

#temp_data.head(15)



sns.lineplot(data=temp_data, dashes=False, sort=False)

plt.legend(loc='lower right')

temp_path = "../input/temperature-timeseries-for-some-brazilian-cities/station_sao_paulo.csv"

temp_data = pd.read_csv(temp_path)

#temp_data.head()

#temp_data.head(10)

plt.figure(figsize=(14,6))



temp_data = temp_data.replace(to_replace = 999.90, value = np.nan)

temp_data.head()



plt.title("10 Years of Temperatures in São Paulo - SP")

plt.ylabel("Temperature(Cº)")

before_2000 = temp_data[temp_data.iloc[:,0].astype(int) <= 2009].index

temp_data = temp_data.loc[:, :"DEC"] 

temp_data = temp_data.drop(before_2000, axis=0)

sns.set_style("whitegrid")



temp_data.set_index("YEAR", inplace=True)

temp_data = temp_data.transpose()

#temp_data.head(15)



sns.lineplot(data=temp_data, dashes=False, sort=False)

plt.legend(loc='lower right')
temp_path = "../input/temperature-timeseries-for-some-brazilian-cities/station_rio.csv"

temp_data = pd.read_csv(temp_path)

#temp_data.head()

#temp_data.head(10)

plt.figure(figsize=(14,6))



temp_data = temp_data.replace(to_replace = 999.90, value = np.nan)

temp_data.head()



plt.title("10 Years of Temperatures in Rio de Janeiro - RJ")

plt.ylabel("Temperature(Cº)")

before_2000 = temp_data[temp_data.iloc[:,0].astype(int) <= 2009].index

temp_data = temp_data.loc[:, :"DEC"] 

temp_data = temp_data.drop(before_2000, axis=0)

sns.set_style("whitegrid")



temp_data.set_index("YEAR", inplace=True)

temp_data = temp_data.transpose()



sns.lineplot(data=temp_data, dashes=False, sort=False)

plt.legend(loc='lower right')
temp_path = "../input/temperature-timeseries-for-some-brazilian-cities/station_sao_paulo.csv"

temp_data = pd.read_csv(temp_path)

x = 1.0

#temp_data.head()

#temp_data.head(10)

plt.figure(figsize=(13,9))

plt.title("Average temperatures per month in São Paulo - SP in decades")

plt.yticks(np.arange(16, 26, step=0.5))

temp_data = temp_data.replace(to_replace = 999.90, value = np.nan)

datatemp = None

for i in range(1951, 2019, 10):

    aft_y = temp_data[temp_data.iloc[:,0].astype(int) > i+9].index

    bef_i = temp_data[temp_data.iloc[:,0].astype(int) < i].index

    temp_data = temp_data.loc[:, :"DEC"] 

    #Código pega por década e retorna a média - 51-60.

    mean_temp_data = temp_data.drop(aft_y, axis=0)

    mean_temp_data = mean_temp_data.drop(bef_i, axis=0)

    mean_temp_data = mean_temp_data.mean(axis=0, skipna=True)

    mean_temp_data = mean_temp_data.to_frame()

    mean_temp_data = mean_temp_data.transpose()

    #print(type(mean_temp_data))

    if(datatemp is None):

        datatemp = mean_temp_data

    else:

        datatemp = pd.concat([datatemp, mean_temp_data], ignore_index=True)

#Trocando as médias de anos pelas respectivas décadas        

for x in np.arange(1955.5, 2015.6, 10.0):

    y = round(x, 0)-6

    if(x == 2015.5):

        datatemp = datatemp.replace(to_replace= 2015.0, value = "10's")

    else:

        datatemp = datatemp.replace(to_replace= x, value = str(y)[-4:][:2] + "'s")

datatemp.set_index("YEAR", inplace=True)

datatemp = datatemp.transpose()



sns.lineplot(data=datatemp, dashes=False, sort=False)

plt.legend(loc='lower right')

temp_path = "../input/temperature-timeseries-for-some-brazilian-cities/station_rio.csv"

temp_data = pd.read_csv(temp_path)

x = 1.0

#temp_data.head()

#temp_data.head(10)

plt.figure(figsize=(13,9))

plt.title("Average temperatures per month in Rio de Janeiro - RJ in decades")

plt.yticks(np.arange(20, 29, step=0.5))

temp_data = temp_data.replace(to_replace = 999.90, value = np.nan)

datatemp = None

for i in range(1951, 2019, 10):

    aft_y = temp_data[temp_data.iloc[:,0].astype(int) > i+9].index

    bef_i = temp_data[temp_data.iloc[:,0].astype(int) < i].index

    temp_data = temp_data.loc[:, :"DEC"] 

    #Código pega por década e retorna a média - 51-60.

    mean_temp_data = temp_data.drop(aft_y, axis=0)

    mean_temp_data = mean_temp_data.drop(bef_i, axis=0)

    mean_temp_data = mean_temp_data.mean(axis=0, skipna=True)

    mean_temp_data = mean_temp_data.to_frame()

    mean_temp_data = mean_temp_data.transpose()

    #print(type(mean_temp_data))

    if(datatemp is None):

        datatemp = mean_temp_data

    else:

        datatemp = pd.concat([datatemp, mean_temp_data], ignore_index=True)

#Trocando as médias de anos pelas respectivas décadas        

for x in np.arange(1955.5, 2015.6, 10.0):

    y = round(x, 0)-6

    if(x == 2015.5):

        datatemp = datatemp.replace(to_replace= 2015.0, value = "10's")

    elif(x == 1975.5):

        datatemp = datatemp.replace(to_replace= 1976.5, value = "70's")

    else:

        datatemp = datatemp.replace(to_replace= x, value = str(y)[-4:][:2] + "'s")

datatemp.set_index("YEAR", inplace=True)

datatemp = datatemp.transpose()

sns.lineplot(data=datatemp, dashes=False, sort=False)

plt.legend(loc='lower right')
temp_path = "../input/temperature-timeseries-for-some-brazilian-cities/station_curitiba.csv"

temp_data = pd.read_csv(temp_path)

x = 1.0

#temp_data.head()

#temp_data.head(10)

plt.figure(figsize=(13,9))

plt.title("Average temperatures per month in Curitiba - PR in decades")

plt.yticks(np.arange(14, 24, step=0.5))

temp_data = temp_data.replace(to_replace = 999.90, value = np.nan)

datatemp = None

for i in range(1951, 2019, 10):

    aft_y = temp_data[temp_data.iloc[:,0].astype(int) > i+9].index

    bef_i = temp_data[temp_data.iloc[:,0].astype(int) < i].index

    temp_data = temp_data.loc[:, :"DEC"] 

    #Código pega por década e retorna a média - 51-60.

    mean_temp_data = temp_data.drop(aft_y, axis=0)

    mean_temp_data = mean_temp_data.drop(bef_i, axis=0)

    mean_temp_data = mean_temp_data.mean(axis=0, skipna=True)

    mean_temp_data = mean_temp_data.to_frame()

    mean_temp_data = mean_temp_data.transpose()

    #print(type(mean_temp_data))

    if(datatemp is None):

        datatemp = mean_temp_data

    else:

        datatemp = pd.concat([datatemp, mean_temp_data], ignore_index=True)

#Trocando as médias de anos pelas respectivas décadas        

for x in np.arange(1955.5, 2015.6, 10.0):

    y = round(x, 0)-6

    if(x == 2015.5):

        datatemp = datatemp.replace(to_replace= 2015.0, value = "10's")

    else:

        datatemp = datatemp.replace(to_replace= x, value = str(y)[-4:][:2] + "'s")

datatemp.set_index("YEAR", inplace=True)

datatemp = datatemp.transpose()

sns.lineplot(data=datatemp, dashes=False, sort=False)

plt.legend(loc='lower right')
temp_path = "../input/temperature-timeseries-for-some-brazilian-cities/station_sao_paulo.csv"

temp_data = pd.read_csv(temp_path)

x = 1.0

#temp_data.head()

#temp_data.head(10)

plt.figure(figsize=(13,9))

plt.title("Average temperatures per month in São Paulo - SP in decades")

#plt.yticks(np.arange(10, 26, step=0.5))

temp_data = temp_data.replace(to_replace = 999.90, value = np.nan)

datatemp = None

for i in range(1951, 2019, 10):

    aft_y = temp_data[temp_data.iloc[:,0].astype(int) > i+9].index

    bef_i = temp_data[temp_data.iloc[:,0].astype(int) < i].index

    temp_data = temp_data.loc[:, :"DEC"] 

    #Código pega por década e retorna a média - 51-60.

    mean_temp_data = temp_data.drop(aft_y, axis=0)

    mean_temp_data = mean_temp_data.drop(bef_i, axis=0)

    mean_temp_data = mean_temp_data.mean(axis=0, skipna=True)

    mean_temp_data = mean_temp_data.to_frame()

    mean_temp_data = mean_temp_data.transpose()

    #print(type(mean_temp_data))

    if(datatemp is None):

        datatemp = mean_temp_data

    else:

        datatemp = pd.concat([datatemp, mean_temp_data], ignore_index=True)

#Trocando as médias de anos pelas respectivas décadas        

for x in np.arange(1955.5, 2015.6, 10.0):

    y = round(x, 0)-6

    if(x == 2015.5):

        datatemp = datatemp.replace(to_replace= 2015.0, value = "10's")

    else:

        datatemp = datatemp.replace(to_replace= x, value = str(y)[-4:][:2] + "'s")

datatemp.set_index("YEAR", inplace=True)

datatemp = datatemp.transpose()

datatemp = datatemp.round(1)



sns.heatmap(datatemp, annot=True, fmt=".1f", cmap="Reds")

plt.xlabel("Decades")

plt.ylabel("Months")
temp_path = "../input/temperature-timeseries-for-some-brazilian-cities/station_rio.csv"

temp_data = pd.read_csv(temp_path)

x = 1.0

#temp_data.head()

#temp_data.head(10)

plt.figure(figsize=(13,9))

plt.title("Average temperatures per month in Rio de Janeiro - RJ in decades")

#plt.yticks(np.arange(10, 26, step=0.5))

temp_data = temp_data.replace(to_replace = 999.90, value = np.nan)

datatemp = None

for i in range(1951, 2019, 10):

    aft_y = temp_data[temp_data.iloc[:,0].astype(int) > i+9].index

    bef_i = temp_data[temp_data.iloc[:,0].astype(int) < i].index

    temp_data = temp_data.loc[:, :"DEC"] 

    #Código pega por década e retorna a média - 51-60.

    mean_temp_data = temp_data.drop(aft_y, axis=0)

    mean_temp_data = mean_temp_data.drop(bef_i, axis=0)

    mean_temp_data = mean_temp_data.mean(axis=0, skipna=True)

    mean_temp_data = mean_temp_data.to_frame()

    mean_temp_data = mean_temp_data.transpose()

    #print(type(mean_temp_data))

    if(datatemp is None):

        datatemp = mean_temp_data

    else:

        datatemp = pd.concat([datatemp, mean_temp_data], ignore_index=True)

#Trocando as médias de anos pelas respectivas décadas        

for x in np.arange(1955.5, 2015.6, 10.0):

    y = round(x, 0)-6

    if(x == 2015.5):

        datatemp = datatemp.replace(to_replace= 2015.0, value = "10's")

    elif(x == 1975.5):

        datatemp = datatemp.replace(to_replace= 1976.5, value = "70's")

    else:

        datatemp = datatemp.replace(to_replace= x, value = str(y)[-4:][:2] + "'s")

datatemp.set_index("YEAR", inplace=True)

datatemp = datatemp.transpose()

datatemp = datatemp.round(1)

#Limpando anos anteriores a 1970 pois são dados vazios

datatemp = datatemp.dropna(axis='columns')



sns.heatmap(datatemp, annot=True, fmt=".1f", cmap="Reds")

plt.xlabel("Decades")

plt.ylabel("Months")
temp_path = "../input/temperature-timeseries-for-some-brazilian-cities/station_salvador.csv"

temp_data = pd.read_csv(temp_path)

#temp_data.head()

#temp_data.head(10)

plt.figure(figsize=(14,6))



temp_data = temp_data.replace(to_replace = 999.90, value = np.nan)

temp_data = temp_data.loc[:, :"DEC"] 

temp_data.tail()



#Orquestrando os dados do jeito necessário

vect_year = []

vect_month = []

vect_temp = []

vect_date = []



for x in range(1, len(temp_data)):

    for y in range(1, 12):

        vect_temp.append(temp_data.loc[x][y])

        vect_year.append(temp_data.loc[x][0])

        vect_month.append(y)

        vect_date.append(dt.datetime(int(temp_data.loc[x][0]), y, 15))

        

        

        

        

d = {'Year': vect_year, 'Temperature': vect_temp, 'Month': vect_month, 'ds': vect_date}



pred_data = pd.DataFrame(data=d)



pred_data.tail() 
#Ajustando os dados para o formato do Prophet

data = pred_data[['Temperature', 'ds']]



data.columns = ['y', 'ds']



data.head()

data = data.dropna()
#Fim do df

data.tail()
# Divisão dos dados em 80% + 20%

divisao = int(data.shape[0] * 4 / 5)

dataa = data[:divisao]

datab = data[divisao:]

print(data.shape, '=', dataa.shape, '+', datab.shape)
model = Prophet(daily_seasonality=False)

model.fit(dataa)
future = datab.drop(['y'], axis=1)

future.head()
forecast = model.predict(future)

forecast[['ds', 'yhat']].tail()
data2 = datab.merge(forecast)[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']]



data2['diff'] = abs(data2['y'] - data2['yhat'])

data2.head()
plt.figure(figsize=(16,9))



data2['y'].plot(alpha=0.5, style='-')

data2['yhat'].plot(style=':')

data2['yhat_lower'].plot(style='--')

data2['yhat_upper'].plot(style='--')



plt.legend(['real', 'previsto', 'pmenor', 'pmaior'], loc='upper left')
def rmse(predictions, targets):

    assert len(predictions) == len(targets)

    return np.sqrt(np.mean((predictions - targets) ** 2))



def rmsle(predictions, targets):

    assert len(predictions) == len(targets)

    return np.sqrt(np.mean((np.log(1 + predictions) - np.log(1 + targets)) ** 2))
print('RMSE:', rmse(data2['yhat'], data2['y']))
model = Prophet(daily_seasonality=False)

model.fit(data)
future = model.make_future_dataframe(periods=24, freq = 'M')

future.head()
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
import os

os.chdir(r'/kaggle/working')
data2 = data.merge(forecast, how= 'right')[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']]



data2.to_csv('/kaggle/working/dados_predict.csv', index=False, sep= '/', decimal= ',')
from IPython.display import FileLink



FileLink(r'dados_predict.csv')