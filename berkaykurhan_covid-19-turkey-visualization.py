



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import warnings

warnings.filterwarnings("ignore")
data_set = pd.read_csv('../input/covid19turkey/Covid19-Turkey.csv')
data_set.head()

data_set.describe()

data_set.columns

sns.set_style('whitegrid')
data_set["Date"]  = pd.DataFrame(np.arange(1,len(data_set.Date)+1,1))
sns.factorplot(x='Date',y='Total Cases',data=data_set,kind='bar')

ax = plt.gca()

ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))

plt.xlabel('Geçen gün')

plt.ylabel('Toplam Vaka')

plt.show()
sns.factorplot(x='Date',y='Daily Cases',data=data_set,kind='bar')

ax = plt.gca()

ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))

plt.xlabel('Geçen gün')

plt.ylabel('Günlük Vaka')

plt.show()
sns.factorplot(x='Date',y='Total Deaths',data=data_set,kind='box')

ax = plt.gca()

ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))

plt.xlabel('Geçen gün')

plt.ylabel('Toplam Ölüm')

plt.show()
yesterday_deaths = 0

Daily_deaths = []

for current_deaths in data_set['Total Deaths']:

    if current_deaths>yesterday_deaths:

        Daily_deaths.append(current_deaths-yesterday_deaths)

    else :

        Daily_deaths.append(0)

    yesterday_deaths = current_deaths

Daily_deaths=pd.DataFrame(Daily_deaths)

data_set['Daily Deaths'] = Daily_deaths


sns.factorplot(x='Date',y='Daily Deaths',data=data_set,kind='box')

ax = plt.gca()

ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))

plt.xlabel('Geçen gün')

plt.ylabel('Günlük Ölüm')

plt.show()
plt.plot(data_set['Date'],data_set['Daily Test Cases'],color ='blue',label ='Günlük Test Sayısı')

plt.plot(data_set['Date'],data_set['Daily Cases'],color ='red',label='Günlük Vaka Sayısı')

plt.title("Günlük Test Sayısına karşın Günlük Vaka Sayısı")

plt.legend()

plt.xlabel('Geçen Gün')

plt.ylabel('Değer')
yesterday_recovered = 0

Daily_recovered = []

for current_recovered in data_set['Total Recovered']:

    if current_recovered>yesterday_recovered:

        Daily_recovered.append(current_recovered-yesterday_recovered)

    else :

        Daily_recovered.append(0)

    yesterday_recovered = current_recovered

Daily_recovered=pd.DataFrame(Daily_recovered)

data_set['Daily Recovered'] = Daily_recovered
plt.plot(data_set['Date'],data_set['Daily Cases'],color ='red',label ='Günlük Vaka Sayısı')

plt.plot(data_set['Date'],data_set['Daily Recovered'],color ='green',label='Günlük İyileşme Sayısı')

plt.title("Günlük Vaka Sayısına Karşın Günlük İyileşme Sayısı")

plt.xlabel('Geçen Gün')

plt.ylabel('Değer')

plt.legend()


Month = []

for current_day in data_set['Date']:

    if  0<=current_day<=20:

        Month.append('Mart')

    elif 20<current_day<=50:

        Month.append('Nisan')

    elif 50<current_day<=80:

        Month.append('Mayıs')

    elif 80<current_day<=112:

        Month.append('Haziran')

Month = pd.DataFrame(Month)

data_set['Month'] = Month

print("mart vaka: {}".format(data_set.iloc[20,2])) 

print("nisan vaka: {}".format(data_set.iloc[50,2]-data_set.iloc[20,2]))

print("mayıs vaka: {}".format(data_set.iloc[80,2]-data_set.iloc[50,2]))

print("haziran vaka: {}".format(data_set.iloc[110,2]-data_set.iloc[80,2])) 

        

    
plt.pie([13531,106673,43373,35510],labels=["Mart","Nisan","Mayıs","Haziran"],autopct='%1.1f%%',shadow=True)

plt.title("Belirlenen Vaka Sayısının Aylara Göre Dağılımı")

plt.show()
data_set["Daily Test Cases"].fillna(0,inplace=True)

sum_test_mart = 0

sum_test_nisan = 0

sum_test_mayıs = 0

sum_test_haziran = 0



for i in data_set["Date"]:

    if data_set.iloc[i-1,16] == "Mart" :

        sum_test_mart = sum_test_mart + data_set.iloc[i-1,6] 

    elif data_set.iloc[i-1,16] == "Nisan" :

        sum_test_nisan = sum_test_nisan + data_set.iloc[i-1,6]

    elif data_set.iloc[i-1,16] == "Mayıs" :

        sum_test_mayıs = sum_test_mayıs + data_set.iloc[i-1,6]

    elif data_set.iloc[i-1,16] == "Haziran" :

        sum_test_haziran = sum_test_haziran + data_set.iloc[i-1,6]



plt.pie([sum_test_mart,sum_test_nisan,sum_test_mayıs,sum_test_haziran],labels=["Mart","Nisan","Mayıs","Haziran"],autopct='%1.1f%%',shadow=True)

plt.title("Yapılan Test Sayısının Aylara Göre Dağılımı")

plt.show()
plt.figure(figsize=(9,3))

plt.scatter(data_set["Date"],data_set["Case incrase rate %"])

plt.plot(data_set["Date"],data_set["Case incrase rate %"])

plt.xlabel("Geçen Gün")

plt.ylabel("%")

plt.title("Vaka Artış Yüzdesi")
