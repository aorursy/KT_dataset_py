import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

def make_DataFrame(path):

    tmp = pd.read_csv(path)

    if ("date" in tmp.columns):

        tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime('%m-%d')

    if ("time" in tmp.columns):

        tmp.drop('time',axis = 1 ,inplace = True)

    print( tmp.tail() )

    return tmp



def new_data(data):

    pre = 0

    tmp = []

    for i in range(61):

        tmp.append(data[i] - pre)

        pre = data[i]

    return tmp



# for comparing with China

def Country_datasets(country):

    tmp = data[data['Country/Region'] == country]

    tmp = tmp.groupby('date',as_index = False).sum()

    tmp["Death_rate"] = tmp["Deaths"]/ tmp["Confirmed"] * 100

    tmp["Recovered_rate"] = tmp["Recovered"]/ tmp["Confirmed"] * 100

    return tmp
main_data = make_DataFrame('/kaggle/input/coronavirusdataset/Time.csv')
time_age = make_DataFrame('/kaggle/input/coronavirusdataset/TimeAge.csv')
age = time_age.groupby(['age']).sum()['confirmed']

plt.title("Confirmed number")

plt.bar(age.index, age.values , color = 'red')
main_data["Confirmed_rate"] = main_data['confirmed'] / main_data['test']

main_data["Death_rate"] = main_data['deceased'] / main_data['confirmed']

main_data["Recovered_rate"] = main_data['released'] / main_data['confirmed']

main_data['negative_rate'] = main_data['negative'] / main_data['test']



main_data['new_confirmed'] = new_data(main_data['confirmed'])

main_data['new_recover'] = new_data(main_data['released'])

main_data['new_death'] = new_data(main_data['deceased'])



main_data['decreasing_confirmed'] = main_data['new_recover'] + main_data['new_death'] - main_data['new_confirmed'] 
main_data.drop(['test','negative'],axis = 1).tail()
#main_data.to_csv('main_data.csv')
# for Outlier

main_data_graph = main_data[main_data["date"] > dt.datetime(2000,1,30,0,0).strftime('%m-%d') ]
x = 'date'

plt.figure(figsize=(15, 7)) 

plt.plot(x,'Confirmed_rate',data = main_data_graph,label = 'confirmed_rate')

plt.plot(x,'Death_rate',data = main_data_graph,label = 'death_rate')

plt.plot(x,'Recovered_rate',data = main_data_graph,label = 'recovered_rate')

plt.xticks(rotation = 90)

plt.xlabel('date')

plt.legend(fontsize = 15 ,loc = 'upper left')

data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data['ObservationDate'] = pd.to_datetime(data["ObservationDate"]) #time_data

data = data[data['ObservationDate'] >= dt.datetime(2020,1,31)]

data['ObservationDate'] = data['ObservationDate'].dt.strftime('%m-%d')

data['date'] = data['ObservationDate']  # rename

data.drop(['SNo','Last Update','Province/State','ObservationDate'],axis = 1, inplace = True)

data_China = Country_datasets('Mainland China')



data_China.tail()
x = 'date'

plt.figure(figsize=(15, 7)) 

plt.plot(x,'Death_rate',data = main_data_graph,label = 'death_rate')

plt.plot(x,'Death_rate',data = data_China,label = 'China_death_rate')



plt.xticks(rotation = 90)

plt.xlabel('date')

plt.title('death_rate')

plt.legend(fontsize = 15 ,loc = 'upper left')
x = 'date'

plt.figure(figsize=(15, 7)) 

plt.plot(x,'Recovered_rate',data = main_data_graph,label = 'recovered_rate')

plt.plot(x,'Recovered_rate',data = data_China,label = 'China_recovered_rate')



plt.plot(main_data_graph[x], main_data_graph['Recovered_rate'] * 100 , label = 'recovered_rate * 100')



plt.xticks(rotation = 90)

plt.xlabel('date')

plt.title('recovered_rate')

plt.legend(fontsize = 15 ,loc = 'upper left')