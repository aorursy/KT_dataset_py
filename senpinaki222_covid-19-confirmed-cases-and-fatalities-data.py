import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



df_train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')



country = df_train['Country/Region']

country_set = list(set(country))

country_set = sorted(country_set)



province = df_train['Province/State']

for i in range(len(province)):

    if(pd.isnull(province[i])):

        province[i] = country[i]



province_set = list(set(province))



date = df_train['Date']



for i in range(len(date)):

    dt = date[i]

    mm = dt[5:7]

    dd = dt[8:10]

    mm= int(mm)

    dd = int(dd)

    if(mm==1):

        day = dd

    elif(mm==2):

        day = 31+dd

    elif(mm==3):

        day = 31+29+dd

    date[i] = day

    

date_set = sorted(list(set(date)))





confirm = df_train['ConfirmedCases']

fatal = df_train['Fatalities']
k=0

key = province[0]

i = 0

l = len(province)

prov_confirm = []

prov_fatal = []



while(i < l):

    if(key==province[i]):

        prov_confirm.append(confirm[i])

        prov_fatal.append(fatal[i])

        i+=1

    else:

        plt.figure(k+1)

        plt.plot(date_set, prov_confirm, label='Confimed cases', markerfacecolor = 'blue')

        plt.plot(date_set, prov_fatal, label='Fatalities', markerfacecolor = 'red')

        plt.xlabel('Day')

        plt.ylabel('count')

        plt.legend(loc='upper left')

        plt.grid(True,linewidth=0.5,color='g', linestyle='--')

    

        if(key == country[i-1]):

            plt.title(key)

        else:

            plt.title(key+' / '+country[i-1])

        plt.show()

        

        k+=1

        key = province[i]

        prov_confirm = []

        prov_fatal = []