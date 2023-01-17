import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



df_train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')



country = df_train['Country_Region']

country_set = list(set(country))

country_set = sorted(country_set)



province = df_train['Province_State']

for i in range(len(province)):

    if(pd.isnull(province[i])):

        province[i] = country[i]



province_set = list(set(province))



date = df_train['Date']



for i in range(len(date)):

    dt = date[i]

    mm = dt[5:7]

    dd = dt[8:10]

    mm = int(mm)

    dd = int(dd)

    if(mm==1):

        day = dd

    elif(mm==2):

        day = 31+dd

    elif(mm==3):

        day = 31+29+dd

    elif(mm==4):

        day = 31+29+31+dd

    date[i] = day

    

date_set = sorted(list(set(date)))





confirm = df_train['ConfirmedCases']

fatal = df_train['Fatalities']
## Plot for India



k=0

key = 'India'

i = 0

l = len(province)

india_confirm = []

india_fatal = []



while(province[i]!='India'):

    i+=1

    

while(province[i]=='India'):

    india_confirm.append(confirm[i])

    india_fatal.append(fatal[i])

    i+=1

    

plt.figure(1)

plt.plot(date_set, india_confirm, label='Confimed cases', markerfacecolor = 'blue')

plt.plot(date_set, india_fatal, label='Fatalities', markerfacecolor = 'red')

plt.xlabel('Day of year 2020')

plt.ylabel('Count')

plt.legend(loc='upper left')

plt.grid(True,linewidth=0.5,color='g', linestyle='--')

plt.title(key)

plt.show()

## Plot for Italy



k=0

key = 'Italy'

i = 0

l = len(province)

italy_confirm = []

italy_fatal = []



while(province[i]!='Italy'):

    i+=1

    

while(province[i]=='Italy'):

    italy_confirm.append(confirm[i])

    italy_fatal.append(fatal[i])

    i+=1

    

plt.figure(1)

plt.plot(date_set, italy_confirm, label='Confimed cases', markerfacecolor = 'blue')

plt.plot(date_set, italy_fatal, label='Fatalities', markerfacecolor = 'red')

plt.xlabel('Day of year 2020')

plt.ylabel('Count')

plt.legend(loc='upper left')

plt.grid(True,linewidth=0.5,color='g', linestyle='--')

plt.title(key)

plt.show()

# Plot for Spain



k=0

key = 'Spain'

i = 0

l = len(province)

Spain_confirm = []

Spain_fatal = []



while(province[i]!='Spain'):

    i+=1

    

while(province[i]=='Spain'):

    Spain_confirm.append(confirm[i])

    Spain_fatal.append(fatal[i])

    i+=1

    

plt.figure(1)

plt.plot(date_set, Spain_confirm, label='Confimed cases', markerfacecolor = 'blue')

plt.plot(date_set, Spain_fatal, label='Fatalities', markerfacecolor = 'red')

plt.xlabel('Day of year 2020')

plt.ylabel('Count')

plt.legend(loc='upper left')

plt.grid(True,linewidth=0.5,color='g', linestyle='--')

plt.title(key)

plt.show()

## plot for different Provinces of China



k=0

key = province[0]

i = 0

l = len(province)

prov_confirm = []

prov_fatal = []



while(i < l):

    

    while(country[i]!='China'):

        i+=1

        if(i==l):

            break

    if(i==l):

        break

    key = province[i]

    while(key==province[i]):

        prov_confirm.append(confirm[i])

        prov_fatal.append(fatal[i])

        i+=1



    plt.figure(k+1)

    plt.plot(date_set, prov_confirm, label='Confimed cases', markerfacecolor = 'blue')

    plt.plot(date_set, prov_fatal, label='Fatalities', markerfacecolor = 'red')

    plt.xlabel('Day')

    plt.ylabel('count')

    plt.legend(loc='upper left')

    plt.grid(True,linewidth=0.5,color='g', linestyle='--')

    plt.title(key+' / '+'China')

    plt.show()

    k+=1

    

    #key = province[i]

    prov_confirm = []

    prov_fatal = []