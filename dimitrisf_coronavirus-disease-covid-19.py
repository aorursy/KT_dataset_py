import pandas as pd

from datetime import datetime as dt

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import functools 
#reading dataset 

df = pd.read_csv(r'../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

#dataset settings

#fixing date formats 

#print(df.iloc[7691])

dates = df['ObservationDate']

formatedDates = []



df = df.sort_values(by='ObservationDate')

countries = ['US','France','Mainland China']

for i in countries:

    dftemp = pd.DataFrame(columns=['ObservationDate', 'Province/State', 'Country/Region', 'Last Update','Confirmed', 'Deaths', 'Recovered'])

    dates = df.loc[(df['Country/Region'] == i)]['ObservationDate'].unique()

    for j in range(len(dates)):

       confirmed = functools.reduce(lambda x,y: x+y,df.loc[(df['Country/Region'] == i) & ((df['ObservationDate'] == dates[j]))]['Confirmed'].tolist())

       deaths = functools.reduce(lambda x,y: x+y,df.loc[(df['Country/Region'] == i) & ((df['ObservationDate'] == dates[j]))]['Deaths'].tolist())

       recovered = functools.reduce(lambda x,y: x+y,df.loc[(df['Country/Region'] == i) & ((df['ObservationDate'] == dates[j]))]['Recovered'].tolist()) 

       dftemp = dftemp.append({'ObservationDate': dates[j], 'Province/State': np.nan, 'Country/Region':i, 'Last Update': np.nan,'Confirmed': confirmed, 'Deaths':deaths, 'Recovered': recovered}, ignore_index=True)

    df = df.append(dftemp,ignore_index=True)

    indexNames = df.loc[(df['Country/Region'] == i) & (df['Province/State'].notnull())].index

    df.drop(indexNames , inplace=True)

df = df.sort_values(by='ObservationDate')
#basic info of the data

print('Basic Info of Data\n\n')

print(df.describe())

print('\n\nPeriod: ' + df.iloc[0]['ObservationDate'] + ' - ' + df.iloc[-1]['ObservationDate'] )
#finding the countires with the most cases 

dfascC = df.sort_values(by ='Confirmed',ascending=False)

allcountries = list(dfascC['Country/Region'].unique())

allcases = [dfascC.loc[dfascC['Country/Region'] == i]['Confirmed'].iloc[0] for i in allcountries]

alldeaths = [dfascC.loc[dfascC['Country/Region'] == i]['Deaths'].iloc[0] for i in allcountries]

allrecovered = [dfascC.loc[dfascC['Country/Region'] == i]['Recovered'].iloc[0] for i in allcountries]

#plotting the first,second and third five countries

f, ax = plt.subplots(3,1,figsize=(20,30))

titles = ['', 'Second', 'Third']

for j in range(0,3):

    pContries = allcountries[0 + j * 4: 4*(j+1)]

    

    #cases

    pCases = allcases[0 + j * 4: 4*(j+1)]

    ax[j].bar(np.arange(0,4), pCases, width = 0.2, label = 'Cases')

    for i in range(0,4):

        ax[j].text(-0.1+i,pCases[i]*1.01,int(pCases[i]),fontsize=15)



    #recovered

    pRecovered = allrecovered[0 + j * 4: 4*(j+1)]

    ax[j].bar(np.arange(0,4) + 0.2,pRecovered, width = 0.2, color='green', label = 'Recovered')

    for i in range(0,4):

        ax[j].text(-0.1+i+0.2,pRecovered[i]*1.01,str(float("{0:.2f}".format(pRecovered[i]/pCases[i]*100)))+'%',fontsize=15)

    

    #deaths

    pDeaths = alldeaths[0 + j * 4: 4*(j+1)]

    ax[j].bar(np.arange(0,4) + 0.4, pDeaths, width = 0.2, color='red', label = 'Deaths')

    for i in range(0,4):

        ax[j].text(-0.1+i+0.44,pDeaths[i]*1.01,str(float("{0:.2f}".format(pDeaths[i]/pCases[i]*100)))+'%',fontsize=15)

    

    #plot settings

    ax[j].set_xticks(np.arange(0,4)+0.2)

    ax[j].set_xticklabels(pContries)

    ax[j].legend(fontsize = 18)

    ax[j].set_title(titles[j] + ' Top 5 countries in CONVID-19 cases',fontsize=20)

    ax[j].set_xlabel('Countries',fontsize=15)

    ax[j].set_ylabel('Confirmed Number',fontsize=15)

    ax[j].tick_params(axis='both', labelsize=15)

    ax[j].set_ylim([0,allcases[(0 + j * 4)]*1.2])

    z = ax[j].plot()
#world-wide stats

totalcases = sum(allcases)

totalRecovered = sum(allrecovered)

totalDeaths = sum(alldeaths)

total = [totalcases,totalRecovered,totalDeaths]

labels = ['Cases','Recovered','Deaths']

#pie 

f, ax = plt.subplots(1,1,figsize=(8,8))

ax.pie(total,labels = labels,colors=['blue','green','red'],radius=1,startangle=100,autopct='%.2f%%',textprops={'fontsize': 14})

#settings

ax.set_title('Wordwide Reference')

x=0
#china time-line

chinadf = df.loc[df['Country/Region'] == 'Mainland China']

nochinadf = df.loc[df['Country/Region'] != 'Mainland China']

dtset = [chinadf,nochinadf]

f, ax = plt.subplots(1,2,figsize=(20,8))

for j in range(0,2):

    #Confirmed cases

    dates = [dt.strptime(i, '%m/%d/%Y') for i in dtset[j]['ObservationDate'].tolist()]

    confirmed = dtset[j]['Confirmed'].tolist()

    #normalize the data 

    minv = confirmed[0]

    for i in range(1,len(confirmed)):

        if(confirmed[i] < minv):

            confirmed[i] = -1

            dates[i] = -1

        else:

            minv = confirmed[i]

    #delete the 'wrong' values

    confirmed = list(filter(lambda x: x != -1,confirmed))

    datesC = list(filter(lambda x: x != -1,dates))

    #Deaths

    dates = [dt.strptime(i, '%m/%d/%Y') for i in dtset[j]['ObservationDate'].tolist()]

    deaths = dtset[j]['Deaths'].tolist()

    #normalize the data 

    minv = deaths[0]

    for i in range(1,len(deaths)):

        if(deaths[i] < minv):

            deaths[i] = -1

            dates[i] = -1

        else:

            minv = deaths[i]

    #delete the 'wrong' values

    deaths = list(filter(lambda x: x != -1,deaths))

    datesD = list(filter(lambda x: x != -1,dates))



    #Recovered

    dates = [dt.strptime(i, '%m/%d/%Y') for i in dtset[j]['ObservationDate'].tolist()]

    recovered = dtset[j]['Recovered'].tolist()

    #normalize the data 

    minv = deaths[0]

    for i in range(1,len(recovered)):

        if(recovered[i] < minv):

            recovered[i] = -1

            dates[i] = -1

        else:

            minv = recovered[i]

    #delete the 'wrong' values

    recovered = list(filter(lambda x: x != -1,recovered))

    datesR = list(filter(lambda x: x != -1,dates))





    #plotting 

    ax[j].plot(datesC, confirmed, color='b', label = 'Confirmed Cases')

    ax[j].plot(datesR, recovered,color='g', label = 'Recovered')

    ax[j].plot(datesD, deaths, color='r', label = 'Deaths')

    ax[j].legend(fontsize=15)

    ax[j].set_xlabel('Time', fontsize=12)

    ax[j].set_ylabel('Value', fontsize=12)

    ax[j].xaxis.set_major_locator(ticker.MaxNLocator(6))

    ax[j].tick_params(axis='both', labelsize=15)



ax[0].set_title('China', fontsize=20)

ax[1].set_title('Out of China', fontsize=20)

ax[0].set_ylim([0,confirmed[-1]+10000])

ax[1].set_ylim([0,confirmed[-1]+10000])

x = 0
#some specific countries...***(i included Greece because is my country)***

#adding US total in the dataframe

countries = ['US','Italy', 'Spain', 'Germany', 'France', 'Iran']

#total

for j in countries:

    dfcountry = df.loc[df['Country/Region'] == j]

    dates = dfcountry['ObservationDate']

    totalconfirmed = dfcountry['Confirmed'].tolist()

    totaldeaths = dfcountry['Deaths'].tolist()

    totalrecovered = dfcountry['Recovered'].tolist()

    dailyconfirmed = [totalconfirmed[0]]

    dailydeaths = [totalconfirmed[0]]

    dailyrecovered = [totalrecovered[0]]

    

    #daily

    for i in range(1,len(dfcountry)):

       dailyconfirmed.append(totalconfirmed[i] - totalconfirmed[i-1])

       dailydeaths.append(totaldeaths[i] - totaldeaths[i-1])

       dailyrecovered.append(totalrecovered[i] - totalrecovered[i-1])



    #plotting

    #total

    f, ax = plt.subplots(1,2,figsize=(20,8))

    ax[0].plot(dates,totalconfirmed,color = 'b',label='Confirmed')

    ax[0].plot(dates,totalrecovered,color = 'g',label='Recovered')

    ax[0].plot(dates,totaldeaths,color = 'r',label='Deaths')



    #daily

    ax[1].plot(dates,dailyconfirmed,color = 'b',label='Confirmed')

    ax[1].plot(dates,dailyrecovered,color = 'g',label='Recovered')

    ax[1].plot(dates,dailydeaths,color = 'r',label='Deaths')



    ax[0].set_title(str(j) + ' total cases timeline',fontsize=20)

    ax[1].set_title(str(j) + ' daily cases',fontsize=20)

    ax[0].legend(fontsize=15)

    ax[1].legend(fontsize=15)

    ax[0].xaxis.set_major_locator(ticker.MaxNLocator(7))

    ax[1].xaxis.set_major_locator(ticker.MaxNLocator(7))

    ax[0].tick_params(labelsize=12)

    ax[1].tick_params(labelsize=12)
