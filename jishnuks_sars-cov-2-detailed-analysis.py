# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # Numerical 

import pandas as pd # data processing, CSV 

import matplotlib.pyplot as plt

import seaborn as sns

import re

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from pathlib import Path

data_dir = Path('../input/novel-corona-virus-2019-dataset')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
plt.rcParams.update({

    "lines.color": "white",

    "patch.edgecolor": "white",

    "text.color": "black",

    "axes.facecolor": "white",

    "axes.edgecolor": "lightgray",

    "axes.labelcolor": "white",

    "xtick.color": "white",

    "ytick.color": "white",

    "grid.color": "lightgray",

    "figure.facecolor": "black",

    "figure.edgecolor": "black",

    "savefig.facecolor": "black",

    "savefig.edgecolor": "black",

    'figure.figsize':(12,8)})
data=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data.rename(columns={'Last Update': 'dateupdated',

                          'ObservationDate' : 'date',

                        'Sno' : 'id',

                         'Province/State':'state',

                         'Country/Region':'country',

                         'Confirmed': 'confirmed',

                         'Deaths':'deaths',

                        'Recovered' : 'recovered'

                    }, inplace=True)

data['active'] = data['confirmed'] - data['deaths'] - data['recovered']

data.tail()
from datetime import datetime

length=len(data['date'])

for i in range(length):

    data['date'][i]=datetime.strptime(data['date'][i], '%m/%d/%Y').date()

first_day=data['date'].min()

data['no_of_days']=data['date']-first_day

for i in range(length):

    data['no_of_days'][i]=data['no_of_days'][i].days

data.tail()
print("External Data")

print(f"Earliest Entry: {data['date'].min()}")

print(f"Last Entry:     {data['date'].max()}")

print(f"Total Days:     {data['date'].max() - data['date'].min()}")
allcountries=set(data['country'])

allcountries=list(allcountries)
subs='Russia'

res = [x for x in allcountries if re.search(subs, x)] 

print(res)
def addcolumn_1(dataframe):

    grouped=dataframe

    length=len(grouped['no_of_days'])

    daily_confirmed=[]

    daily_deaths=[]

    daily_recovered=[]

    active_cases=[]

    for i in range(length):

        if i==0:

            daily_confirmed.append(grouped['confirmed'][i])

            daily_deaths.append(grouped['deaths'][i])

            daily_recovered.append(grouped['recovered'][i])

            active_cases.append(grouped['confirmed'][i])

        else:

            daily_confirmed.append(grouped['confirmed'][i]-grouped['confirmed'][i-1])

            daily_deaths.append(grouped['deaths'][i]-grouped['deaths'][i-1])

            daily_recovered.append(grouped['recovered'][i]-grouped['recovered'][i-1])

            active_cases.append(active_cases[i-1]+daily_confirmed[i]-daily_deaths[i]-daily_recovered[i])

    grouped['daily_confirmed']=daily_confirmed

    grouped['daily_deaths']=daily_deaths

    grouped['daily_recovered']=daily_recovered

    grouped['active_cases']=active_cases

    return grouped



def group_country(country):

    string=str(country)

    grouped_country = data[data['country'] ==string].reset_index()

    grouped_country_date = grouped_country.groupby('no_of_days')['no_of_days', 'confirmed', 'deaths','recovered'].sum().reset_index()

    grouped_country_date = addcolumn_1(grouped_country_date)

    return grouped_country_date



def thousandcaseplot(dataset):

    grouped_=dataset[dataset['confirmed'] >1000].reset_index()

    length=len(grouped_['no_of_days'])

    noofdays=np.arange(length)

    grouped_['no_of_days']=noofdays

    plt.plot(grouped_['no_of_days'],grouped_['confirmed'],linewidth=2)

    

def hundredcaseplot(dataset):

    grouped_=dataset[dataset['confirmed'] >100].reset_index()

    length=len(grouped_['no_of_days'])

    noofdays=np.arange(length)

    grouped_['no_of_days']=noofdays

    plt.plot(grouped_['no_of_days'],grouped_['confirmed'],linewidth=2)



def thousandcasebarplot(dataset,country):

    grouped_=dataset[dataset['confirmed'] >1000].reset_index()

    length=len(grouped_['no_of_days'])

    noofdays=np.arange(length)

    grouped_['no_of_days']=noofdays

    barplot(grouped_,country)



def barplot(dataset,country):

    dataset=dataset

    country=str(country)

    fig = plt.figure()

    ax = fig.add_axes([0,0,1,1])

    ax.bar(dataset['no_of_days']+0.25, dataset['daily_confirmed'], color = 'b',width=0.20)

    ax.bar(dataset['no_of_days']+0.50, dataset['daily_deaths'], color = 'r',width=0.20)

    ax.bar(dataset['no_of_days']+0.75, dataset['daily_recovered'], color = 'g',width=0.20)

    plt.legend(["confirmed","deaths","recovered"])

    plt.xlabel("no of days")

    plt.ylabel("Confirmed cases and deaths")

    plt.title(country + "confirmed and deaths")

    fig.show()



def moving_avg(dataset,country):

    country=str(country)

    length=len(dataset['no_of_days'])

    if length%7 ==0:

        no_of_weeks=length//7

        active=np.array(dataset['active_cases'])

    else:

         no_of_weeks=(length//7) +1  

         active=list(dataset['active_cases'])  

         z=np.zeros((no_of_weeks*7)-(length))

         z[z==0]=np.nan

         active[length:(no_of_weeks*7)]=z

    active=np.array(active)

    active=active.reshape(no_of_weeks,7)

    mvgavg=np.nanmean(active,axis=1)

    plt.plot(np.arange(1,no_of_weeks+1),mvgavg,linewidth=2,label=country)

    plt.text(no_of_weeks,mvgavg[-1],"{}".format(country),fontsize=12)

    

def week_avg(dataset,country):

    country=str(country)

    length=len(dataset['no_of_days'])

    if length%7 ==0:

        no_of_weeks=length//7

        active=np.array(dataset['daily_confirmed'])

    else:

         no_of_weeks=(length//7) +1  

         active=list(dataset['daily_confirmed'])  

         z=np.zeros((no_of_weeks*7)-(length))

         z[z==0]=np.nan

         active[length:(no_of_weeks*7)]=z

    active=np.array(active)

    active=active.reshape(no_of_weeks,7)

    mvgavg=np.nanmean(active,axis=1)   

    return mvgavg
allcountry_data=pd.DataFrame({"countries":allcountries})

xco=[]

xrate=[]

xdouble=[]

for country in allcountries:

    gp_=group_country(country)

    mvgavg=week_avg(gp_,country)

    l=len(mvgavg)

    xco.append(format(country))

    dcdt=mvgavg[l-1]*(mvgavg[l-1]/mvgavg[l-2])

    xrate.append(dcdt)

    x2=(gp_['confirmed'].iloc[-1]/dcdt) + 0

    xdouble.append(x2)

allcountry_data["countries"]=xco

allcountry_data["current rate"]=xrate

allcountry_data["no of days for doubling"]=xdouble
fig,ax = plt.subplots()

bar1=allcountry_data[(allcountry_data["no of days for doubling"]>0) & (allcountry_data["no of days for doubling"]<5)].sort_values(by=["no of days for doubling"])

sns.barplot(bar1["countries"],bar1["no of days for doubling"])

plt.xticks(rotation=90)

plt.title("Countries having doubling time less than 5 days")

plt.text(.4,.93,"Countries with Case doubling time less than 5 days",

          horizontalalignment='center',

          verticalalignment='center',

          transform = ax.transAxes,fontsize=14)

plt.show()

fig,ax = plt.subplots()

bar2=allcountry_data[(allcountry_data["no of days for doubling"]>=5) & (allcountry_data["no of days for doubling"]<10)].sort_values(by=["no of days for doubling"])

sns.barplot(bar2["countries"],bar2["no of days for doubling"])

plt.xticks(rotation=90)

plt.title("Countries with Case Countries having doubling in 5-10 days")

plt.ylim([4,10])

plt.text(.4,.93,"Countries with Case Case doubling in 5 to 10 days",

          horizontalalignment='center',

          verticalalignment='center',

          transform = ax.transAxes,fontsize=14)

plt.show()

fig,ax = plt.subplots()

bar3=allcountry_data[(allcountry_data["no of days for doubling"]>=10) & (allcountry_data["no of days for doubling"]<15)].sort_values(by=["no of days for doubling"])

sns.barplot(bar3["countries"],bar3["no of days for doubling"])

plt.xticks(rotation=90)

plt.title("Countries having doubling time less than 15 days")

plt.ylim([9,15])

plt.text(.4,.93,"Countries with Case Case doubling in 10 to 15 days",

          horizontalalignment='center',

          verticalalignment='center',

          transform = ax.transAxes,fontsize=14)

plt.show()

bar3[bar3["countries"]=="India"]
fig,ax = plt.subplots()

bar4=allcountry_data[(allcountry_data["no of days for doubling"]>=15) & (allcountry_data["no of days for doubling"]<20)].sort_values(by=["no of days for doubling"])

sns.barplot(bar4["countries"],bar4["no of days for doubling"])

plt.xticks(rotation=90)

plt.title("Countries having doubling time less than 20 days")

plt.ylim([14,20])

plt.text(.4,.94,"Countries with Case Case doubling in 15 to 20 days",

          horizontalalignment='center',

          verticalalignment='center',

          transform = ax.transAxes,fontsize=14)

plt.show()

fig,ax = plt.subplots()

bar5=allcountry_data[(allcountry_data["no of days for doubling"]>=20) & (allcountry_data["no of days for doubling"]<50)].sort_values(by=["no of days for doubling"])

sns.barplot(bar5["countries"],bar5["no of days for doubling"])

plt.xticks(rotation=90)

plt.text(.4,.93,"Countries with Case Case doubling in 20 to 50 days",

          horizontalalignment='center',

          verticalalignment='center',

          transform = ax.transAxes,fontsize=14)

plt.ylim([19,50])

plt.show()
case1k=[]

case10k=[]

case25k=[]

case50k=[]

case100k=[]

case500k=[]

case1m=[]

case2m=[]

for country in allcountries:

    gp_=group_country(country)

    l=len(gp_['confirmed'])

    if gp_['confirmed'][l-1] >0 and gp_['confirmed'][l-1] < 1000:

        case1k.append(format(country))

    elif gp_['confirmed'][l-1] >1000 and gp_['confirmed'][l-1] < 10000:

        case10k.append(format(country))

    elif gp_['confirmed'][l-1] >10000 and gp_['confirmed'][l-1] < 25000:

        case25k.append(format(country))

    elif gp_['confirmed'][l-1] >25000 and gp_['confirmed'][l-1] < 50000:

        case50k.append(format(country))

    elif gp_['confirmed'][l-1] >50000 and gp_['confirmed'][l-1] < 100000:

        case100k.append(format(country))

    elif gp_['confirmed'][l-1]>100000 and gp_['confirmed'][l-1] < 500000:

        case500k.append(format(country))

    elif gp_['confirmed'] [l-1]>500000 and gp_['confirmed'] [l-1]< 1000000:

        case1m.append(format(country))

    elif gp_['confirmed'][l-1]>1000000 and gp_['confirmed'][l-1] < 2000000:

        case2m.append(format(country))

    
for x in case25k:

    gpm=group_country(x)

    sns.lineplot(gpm["no_of_days"],gpm["confirmed"],linewidth=2,label=str(x))

plt.title("number of countries having cases bestween 10k and 25k")

plt.show()

print("number of countries having cases bestween 10k and 25k " +str(len(case25k)))
for x in case50k:

    gpm=group_country(x)

    l=len(gpm["no_of_days"])

    sns.lineplot(gpm["no_of_days"],gpm["confirmed"],linewidth=2,label=str(x))

    plt.text(gpm["no_of_days"][l-4],gpm["confirmed"][l-4],"{}".format(x))

plt.title("number of countries having cases bestween 25k and 50k")

plt.xlim([0,gpm["no_of_days"][l-1]+5])

plt.show()

print("number of countries having cases bestween 20k and 50k " +str(len(case50k)))
for x in case100k:

    gpm=group_country(x)

    sns.lineplot(gpm["no_of_days"],gpm["confirmed"],linewidth=2,label=str(x))

    plt.text(gpm["no_of_days"].iloc[-4],gpm["confirmed"].iloc[-4],"{}".format(x))

plt.show()

print("number of countries having cases bestween 50k and 100k " +str(len(case100k)))
for x in case500k:

    gpm=group_country(x)

    sns.lineplot(gpm["no_of_days"],gpm["confirmed"],linewidth=2,label=str(x))

    plt.text(gpm["no_of_days"].iloc[-4],gpm["confirmed"].iloc[-4],"{}".format(x))

plt.show()

print("number of countries having cases bestween 100k and 500k " +str(len(case500k)))
grouped = data.groupby('no_of_days')['no_of_days', 'confirmed', 'deaths','recovered'].sum().reset_index()

grouped=addcolumn_1(grouped)

plt.plot(grouped['no_of_days'],grouped['confirmed'],color='b')

plt.plot(grouped['no_of_days'],grouped['deaths'],color='r')

plt.plot(grouped['no_of_days'],grouped['recovered'],color='g')

plt.xlabel('no of days from first reported case')

plt.ylabel('number of confirmed cases')

plt.title('SARS-NCOVID 19 confirmed cases all around the world', color="w")

plt.legend()

plt.tight_layout()

plt.show()
plt.plot(grouped['no_of_days'],grouped['confirmed'],color='b')

plt.plot(grouped['no_of_days'],grouped['deaths'],color='r')

plt.plot(grouped['no_of_days'],grouped['recovered'],color='g')

plt.xlabel('no of days from first reported case')

plt.ylabel('number of confirmed cases')

plt.title('SARS-NCOVID 19 confirmed cases all around the world', color="w")

plt.yscale('log')

plt.legend()

plt.tight_layout()

plt.show()
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.bar(grouped['no_of_days']+0.25, grouped['daily_confirmed'], color = 'b',width=0.25)

ax.bar(grouped['no_of_days']+0.50, grouped['daily_deaths'], color = 'r',width=0.25)

plt.legend(["confirmed","deaths"])

plt.title("Number of new cases per day")

fig.show()
country="Mainland China"

grouped_china=group_country(country)

plt.plot(grouped_china['no_of_days'],grouped_china['confirmed'],linewidth=2)



country="US"

grouped_US=group_country(country)

plt.plot(grouped_US['no_of_days'],grouped_US['confirmed'],linewidth=2)



country="India"

grouped_india=group_country(country)

plt.plot(grouped_india['no_of_days'],grouped_india['confirmed'],linewidth=2)



country="Spain"

grouped_spain=group_country(country)

plt.plot(grouped_spain['no_of_days'],grouped_spain['confirmed'],linewidth=2)



country="France"

grouped_france=group_country(country)

plt.plot(grouped_france['no_of_days'],grouped_france['confirmed'],linewidth=2)



country="Italy"

grouped_italy=group_country(country)

plt.plot(grouped_italy['no_of_days'],grouped_italy['confirmed'],linewidth=2)



plt.xlabel("no of days")

plt.ylabel("Confirmed cases")

plt.legend(["China","US","India","Spain","France","Italy"])

plt.title("Confirmed cases in countries")
country="Mainland China"

grouped_china=group_country(country)

plt.plot(grouped_china['no_of_days'],grouped_china['active_cases'],linewidth=2)



country="Iran"

grouped_Iran=group_country(country)

plt.plot(grouped_Iran['no_of_days'],grouped_Iran['active_cases'],linewidth=2)



country="Spain"

grouped_spain=group_country(country)

plt.plot(grouped_spain['no_of_days'],grouped_spain['active_cases'],linewidth=2)



country="France"

grouped_france=group_country(country)

plt.plot(grouped_france['no_of_days'],grouped_france['active_cases'],linewidth=2)



country="Italy"

grouped_italy=group_country(country)

plt.plot(grouped_italy['no_of_days'],grouped_italy['active_cases'],linewidth=2)



country="Germany"

grouped_Germany=group_country(country)

plt.plot(grouped_Germany['no_of_days'],grouped_Germany['active_cases'],linewidth=2)



country="UK"

grouped_UK=group_country(country)

plt.plot(grouped_UK['no_of_days'],grouped_UK['active_cases'],linewidth=2)



country="Russia"

grouped_Russia=group_country(country)

plt.plot(grouped_Russia['no_of_days'],grouped_Russia['active_cases'],linewidth=2)



plt.xlabel("no of days")

plt.ylabel("Confirmed cases")

plt.legend(["China","Iran","Spain","France","Italy","Germany","UK","Russia"])

plt.title("Active cases in countries")
country="US"

grouped_US=group_country(country)

plt.plot(grouped_US['no_of_days'],grouped_US['active_cases'],linewidth=2)
country="India"

grouped_india=group_country(country)

plt.plot(grouped_india['no_of_days'],grouped_india['active_cases'],linewidth=2)



country="Singapore"

grouped_Singapore=group_country(country)

plt.plot(grouped_Singapore['no_of_days'],grouped_Singapore['active_cases'],linewidth=2)



country="Japan"

grouped_Japan=group_country(country)

plt.plot(grouped_Japan['no_of_days'],grouped_Japan['active_cases'],linewidth=2)



country="Brazil"

grouped_Brazil=group_country(country)

plt.plot(grouped_Brazil['no_of_days'],grouped_Brazil['active_cases'],linewidth=2)



country="Pakistan"

grouped_Pakistan=group_country(country)

plt.plot(grouped_Pakistan['no_of_days'],grouped_Pakistan['active_cases'],linewidth=2)



country="South Korea"

grouped_Southkorea=group_country(country)

plt.plot(grouped_Southkorea['no_of_days'],grouped_Southkorea['active_cases'],linewidth=2)



plt.xlabel("no of days")

plt.ylabel("Confirmed cases")

plt.legend(["India","Singapore","Japan","Brazil","Pakistan","Southkorea"])

plt.title("Active cases countries at drim of community transmission")
# country="Mainland China"

# moving_avg(grouped_china,country)



# country="US"

# moving_avg(grouped_US,country)



country="India"

moving_avg(grouped_india,country)



country="Spain"

moving_avg(grouped_spain,country)



country="France"

moving_avg(grouped_france,country)



country="Italy"

moving_avg(grouped_italy,country)



country="Iran"

moving_avg(grouped_Iran,country)



country="Russia"

moving_avg(grouped_Russia,country)



country="Germany"

moving_avg(grouped_Germany,country)



plt.xlabel("no of weeks")

plt.ylabel("Moving average of Confirmed cases per week")

plt.legend()

plt.title("Moving average of Confirmed cases in countries")

plt.text(.4,.79,"Seven day Moving Average",

          horizontalalignment='center',

          verticalalignment='center',

          transform = ax.transAxes,fontsize=14)
country="India"

moving_avg(grouped_india,country)



country="Singapore"

moving_avg(grouped_Singapore,country)



country="Japan"

moving_avg(grouped_Japan,country)



country="Brazil"

moving_avg(grouped_Brazil,country)



country="Pakistan"

moving_avg(grouped_Pakistan,country)



country="Southkorea"

moving_avg(grouped_Southkorea,country)



plt.xlabel("no of days")

plt.ylabel("Confirmed cases")

plt.legend(["India","Singapore","Japan","Brazil","Pakistan","South Korea"])

plt.title("Active cases countries at drim of community transmission")


# country="Mainland China"

# grouped_china=group_country(country)

# thousandcaseplot(grouped_china)



# country="US"

# grouped_US=group_country(country)

# thousandcaseplot(grouped_US)



country="India"

grouped_india=group_country(country)

thousandcaseplot(grouped_india)



country="Spain"

grouped_spain=group_country(country)

thousandcaseplot(grouped_spain)



country="France"

grouped_france=group_country(country)

thousandcaseplot(grouped_france)



country="Italy"

grouped_italy=group_country(country)

thousandcaseplot(grouped_italy)







plt.xlabel("no of days")

plt.ylabel("Confirmed cases")

plt.legend(["India","Spain","France","Italy"])

plt.title("Confirmed cases in countries after 100 conformations")
country="Spain"

grouped_spain=group_country(country)

thousandcasebarplot(grouped_spain,country)

plt.text(0.5, 1,'Covid-19 Spani',

     horizontalalignment='center',

     verticalalignment='center',

     transform = ax.transAxes,fontsize=20)

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams["font.size"] = "10"

# plt.savefig("Covid-19 Spain")
country="Italy"

grouped_italy=group_country(country)

thousandcasebarplot(grouped_italy,country)
country="US"

grouped_US=group_country(country)

thousandcasebarplot(grouped_US,country)
country="France"

grouped_france=group_country(country)

thousandcasebarplot(grouped_france,country)
country="India"

grouped_india=group_country(country)

thousandcasebarplot(grouped_india,country)