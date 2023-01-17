#Importing packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os



import seaborn as sns

from datetime import datetime



plt.rcParams.update({'font.size': 12})





from datetime import datetime

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime



plt.rcParams.update({'font.size': 12})







# Load the Covid19 dataset

#df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', index_col="SNo", )

#df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates =["ObservationDate"])

df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates = ['ObservationDate','Last Update'])

#Explore the dataset

df.head(10)
#The shape of the dataset

df.shape
#Report dataset



a = df['Country/Region']

country = len(np.unique(a))

print( '** This Dataset is Updated Daily')



## Mercy's Comment: I edited this to add the variable that holds the number of countries. It was giving an empty list initally because of the way `a` was defined

print ('Updates on Covid19 pandemic from ' + str(df.ObservationDate.min())+' to '+str(df.ObservationDate.max())+' from ' + str(country) + ' Countries/Region around the world')
print("Number of observations in the Dataset = ", df.shape[0])

print("Number of Features for each observation = ", df.shape[1]) 
#The features of the dataset

col =(df.columns[:,])

print([col])



## Mercy's Comment: df.columns also works fine here.
#Filter Nigeria

#df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates =["ObservationDate"], index_col="ObservationDate")



nigeria =df[df['Country/Region'] == "Nigeria"]



#Explore the dataset for Nigeria

nigeria.head(10)

#All the Province/State Colums are dropped because all are NaN

nigeria3=nigeria.drop(['SNo','Last Update','Province/State','Country/Region'], axis = 1)



nigeria3.tail()

nigeria3.shape
#The quoted death rate is taken from confirmed deaths over total reported cases. 

update = nigeria3[nigeria3.ObservationDate == nigeria3.ObservationDate.max()]

print ('Total confirmed cases in Nigeria: %.d' %np.sum(update['Confirmed']))

print ('Total death cases in Nigeria: %.d' %np.sum(update['Deaths']))

print ('Total recovered cases in Nigeria: %.d' %np.sum(update['Recovered']))

print ('Death rate in Nigeria  %%: %.2f' % (np.sum(update['Deaths'])/np.sum(update['Confirmed'])*100))



## Mercy's Comment: This was nicely executed. Another alternative would be to use .iloc to 

## select the values from the last row of the dataframe i.e 



## confirmed = nigeria3.iloc[-1,:]['Confirmed']

## deaths = nigeria3.iloc[-1,:]['Deaths']

## recovered = nigeria3.iloc[-1,:]['Recovered']
#Data Visualization



import matplotlib

x1 =np.sum(update['Confirmed'])

x2 =sum(update['Deaths'])

x3 = np.sum(update['Recovered'])

x_list =[x1,x2,x3]

colors = ['#ff9999','#66b3ff','#99ff99']

labels = ['Confirmed','Dead', 'Recovered']

label = 'Visual Representation of Global Covid19 trend in Nigeria'

# ,'#ffcc99'

fig1, ax1 = plt.subplots()

plt.rcParams["figure.figsize"] = [16,9]

plt.pie(x_list,  labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)

#ax1.pie( colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)explode=explode,

#draw circle

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.legend(labels, loc="upper left")

plt.axis('equal')

plt.title(label,fontsize = 16)


l=len(nigeria3)

i = 0

y = 1

c=nigeria3.Confirmed.iloc[0]

d=nigeria3.Deaths.iloc[0]

r= nigeria3.Recovered.iloc[0]

 

daily_confirm =[c]

daily_deaths=[d]

daily_recovery=[r]

#daily_active =[]

#growth_Rate = [c]



while ((i <= (l-2)) & (y <= (l-1))):

    day_confirm  =(nigeria3.Confirmed.iloc[y] -nigeria3.Confirmed.iloc[i])

    #growthRate = (((nigeria3.Confirmed.iloc[y]) / (nigeria3.Confirmed.iloc[i]))-1)

    day_deaths  =(nigeria3.Deaths.iloc[y] -nigeria3.Deaths.iloc[i])

    #day_active =(nigeria3.Confirmed.iloc[i] -nigeria3.Deaths.iloc[i]-nigeria3.Recovered.iloc[i])

      

    day_recovery  =(nigeria3.Recovered.iloc[y] -nigeria3.Recovered.iloc[i])  

    daily_confirm.append(day_confirm)

    daily_deaths.append(day_deaths)

    daily_recovery.append(day_recovery)

    #daily_active.append(day_active)

    #growth_Rate.append(growthRate)

    i += 1

    y+=1



nigeria3['daily_confirm'] = daily_confirm 

nigeria3['daily_deaths'] = daily_deaths

nigeria3['daily_recovery'] = daily_recovery

#nigeria3['daily_active'] = daily_active

#nigeria3['growth_Rate'] = growthRate   

nigeria3.tail(20)
# Calculating daily active cases in Nigeria

l=len(nigeria3)

i = 0

#actv= (nigeria3.Confirmed.iloc[i] -nigeria3.Deaths.iloc[i]-nigeria3.Recovered.iloc[i])

daily_active =[]

while ((i <= (l-1))):

    day_active =(nigeria3.Confirmed.iloc[i] -nigeria3.Deaths.iloc[i]-nigeria3.Recovered.iloc[i])

    daily_active.append(day_active)

    i+=1

nigeria3['daily_active'] = daily_active



## Mercy's Comment: To calculate the daily_active cases, you don't need a while loop, 

## pandas handles this type of data manipulation by default internally.



## Here's an example below:

nigeria3['daily_active_2'] = nigeria3['Confirmed'] - nigeria3['Deaths'] - nigeria3['Recovered']
nigeria3
new = nigeria3[nigeria3['daily_confirm'] == nigeria3.daily_confirm.max()]

print('Higheset new confirmed cases in a day in Nigeria =',(new['daily_confirm']).to_string(index = False))

print('Day with Higheset New Confirmed cases in Nigeria =',(new['ObservationDate']).to_string(index = False))
new = nigeria3[nigeria3['daily_deaths'] == nigeria3.daily_deaths.max()]

print('Higheset Deaths in Nigeria in a day =',(new['daily_deaths']).to_string(index = False))

print('Day with Higheset Deaths in Nigeria =',(new['ObservationDate']).to_string(index = False))
new = nigeria3[nigeria3['daily_recovery'] == nigeria3.daily_recovery.max()]

print('Higheset Number of Recovery in a day =',(new['daily_recovery']).to_string(index = False))

print('Day with Higheset Recovery in Nigeria =',(new['ObservationDate']).to_string(index = False))
new = nigeria3[nigeria3['daily_active'] == nigeria3.daily_active.max()]

print('Higheset Number of Active Cases in a day =',(new['daily_active']).to_string(index = False))

print('Day with Higheset Active Cases in Nigeria =',(new['ObservationDate']).to_string(index = False))
import matplotlib

%matplotlib inline

label = "Daily Growth Rate of Covid19 Pandemic in Nigeria"



plt.rcParams["figure.figsize"] = [16,9]

ax = plt.gca()

nigeria3.plot(kind='line',x ='ObservationDate',y='daily_confirm',color='red',ax=ax)

nigeria3.plot(kind='line',x ='ObservationDate',y='daily_deaths',color='green',ax=ax)

nigeria3.plot(kind='line',x ='ObservationDate',y='daily_recovery',ax=ax)

#nigeria3.plot(kind='line',x ='ObservationDate',y='daily_active',ax=ax)

plt.title(label,fontsize = 16)
#Daily New Cases of Covid19 Pandemic in Nigeria

import matplotlib

%matplotlib inline

label = "Daily New Confirmed Cases of Covid19 Pandemic in Nigeria"



plt.rcParams["figure.figsize"] = [16,9]

ax = plt.gca()

nigeria3.plot(kind='line',x ='ObservationDate',y='daily_confirm',color='red',ax=ax)

plt.title(label,fontsize = 16)
#Daily Cummulative Cases of Covid19 Pandemic in Nigeria

import matplotlib

%matplotlib inline

label = "Daily Cummulative Cases of Covid19 Pandemic in Nigeria"



plt.rcParams["figure.figsize"] = [16,9]

ax = plt.gca()

nigeria3.plot(kind='line',x ='ObservationDate',y='Confirmed',color='red',ax=ax)

plt.title(label,fontsize = 16)
#Daily New Deaths from Covid19 Pandemic in Nigeria

import matplotlib

%matplotlib inline

label = "Daily New Deaths from Covid19 Pandemic in Nigeria"



plt.rcParams["figure.figsize"] = [16,9]

ax = plt.gca()

nigeria3.plot(kind='line',x ='ObservationDate',y='daily_deaths',color='green',ax=ax)

plt.title(label,fontsize = 16)
#Daily Cummulative Deaths of Covid19 Pandemic in Nigeria

import matplotlib

%matplotlib inline

label = "Daily Cummulative Deaths of Covid19 Pandemic in Nigeria"



plt.rcParams["figure.figsize"] = [16,9]

ax = plt.gca()

nigeria3.plot(kind='line',x ='ObservationDate',y='Deaths',color='green',ax=ax)

plt.title(label,fontsize = 16)
#Daily NewRecovery from Covid19 Pandemic in Nigeria

import matplotlib

%matplotlib inline

label = "Daily New Recovery from Covid19 Pandemic in Nigeria"



plt.rcParams["figure.figsize"] = [16,9]

ax = plt.gca()

nigeria3.plot(kind='line',x ='ObservationDate',y='daily_recovery',color='blue',ax=ax)

plt.title(label,fontsize = 16)
#Daily NewRecovery from Covid19 Pandemic in Nigeria

import matplotlib

%matplotlib inline

label = "Cummulative Daily Recovery from Covid19 Pandemic in Nigeria"



plt.rcParams["figure.figsize"] = [16,9]

ax = plt.gca()

nigeria3.plot(kind='line',x ='ObservationDate',y='Recovered',color='blue',ax=ax)

plt.title(label,fontsize = 16)
#Daily Active Cases from Covid19 Pandemic in Nigeria

import matplotlib

%matplotlib inline

label = "Daily Active Cases of Covid19 Pandemic in Nigeria"



plt.rcParams["figure.figsize"] = [16,9]

ax = plt.gca()

nigeria3.plot(kind='line',x ='ObservationDate',y='daily_active',color='orange',ax=ax)

plt.title(label,fontsize = 16)
label = "Daily New Cases of Covid19 in Nigeria "

plt.rcParams["figure.figsize"] = [16,9]

ax = plt.gca()

nigeria3.plot(kind='bar',x ='ObservationDate',y='daily_confirm', ax=ax)

plt.title(label,fontsize = 16)
#Drop the Last Update column

df.drop(["Last Update"], axis = 1, inplace = True)

df.head(10)
#Drop the Last Serial no

df.drop(["SNo"], axis = 1, inplace = True)

df.head(10)
#Verifying that each Country/Region has only one observation each day

verify = df.groupby(['Country/Region','Province/State','ObservationDate']).count().iloc[:,0]

verify[verify>1]
# Clean data

#Remove zero values

df = df[(df.Confirmed>0) | (df['Province/State'] == 'Recovered')]



# To check null values

df.isnull().sum()
# Sort data

df = df.sort_values(['ObservationDate','Country/Region','Province/State'])

# Add column for the first confirmed case of Covid19  

df['1st_date'] = df.groupby('Country/Region')['ObservationDate'].transform('min')



# Add column for the number of days since the first confirmed case of Covid19

df['days'] = (df['ObservationDate'] - df['1st_date']).dt.days



df

evo=df

evo
df.head()
update = df[df.ObservationDate == df.ObservationDate.max()]

print ('Total Confirmed Cases: %.d' %np.sum(update['Confirmed']))

print ('Total Death Cases: %.d' %np.sum(update['Deaths']))

print ('Total Recovered Cases: %.d' %np.sum(update['Recovered']))

print ('Global Death Rate %%: %.2f' % (np.sum(update['Deaths'])/np.sum(update['Confirmed'])*100))
update.head()
import matplotlib

x1 =np.sum(update['Confirmed'])

x2 =sum(update['Deaths'])

x3 = np.sum(update['Recovered'])

x_list =[x1,x2,x3]

colors = ['#ffff88','#66b3ff','#ff9999']

labels = ['Comfirmed','Deaths', 'Recovered']

label = 'Visual Representation of the Global Covid19 trend'

# ,'#ffcc99''#99ff99'

fig1, ax1 = plt.subplots()

plt.rcParams["figure.figsize"] = [16,3]

plt.pie(x_list,  labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)

#ax1.pie( colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)explode=explode,

#draw circle

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()



plt.legend(labels, loc="upper left")

plt.axis('equal')

plt.title(label,fontsize = 16)
#China Covid19 Update

a =update[update['Country/Region'] == 'Mainland China']

a.sum()
update.head()
ab =update['Country/Region']

countries = np.unique(ab)

update.tail(20)
update.shape
# finding the total Number of Countries/Regions Worldwide

df = df.sort_values(['ObservationDate','Country/Region','Province/State'])

update = df[df.ObservationDate == df.ObservationDate.max()]

ab =update['Country/Region']

countries = np.unique(ab).copy()

countries

print('Total Number of Countries/Regions in the Dataset: ',len(countries))
#List of countries/Region

countries
#Creating a new dataframe 

l=len(countries)-1

confirmsum =[]

deathsum=[]

recoveredsum=[]

country = []

i = 0

y=0

while (i <= l):

    t=update[update['Country/Region'] == countries[i]]

    

    #print(countries[i])



    t1 = t.Confirmed.sum()

    t2 = t.Deaths.sum()

    t3 = t.Recovered.sum()

    #print('Total Confirmed',t1,'Total Deaths',t2,'Total Recovered',t3)

    #print()

   

    confirmsum.append(t1)

    deathsum.append(t2)

    recoveredsum.append(t3)

    i+=1    

   
data = pd.DataFrame({

    'Country/Region':countries,

    'Comfirmed':confirmsum,

    'Deadth':deathsum,

    'Recovery':recoveredsum

  

})
# List of Comfirmed cases, death and recovery by country

data
# Country with highest number of deaths

a=data[data['Deadth'] == data.Deadth.max()]

a
high =(a['Country/Region']).to_string(index = False)

print('Country with highest number of deaths is',high)

# Top 10 Countries with highest number of deaths

data = data.sort_values(['Deadth'], ascending=False)

data.head(10)
#List of contries with the sum of Death Rate	Recovery Rate	Active



data['Death Rate'] = data['Deadth'] / data['Comfirmed'] * 100

data['Recovery Rate'] = data['Recovery'] / data['Comfirmed'] * 100

data['Active'] = data['Comfirmed'] - data['Deadth'] - data['Recovery']





#Group by country/region

region = data.groupby('Country/Region').sum()

region.sort_values(['Deadth'], ascending=False)





region.head(10)


x1 = np.sum(region['Comfirmed'])

x2 = sum(region['Deadth'])

x3 = np.sum(region['Active'])

x_list =[x1,x2,x3]

colors = ['#bbbb88','#66b3ff','#999999']

#'#ffff88','#66b3ff','#ff9999'

labels = ['Comfirmed','Deaths', 'Active']

label = 'Visual Representation of the Global Covid19 Trend'

# ,'#ffcc99''#99ff99'

fig1, ax1 = plt.subplots()

plt.rcParams["figure.figsize"] = [26,3]



plt.pie(x_list,  labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)



#draw circle

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()



plt.legend(labels, loc="upper right")

plt.axis('equal')

plt.title(label,fontsize = 16)



#Total active cases of Covid19 globally

active =data['Active']

act=active.sum()



print('Total active cases of Covid19 globally is',act)
# Top 50 Countries with highest confirmed cases

a1 = data.sort_values(['Deadth'], ascending=False)

a1.head(10)
#Visualizing the Trend using bar plots

#Top 50 Countries with highest deaths from Covid19 Globally

import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt

label = "50 Countries with Highest Deaths from Covid19 Globally "



#a1 =a1[0:50]

d =data[0:50]

d10 = data[0:10]

plt.rcParams["figure.figsize"] = [16,9]

ax = plt.gca()

d.plot(kind='barh',x ='Country/Region',y='Deadth', color = 'red', ax=ax)

plt.title(label,fontsize = 16)

plt.show()
#Comparison of Death Since First Case for top 10 Countrie/Region

df_break = df.groupby(['Country/Region','days'])['Confirmed','Deaths'].sum().reset_index()

top10case = update.sort_values('Confirmed', ascending=False).head(10)['Country/Region'].to_list()

top10death = update.sort_values('Deaths', ascending=False).head(10)['Country/Region'].to_list()



_ = df_break[df_break['Country/Region'].isin(top10death)]

plt.figure(figsize=(10,6))

sns.lineplot(x='days',y='Deaths', data=_, hue='Country/Region', lw=2)

plt.legend(bbox_to_anchor=(1.02, 1), fontsize=10)

plt.grid(True)

plt.title('Comparison of Death Since First Case')

plt.show()
# Top 50 Countries with highest confirmed cases

a2 = data.sort_values(['Comfirmed'], ascending=False)

a2.head(50)
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt

label = "50 Countries with the Highest Confirmed Cases of Covid19 Globally "



d =data[0:50]



plt.rcParams["figure.figsize"] = [16,9]

ax = plt.gca()

d.plot(kind='barh',x ='Country/Region',y='Comfirmed', color = 'green', ax=ax)

plt.title(label,fontsize = 16)

plt.show()

#Comparison of Confirmed Cases Since First Case for the top 10 countries/region

_ = df_break[df_break['Country/Region'].isin(top10case)]

plt.figure(figsize=(10,6))

sns.lineplot(x='days',y='Confirmed', data=_, hue='Country/Region', lw=2)

plt.legend(bbox_to_anchor=(1.02, 1), fontsize=10)

plt.grid(True)

plt.title('Comparison of Number of Confirmed Cases Since First Case')

plt.show()
#Comparison of First 10 Countries with Confirmed Cases'

early_10 = df.groupby('Country/Region')['ObservationDate'].min().sort_values().head(10).index

_ = df_break[df_break['Country/Region'].isin(early_10)]

plt.figure(figsize=(10,6))

sns.lineplot(x='days',y='Confirmed', data=_, hue='Country/Region', lw=2)

plt.legend(bbox_to_anchor=(1.02, 1), fontsize=10)

plt.yscale('log')

plt.grid(True)

plt.title('Comparison of First 10 Countries with Confirmed Cases')

plt.show()
# Top 50 Countries with highest active cases

a3 = data.sort_values(['Active'], ascending=False)

a3.head(50)
# 50 Countries with the Highest Active Cases of Covid19 Globally

import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt

label = "50 Countries with the Highest Active Cases of Covid19 Globally "



d =data[0:50]



plt.rcParams["figure.figsize"] = [16,9]

ax = plt.gca()

d.plot(kind='barh',x ='Country/Region',y='Active', ax=ax)

plt.title(label,fontsize = 16)

plt.show()
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime



plt.rcParams.update({'font.size': 12})

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
# Total Confirmed	Deaths	Recovered	Active	Death Rate	Recover Rate daily

updated = df.groupby('ObservationDate')[['Confirmed','Deaths','Recovered']].sum()



updated['Active'] =updated['Confirmed'] - updated['Deaths'] - updated['Recovered']

updated['Death Rate'] = updated['Deaths'] / updated['Confirmed'] * 100

updated['Recover Rate'] = updated['Recovered'] / updated['Confirmed'] * 100

updated.head(20)



import matplotlib.dates as mdates

months_fmt = mdates.DateFormatter('%b-%e')

def plot_updated(num, col, title):

    ax[num].plot(updated[col], lw=3, color = 'red')

    ax[num].set_title(title)

    ax[num].xaxis.set_major_locator(plt.MaxNLocator(7))

    ax[num].xaxis.set_major_formatter(months_fmt)

    ax[num].grid(True)

fig, ax = plt.subplots(2, 2, figsize=(12,9))  

plot_updated((0,0), 'Confirmed', 'Confirmed cases')

plot_updated((0,1), 'Deaths', 'Death cases')

plot_updated((1,0), 'Active', 'Active cases')

plot_updated((1,1), 'Death Rate', 'Death rate')
def plot_country(num, evo_col, title):

    ax[num].plot(evo_col, lw=3, color = 'green')

    ax[num].set_title(title)

    ax[num].xaxis.set_major_locator(plt.MaxNLocator(7))

    ax[num].xaxis.set_major_formatter(months_fmt)

    ax[num].grid(True)



def evo_country(country):

    evo_country = df[df['Country/Region']==country].groupby('ObservationDate')[['Confirmed','Deaths','Recovered']].sum()

    evo_country['Active'] = evo_country['Confirmed'] - evo_country['Deaths'] - evo_country['Recovered']

    evo_country['Death Rate'] = evo_country['Deaths'] / evo_country['Confirmed'] * 100

    plot_country((0,0), evo_country['Confirmed'], 'Confirmed cases')

    plot_country((0,1), evo_country['Deaths'], 'Death cases')

    plot_country((1,0), evo_country['Active'], 'Active cases')

    plot_country((1,1), evo_country['Death Rate'], 'Death rate')

    fig.suptitle(country, fontsize=16)

    plt.show()
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_country('Nigeria')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_country('Canada')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_country('US')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_country('UK')