import numpy as np

import pandas as pd

import datetime

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')



def getMiliSeconds(time):

    try:

        if '.' in time:

            x = datetime.datetime.strptime(time,'%M:%S.%f')

        elif ',' in time:

            x = datetime.datetime.strptime(time,'%M:%S,%f')

        else:

            x = datetime.datetime.strptime(time,'%M:%S:%f')

        return datetime.timedelta(minutes=x.minute,seconds=x.second, microseconds = x.microsecond).total_seconds()

    except:

        x = datetime.datetime.strptime(str(time).split('.')[0],'%M:%S:%f')

        return datetime.timedelta(minutes=x.minute,seconds=x.second, microseconds = x.microsecond).total_seconds()



def wins_per_year(driverRef, year):

    try:

        return topTenYears.loc[(topTenYears['driverRef'] == driverRef) & (topTenYears['positionOrder'] == 1) & (topTenYears['year'] == year)].groupby('driverId')['raceId'].count().values[0]

    except:

        return 0



def championsInYears(years, driverRef):

    total = []

    t = 0

    for year in years:

        winner = r.loc[r['year']== year].groupby('driverRef')['points'].sum().sort_values(ascending = False).index[0]

        if winner == driverRef:

            t = t + 1

        total.append(t)

    return total



df_races  = pd.read_csv('/kaggle/input/races.csv')

df_drivers = pd.read_csv('/kaggle/input/drivers.csv',encoding='latin1')

df_circuit = pd.read_csv('/kaggle/input/circuits.csv',encoding='latin-1')

df_constructors = pd.read_csv('/kaggle/input/constructors.csv',encoding='latin-1')

df_qualy = pd.read_csv('/kaggle/input/qualifying.csv',encoding='latin-1')

df_results = pd.read_csv('/kaggle/input/results.csv',encoding='latin-1')

df_status = pd.read_csv('/kaggle/input/status.csv')

df_timeLaps = pd.read_csv('/kaggle/input/lapTimes.csv')



df_races = df_races.drop(['date','time','url','round','circuitId'],axis=1)

df_drivers = df_drivers.drop(['number','code','dob','url'], axis=1)

df_constructors = df_constructors.drop(['nationality','Unnamed: 5','url','name'], axis=1)

df_results = df_results.drop(['number','grid','positionText','position','laps','time','milliseconds','rank','fastestLap','fastestLapSpeed','resultId'], axis=1)

df_timeLaps = df_timeLaps.drop(['milliseconds'],axis = 1)



df_results['fastestLapTime'] = df_results['fastestLapTime'].fillna('00:00.0')

df_qualy['q1'] = df_qualy['q1'].fillna('00:00.0')

df_qualy['q2'] = df_qualy['q2'].fillna('00:00.0')

df_qualy['q3'] = df_qualy['q3'].fillna('00:00.0')

df_timeLaps['time'] = df_timeLaps['time'].fillna('00:00.0')



df_qualy['q1'] = df_qualy['q1'].apply(lambda x: getMiliSeconds(x))

df_qualy['q2'] = df_qualy['q2'].apply(lambda x: getMiliSeconds(x))

df_qualy['q3'] = df_qualy['q3'].apply(lambda x: getMiliSeconds(x))

df_timeLaps['time'] = df_timeLaps['time'].apply(lambda x: getMiliSeconds(x))

df_results['fastestLapTime'] = df_results['fastestLapTime'].apply(lambda x: getMiliSeconds(x))



df_drivers['fullName'] = df_drivers['forename'] +" "+ df_drivers['surname']

    
results = pd.merge(df_drivers,df_results, on='driverId',how='inner')

len(results.loc[(results['positionOrder'] == 1) & (results['driverId'] == 1)])
from wordcloud import WordCloud, STOPWORDS

from collections import Counter

import random

def grey_color_func(word, font_size, position, orientation, random_state=None,

                    **kwargs):

    return "hsl(10, 100%%, %d%%)" % random.randint(15, 90)



q = pd.merge(df_qualy,df_drivers,how='inner',on='driverId')

q = pd.merge(q,df_races,how='inner',on=['raceId'])

q = q.loc[q['position']==1]



array = list(q.loc[q['position']== 1]['fullName'].values)



word_could_dict=Counter(array)

##CORRECTING THE DATASET, BECAUSE DATASET OF QUALIFICATION IS MISSING SOME DATA

word_could_dict['Lewis Hamilton'] = 72

word_could_dict['Michael Schumacher'] = 68

word_could_dict['Ayrton Senna'] = 65

word_could_dict['Alain Prost'] = 33

word_could_dict['Jim Clark'] = 33

word_could_dict['Nigel Mansell'] = 32

word_could_dict['Kimi RÌ_ikkÌ¦nen'] = 0

word_could_dict['Kimi Raikkonen'] = word_could_dict['Kimi RÌ_ikkÌ¦nen']

word_could_dict['Mika Hakkinen'] = word_could_dict['Mika HÌ_kkinen']

word_could_dict.pop('Kimi RÌ_ikkÌ¦nen')

word_could_dict.pop('Mika HÌ_kkinen')

################################################################################

wordcloud = WordCloud(width = 1000, height = 1000, max_font_size=100).generate_from_frequencies(word_could_dict)

default_colors = wordcloud.to_array()

plt.figure(figsize=(40,50))

plt.title("F1 Pole Quantity",fontsize=100)

plt.axis("off")

plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),

           interpolation="bilinear")

plt.show()
gb = pd.merge(results.loc[results['driverId'] == 1],df_races, on='raceId',how='inner')

gb = gb.loc[gb['positionOrder']==1].groupby('name')['driverId'].count().sort_values()



f,ax = plt.subplots(figsize =(25,15))

ax.tick_params(axis="x", labelsize=20)

ax.tick_params(axis="y", labelsize=30)



ax.barh(list(gb.index), list(gb.values),height = 0.9, color = ['c','gray'])

ax.set_facecolor('whitesmoke')

ax.patch.set_alpha(0.9)

plt.title("Victory by Gran Prix",fontsize=30)

plt.ylabel('Gran Prix',fontsize = 20)

plt.xlabel('Total',fontsize = 20,)

plt.grid()

plt.show()
topBritish = results.loc[(results['nationality'] == 'British') & (results['positionOrder'] == 1)].groupby('driverRef')['raceId'].count().sort_values(ascending=False)[:10]

topBritish = topBritish.sort_values()

topWorld = results.loc[(results['positionOrder'] == 1)].groupby('driverRef')['raceId'].count().sort_values(ascending=False)[:20]

topWorld = topWorld.sort_values()



f,ax = plt.subplots(figsize =(25,15))

ax.tick_params(axis="x", labelsize=20)

ax.tick_params(axis="y", labelsize=30)



ax.barh(list(topBritish.index), list(topBritish.values),height = 0.9, color = ['darkblue','crimson'])

ax.set_facecolor('whitesmoke')

ax.patch.set_alpha(0.9)

plt.title("Top British Race Winners",fontsize=30)

plt.ylabel('Drivers',fontsize = 15)

plt.xlabel('Total Wins',fontsize = 15,)

plt.grid()

plt.show()



f,ax = plt.subplots(figsize =(25,15))

ax.tick_params(axis="x", labelsize=20)

ax.tick_params(axis="y", labelsize=30)



ax.barh(list(topWorld.index), list(topWorld.values),height = 0.9, color = ['c','gray'])

ax.set_facecolor('k')

ax.patch.set_alpha(0.9)

plt.title("Top World Race Winners",fontsize=30)

plt.ylabel('Drivers',fontsize = 15)

plt.xlabel('Total Wins',fontsize = 15,)

plt.grid()

plt.show()



topTenYears = pd.merge(results,df_races,on='raceId',how='inner')

hamWinsPerYear = []

vetWinsPerYear = []

rosWinsPerYear = []

butWinsPerYear = []

aloWinsPerYear = []

for year in range(2007,2018):

    hamWinsPerYear.append(wins_per_year('hamilton',year))

    vetWinsPerYear.append(wins_per_year('vettel',year))

    rosWinsPerYear.append(wins_per_year('rosberg',year))

    butWinsPerYear.append(wins_per_year('button',year))

    aloWinsPerYear.append(wins_per_year('alonso',year))   



x = range(2007,2018)

f,ax = plt.subplots(figsize =(20,10))

ax.tick_params(axis="x", labelsize=15)

ax.tick_params(axis="y", labelsize=30)



ax.set_facecolor('whitesmoke')

line1, = ax.plot(x, hamWinsPerYear, label='Hamilton Wins', color='c',linewidth=4,)

line2, = ax.plot(x, vetWinsPerYear, label='Vettel Wins' ,linewidth=2, color='r')

line3, = ax.plot(x, rosWinsPerYear, label='Rosberg Wins', color='k',linewidth=2,)

line4, = ax.plot(x, butWinsPerYear, label='Button Wins' ,linewidth=2, color='gold')

line5, = ax.plot(x, aloWinsPerYear, label='Alonso Wins', color='deeppink',linewidth=2,)

plt.title("Top 5 from the Last 10 years - Wins per Year ",fontsize=30)

plt.ylabel('Total',fontsize = 15)

plt.xlabel('Years',fontsize = 15,)

plt.grid()

ax.legend()

plt.show()



r = pd.merge(df_races,df_results)

r = pd.merge(r,df_drivers)

# winnersYear = []

# for year in :

#     r.loc[r['year']== year].groupby('driverRef')['points'].sum().sort_values(ascending = False).index[0]



schummyCP = championsInYears(np.sort(r['year'].unique()),'michael_schumacher')

fangioCP = championsInYears(np.sort(r['year'].unique()),'fangio')

prostCP = championsInYears(np.sort(r['year'].unique()),'prost')

hamiltonCP = championsInYears(np.sort(r['year'].unique()),'hamilton')

vettelCP = championsInYears(np.sort(r['year'].unique()),'vettel')



x = range(1950,2018)

f,ax = plt.subplots(figsize =(20,10))

ax.tick_params(axis="x", labelsize=15)

ax.tick_params(axis="y", labelsize=30)



ax.set_facecolor('whitesmoke')

line1, = ax.plot(x, schummyCP, label='Schumacher World Championships Count', color='r',linewidth=2,)

line2, = ax.plot(x, fangioCP, label='Fangio World Championships Count' ,linewidth=2, color='magenta')

line3, = ax.plot(x, prostCP, label='Prost World Championships Count', color='k',linewidth=2,)

line4, = ax.plot(x, hamiltonCP, label='Hammilton World Championships Count' ,linewidth=4, color='aqua')

line5, = ax.plot(x, vettelCP, label='Vettel World Championships Count', color='darkblue',linewidth=2,)

plt.title("Top 5 from All Time - Championships over the Years ",fontsize=30)

plt.ylabel('Total',fontsize = 15)

plt.xlabel('Years',fontsize = 15,)

plt.grid()

ax.legend()

plt.show()
result_races = pd.merge(df_results,df_races, how='inner',on='raceId')

mercedesId = df_constructors.loc[(df_constructors['constructorRef']=='mercedes')]['constructorId'].values[0]  #131

hamQtdQualy = len(df_qualy.loc[(df_qualy['driverId']==1) & (df_qualy['position']==1) & (df_qualy['constructorId']==mercedesId)]) # 46

rosQtdQualy = len(df_qualy.loc[(df_qualy['driverId']==3 ) & (df_qualy['position']==1) & (df_qualy['constructorId']==mercedesId)]) # 30

hamQtdRaces = len(results.loc[(results['positionOrder'] == 1) & (results['driverId'] == 1) & (results['constructorId'] == 131)])

rosQtdRaces = len(results.loc[(results['positionOrder'] == 1) & (results['driverId'] == 3) & (results['constructorId'] == 131)])

hamWinsPerYear = result_races.loc[(result_races['driverId']==1) & (result_races['constructorId'] == 131) & (result_races['positionOrder'] == 1) & (result_races['year'] < 2017)].groupby('year')['driverId'].count()

rosWinsPerYear = result_races.loc[(result_races['driverId']==3) & (result_races['constructorId'] == 131) & (result_races['positionOrder'] == 1) & (result_races['year'] > 2012)].groupby('year')['driverId'].count()



f,ax = plt.subplots(figsize =(25,15))

ax.tick_params(axis="x", labelsize=30)

ax.tick_params(axis="y", labelsize=20)



ax.bar(['Hamilton','Rosberg'], [hamQtdQualy,rosQtdQualy], width = 0.25, color = ['c','gray'])

ax.set_facecolor('whitesmoke')

plt.title("Total of Poles in Mercedes",fontsize=30)

plt.ylabel('Total',fontsize = 15)

plt.xlabel('Pilots',fontsize = 15,)

plt.show()



f,ax = plt.subplots(figsize =(25,15))

ax.tick_params(axis="x", labelsize=30)

ax.tick_params(axis="y", labelsize=20)



ax.bar(['Hamilton','Rosberg'], [hamQtdRaces,rosQtdRaces], width = 0.25, color = ['c','gray'], align='center')

ax.set_facecolor('whitesmoke')

plt.title("Total of Races Won in Mercedes",fontsize=30)

plt.ylabel('Total',fontsize = 15)

plt.xlabel('Pilots',fontsize = 15,)

plt.show()



x = list(hamWinsPerYear.index)

y = hamWinsPerYear

f,ax = plt.subplots(figsize =(20,10))

ax.tick_params(axis="x", labelsize=15)

ax.tick_params(axis="y", labelsize=30)



ax.set_facecolor('k')

line1, = ax.plot(x, hamWinsPerYear.values, label='Hamilton Wins', color='c',linewidth=4,)

line2, = ax.plot(x, rosWinsPerYear.values, label='Rosberg Wins' ,linewidth=4, color='lightgray')

plt.title("Total Wins per Year in Mercedes",fontsize=30)

plt.ylabel('Total',fontsize = 15)

plt.xlabel('Years',fontsize = 15,)

ax.legend()

plt.show()
pd.merge(df_drivers,df_results.loc[(df_results['raceId'] == 940)],on='driverId',how='inner').sort_values(['positionOrder'])[:3]
timeHamilton = df_timeLaps.loc[((df_timeLaps['driverId'] == 1) & (df_timeLaps['raceId']==940))]['time'].iloc[:].values

timeRos = df_timeLaps.loc[(df_timeLaps['driverId'] == 3) & (df_timeLaps['raceId']==940)]['time'].iloc[:].values



x = range(0,53)

f,ax = plt.subplots(figsize =(20,10))

ax.tick_params(axis="x", labelsize=15)

ax.tick_params(axis="y", labelsize=20)



ax.set_facecolor((0, 0, 0))

ax.patch.set_alpha(0.9)

line1, = ax.plot(x, timeHamilton, label='Hamilton Time', color='gray',linewidth=4,)

line2, = ax.plot(x, timeRos, label='Rosberg Time' ,linewidth=4,)

plt.title("Suzuka GP - Time Laps Comparison - 2014 ",fontsize=30)

plt.ylabel('Total Seconds(s)',fontsize = 15)

plt.xlabel('Lap',fontsize = 15,)

ax.legend()

plt.show()

timeHamilton = df_timeLaps.loc[((df_timeLaps['driverId'] == 1) & (df_timeLaps['raceId']==853))]['time'].iloc[:].values

timeShummy = df_timeLaps.loc[(df_timeLaps['driverId'] == 30) & (df_timeLaps['raceId']==853)]['time'].iloc[:].values

x = range(0,53)

f,ax = plt.subplots(figsize =(20,10))

ax.tick_params(axis="x", labelsize=15)

ax.tick_params(axis="y", labelsize=20)



ax.set_facecolor((0, 0, 0))

ax.patch.set_alpha(0.9)

line1, = ax.plot(x, timeHamilton, label='Hamilton Time', color='red',linewidth=4,)

line2, = ax.plot(x, timeShummy, label='Schumacher Time' ,linewidth=4,)

plt.title("Monza GP - Time Laps Comparison - 2011 ",fontsize=15)

plt.ylabel('Total Seconds(s)',fontsize = 15)

plt.xlabel('Lap',fontsize = 15,)

ax.legend()

plt.show()
positionsHamilton = df_timeLaps.loc[((df_timeLaps['driverId'] == 1) & (df_timeLaps['raceId']==853))]['position'].iloc[:].values

positionsShummy = df_timeLaps.loc[(df_timeLaps['driverId'] == 30) & (df_timeLaps['raceId']==853)]['position'].iloc[:].values



x = range(0,53)

f,ax = plt.subplots(figsize =(20,10))

ax.tick_params(axis="x", labelsize=15)

ax.tick_params(axis="y", labelsize=20)



ax.set_facecolor((0, 0, 0))

ax.patch.set_alpha(0.9)

line1, = ax.plot(x, positionsHamilton, label='Hamilton Position', color='red',linewidth=4,)

line2, = ax.plot(x, positionsShummy, label='Schumacher Position' ,linewidth=4,)

plt.title("Monza Gran Prix 2011- Fight for position",fontsize=15)

plt.ylabel('Position',fontsize = 15)

plt.xlabel('Lap',fontsize = 15,)

ax.legend()

plt.ylim(10, 1)

plt.show()
racesQualysHam = pd.merge(r.loc[(r['driverRef']=='hamilton') & (r['year']==2017)], df_qualy.loc[(df_qualy['q3']!= 0) & (df_qualy['driverId'] == 1)], on='raceId', how='inner')

q3s = racesQualysHam['q3'].values

raceTimes = racesQualysHam['fastestLapTime'].values

x = racesQualysHam.name.values



f,ax = plt.subplots(figsize =(20,10))

ax.tick_params(axis="x", labelsize=15)

ax.tick_params(axis="y", labelsize=15)



ax.set_facecolor('whitesmoke')

line1, = ax.plot(x, raceTimes, 'bs',c='red',label="Race Time")

line2, = ax.plot(x, q3s,'bs', c='blue', label="Qualy time")

plt.title("Race vs Qualification",fontsize=30)

plt.ylabel('Time (s)',fontsize = 15)

plt.xlabel('Código da Corrida',fontsize = 15,)

plt.xticks(rotation=45, ha='right')

ax.legend()

plt.show()

qtdMcl = len(r.loc[(r['driverRef']=='hamilton') & (r['positionOrder']==1) & (r['constructorId']==1)])

qtdMcd = len(r.loc[(r['driverRef']=='hamilton') & (r['positionOrder']==1) & (r['constructorId']==131)])

qtdQualMCL = len(df_qualy.loc[(df_qualy['driverId'] == 1) & (df_qualy['position'] == 1) & (df_qualy['constructorId'] == 1)])

qtdQualMCD = len(df_qualy.loc[(df_qualy['driverId'] == 1) & (df_qualy['position'] == 1) & (df_qualy['constructorId'] == 131)])



f,ax = plt.subplots(figsize =(20,10))

ax.tick_params(axis="x", labelsize=15)

ax.tick_params(axis="y", labelsize=30)



ax.pie([qtdMcl,qtdMcd], labels=['McClaren','Mercedes'],explode=(0, 0.1),shadow=True,colors=['r','c'],autopct='%1.1f%%')

ax.set_facecolor('whitesmoke')

plt.title("Total of Hamilton's wins per Constructors",fontsize=20)

plt.show()



f,ax = plt.subplots(figsize =(20,10))

ax.tick_params(axis="x", labelsize=15)

ax.tick_params(axis="y", labelsize=30)



ax.pie([qtdQualMCL,qtdQualMCD], labels=['McClaren','Mercedes'],explode=(0, 0.1),shadow=True,colors=['r','c'],autopct='%1.1f%%')

ax.set_facecolor('whitesmoke')

plt.title("Total of Hamilton's Poles per Constructor",fontsize=20)

plt.show()
mclarensRacers = r.loc[(r['constructorId']==1)&(r['positionOrder']==1)].groupby('driverRef')['raceId'].count().sort_values(ascending=False)[:10]

mclarensRacers = mclarensRacers.sort_values()



mercedesRacers = r.loc[(r['constructorId']==131)&(r['positionOrder']==1)].groupby('driverRef')['raceId'].count().sort_values(ascending=False)[:10]

mercedesRacers = mercedesRacers.sort_values()



f,ax = plt.subplots(figsize =(20,10))

ax.tick_params(axis="x", labelsize=15)

ax.tick_params(axis="y", labelsize=20)



ax.barh(list(mclarensRacers.index), list(mclarensRacers.values),height = 0.9, color = ['whitesmoke','crimson'])

ax.set_facecolor('k')

ax.patch.set_alpha(0.9)

plt.title("Top McLaren's Race Winners",fontsize=30)

plt.ylabel('Drivers',fontsize = 15)

plt.xlabel('Total Wins',fontsize = 15,)

plt.grid()

plt.show()



f,ax = plt.subplots(figsize =(20,10))

ax.tick_params(axis="x", labelsize=15)

ax.tick_params(axis="y", labelsize=20)



ax.barh(list(mercedesRacers.index), list(mercedesRacers.values),height = 0.9, color = ['c','whitesmoke'])

ax.set_facecolor('k')

ax.patch.set_alpha(0.9)

plt.title("Top Mercedes' Race Winners",fontsize=30)

plt.ylabel('Drivers',fontsize = 15)

plt.xlabel('Total Wins',fontsize = 15,)

plt.grid()

plt.show()