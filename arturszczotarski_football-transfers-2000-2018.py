import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/top-250-football-transfers-from-2000-to-2018/top250-00-19.csv")

data['Transfer_value'] = data['Transfer_fee'].apply(lambda x: "€{:.2f}M".format((x/1000000)))

data.head()
laliga =data.loc[data['League_to']=="LaLiga"]

spain_market_value = laliga["Transfer_fee"].sum()

spain_market_format = "€{:.3f}M".format(spain_market_value/1000000)

print(f'LaLiga: {spain_market_format}')



serie_a =data.loc[data['League_to']=="Serie A"]

italy_market_value = serie_a.Transfer_fee.sum()

italy_market_format = "€{:.3f}M".format(italy_market_value/1000000)

print(f'Serie A:  {italy_market_format}')



Premier =data.loc[data['League_to']=="Premier League"]

england_market_value = Premier.Transfer_fee.sum()

england_market_format = "€{:.3f}M".format(england_market_value/1000000)

print(f'Premier League: {england_market_format}')



Ligue1 =data.loc[data['League_to']=="Ligue 1"]

france_market_value = Ligue1.Transfer_fee.sum()

france_market_format = "€{:.3f}M".format(france_market_value/1000000)

print(f'Ligue 1: {france_market_format}')
leagues = ['Laliga','Serie A','Premier League', 'Ligue1']

money_sum = [spain_market_value,italy_market_value,england_market_value,france_market_value]

money_sum_milions = []



for x in money_sum:

    money_sum_milions.append(x/1000000)
fig = plt.figure(figsize=[15,7])

ax = fig.add_subplot(111)

ax.set_title("Top 4 football leagues transfer spending 2000 - 2018",fontsize = 20, color = "blue",y =1.05)

ax.set_xlabel("League",fontsize = 18, color = "blue")

ax.set_ylabel("Transfer expenses [M€]",fontsize = 18,color = "blue")

plt.xticks(fontsize = 18)

plt.yticks(fontsize = 18)

ax.ticklabel_format(useOffset=False, style='plain')

ax.bar(leagues,money_sum_milions,

       color = "teal")
england_by_year =[]

spain_by_year =[]

italy_by_year =[]

france_by_year = []

seasons = data["Season"].unique()

seasons = np.delete(seasons,18)
def year():

    

    for season in seasons:

        england =data.loc[(data['League_to']== "Premier League") & (data['Season'] == season)]

        england_market = england.Transfer_fee.sum()

        england_by_year.append(england_market/1000000)

    

    for season in seasons:

        spain =data.loc[(data['League_to']== "LaLiga") & (data['Season'] == season)]

        spain_market = spain.Transfer_fee.sum()

        spain_by_year.append(spain_market/1000000)

        

    for season in seasons:

        italy =data.loc[(data['League_to']== "Seria A") & (data['Season'] == season)]

        italy_market = italy.Transfer_fee.sum()

        italy_by_year.append(italy_market/1000000)

        

    for season in seasons:

        france =data.loc[(data['League_to']== "Ligue 1") & (data['Season'] == season)]

        france_market = france.Transfer_fee.sum()

        france_by_year.append(france_market/1000000)

        

        

year()
fig1 = plt.figure(figsize = [20,8])

ax = fig1.add_subplot(111)

ax.set_title("Total Premier League spending across transfer windows [€m]", fontsize = 20,color = "blue")

ax.set_xlabel("Season",fontsize =20,color = "blue")

ax.set_ylabel("Transfer expenses [M€]",fontsize = 20,color = "blue")

ax.ticklabel_format(useOffset=False, style='plain')

ax.bar(seasons,england_by_year,

       width = 0.8,color = "teal")

plt.xticks(fontsize = 15,  rotation = 45)

plt.yticks(fontsize = 15)



i = 0 

for x in england_by_year:

    ax.text(-0.42+i,x,"{:.0f}M".format(x),fontsize = 15,color = "black",weight = 'bold')

    i+=0.995
fig1 = plt.figure(figsize = [20,8])

ax = fig1.add_subplot(111)

ax.set_title("Total LaLiga spending across transfer windows", fontsize = 20,color = "blue")

ax.set_xlabel("Season",fontsize =20,color = "blue")

ax.set_ylabel("Transfer expenses [M€]",fontsize = 20,color = "blue")

ax.ticklabel_format(useOffset=False, style='plain')

ax.bar(seasons,spain_by_year,width = 0.8,color = "teal")

plt.xticks(fontsize = 15,  rotation = 45)

plt.yticks(fontsize = 15)

i = 0

for x in spain_by_year:

    ax.text(-0.42+i,x,"{:.0f}M".format(x),fontsize = 15,color = "black",weight = 'bold')

    i+=0.999
fig1 = plt.figure(figsize = [20,8])

ax = fig1.add_subplot(111)

ax.set_title("Total Ligue 1 spending across transfer windows ", fontsize = 20,color = "blue")

ax.set_xlabel("Season",fontsize =20,color = "blue")

ax.set_ylabel("Transfer expenses [M€]",fontsize = 20,color = "blue")

ax.ticklabel_format(useOffset=False, style='plain')

ax.bar(seasons,france_by_year,width = 0.8,color = "teal")

plt.xticks(fontsize = 15,  rotation = 45)

plt.yticks(fontsize = 15)

i = 0 

for x in france_by_year:

    ax.text(-0.42+i,x,"{:.0f}M".format(x),fontsize = 15,color = "black",weight = 'bold')

    i+=0.999
clubs_Sum = data.groupby('Team_to')['Transfer_fee'].sum()/1000000

top15 = clubs_Sum.sort_values(ascending = False).head(15)

clubs = top15.index.tolist()

values = top15.values.tolist()
fig5 = plt.figure(figsize = [20,6])

ax = fig5.add_subplot(111)

ax.barh(y = clubs,width = values,height = 0.9,color = np.random.rand(2,3))

ax.set_xlabel("Transfer expenses [M€]",fontsize = 20,color = "blue")

plt.xticks(fontsize = 15)

ax.set_ylabel("Clubs",fontsize = 20,color ="blue")

plt.yticks(fontsize = 15)

ax.set_title("Transfer Spending combine 2000 - 2019", fontsize = 20,color = "blue")