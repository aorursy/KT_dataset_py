import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
players = pd.read_csv('../input/Players.csv')

seasons = pd.read_csv('../input/Seasons_Stats.csv')

seasons = seasons.drop(seasons.index[len(seasons) - 1])
seasons['ppg'] = seasons.PTS/seasons.G
plt.figure(figsize = (10,10))

plt.plot(seasons.Year,seasons.ppg,'o', alpha = 0.1)

plt.plot(seasons.Year[seasons.Player == 'Kobe Bryant'],seasons.ppg[seasons.Player == 'Kobe Bryant'], color = 'y')

plt.plot(seasons.Year[seasons.Player == 'Michael Jordan*'],seasons.ppg[seasons.Player == 'Michael Jordan*'], color = 'r')

plt.plot(seasons.Year[seasons.Player == 'LeBron James'],seasons.ppg[seasons.Player == 'LeBron James'])

plt.plot(seasons.Year[seasons.Player == 'Magic Johnson*'],seasons.ppg[seasons.Player == 'Magic Johnson*'])

plt.plot(seasons.Year[seasons.Player == 'Larry Bird*'],seasons.ppg[seasons.Player == 'Larry Bird*'])

plt.plot(seasons.Year[seasons.Player == 'Kareem Abdul-Jabbar*'],seasons.ppg[seasons.Player == 'Kareem Abdul-Jabbar*'])

plt.plot(seasons.Year[seasons.Player == 'Wilt Chamberlain*'],seasons.ppg[seasons.Player == 'Wilt Chamberlain*'])

plt.plot(seasons.Year[seasons.Player == 'Bill Russell*'],seasons.ppg[seasons.Player == 'Bill Russell*'], color = 'g')

plt.plot(seasons.Year[seasons.Player == 'Kevin Durant'],seasons.ppg[seasons.Player == 'Kevin Durant'])

plt.legend(['All','Kobe Bryant','Michael Jordan','LeBron James','Magic Johnson','Larry Bird','Kareem Abdul Jabbar','Wilt Chamberline','Bill Russel','Kevin Durant'], fontsize = 10)

plt.xlabel('Year')

plt.ylabel('Points Per Game')

plt.title('Points Leaders')
seasons['apg'] = seasons.AST/seasons.G



plt.figure(figsize = (10,10))

plt.plot(seasons.Year,seasons.apg,'o', alpha = 0.1)

plt.plot(seasons.Year[seasons.Player == 'Magic Johnson*'],seasons.apg[seasons.Player == 'Magic Johnson*'])

plt.plot(seasons.Year[seasons.Player == 'John Stockton*'],seasons.apg[seasons.Player == 'John Stockton*'])

plt.plot(seasons.Year[seasons.Player == 'Chris Paul'],seasons.apg[seasons.Player == 'Chris Paul'])

plt.plot(seasons.Year[seasons.Player == 'Oscar Robertson*'],seasons.apg[seasons.Player == 'Oscar Robertson*'])

plt.plot(seasons.Year[seasons.Player == 'Steve Nash'],seasons.apg[seasons.Player == 'Steve Nash'])

plt.plot(seasons.Year[seasons.Player == 'Jason Kidd'],seasons.apg[seasons.Player == 'Jason Kidd'])

plt.legend(['All','Magic Johnson','John Stockton','Chris Paul','Oscar Robertson','Steve Nash','Jason Kidd'])

plt.xlabel('Year')

plt.ylabel('Assists Per Game')

plt.title('Assists Leaders')
seasons['rbg'] = seasons.TRB/seasons.G

plt.figure(figsize = (10,10))

plt.plot(seasons.Year,seasons.rbg,'o', alpha = 0.1)

plt.plot(seasons.Year[seasons.Player == 'Dennis Rodman*'],seasons.rbg[seasons.Player == 'Dennis Rodman*'])

plt.plot(seasons.Year[seasons.Player == 'Wilt Chamberlain*'],seasons.rbg[seasons.Player == 'Wilt Chamberlain*'])

plt.plot(seasons.Year[seasons.Player == 'Kareem Abdul-Jabbar*'],seasons.rbg[seasons.Player == 'Kareem Abdul-Jabbar*'])

plt.plot(seasons.Year[seasons.Player == 'Bill Russell*'],seasons.rbg[seasons.Player == 'Bill Russell*'])

plt.plot(seasons.Year[seasons.Player == 'Dwight Howard'],seasons.rbg[seasons.Player == 'Dwight Howard'])

plt.plot(seasons.Year[seasons.Player == 'Tim Duncan'],seasons.rbg[seasons.Player == 'Tim Duncan'])

plt.plot(seasons.Year[seasons.Player == 'Hakeem Olajuwon*'],seasons.rbg[seasons.Player == 'Hakeem Olajuwon*'],'r')



plt.legend(['All','Dennis Rodman','Wilt Chamberlain','Kareem Abdul-Jabbar','Bill Russell','Dwight Howard','Tim Duncan','Hakeem Olajuwon'])

plt.xlabel('Year')

plt.ylabel('Rebouns Per Game')

plt.title('Rebounds Leaders')
vorp = pd.DataFrame()

vorp['player'] = seasons.groupby('Player').mean()['VORP'].index

vorp['vorp'] = seasons.groupby('Player').mean()['VORP'].values

vorp.sort('vorp', ascending = False).head(10)
ws = pd.DataFrame()

ws['player'] = seasons.groupby('Player').mean()['WS'].index

ws['ws'] = seasons.groupby('Player').mean()['WS'].values

ws.sort('ws', ascending = False).head(10)