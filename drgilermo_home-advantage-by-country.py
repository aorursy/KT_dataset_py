import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sqlite3

import seaborn as sns

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

conn = sqlite3.connect('../input/database.sqlite')

cur = conn.cursor()

countries = cur.execute('select id,name from Country').fetchall()
plt.figure()

plt.hold(True)

county_names = []

Home_factor = np.zeros(len(countries))

home_dict = dict()

away_dict = dict()



for j,row in enumerate(countries):

    goals_home_list = []

    goals_away_list = []

    goals_home = cur.execute('select home_team_goal from Match where country_id =' + str(row[0])).fetchall()

    goals_away = cur.execute('select away_team_goal from Match where country_id =' + str(row[0])).fetchall()

    for i,game in enumerate(goals_home):

        goals_home_list.append(goals_home[:][i][0])

        goals_away_list.append(goals_away[:][i][0])

        

    Diff =  np.array(goals_home_list) - np.array(goals_away_list)

    Home_pct = np.true_divide(len(Diff[Diff>0]),len(Diff))

    Away_pct = np.true_divide(len(Diff[Diff<0]),len(Diff))

    Draw_pct = np.true_divide(len(Diff[Diff == 0]),len(Diff))

    

    away_expect = Away_pct*3 + Draw_pct

    home_expect = Home_pct*3 + Draw_pct

    

    home_dict[row[1]] = home_expect

    away_dict[row[1]] = away_expect



    if (row[1] == 'Spain') | (row[1] == 'Scotland'):

       sns.distplot(Diff,hist = False,kde_kws={"shade": True})



    print(row[1], '   Home Win:', round(Home_pct,2), '   Draw:', round(Draw_pct,2),'   Away Win:', round(Away_pct,2), '   Average Difference:',round(np.mean(Diff),2))



plt.legend(['Scotland', 'Spain'], fontsize = 20)

plt.xlim([-10,10])

plt.title('Goals Difference between Home and Away teams distribution')

plt.show()
l = sorted(home_dict, key= home_dict.get)

y1 = []

y2 = []

for country in l:

    y1.append(home_dict[country])

    y2.append(away_dict[country])





plt.style.use('fivethirtyeight')

plt.bar(np.linspace(1,20,len(countries)),y1, width = 0.5, color = 'b')

plt.bar(np.linspace(1,20,len(countries)) + 0.5*np.ones(len(countries)),y2, width = 0.5, color = 'red')

plt.xticks(np.linspace(1,20,len(countries))+0.25*np.ones(len(countries)) , l, size='small',rotation= 45)

plt.ylabel('Expected Points')

plt.title('Home Advantage')

plt.legend(['Home Expected Points','Away Expected Points'], loc = 2)