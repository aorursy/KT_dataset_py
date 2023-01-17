import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



Countries = pd.read_csv("../input/Country_facts.csv")

NBA = pd.read_csv("../input/NBA.csv", encoding = "ISO-8859-1") 

Soccer = pd.read_csv("../input/SoccerLEagues.csv",encoding = "ISO-8859-1")
plt.plot(NBA.HomePCT + np.random.normal(0,0.01,len(NBA)),NBA.AwayPCT + np.random.normal(0,0.01,len(NBA)),'o', alpha = 0.4)

plt.plot([0,1],[0,1])

plt.xlabel('Home PCT')

plt.ylabel('Away PCT')

plt.title('Home vs Away PCT in the NBA')
Soccer['HomePCT'] = np.true_divide(Soccer['HomeWins'],Soccer['HomeWins'] + Soccer['HomeLoss'] + Soccer['HomeDraw'])

Soccer['AwayPCT'] = np.true_divide(Soccer['AwayWins'],Soccer['AwayWins'] + Soccer['AwayLoss'] + Soccer['AwayDraw'])
plt.plot(Soccer.HomePCT + np.random.normal(0,0.01,len(Soccer)),Soccer.AwayPCT + np.random.normal(0,0.01,len(Soccer)),'o', alpha = 0.1)

plt.plot([0,1],[0,1])

plt.xlabel('Home PCT')

plt.ylabel('Away PCT')

plt.title('Home vs Away PCT in Soccer Leagues')

plt.xlim([0,1])

plt.ylim([0,1])
NBA['Home_Factor'] = np.true_divide((NBA.HomePCT - NBA.AwayPCT),(NBA.HomePCT + NBA.AwayPCT))

Soccer['Home_Factor'] = np.true_divide((Soccer.HomePCT - Soccer.AwayPCT),(Soccer.HomePCT + Soccer.AwayPCT))
import seaborn as sns

sns.kdeplot(Soccer.Home_Factor, shade = True)

sns.kdeplot(NBA.Home_Factor, shade = True)

plt.xlim([-0.8,0.8])

plt.legend(['Soccer Leagues','NBA'])

plt.plot([0,0],[0,3])

plt.xlabel('Home Factor')
plt.style.use('fivethirtyeight')

plt.plot(Soccer.HomeRatio + np.random.normal(0,0.5,len(Soccer)),Soccer.AwayGoalsDiff+ np.random.normal(0,0.5,len(Soccer)),'o', alpha = 0.2)

plt.plot([-100,80],[-100,80])

plt.xlim([-40,40])

plt.ylim([-40,40])

plt.xlabel('Home Goals Difference')

plt.ylabel('Away Goals Difference')
plt.plot(NBA.Year + np.random.normal(0,0.5,len(NBA)),NBA.Home_Factor,'o', alpha = 0.8)

x = NBA.Year

y = NBA.Home_Factor

fn = np.polyfit(x,y,1)

fit_fn = np.poly1d(fn) 

plt.plot(x,fit_fn(x))

plt.xlabel('Year')

plt.ylim([-1,1])

plt.xlim([1950,2010])


plt.figure(figsize = (10,10))

plt.subplot(2,1,1)

plt.title('Home Advantage vs FIFA Rank')

plt.plot(Countries.FIFA_Rank, Countries.Home_Away_Contrast,'o')

x = Countries.FIFA_Rank

y = Countries.Home_Away_Contrast

fn = np.polyfit(x,y,1)

fit_fn = np.poly1d(fn) 

plt.plot(x,fit_fn(x))

plt.xlabel('FIFA Rank')



plt.subplot(2,1,2)

plt.title('Home Advantage vs Crowd Attendance')

plt.plot(Countries.Attendance, Countries.Home_Away_Contrast,'o')

x = Countries.Attendance[Countries.Attendance>0]

y = Countries.Home_Away_Contrast[Countries.Attendance>0]

fn = np.polyfit(x,y,1)

fit_fn = np.poly1d(fn) 

plt.plot(x,fit_fn(x))

plt.xlabel('Average Attandance per game in the leagure')
plt.figure(figsize = (10,10))

plt.subplot(2,2,1)

x = Countries.Phones[Countries.Phones>0]

y = Countries.Home_Away_Contrast[Countries.Phones>0]

fn = np.polyfit(x,y,1)

fit_fn = np.poly1d(fn) 

plt.plot(x,y,'o')

plt.plot(x,fit_fn(x))

plt.title('# of Phones per person')



plt.subplot(2,2,2)

x = Countries.GDP[Countries.GDP>0]

y = Countries.Home_Away_Contrast[Countries.GDP>0]

fn = np.polyfit(x,y,1)

fit_fn = np.poly1d(fn) 

plt.plot(x,y,'o')

plt.plot(x,fit_fn(x))

plt.title('GDP')



plt.subplot(2,2,3)

x = Countries.Agriculture[Countries.Agriculture>0]

y = Countries.Home_Away_Contrast[Countries.Agriculture>0]

fn = np.polyfit(x,y,1)

fit_fn = np.poly1d(fn) 

plt.plot(x,y,'o')

plt.plot(x,fit_fn(x))

plt.title('Agriculture')



plt.subplot(2,2,4)

x = Countries.Literacy[Countries.Literacy>0]

y = Countries.Home_Away_Contrast[Countries.Literacy>0]

fn = np.polyfit(x,y,1)

fit_fn = np.poly1d(fn) 

plt.plot(x,y,'o')

plt.plot(x,fit_fn(x))

plt.title('Literacy')