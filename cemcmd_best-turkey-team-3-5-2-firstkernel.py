# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/fifa19/data.csv')
data.head()
data.columns
akh = data[data['Club']=='Akhisar Belediyespor']
alan = data[data['Club']=='Alanyaspor'] 
ant = data[data['Club']=='Antalyaspor'] 
baş = data[data['Club']=='Medipol Başakşehir FK'] 
erz = data[data['Club']=='BB Erzurumspor'] 
fb = data[data['Club']=='Fenerbahçe SK'] 
beş = data[data['Club']=='Beşiktaş JK'] 
bur = data[data['Club']=='Bursaspor'] 
çay = data[data['Club']=='Çaykur Rizespor'] 
gal = data[data['Club']=='Galatasaray SK'] 
göz = data[data['Club']=='Göztepe SK'] 
kas = data[data['Club']=='Kasımpaşa SK'] 
kay = data[data['Club']=='Kayserispor'] 
kon = data[data['Club']=='Atiker Konyaspor'] 
ank = data[data['Club']=='MKE Ankaragücü'] 
siv = data[data['Club']=='Sivasspor'] 
trb = data[data['Club']=='Trabzonspor'] 
mal = data[data['Club']=='Yeni Malatyaspor']

teamtr = pd.concat([akh, alan, ant, baş, erz, fb, beş, bur, çay, gal, göz,
                   kas, kay, kon, ank, siv, trb, mal])


trage = teamtr.Age

f, ax = plt.subplots(figsize = (15, 7))
ax = sns.distplot(trage, bins = 58, kde = False, color = 'g')
ax.set_xlabel(xlabel = "Player\'s age", fontsize = 16)
ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)
ax.set_title(label = 'Histogram of players age', fontsize = 20)

plt.show()
f, ax = plt.subplots(figsize = (8, 6))
ax = sns.countplot(y = 'Preferred Foot', data = teamtr, color = 'b')
plt.show()
plt.figure(figsize = (18, 8))
ax = sns.countplot('Position', data = teamtr, palette = 'bone')
ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)
plt.show()
ax = teamtr['Nationality'].value_counts().head(50).plot.bar(color = 'orange', figsize = (20, 7))
plt.title('Different Nations Participating in FIFA 2019', fontsize = 30, fontweight = 20)
plt.xlabel('Name of The Country')
plt.ylabel('count')
plt.show()
plt.figure(figsize = (30, 30))
sns.heatmap(teamtr[['Age','Overall','Potential', 'Crossing','Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression','Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'Penalties', 'GKDiving', 'GKHandling','GKKicking', 'GKPositioning', 'GKReflexes']].corr(), annot = True)
plt.show()
ww = teamtr['Wage']
ww = ww.tolist()
plt.figure(figsize = (22, 8))
res = list(map(lambda sub:int(''.join([ele for ele in sub if ele.isnumeric()])), ww)) 
ax = sns.boxplot(x = teamtr['Club'], y = res)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title(label = 'Wage Distribution of Teams', fontsize = 20)

plt.show()
ww = teamtr['Weight']
ww = ww.tolist()

plt.figure(figsize = (22, 8))
res = list(map(lambda sub:int(''.join([ele for ele in sub if ele.isnumeric()])), ww)) 
resy = np.array(res)
ax = sns.violinplot(x = teamtr['Club'], y = resy*0.454, dodge=False)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_xlabel(xlabel = 'Club', fontsize = 9)
ax.set_ylabel(ylabel = 'Weight in kg', fontsize = 9)
ax.set_title(label = 'Distribution of Weight of players from different countries', fontsize = 20)

plt.show()
eldest = teamtr.sort_values('Age', ascending = False)[['Name','Overall', 'Age', 'Club', 'Nationality']].head(15)
print(eldest)

young = teamtr.sort_values('Age', ascending = True)[['Name', 'Overall','Potential','Age', 'Club', 'Position']].head(15)
print(young)
plt.figure(figsize = (22, 8))
ax = sns.lineplot(teamtr['Age'], teamtr['Overall'])
plt.title('Age - Rating Graph', fontsize = 20)

plt.show()
def playerdata(x):
    return teamtr.loc[x,:]

x = playerdata(725)  #emre belözoğlu, id = 725.
#pd.set_option('display.max_rows', 200)
pd.DataFrame(x).head(25)

plt.show()
f, ax = plt.subplots(figsize=(10, 10))
sns.stripplot(x = 'International Reputation', y = 'Club' ,data = teamtr, jitter=10, linewidth=1)
plt.title('International Reputation Disturbitions', fontsize = 20)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x = teamtr["Potential"])
plt.title('Potential of league', fontsize = 20)

plt.show()
pottr = teamtr['Potential']
pottr = pottr.tolist()
pottr = np.array(pottr)
plt.figure(figsize = (22, 8))
#res = list(map(lambda sub:int(''.join([ele for ele in sub if ele.isnumeric()])), ww)) 
ax = sns.boxplot(x = teamtr['Club'], y = pottr)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
plt.title('Potential Distributions of Teams', fontsize = 20)

plt.show()

pos = 'GK' # GK, CB, LB, CM, RW, RM, CAM, ST ....

world = data[data['Position'] == pos]
worldpos = world.sort_values('Overall', ascending = False)[['Crossing','Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression','Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'Penalties', 'GKDiving', 'GKHandling','GKKicking', 'GKPositioning', 'GKReflexes' ]].head(7)
#best 7 player of worlds (GK). 

worldpos = worldpos.mean(axis = 0) #mean of features
indcb = worldpos.sort_values(ascending = False).head(8) #Top 8 best features of best players. 
#GKReflexes       89.857143
#GKDiving         87.571429
#GKHandling       87.000000
#GKPositioning    86.571429
#Reactions        85.285714
#GKKicking        79.000000
#Jumping          74.142857
#Strength         69.285714
indcb = indcb.index.values
indcb = list(indcb)


trr = teamtr[teamtr['Position'] == pos]  #GK player of Turkey.
rr = trr.sort_values('Overall', ascending = False)[indcb].head(10) #best 10 player of Turkey (GK).
rr2 = rr.mean(axis=1) #mean of features
seccb = rr2.sort_values(ascending = False).head(3) #sorting top 3 player
ind = seccb.index.values #id of top 5 player

teamtr.loc[ind,:][['Name', 'Overall', 'Potential','Age', 'Club', 'Position']] #To find the GK which is konwn his id.