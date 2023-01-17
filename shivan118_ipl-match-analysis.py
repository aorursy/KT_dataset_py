from IPython.display import Image

Image(filename='/kaggle/input/ipl-dataset/ipl-bcc.jpg', width="800", height='50')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("whitegrid")

plt.style.use("fivethirtyeight")



%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path0 = '/kaggle/input/ipl/deliveries.csv'

path1 = '/kaggle/input/ipl/matches.csv'
score = pd.read_csv(path0)

match = pd.read_csv(path1)

match.head()

# print(match.shape)
match.columns
score.head()

# print(score.shape)
score.shape
score.columns
match.isnull().sum()
match.shape
sns.heatmap(match.isnull(), yticklabels = False)  #checking NULl Values via graph,where you can find yellow colour which means that column contains NUll values
match.drop('umpire3',axis = 1, inplace=True)
score.isnull().sum()
match['team1'].unique()

match.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',

       'Rising Pune Supergiant', 'Royal Challengers Bangalore',

       'Kolkata Knight Riders', 'Delhi Daredevils', 'Kings XI Punjab',

       'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',

       'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],['SRH','MI','GL','RPS','RCB','KKR','DD','KXIP','CSK','RR','DC','KTK','PW','RPS'],inplace = True)



score.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',

       'Rising Pune Supergiant', 'Royal Challengers Bangalore',

       'Kolkata Knight Riders', 'Delhi Daredevils', 'Kings XI Punjab',

       'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',

       'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],['SR','MI','GL','RPS','RCB','KKR','DD','KXIP','CSK','RR','DC','KTK','PW','RPS'],inplace = True)
df=match.iloc[[match['win_by_runs'].idxmax()]]

df[['season','team1','team2','winner','win_by_runs']]
df=match.iloc[[match['win_by_wickets'].idxmax()]]

df[['season','team1','team2','winner','win_by_wickets']]
# lets see how many matches are being played every season

plt.subplots(figsize=(15,5))

sns.countplot(x = 'season', data = match, palette = 'dark')

plt.show()
# By this graph you ca nsee that Mumbai Indians have won most of the matches in the IPL

plt.figure(figsize=(15,8))

sns.countplot(x = 'winner', data = match, palette = ['darkorange','#d11d9b','purple',

                                                       'tomato','gold','royalblue','red','#e04f16','yellow','gold'

                                                       ,'black','silver','b'])

plt.show()
df=match[match['toss_winner']==match['winner']]

slices=[len(df),(577-len(df))]

labels=['yes','no']

plt.pie(slices,labels=labels,startangle=90,shadow=True,explode=(0,0.05),autopct='%1.1f%%',colors=['r','g'])

fig = plt.gcf()

fig.set_size_inches(9,9)

plt.title("Toss winner is match winner")

plt.show()
plt.subplots(figsize=(15,15))

ax = match['venue'].value_counts().sort_values(ascending=True).plot.barh(width=.9,color=sns.color_palette('RdBu',40))

ax.set_xlabel('Grounds')

ax.set_ylabel('count')

plt.show()
# Top cities where the matches are held

plt.figure(figsize=(15,8))

fav_cities = match['city'].value_counts().reset_index()

fav_cities.columns = ['city','count']

sns.barplot(y = 'count',x = 'city', data = fav_cities[:10])

plt.show()
plt.figure(figsize=(15,6))

fav_ground = match['venue'].value_counts().reset_index()

fav_ground.columns = ['venue','count']

sns.barplot(x = 'count',y = 'venue', data = fav_ground[:10], palette = ['darkorange','#d11d9b','purple',

                                                       'tomato','gold','royalblue','red','#e04f16','yellow','gold'

                                                       ])

plt.show()
print('Toss Decisions in %\n',((match['toss_decision']).value_counts())/577*100)
plt.subplots(figsize=(15,8))

sns.countplot(x='season',hue='toss_decision',data=match)

plt.show()

plt.subplots(figsize=(15,9))

temp_series = match.toss_decision.value_counts()

labels = (np.array(temp_series.index))

sizes = (np.array((temp_series / temp_series.sum())*100))

colors = ['gold', 'lightskyblue']

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.title("Toss decision percentage")

plt.show()
plt.subplots(figsize=(15,8))

ax=match['toss_winner'].value_counts().plot.bar(width=0.9,color=sns.color_palette('Blues_d',20))

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))

plt.show()
plt.subplots(figsize=(15,8))

#the code used is very basic but gets the job done easily

ax = match['player_of_match'].value_counts().head(10).plot.bar(width=.8, color=sns.color_palette('inferno',10))  #counts the values corresponding 

# to each batsman and then filters out the top 10 batsman and then plots a bargraph 

ax.set_xlabel('player_of_match') 

ax.set_ylabel('count')

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.25))

plt.show()
for i in range(2008,2017):

    df=((match[match['season']==i]).iloc[-1]) 

    print(df[[1,10]].values)
player = match.player_of_match.value_counts()[:10]

labels = np.array(player.index)

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(player), width=width, color='y')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Top player of the match awardees")

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.25))

plt.show()
# player with most boundries

data = score[(score['batsman_runs'] == 4) | (score['batsman_runs'] == 6)][['batsman','batsman_runs']].groupby('batsman').count().reset_index().sort_values(ascending = False, by = 'batsman_runs')

plt.subplots(figsize=(15,9))

sns.set_style("whitegrid")

sns.despine()

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

sns.barplot(x = 'batsman_runs', y = 'batsman', data = data[:10],palette="Reds")