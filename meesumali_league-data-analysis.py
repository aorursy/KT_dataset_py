# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')



data.head()
data.info()
#exploring impact of kills on game wins

#create histogram to see spread of game kills





#combine blue and red kills into single series for larger data sample



kills = data['blueKills'].append(data['redKills'],ignore_index = True)



#kills.hist(bins=23)

plt.figure(figsize=(9, 6))

plt.rcParams.update({'font.size': 12})

# Add title and axis names

plt.title('kills by 10 min')

plt.xlabel('Number of kills')

plt.ylabel('Number of games')

plt.xticks(np.arange(0,23,2))



plt.hist(kills , bins = 23)



plt.show()
#print(kills.describe())



plt.figure(figsize=(9, 6))

plt.title('kills by 10 min')

plt.xlabel('Number of kills')

plt.xticks(np.arange(0,23,2))

plt.ylim(-0.001 , 0.14)

kills.plot.kde(bw_method=0.2,ind = np.arange(-1,23,0.1))





plt.show()
plt.figure(figsize=(4, 8))

plt.boxplot(kills , autorange = True)

plt.title('kills by 10 min')

plt.ylabel('Number of kills')

plt.yticks(np.arange(0,24,2))



plt.show()
kills.describe()
print('Statistical analysis of kills :')

stats = kills.describe()

print()

print(f'Total number of kills in all games : {stats["count"]}')

print(f'Average number of kills per game : {stats["mean"]}')

print(f'Standard Deviation of kills per game : {stats["std"]}')

print()

print('Spread of the data given below :')

print()

print(f'{stats["25%"]} or less kills occured in 25 percent of games')

print(f'{stats["50%"]} or less kills occured in 50 percent of games')

print(f'{stats["75%"]} or less kills occured in 75 percent of games')

print(f'Maximum number of kill in a game : {stats["max"]}')

print(f'Minimum number of kill in a game : {stats["min"]}')
print('Now I will test the hypothesis that if a team has higher then normal number of kills they are more likely to win.')

print()

print('To test this I will split back into analying the data at team level with the assumption that the stats found in aggrigate above are roughtly the same at the indivudal team level.')

print("I will define a higher than average number of kills per game as 6 or greater")

print()

print("therefore I will split the data off into those higher then normal kill games")
#setting plot variables how win percentage changes as number of kills changes

red_win = []

blue_win = []


kig = 6



#BLUE TEAM BLOW

h_norm_blue = data[data['blueKills'] >= kig]



count = h_norm_blue['blueWins'].value_counts()

blue_win.append(count[1]/count.sum())

print(f"Games in which blue team had {kig} or more kills they ended up winning {count[1]/count.sum()} percent of the time")



#creating pie plot

plt.figure(figsize=(6, 6))

plt.title(f'Blue team has {kig} or more kills')

plt.pie(count , labels = ['Win','Loss'] , autopct='%1.1f%%')



plt.show()



#RED TEAM BELOW

h_norm_blue = data[data['redKills'] >= kig]



count = h_norm_blue['blueWins'].value_counts()

red_win.append(count[0]/count.sum())

print(f"Games in which red team had {kig} or more kills they ended up winning {count[0]/count.sum()} percent of the time")



#creating pie plot

plt.figure(figsize=(6, 6))

plt.title(f'Red team has {kig} or more kills')

plt.pie(count , labels = ['Win','Loss'] , autopct='%1.1f%%')



plt.show()
print("Given the above findings we can see that when we have a higher then normal number of kills this increase our chances of winning the game regardless of team")

print()

print("Just to drive this point more home I will move one and two standard deviations away from the mean to see how such high kill values impact the game")

print()

print("the mean of the kills is 6 and the standard devation is roughly 3 therefore 2 sdve above the mean would be 12 kills in a game")
kig = 9



#BLUE TEAM BLOW

h_norm_blue = data[data['blueKills'] >= kig]



count = h_norm_blue['blueWins'].value_counts()

blue_win.append(count[1]/count.sum())

print(f"Games in which blue team had {kig} or more kills they ended up winning {count[1]/count.sum()} percent of the time")



#creating pie plot

plt.figure(figsize=(6, 6))

plt.title(f'Blue team has {kig} or more kills')

plt.pie(count , labels = ['Win','Loss'] , autopct='%1.1f%%')



plt.show()



#RED TEAM BELOW

h_norm_blue = data[data['redKills'] >= kig]



count = h_norm_blue['blueWins'].value_counts()

red_win.append(count[0]/count.sum())

print(f"Games in which red team had {kig} or more kills they ended up winning {count[0]/count.sum()} percent of the time")



#creating pie plot

plt.figure(figsize=(6, 6))

plt.title(f'Red team has {kig} or more kills')

plt.pie(count , labels = ['Win','Loss'] , autopct='%1.1f%%')



plt.show()
kig = 12



#BLUE TEAM BLOW

h_norm_blue = data[data['blueKills'] >= kig]



count = h_norm_blue['blueWins'].value_counts()

blue_win.append(count[1]/count.sum())

print(f"Games in which blue team had {kig} or more kills they ended up winning {count[1]/count.sum()} percent of the time")



#creating pie plot

plt.figure(figsize=(6, 6))

plt.title(f'Blue team has {kig} or more kills')

plt.pie(count , labels = ['Win','Loss'] , autopct='%1.1f%%')



plt.show()



#RED TEAM BELOW

h_norm_blue = data[data['redKills'] >= kig]



count = h_norm_blue['blueWins'].value_counts()

red_win.append(count[0]/count.sum())

print(f"Games in which red team had {kig} or more kills they ended up winning {count[0]/count.sum()} percent of the time")



#creating pie plot

plt.figure(figsize=(6, 6))

plt.title(f'Red team has {kig} or more kills')

plt.pie(count , labels = ['Win','Loss'] , autopct='%1.1f%%')



plt.show()
print("As we can see from the above charts of when the number of kills per game are 9 or 12 the likelyhood of winning the game also increases respectively")

print()

print("Therefore we can pretty confidently say that there is a strong relationship between the number of kills a team has and there likelyhood of winning the game")

print()

print("this is also demonstrated by bar plot below")
# Fake dataset

height = red_win

bars = ('6', '9', '12')

y_pos = np.arange(len(bars))

 

# Create bars and choose color

plt.bar(y_pos, height, color = 'r')

 

# Add title and axis names

plt.title('Red team kills vs % games won')

plt.xlabel('kills per game')

plt.ylabel('% of games won')

plt.yticks(np.arange(0,1,0.15))

# Limits for the Y axis

plt.ylim(0,1)

 

# Create names

plt.xticks(y_pos, bars)

 

# Show graphic

plt.show()



# Fake dataset

height = blue_win

bars = ('6', '9', '12')

y_pos = np.arange(len(bars))

 

# Create bars and choose color

plt.bar(y_pos, height, color = 'b')

 

# Add title and axis names

plt.title('blue team kills vs % games won')

plt.xlabel('kills per game')

plt.ylabel('% of games won')

plt.yticks(np.arange(0,1,0.15))

# Limits for the Y axis

plt.ylim(0,1)

 

# Create names

plt.xticks(y_pos, bars)

 

# Show graphic

plt.show()
