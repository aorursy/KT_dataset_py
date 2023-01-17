import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date

#style for charts
plt.style.use('ggplot')

#draw charts right away
%matplotlib inline
#open csv file inside program to work with
df = pd.read_csv('/kaggle/input/nba2k20-player-dataset/nba2k20-full.csv')
#show file
df
#show information about file
df.info()
print('There are ' + str(df.shape[0]) +' raws and ' + str(df.shape[1]) + ' columns.')
print('Each row has information like: ')
print(df.columns.values)
print('The best rating player is ' + df[df.rating == df.rating.max()].full_name.values[0] + ' and ', end='')
print(df[df.rating == df.rating.max()].full_name.values[1], end=', ')
print('they have ' + str(df.rating.max()) + ' points.') 

print('The worst rating player is ' + df[df.rating == df.rating.min()].full_name.values[0], end=', ')
print('he has ' + str(df.rating.min()) + ' points.')
print('The highest player is ' + df[df.height == df.height.max()].full_name.values[0] + ', he is ', end='')
print(str(df.height.max()) + ' metres.')
print('The lowest player is ' + df[df.height == df.height.min()].full_name.values[0] + ', he is ', end='')
print(str(df.height.min()) + ' metres.')

#Split column 'height' to height in metres and ft
height = df['height'].str.split('/', expand=True)

#make figure with size 10x5
fig = plt.figure(figsize = (10,5))

#show histogram about height of all players in metres
plt.hist(np.sort(height[1]), color='blue')
plt.xlabel('Height in metres')
plt.ylabel('Amount of players')
plt.title('Height of players')
plt.show()
print('The heaviest player is ' + df[df.weight == df.weight.max()].full_name.values[0] + ', he is ', end ='')
print(str(df.weight.max()))
print('The easiest player is ' + df[df.weight == df.weight.min()].full_name.values[0] + ', he is ', end ='')
print(str(df.weight.min()))

#Split column 'weight' to weight in kg and lbs
weight = df['weight'].str.split('/', expand=True)
weight[1] = weight[1].str.replace(' kg', '')

#make figure with size 15x5
fig = plt.figure(figsize = (15,5))
plt.xticks(np.arange(0, 100, 5.0))

#show histogram about weight of all players in kg
plt.hist(np.sort(weight[1]), color='red')
plt.xlabel('Weight in kg')
plt.ylabel('Amount of players')
plt.title('Weight of players')
plt.show()
print('The most popular jersey is ' + df.jersey.value_counts().index[0] + ', it counts ', end='')
print(str(df.jersey.value_counts().max()) + ' players.')
print('The most rare jersey is ' + df.jersey.value_counts().index[-1] + ', it counts ', end='')
print(str(df.jersey.value_counts().min()) + ' player.')

fig = plt.figure(figsize=(10,5))

#show histogram about popularity of jersey
plt.bar(df.jersey.value_counts().index, df.jersey.value_counts(), color='orange')
plt.xticks(rotation=90)
plt.xlabel('Jersey')
plt.ylabel('Player count')
plt.title('Jersey Based on Player Count')
plt.show()



print('The biggest team by amount of players is ' + df.team.value_counts().index[0] + ', which counts', end=' ')
print(str(df.team.value_counts()[0]) + ' players.')
print('The smallest team by amount of players is ' + df.team.value_counts().index[-1] + ', which counts', end=' ')
print(str(df.team.value_counts()[-1]) + ' players.')

fig = plt.figure(figsize=(10,5))

#show histogram about amount of players in team
plt.bar(df.team.value_counts().index, df.team.value_counts(), color='green')
plt.xticks(rotation=90)
plt.xlabel('Team')
plt.ylabel('Player count')
plt.title('Teams Based on Player Count')
plt.show()

print('The most paid player is ' + df[df.salary == df.salary.max()].full_name.values[0], end='') 
print(', who earn ' + df.salary.max())
print('The lowest paid player is ' + df[df.salary == df.salary.min()].full_name.values[-1], end= '')
print(', who earn ' + df.salary.min())

#make 3 lists which contains players and their salary
players_salary1 = []
players_salary5 = []
players_salary10 = []

#loop which distribute players by salary, less than 1mln, between 1mln and 5 mln, 5 mln and more
for i in range(df.salary.size):
    if(int(df.salary.values[i].replace('$', '')) < 1000000):
        players_salary1.append(df.salary.values[i])
    elif(1000000 <= int(df.salary.values[i].replace('$', '')) < 5000000):
        players_salary5.append(df.salary.values[i])
    else:
        players_salary10.append(df.salary.values[i])  

players = [len(players_salary1), len(players_salary5), len(players_salary10)]
        
fig = plt.figure(figsize=(10,5))

#Show diagram about salary of players
plt.pie(players, labels=['less than 1mln', 'less than 5mln', '10mln and above'])
plt.title('Pie chart of salary players')
plt.show()
fig = plt.figure(figsize=(10,5))

#Histogram about players country
plt.bar(df.country.value_counts().index, df.country.value_counts())
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('Count of players')
plt.title('Players Country')
plt.show()
#function which calculate and return age of player
def get_age(b_day):
    today = date.today()
    age = today.year - b_day.year - ((today.month, today.day) < (b_day.month, b_day.day))
    return age

df['b_day'] = pd.to_datetime(df['b_day']) 

#make list of age each player
age = df['b_day'].apply(lambda x : get_age(x))
print('The oldest player is ' + str(df[age == age.max()].full_name.values[0]) + ', he is ', end='')
print(str(age.max()) + ' years old.')
print('The youngest player is ' + str(df[age == age.min()].full_name.values[0]) + ', he is ', end='')
print(str(age.min()) + ' years old.')

fig = plt.figure(figsize=(10,5))
#Histogram about player age
plt.hist(age, color='darkred')
plt.xlabel('Age')
plt.ylabel('Amount of Players')
plt.title('Players Age')
plt.show()
print('The biggest amount of players are graduated from ' + df.college.value_counts().index[0], end='')
print(', there are ' + str(df.college.value_counts()[0]) + ' players.')
fig = plt.figure(figsize=(20,7))

#Histogram about players college
plt.bar(df.college.value_counts().index, df.college.value_counts())
plt.xticks(rotation=90)
plt.xlabel('College')
plt.ylabel('Count of players')
plt.title('Players College')
plt.show()
fig = plt.figure(figsize=(10,5))

#Histogram about players draft year
plt.bar(df.draft_year.value_counts().index, df.draft_year.value_counts(), color = 'lightblue')
plt.xticks(rotation=90)
plt.xlabel('Year')
plt.ylabel('Count of players')
plt.title('Count of Players based on Draft year')
plt.show()