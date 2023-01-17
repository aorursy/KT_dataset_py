# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/data.csv')
data.shape
data.head()
data.info()
data.describe()
print(data['Nationality'].value_counts(dropna =False))
data.columns
data.drop(["Photo"],axis=1, inplace=True)

data.drop(["Flag"],axis=1, inplace=True)

data.drop(["Club Logo"],axis=1, inplace=True)
data.head()
def club(x):

    return data[data['Club'] == x][['Name','Jersey Number','Position','Overall','Nationality','Age','Wage',

                                    'Value','Contract Valid Until']]

#We created a function to bring a club data whenever we want
#for example

club('Arsenal')
xarsenal = club('Arsenal')

xarsenal.shape
# checking if the data contains any NULL value



data.isnull().sum()
# filling the missing value for the continous variables for proper data visualization



data['ShortPassing'].fillna(data['ShortPassing'].mean(), inplace = True)

data['Volleys'].fillna(data['Volleys'].mean(), inplace = True)

data['Dribbling'].fillna(data['Dribbling'].mean(), inplace = True)

data['Curve'].fillna(data['Curve'].mean(), inplace = True)

data['FKAccuracy'].fillna(data['FKAccuracy'], inplace = True)

data['LongPassing'].fillna(data['LongPassing'].mean(), inplace = True)

data['BallControl'].fillna(data['BallControl'].mean(), inplace = True)

data['HeadingAccuracy'].fillna(data['HeadingAccuracy'].mean(), inplace = True)

data['Finishing'].fillna(data['Finishing'].mean(), inplace = True)

data['Crossing'].fillna(data['Crossing'].mean(), inplace = True)

data['Weight'].fillna('200lbs', inplace = True)

data['Contract Valid Until'].fillna(2019, inplace = True)

data['Height'].fillna("5'11", inplace = True)

data['Loaned From'].fillna('None', inplace = True)

data['Joined'].fillna('Jul 1, 2018', inplace = True)

data['Jersey Number'].fillna(8, inplace = True)

data['Body Type'].fillna('Normal', inplace = True)

data['Position'].fillna('ST', inplace = True)

data['Club'].fillna('No Club', inplace = True)

data['Work Rate'].fillna('Medium/ Medium', inplace = True)

data['Skill Moves'].fillna(data['Skill Moves'].median(), inplace = True)

data['Weak Foot'].fillna(3, inplace = True)

data['Preferred Foot'].fillna('Right', inplace = True)

data['International Reputation'].fillna(1, inplace = True)

data['Wage'].fillna('€200K', inplace = True)
data.corr()
f,ax = plt.subplots(figsize=(12,12)) 

sns.heatmap(data.corr(), annot=False, linewidths=.5, fmt= '.1f',ax=ax)

# Annot=False means, we don't want to see numbers in boxes because there are too many features it can get complicated.

plt.show()
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Potential.plot(kind = 'line', color = 'b',label = 'Potential',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Overall.plot(color = 'r',label = 'Overall',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()



# x axis = shows frequency.
data1 = data.loc[:,["Potential","Overall","Age"]]

data1.plot()

plt.show()
data1.plot(subplots = True)

plt.show()
plt.scatter(data.Potential, data.Overall, alpha=0.5)

plt.show()
# To show that there are people having same age

# Histogram: number of players's age



sns.set(style = "dark", palette = "colorblind", color_codes = True)

x = data.Age

plt.figure(figsize = (15,8))

ax = sns.distplot(x, bins = 58, kde = False, color = 'g')

ax.set_xlabel(xlabel = "Player\'s age", fontsize = 16)

ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)

ax.set_title(label = 'Histogram of players age', fontsize = 20)

plt.show()
data.Acceleration.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.xlabel('Acceleration')

plt.show()



# or

# data1.plot(kind = "hist",y = "Acceleration",bins = 50,range= (0,250),normed = True)

overall85 = data['Overall']>85     # There are only 77 players with Overall higher than 85

data85=data[overall85]

data85#(dataframe)
data[np.logical_and(data['Overall']>90, data['Age']<30 )]

# There are only 4 players with Overall higher than 90 and Age lower than 30.
PositionGK= data['Club']=='Arsenal'

dataGK=data[PositionGK]

#We take the Arsenal club for bar plot
player_list = list(dataGK['Nationality'].unique())    #we take the Nationality feature's unique values

player_ratio=[]   #we create a empty list

for i in player_list:

    x = dataGK[dataGK['Nationality']==i]

    player_rate = sum(x.Overall)/len(x)

    player_ratio.append(player_rate)

#we take the mean of the player overalls by Nationality

    

data2 = pd.DataFrame({'player_list': player_list,'player_ratio':player_ratio})

new_index = (data2['player_ratio'].sort_values(ascending=False)).index.values   #"ascending=False" =downward

sorted_data = data2.reindex(new_index) 

#created a new dataFrame
# visualization (Bar Plot)

plt.figure(figsize=(15,15))

sns.barplot(x=sorted_data['player_list'], y=sorted_data['player_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('Nationality')

plt.ylabel('Arsenal Player Overalls Ratio')

plt.title('Arsenal Player Overalls Ratio According to Nationality')

plt.show()
data85.head(10)
teams85=data85.Club.value_counts()    #we count the Clubs

#print

plt.figure(figsize=(20,7))

sns.barplot(x=teams85[:10].index,y=teams85[:10].values)

plt.ylabel('Number of Players Higher Overall Than 85')

plt.xlabel('Clubs')

plt.title('Team Counts(First 10)',color = 'blue',fontsize=15)
# comparison of preferred foot over the different players



plt.rcParams['figure.figsize'] = (10, 5)

sns.countplot(data['Preferred Foot'], palette = 'dark')

plt.title('Most Preferred Foot of the Players', fontsize = 20)

plt.show()
# different positions acquired by the players 



plt.figure(figsize = (18, 8))

plt.style.use('fivethirtyeight')

ax = sns.countplot('Position', data = data, palette = 'bone')

ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)

ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)

plt.show()
datapair=data.loc[:,["Age","Potential"]] # we take Age and Potential features

#Print

sns.pairplot(datapair)
overall87 = data['Overall']>87    # There are only 9 players with Overall higher than 90

data87=data[overall87]

data87.head()
plt.figure(figsize=(25,10))

sns.swarmplot(x="Nationality", y="Age",hue="Club", data=data87, palette= "colorblind")

plt.show()
# Distribution of Wages in some Popular clubs



some_clubs = ('CD Leganés', 'Southampton', 'RC Celta','Arsenal', 'Manchestar City',

              'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid','Galatasaray SK')



data_club = data.loc[data['Club'].isin(some_clubs) & data['International Reputation']]



plt.rcParams['figure.figsize'] = (16, 8)

ax = sns.violinplot(x = 'Club', y = 'International Reputation', data = data_club, palette = 'bright')

ax.set_xlabel(xlabel = 'Names of some popular Clubs', fontsize = 11)

ax.set_ylabel(ylabel = 'Distribution of Reputation', fontsize = 11)

ax.set_title(label = 'Disstribution of International Reputation in some Popular Clubs', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()