# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Read data from csv
dataFrame = pd.read_csv("../input/NBA_player_of_the_week.csv")
#Information about data
dataFrame.info()
#review the first 10 rows of data
dataFrame.head(10)
dataFrame.corr()
#correlation map
f,ax = plt.subplots(figsize = (13,13))
sns.heatmap(dataFrame.corr(),annot=True,linewidths=.5,linecolor='black',fmt= '.2f',ax=ax)
plt.show()


#data coloumns
dataFrame.columns
dataFrame.Age.plot(kind = 'line',color = 'red',figsize=(13,13),grid=True,label="Age",legend=True)
plt.ylabel("Player's Age ")

plt.show()
#First method
#x = Age y = Seasons of league
dataFrame.plot(kind='scatter' , x = 'Age',y = 'Seasons in league',figsize=(12,12),title="Scatter Plot",fontsize=16,color = 'orange')
plt.show()
#Another Method
plt.scatter(x=dataFrame.Age , y = dataFrame['Seasons in league'],color = 'black',linewidths=3,edgecolors='red')
plt.show()
#First method
dataFrame['Draft Year'].plot(kind='hist',color ='#000080',figsize = (12,12))
plt.xlabel("Draft Year")
plt.show()

#Another Method
plt.hist(dataFrame["Draft Year"],color='#008080',label='Draft Year',bins = 20)
plt.show()
players = dataFrame.Player.unique()
x = list()
y = list()
for each in players:
    filters_data = dataFrame[dataFrame.Player == each]
    if(filters_data.Player.count() >20):
        x.append(each)
        y.append(filters_data.Player.count())

#Figure size 12,9        
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
plt.title("Week's Player Count")
plt.xlabel("Players")
plt.bar(x,y)
plt.show()
#Another bar plot
teams = dataFrame.Team.unique()
x = list()
y = list()
for each in teams:
    filters_data = dataFrame[dataFrame.Team == each]
    if(filters_data.Team.count() > 45):
        x.append(each)
        y.append(filters_data.Team.count())

plt.bar(x,y,color = 'red')
plt.xlabel("Teams")
plt.ylabel("Player Of The Week Count")

#figure size
plt.rcParams["figure.figsize"] = (11,10)
plt.show()


fname = ["Onur","Güray","Ömer","Burak","Kadir","Alperen","Ömer","Doğukan","Metehan"]
lname = ["Öztunç","Özgödek","Turan","Can","Köse","Özdoğan","Evin","Öz","Batmaz"]
age = np.random.randint(19,23,(9,))
#or
#age = np.linspace(dtype=int,start=19,stop=22,num=9)

my_dict = dict([("Name",fname),("Last Name",lname),("Age",age)])
my_data = pd.DataFrame(my_dict)
my_data
my_data["Sex"] = ['M' for i in range(9)]
my_data
#loc function
my_data.loc[:,'Name':'Last Name'] #including borders
#iloc
my_data.iloc[:,::-1] #use index
#One method
means_age = my_data.Age.mean()
means_age
#Another method -> use numpy
means_age = np.mean(my_data.Age)
round(means_age,2)
my_data.columns = [each.upper().replace(" ","_") for each in my_data.columns]
my_data.columns
my_data["BEGINS_WITH_'OZ'"] = [True if each.lower().startswith("öz") else False for each in my_data.LAST_NAME]
my_data
#drop
my_data.drop(my_data.index[:3],axis = 0) 
my_data.drop(columns='AGE',axis = 1)
my_data.drop(["SEX"],axis = 1)
#Vertical
new_data = pd.concat([my_data.SEX,my_data.NAME],axis = 1)
new_data
#Horizontal
new_data = pd.concat((my_data.tail(3),my_data.head(5)),axis = 0,sort =True,ignore_index= True)
new_data
#Sort by name
my_data.sort_values(ascending=True,by = 'NAME',inplace=True)
my_data
#Find the youngest player of the week
min_age = np.min(dataFrame.Age) #find minimum age
young_player = dataFrame[dataFrame.Age == min_age] #find the youngest player 
player_name = young_player.Player.unique() #the youngest player's name
player_name[0] 
#Find the oldest player of the week
max_age = np.max(dataFrame.Age) #find maximum age
oldest_player = dataFrame[dataFrame.Age == max_age] # find the oldest player
player_name = oldest_player.Player.unique() #the oldest player's name
player_name[0]

seasons = dataFrame["Season short"].unique()
season_list = list() #for new data frame
player_list = list() #for new data frame
team_list = list() #for new data frame
count_list = list() #for new data frame
for years in seasons:
    filter_data = dataFrame[dataFrame["Season short"] == years] #filtering by years
    players = filter_data.Player.unique() #players in that season
    season_list.append(years)
    maximum = 0
    for i in players:
        filter_data_2 = dataFrame[(dataFrame["Season short"] == years ) & (dataFrame["Player"] == i)] 
        count_maximum = filter_data_2.Player.count() #how many times per year
        if(count_maximum > maximum): 
            maximum = count_maximum
            player = i
            team = filter_data_2.Team.unique()[0]
    player_list.append(player)
    count_list.append(maximum)
    team_list.append(team)
new_dataFrame = pd.DataFrame({"Season":season_list,"Player":player_list,"Team":team_list,"Count Week's Player":count_list})

#Top 10
new_dataFrame.sort_values(inplace=True,by = "Count Week's Player",ascending=False)
new_dataFrame.head(10)
east = dataFrame[dataFrame.Conference == "East"]
east_count = east.Conference.count()
west = dataFrame[dataFrame.Conference == "West"]
west_count = west.Conference.count()
x = ["East","West"]
y = [east_count,west_count]
plt.bar(x,y,color = ["Blue","Red"])
plt.xlabel("Conference")
plt.title("Compare East and West")
plt.rcParams["figure.figsize"] = [7,8]
plt.show()
