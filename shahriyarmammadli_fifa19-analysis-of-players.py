# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls","../input"]).decode("utf8"))



#import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#read data

data = pd.read_csv("../input/data.csv")
data.head()
#data.Positioning.value_counts()
data.info()
# Create new dataframe for player overall

player_overall = pd.DataFrame({"Player_Name":data.Name,"Player_Overall":data.Overall})

new_index = (player_overall['Player_Overall'].sort_values(ascending=False)).index.values

player_overall=player_overall.reindex(new_index)

player_overall=player_overall.head(50)



# visualisation data

plt.figure(figsize=(15,10))

sns.barplot(x = player_overall['Player_Name'],y = player_overall['Player_Overall'])

plt.xticks(rotation = 75)

plt.xlabel("Player_Name")

plt.ylabel("Player_Overall")

plt.show()


#data.head()
#data.Nationality.value_counts()
# Player Overalls by Country

country_list = list(data.Nationality.unique())

country_overall = []

country_list10 = []



# setting and sorting dataframe overall by countries

for i in country_list:

    x = data[data["Nationality"]==i]

    overall_rate = sum(x.Overall)/len(x)

    country_overall.append(overall_rate)

    country_list10.append(len(x))

data_country = pd.DataFrame({"country_list":country_list,"country_overall":country_overall})

data_country1 = pd.DataFrame({"country_list":country_list,"country_overall":country_overall,"country_count":country_list10})

data_country1 = data_country1[data_country1.country_count>20]

new_index = (data_country1["country_overall"].sort_values(ascending=False)).index.values

data_country1=data_country1.reindex(new_index)

data_country1 = data_country1.head(40)

#data_country1 = data_country.head(int(len(data_country)/4))



# visualisation

plt.figure(figsize=(15,10))

sns.barplot(x = "country_list",y = "country_overall",data=data_country1)

plt.xticks(rotation=75)

plt.xlabel("country_list")

plt.ylabel("country_overall")

plt.title("30 countries with the hig")
data_country1.head(10)
#40 countries with the highest number of players

country_count = Counter(data.Nationality)

most_common_countries = country_count.most_common(40)

x,y = zip(*most_common_countries)

x,y = list(x),list(y)



#visualisation



plt.figure(figsize=(15,10))

sns.barplot(x = x,y = y,palette=sns.cubehelix_palette(len(x)))

plt.xticks(rotation=80)

plt.xlabel("Countries")

plt.ylabel("Player counts")

plt.title("Most Commont 40 Countries in FIFA19")
# list_clubs_40 = ['FC Barcelona',

#  'Real Madrid',

#  'Juventus',

#  'Paris Saint-Germain',

#  'Manchester City',

#  'FC Bayern München',

#  'Manchester United',

#  'Liverpool',

#  'Napoli',

#  'Arsenal',

#  'Milan',

#  'Inter',

#  'Chelsea',

#  'Atlético Madrid',

#  'Tottenham Hotspur',

#  'Roma',

#  'Borussia Dortmund',

#  'FC Porto',

#  'AS Monaco',

#  'Ajax',

#  'PSV',

#  'VfL Wolfsburg',

#  'Shakhtar Donetsk',

#  'Leicester City',

#  'RB Leipzig',

#  'OGC Nice',

#  'Toronto FC',

#  'Galatasaray SK',

#  'Fenerbahçe SK',

#  'Beşiktaş JK',

#  'Real Betis',

#  'Olympique de Marseille',

#  'RC Celta',

#  'Shakhtar Donetsk',

#  'Lokomotiv Moscow',

#  'LA Galaxy',

#  'New York City FC',

#  'Atlanta United',

#  'Guangzhou Evergrande Taobao FC',

#  'Shanghai SIPG FC']
list_clubs_33 = ['FC Barcelona',

 'Real Madrid',

 'Juventus',

 'Paris Saint-Germain',

 'Manchester City',

 'FC Bayern München',

 'Manchester United',

 'Liverpool',

 'Napoli',

 'Arsenal',

 'Milan',

 'Inter',

 'Chelsea',

 'Atlético Madrid',

 'Tottenham Hotspur',

 'Roma',

 'Borussia Dortmund',

 'FC Porto',

 'AS Monaco',

 'Ajax',

 'PSV',

 'VfL Wolfsburg',

 'Shakhtar Donetsk',

 'Leicester City',

 'RB Leipzig',

 'OGC Nice',

 'Galatasaray SK',

 'Fenerbahçe SK',

 'Beşiktaş JK',

 'Real Betis',

 'Olympique de Marseille',

 'RC Celta']
# Overalls according to club

club_overall = []



for i in list_clubs_33:

    x = data[data.Club==i]

    club_overall.append(sum(x.Overall)/len(x))



# Sorting

sorted_club_overall = pd.DataFrame({"Clubs":list_clubs_33,"Club_overall":club_overall})

new_index = (sorted_club_overall["Club_overall"].sort_values(ascending=False)).index.values

sorted_club_overall = sorted_club_overall.reindex(new_index)



#Visualization

plt.figure(figsize=(20,10))

sns.barplot(x = sorted_club_overall["Clubs"],y = sorted_club_overall["Club_overall"])

plt.xticks(rotation=80)

plt.xlabel("Clubs")

plt.ylabel("Average Overalls")

plt.title("Percentage of 33 Clubs' Overall of Players")
# Ages according to club

club_age = []



for i in list_clubs_33:

    x = data[data.Club==i]

    club_age.append(sum(x.Age)/len(x))



# Sorting

sorted_club_age = pd.DataFrame({"Clubs":list_clubs_33,"Ages":club_age})

new_index = (sorted_club_age["Ages"].sort_values(ascending=True)).index.values

sorted_club_age = sorted_club_age.reindex(new_index)



#Visualization

plt.figure(figsize=(15,10))

sns.barplot(x = sorted_club_age["Clubs"],y = sorted_club_age["Ages"])

plt.xticks(rotation=80)

plt.xlabel("Clubs")

plt.ylabel("Average Overalls")

plt.title("Percentage of 33 Clubs' Ages of Players")
#Close warnings

import warnings

warnings.filterwarnings("ignore")



# 40 Clubs according to body types

data_body = data[["Club","Nationality","Body Type"]]

data_body["Body Type"].replace(["PLAYER_BODY_TYPE_25"],"Normal",inplace=True)

data_body["Body Type"].replace(["Shaqiri"],"Stocky",inplace=True)

data_body["Body Type"].replace(["Courtois"],"Normal",inplace=True)

data_body["Body Type"].replace(["Messi"],"Normal",inplace=True)

data_body["Body Type"].replace(["Neymar"],"Normal",inplace=True)

data_body["Body Type"].replace(["C. Ronaldo"],"Normal",inplace=True)

data_body["Body Type"].replace(["Akinfenwa"],"Stocky",inplace=True)



# Adding three columns according to body types



data_body["Normal_body"] = [1 if i == "Normal" else 0 for i in data["Body Type"]]

data_body["Lean_body"] = [1 if i == "Lean" else 0 for i in data["Body Type"]]

data_body["Stocky_body"] = [1 if i == "Stocky" else 0 for i in data["Body Type"]]



# Added three lists for each of body types



Normal_body = []

Lean_body = []

Stocky_body = []



for i in list_clubs_33:

    x = data_body[data_body.Club==i]

    Normal_body.append(sum(x.Normal_body)/len(x))

    Lean_body.append(sum(x.Lean_body)/len(x))

    Stocky_body.append(sum(x.Stocky_body)/len(x))



# Visualization



f,ax=plt.subplots(figsize = (9,12))

sns.barplot(x = Normal_body,y = list_clubs_33,color="red",alpha = 0.6,label = "Normal_Body")

sns.barplot(x = Lean_body,y = list_clubs_33,color="blue", alpha = 0.5,label = "Lean_Body")

sns.barplot(x = Stocky_body,y = list_clubs_33,color="cyan",alpha = 0.6,label = "Stock_Body")



ax.legend(loc = "upper right",frameon = True)

ax.set(xlabel="Percentage of Body Types",ylabel="Clubs",title = "Percentage of Clubs' Players of Body Type")

plt.show()
#sorted_club_age.head()

#sorted_club_overall.head()
# Club Overall vs Club Age of each club

sorted_club_age["Ages"] = sorted_club_age["Ages"]/max(sorted_club_age["Ages"])

sorted_club_overall["Club_overall"] = sorted_club_overall["Club_overall"]/max(sorted_club_overall["Club_overall"])

data = pd.concat([sorted_club_age,sorted_club_overall["Club_overall"]],axis = 1)

data.sort_values("Club_overall",inplace=False)



#visualization

plt.subplots(figsize =(20,10))

sns.pointplot(x = "Clubs",y = "Ages",data=data,color="red",alpha=0.5)

sns.pointplot(x = "Clubs",y = "Club_overall",data=data,color="lime",alpha = 0.8)

plt.xticks(rotation=90)

plt.text(20,0.99,'Club Ages',color='red',fontsize = 25,style = 'italic')

plt.text(20,0.96,'Club Overalls',color='lime',fontsize = 25,style = 'italic')

plt.xlabel('Clubs',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('Club Ages vs Club Overalls',fontsize = 20,color='blue')

plt.grid()
# joint kernel density

g = sns.jointplot(data.Club_overall,data.Ages,kind="kde",size=7)

plt.savefig("graph.png")

plt.show()
# you can change parameters of joint plot

g = sns.jointplot("Club_overall","Ages",data=data,size=5,ratio=3,color="red")
data_new = pd.read_csv("../input/data.csv")
data_new.head()
data_new['Preferred Foot'].value_counts()
#Players according to Preferred Foot

labels = data_new['Preferred Foot'].value_counts().index

colors = ["red","green"]

explode=[0,0]

sizes = data_new['Preferred Foot'].value_counts().values

# visualisation

plt.figure(figsize=(6,6))

plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct="%1.1f%%")

plt.title("Players Accroding to Foots",color = "blue",fontsize=15)
data_new['Body Type'].value_counts()
#Players according to International Reputation

data_new["Body Type"].replace(["PLAYER_BODY_TYPE_25"],"Normal",inplace=True)

data_new["Body Type"].replace(["Shaqiri"],"Stocky",inplace=True)

data_new["Body Type"].replace(["Courtois"],"Normal",inplace=True)

data_new["Body Type"].replace(["Messi"],"Normal",inplace=True)

data_new["Body Type"].replace(["Neymar"],"Normal",inplace=True)

data_new["Body Type"].replace(["C. Ronaldo"],"Normal",inplace=True)

data_new["Body Type"].replace(["Akinfenwa"],"Stocky",inplace=True)



labels = data_new['Body Type'].value_counts().index

explode = [0,0,0]

sizes = data_new['Body Type'].value_counts().values

colors = ["red","green","blue"]

# visalization

plt.figure(figsize=(7,7))

plt.pie(sizes,labels=labels,colors=colors,explode=explode,autopct="%1.1f%%")

plt.title("Players According to Body Type",color = "blue",fontsize=15)
data.head()
# Show the results of a linear regression within each dataset

sns.lmplot("Club_overall","Ages",data=data)

plt.show
# cubehelix plot

sns.kdeplot(data.Club_overall,data.Ages,shade=True,cut = 3)

plt.show()
# Use cubehelix to get a custom sequential palette

pal = sns.cubehelix_palette(2, rot=-.5,dark=.4)

sns.violinplot(data=data,palette=pal,inner="points")

plt.show()
data.corr()
f,ax = plt.subplots(figsize = (5,5))

sns.heatmap(data.corr(),annot=True,linewidths=0.5,linecolor="red",fmt = '.2f',ax = ax)

plt.show()
data_new.head()
data_new["Body Type"].value_counts()
# Plot the orbital period with horizontal boxe

plt.figure(figsize=(12,6))

sns.boxplot(x = "Preferred Foot",y = "Overall",hue='Body Type',data=data_new,palette="PRGn")

plt.show()
# swarm plot

sns.swarmplot(x = "Preferred Foot",y = "Overall",hue="Body Type",data=data_new)

plt.show()
data.head()
# Pair plot

sns.pairplot(data)

plt.show()
# Foot properties

#data_new.Position.value_counts()

plt.figure(figsize=(10,6))

sns.countplot(data_new.Position,order=data_new.Position.value_counts().index)

#sns.countplot(data_new['Preferred Foot'])

plt.xticks(rotation=90)

plt.title("Position",color="blue",fontsize=15)
data_new.head()
#Jersey number

player_number = data_new['Jersey Number'].value_counts()

#visualization

plt.figure(figsize=(10,7))

sns.barplot(x = player_number[:1].index,y = player_number[:1].values)

plt.xlabel("Number of Jersey Number")

plt.ylabel("Numbers")

plt.title("Jersey Numers",color="blue",fontsize=15)
# Ages of Players

above25 = [ "above25" if i>25 else "equal25" if i==25 else "below25" for i in data_new.Age]

df = pd.DataFrame({"Age":above25})

sns.countplot(x = df.Age)

plt.ylabel("Number of Players Age")

plt.title("Age of Players",color="blue",fontsize=15)