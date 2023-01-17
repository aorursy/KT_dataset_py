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
Fifa_19_player=pd.read_csv('../input/data.csv')

Fifa_19_player=pd.DataFrame(Fifa_19_player)

Fifa_19_player.head()
Fifa_19_player.columns

kolonlar=['Unnamed: 0', 'ID', 'Name', 'Age', 'Photo', 'Nationality', 'Flag',

       'Overall', 'Potential', 'Club', 'Club_Logo', 'Value', 'Wage', 'Special',

       'Preferred_Foot', 'International_Reputation', 'Weak_Foot',

       'Skill_Moves', 'Work_Rate', 'Body_Type', 'Real_Face', 'Position',

       'Jersey_Number', 'Joined', 'Loaned_From', 'Contract_Valid_Until',

       'Height', 'Weight', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',

       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',

       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Crossing',

       'Finishing', 'Heading_Accuracy', 'Short_Passing', 'Volleys', 'Dribbling',

       'Curve', 'FK_Accuracy', 'Long_Passing', 'Ball_Control', 'Acceleration',

       'Sprint_Speed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'Standing_Tackle', 'Sliding_Tackle', 'GK_Diving', 'GK_Handling',

       'GK_Kicking', 'GK_Positioning', 'GK_Reflexes', 'Release_Clause']

Fifa_19_player.columns=kolonlar

#Fifa_19_player.columns
Fifa_19_player.info()
#Fifa_19_player.Name.value_counts()

#Fifa_19_player.Age.value_counts()

#Fifa_19_player.Nationality.value_counts()

#Fifa_19_player.Nationality.value_counts()

#Fifa_19_player.Club.value_counts()

#Fifa_19_player.Value.value_counts()

#Fifa_19_player.Preferred_Foot.value_counts()

#Fifa_19_player.Position.value_counts()

#Fifa_19_player.Jersey_Number.value_counts()

#Fifa_19_player.Dribbling.value_counts()

#Fifa_19_player.Sprint_Speed.value_counts()
#ÜLKELERE GÖRE FUTBOLCU SAYILARI
Nationality_list=list(Fifa_19_player.Nationality.unique())

Nationality_player_frequency=Fifa_19_player['Nationality'].value_counts()





data1=pd.DataFrame({'Nationality_list':Nationality_list,'Nationality_player_frequency':Nationality_player_frequency})

new_data=(data1['Nationality_player_frequency'].sort_values(ascending=False)).index.values

data1['Nationality_list']=new_data

sorted_data=data1.reindex(new_data)



plt.figure(figsize=(15,25))

sns.barplot(y=sorted_data['Nationality_list'],x=sorted_data['Nationality_player_frequency'])

plt.xlabel('Nationality_list')

plt.ylabel('Nationality_player_frequency')

plt.show()
sorted_data.head(10)
#Nationality_frequency

#data1
Fifa_19_player.columns
#ÜLKELERİN TOPLAM FUTBOLCULARININ DRİBLİNG VE SPRİNT SPEED ORANI
Nationality_list=list(Fifa_19_player.Nationality.unique())

Sprint_Speed_ratio=[]

for i in Nationality_list:

    x=Fifa_19_player[Fifa_19_player['Nationality']==i]

    Sprint_Speed_rate=sum(x.Sprint_Speed)/len(x)

    Sprint_Speed_ratio.append(Sprint_Speed_rate)

data3=pd.DataFrame({'Nationality_list':Nationality_list,'Sprint_Speed_ratio':Sprint_Speed_ratio})

new_data=(data3['Sprint_Speed_ratio'].sort_values(ascending=False)).index.values

sorted_data3=data3.reindex(new_data)



Nationality_list=list(Fifa_19_player.Nationality.unique())

dribbling_ratio=[]

for i in Nationality_list:

    x=Fifa_19_player[Fifa_19_player['Nationality']==i]

    dribbling_rate=sum(x.Dribbling)/len(x)

    dribbling_ratio.append(dribbling_rate)

data2=pd.DataFrame({'Nationality_list':Nationality_list,'dribbling_ratio':dribbling_ratio})

new_data=(data2['dribbling_ratio'].sort_values(ascending=False)).index.values

sorted_data2=data2.reindex(new_data)
sorted_data3.head()
sorted_data2.head()
plt.figure(figsize=(15,25))

sns.pointplot(y=sorted_data2['Nationality_list'],x=sorted_data2['dribbling_ratio'],color='red',alpha=0.5)

sns.pointplot(y=sorted_data2['Nationality_list'],x=sorted_data3['Sprint_Speed_ratio'],color='lime',alpha=0.5)

plt.text(40,0.6,'dribbling_ratio',color='red',fontsize = 17,style = 'italic')

plt.text(10,0.55,'Sprint_Speed_ratio',color='lime',fontsize = 18,style = 'italic')

plt.grid()

plt.show()
Fifa_19_player.head()
#ülkelerin oyuncularına göre itibarlarının oranı
Nationality_list=list(Fifa_19_player.Nationality.unique())

International_Reputation_ratio=[]

for i in Nationality_list:

    x=Fifa_19_player[Fifa_19_player['Nationality']==i]

    International_Reputation_rate=sum(x.International_Reputation)/len(x)

    International_Reputation_ratio.append(International_Reputation_rate)

data4=pd.DataFrame({'Nationality_list':Nationality_list,'International_Reputation_ratio':International_Reputation_ratio})

new_data=(data4['International_Reputation_ratio'].sort_values(ascending=False)).index.values

sorted_data4=data4.reindex(new_data)



plt.figure(figsize=(15,25))

sns.barplot(y=sorted_data4['Nationality_list'],x=sorted_data4['International_Reputation_ratio']) 

plt.xlabel('International_Reputation_ratio')

plt.ylabel('Nationality_list')

plt.xticks(rotation=45)

plt.title('International_Reputation_ratio')

plt.show()