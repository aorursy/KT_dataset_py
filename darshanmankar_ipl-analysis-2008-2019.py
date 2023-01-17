# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np              # used to read and preprocess data

import seaborn as sns           # used to working with arrays Single or MultiDiementional

import pandas as pd             # Visualisation of data

import matplotlib.pyplot as plt # Visualisation of data

%matplotlib inline
data=pd.read_csv('/kaggle/input/ipldata/matches.csv')
data.head()
data.shape
# removing unwanted coloumns

columns_to_remove = ['id','umpire1','umpire2','umpire3']

data.drop(labels=columns_to_remove,axis=1,inplace=True)
data.head()
data['team1'].unique()
data['team2'].unique()
data.at[data['team1']=='Delhi Daredevils','team1']='Delhi Capitals'

data.at[data['team2']=='Delhi Daredevils','team2']='Delhi Capitals'

data.at[data['winner']=='Delhi Daredevils','winner']='Delhi Capitals'
consistent_teams = ['Royal Challengers Bangalore',

       'Kolkata Knight Riders', 'Kings XI Punjab',

       'Sunrisers Hyderabad', 'Mumbai Indians', 

       'Rajasthan Royals', 'Chennai Super Kings',      

       'Delhi Capitals']
data = data[(data['team1'].isin(consistent_teams)) & (data['team2'].isin(consistent_teams))]
print(data['team1'].unique())

print(data['team2'].unique())
data.head()
data.shape
sns.set_style("darkgrid")

fig=plt.gcf()

fig.set_size_inches(15,5)

plt.xticks(rotation=0,fontsize=12)

plt.yticks(fontsize=16)

results=pd.DataFrame(data['result'].value_counts())

results['name']=results.index  # store index as ht in name

plt.bar(results['name'],results['result'],color=['orange','green'])

count=0

for i in results['result']:

    plt.text(count-0.10,i+0.1,str(i),size=15,color='black',rotation=0)

    count+=1

    

#  count-0.15 for center align

#  i+0.1 for Vertical Alignment



plt.title('Final Result',fontsize=20)

plt.xlabel('Result',fontsize=15)

plt.ylabel('Total no. of matches (2008-2019)',fontsize=15)
sns.set(style='darkgrid')

fig=plt.gcf()

fig.set_size_inches(18.5,10.5)

wins=pd.DataFrame(data['winner'].value_counts())

wins['name']=wins.index

plt.xticks(rotation=90,fontsize=12)

plt.yticks(fontsize=16)

plt.bar(wins['name'],

        wins['winner'],

        color=['#15244C','#FFFF48','#292734','#EF2920','#CD202D','#ECC5F2',

               '#294A73','#D4480B','#242307','#FD511F','#158EA6','#E82865',

               '#005DB7','#C23E25','#E82865']

        ,alpha=0.8)

count=0

for i in wins['winner']:

    plt.text(count-0.15,i-4,str(i),size=15,color='black',rotation=90)

    count+=1

plt.title('Total wins by each team',fontsize=20)

plt.xlabel('Teams',fontsize=15)

plt.ylabel('Total no. of matches won(2008-2019)',fontsize=14)

#plt.show()
sns.set(style='darkgrid')

fig=plt.gcf()

fig.set_size_inches(18.5,10.5)

M_O_M=pd.DataFrame(data['player_of_match'].value_counts())

M_O_M['name']=M_O_M.index

M_O_M=M_O_M.head(10)

plt.xticks(rotation=90,fontsize=12)

plt.yticks(fontsize=16)

plt.bar(M_O_M['name'],M_O_M['player_of_match'],

        color=['#CD202D','#EF2920','#D4480B','#15244C','#FFFF48','#EF2920',

               '#FFFF48','#FFFF48','#292734','#FFFF48','#ECC5F2','#EF2920',

               '#292734','#15244C','#005DB7','#005DB7','#292734','#15244C',

               '#FFFF48','#CD202D'],alpha=0.8)

count=0

for i in M_O_M['player_of_match']:

    plt.text(count-0.15,i+0.1,str(i),size=15,color='black',rotation=0)

    count+=1

    

#  count-0.15 for center align

#  i+0.1 for Vertical Alignment



plt.title('Top 20 Man Of The Match Winners(2008-2019)',fontsize=20)

plt.xlabel('Players Name',fontsize=15)

plt.ylabel('Total Awards Count',fontsize=14)
data.at[data['city']=='Bengaluru','city']='Bangalore'
sns.set(style='darkgrid')

fig=plt.gcf()

fig.set_size_inches(18.5,10.5)

cities=pd.DataFrame(data['city'].value_counts())

cities['name']=cities.index

#cities=cities.head(10)

plt.xticks(rotation=90,fontsize=12)

plt.yticks(fontsize=16)

plt.bar(cities['name'],cities['city'],alpha=0.8)

count=0

for i in cities['city']:

    plt.text(count-0.18,i+0.1,str(i),size=15,color='black',rotation=0)

    count+=1

plt.title('Total Matches Hosted At Each City ',fontsize=20)

plt.xlabel('City',fontsize=15)

plt.ylabel('Total Number Of Matches Hosted',fontsize=14)
sns.set(style='darkgrid')

fig=plt.gcf()

fig.set_size_inches(18.5,10.5)

Venue=pd.DataFrame(data['venue'].value_counts())

Venue['name']=Venue.index

plt.xticks(rotation=90,fontsize=12)

plt.yticks(fontsize=16)

plt.bar(Venue['name'],Venue['venue'],alpha=0.8)

count=0

for i in Venue['venue']:

    plt.text(count-0.18,i+0.1,str(i),size=15,color='black',rotation=0)

    count+=1

plt.title('Total Matches Hosted At Each venue ',fontsize=20)

plt.xlabel('Venue',fontsize=15)

plt.ylabel('Total Number Of Matches Hosted',fontsize=14)
head_to_head = ['Mumbai Indians','Chennai Super Kings',]
# we consider only those matches played bitween MI and CSK

data_MIvsCSK = data[(data['team1'].isin(head_to_head)) & (data['team2'].isin(head_to_head))]
# we can also use this method to find head to head clash

# we consider only those matches played bitween MI and CSK

# data_MIvsCSK=data[np.logical_or

#       (np.logical_and(data['team1']=='Mumbai Indians',data['team2']=='Chennai Super Kings')

#                  ,np.logical_and(data['team2']=='Mumbai Indians',data['team1']=='Chennai Super Kings'))]
print(data_MIvsCSK['team1'].unique())

print(data_MIvsCSK['team2'].unique())
sns.set(style='dark')

fig=plt.gcf()

fig.set_size_inches(10,8)

sns.countplot(data_MIvsCSK['winner'],order=data_MIvsCSK['winner'].value_counts().index)

plt.text(-0.1,15,str(data_MIvsCSK['winner'].value_counts()['Mumbai Indians']),size=29,color='white')

plt.text(0.9,9,str(data_MIvsCSK['winner'].value_counts()['Chennai Super Kings']),size=29,color='white')

plt.xlabel('Winner',fontsize=15)

plt.ylabel('Count',fontsize=15)

plt.yticks(fontsize=0)

plt.title('MI vs CSK - head to head')

plt.show()
sns.set(style='darkgrid')

fig=plt.gcf()

fig.set_size_inches(18.5,8)

sns.countplot(data_MIvsCSK['player_of_match'],order=data_MIvsCSK['player_of_match'].value_counts().index,palette='Set2')

plt.title('All man of the match awards in MI-CSK games',fontsize=15)

plt.yticks([1,2,3],[1,2,3],fontsize=15)

plt.xticks(fontsize=15,rotation=90)

plt.xlabel('Man of the match',fontsize=15)

plt.ylabel('Count',fontsize=15)

plt.show()
sns.set(style='darkgrid')

fig=plt.gcf()

fig.set_size_inches(18.5,8)

sns.countplot(data_MIvsCSK['venue'],order=data_MIvsCSK['venue'].value_counts().index,palette='Set2',hue=data['toss_decision'])

plt.title('Toss decision at each venue in MIvCSK matches',fontsize=15)

plt.yticks(fontsize=15)

plt.xticks(fontsize=15,rotation=90)

plt.xlabel('Venue',fontsize=15)

plt.ylabel('Count',fontsize=15)

plt.legend(loc=5,fontsize=15)

plt.show()
sns.set(style='darkgrid')

fig=plt.gcf()

fig.set_size_inches(18.5,8)

sns.countplot(data_MIvsCSK['toss_winner'],order=data_MIvsCSK['toss_winner'].value_counts().index,palette='Set2',hue=data['toss_decision'])

plt.title('Toss decision statistics for both team',fontsize=15)

plt.yticks(fontsize=15)

plt.xticks(fontsize=15)

plt.xlabel('Toss winner',fontsize=15)

plt.ylabel('Count',fontsize=15)

plt.text(-0.25,6,str(int((7/15)*100)+1)+'%',fontsize=29)

plt.text(0.15,7,str(int((8/15)*100))+'%',fontsize=29)

plt.text(0.75,7,str(int((8/13)*100)+1)+'%',fontsize=29)

plt.text(1.15,4,str(int((5/13)*100))+'%',fontsize=29)

plt.legend(['Field first','Bat first'],loc='best',fontsize=15)

plt.show()