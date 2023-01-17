# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.
data_cups=pd.read_csv('../input/WorldCups.csv')
#print(data_matches.info())
print(data_cups.info())
#data_players.info()
data_cups.corr()
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data_cups.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()
#columns - sütunlar
#print(data_matches.columns)
print(data_cups.columns)
#first 10 results at matches
#ilk 10 sonuç
data_cups.head(10)
data_cups.plot(color = 'r', figsize= [18,8], x='Year', y='GoalsScored', linewidth=3, alpha = 1,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Goaals Scored')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
data_cups.plot(kind='scatter', figsize= [18,8],  x='Year', y='GoalsScored',alpha = 0.75,color = 'blue')
plt.xlabel('Year')              # label = name of label
plt.ylabel('GoalsScored')
plt.title('GoalsScored by Year Scatter Plot')            # title = title of plot
#data_cups.columns = [c.replace(' ', '_') for c in data_matches.columns]
#data_cups.columns = [c.replace('-', '_') for c in data_matches.columns]
data_cups.MatchesPlayed.plot(kind = 'hist',bins = 25,figsize = (14,8))
plt.show()
data_cups[np.logical_and(data_cups['GoalsScored']>100, data_cups['Year']>1930 )]
def encokgol(count=5):
    """returns a list of top (count) cups that finished with the highest amount of goals (default:5)"""
    ecg=data_cups.sort_values(by=['GoalsScored'],ascending=False).head(count)
    return ecg
encokgol()
def goals(country=16):
    Country=data_cups.at[country-1,'Country']
    year=data_cups.at[country-1,'Year']
    goals=data_cups.at[country-1,'GoalsScored']
    matches=data_cups.at[country-1,'MatchesPlayed']
    def AvgGoal(goals, matches):
        avg=goals/matches
        return avg
    print(Country, year)
    print("Average goal per match:",AvgGoal(goals,matches))
goals()
def ulkeler(*args):
    for i in args:
        print(i)
countries=tuple(data_cups.iloc[:, data_cups.columns.get_loc('Country')])
ulkeler(countries)

dict=data_cups.set_index('Country').to_dict()['Winner']
def f(**kwargs):
    for key, value in kwargs.items():
        print(key, " ", value)
    
f(**dict)
#cleaning data and make it more usable and readable 
#we will learn more about cleaning at below...
data_cups.Attendance = data_cups.Attendance.astype(str)
data_cups.Attendance = [c.replace('.', '') for c in data_cups.Attendance]
data_cups.Attendance = data_cups.Attendance.astype(int)
#data_cups.Attendance
ort=sum(data_cups.Attendance)/len(data_cups.Attendance)
print("Ortalama", ort)
kalabalik = list(filter(lambda x: (x>ort) , data_cups.Attendance))
print("Yoğun katılımlı kupalar:", kalabalik)
# zip example
list1 = data_cups.groupby('Winner').count().sort_values('Year',ascending=False)
list2 = data_cups.groupby('Winner').count().sort_values('Year',ascending=False)
z = zip(list1.index,list2.Year)
z_list = list(z)
print("yanıtlar", z_list)
import math
esik = sum(data_cups.GoalsScored)/len(data_cups.GoalsScored)
#print(data_cups.GoalsScored)
data_cups["BolGol"] = ["Bol Gollü" if i > esik else "Az Gollü" for i in data_cups.GoalsScored]
data_cups.reindex(columns=["Year","Country","BolGol","GoalsScored"]).sort_values('GoalsScored',ascending=False)
data_matches = pd.read_csv('../input/WorldCupMatches.csv')
data_matches.head()
data_matches.tail(30)
data_matches.columns
data_matches.shape
data_matches.info()
print(data_matches.Attendance.value_counts(dropna =False))  # if there are nan values that also be counted
data_matches.describe
data_matches.boxplot(column='Home Team Goals',by = 'Away Team Goals')
data = data_matches.head(10)    # I only take 10 rows into new data
data
melted = pd.melt(frame=data,id_vars = 'MatchID', value_vars= ['City','Stadium'])
melted

melted.pivot(index = 'MatchID', columns = 'variable',values='value')
data1 = data_matches.head()
data2= data_matches.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row
data1 = data_matches['Stadium'].head()
data2= data_matches['City'].head()
data3= data_matches['Referee'].head()
conc_data_col = pd.concat([data1,data2,data3],axis =1) # axis = 0 : adds dataframes in row
conc_data_col
#data_matches.dtypes
data_matches.info()
data_matches['Attendance'].value_counts(dropna =False)
data_matches=data_matches.dropna(thresh=1) #even if a row includes a not nan value cell, dropna keeps that row.
data_matches.columns = [c.replace(' ', '') for c in data_matches.columns]
data_matches.columns = [c.replace('-', '') for c in data_matches.columns]
data_matches.columns
data_matches.Year.fillna(0.0 ,inplace=True)
data_matches.Year=data_matches.Year.astype(int)
data_matches.HomeTeamGoals.fillna(0.0 ,inplace=True)
data_matches.HomeTeamGoals=data_matches.HomeTeamGoals.astype(int)

data_matches.AwayTeamGoals.fillna(0.0 ,inplace=True)
data_matches.AwayTeamGoals=data_matches.AwayTeamGoals.astype(int)

data_matches.HalftimeHomeGoals.fillna(0.0 ,inplace=True)
data_matches.HalftimeHomeGoals=data_matches.HalftimeHomeGoals.astype(int)

data_matches.HalftimeAwayGoals.fillna(0.0 ,inplace=True)
data_matches.HalftimeAwayGoals=data_matches.HalftimeAwayGoals.astype(int)

data_matches.Attendance
data_matches.Attendance.fillna(0.0 ,inplace=True)
data_matches.Attendance = data_matches.Attendance.astype(int)
data_matches.Attendance.fillna(0 ,inplace=True)
data_matches.Attendance

data_matches.info()
# Assert statement:
# return nothing if True
# return error if False
assert data_matches['Winconditions'].notnull().all()
# no errors because no nan values anymore :)
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data_cups.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()