#imports
from PIL import  Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pylab
import warnings
warnings.filterwarnings('ignore')


plt.figure(figsize=(15,10))
img = np.array(Image.open(r"../input/images-fifa/fifa.jpg"))
plt.imshow(img,interpolation="bilinear")
plt.axis("off")
plt.show()
worldCup = pd.read_csv('../input/fifa-world-cup/WorldCups.csv')
matches = pd.read_csv('../input/fifa-world-cup/WorldCupMatches.csv')
players = pd.read_csv('../input/fifa-world-cup/WorldCupPlayers.csv')
worldCup.head()
worldCup.replace(to_replace="Germany FR", value="Germany",inplace=True)
players.head()
matches.head()

plt.figure(figsize=(10, 10))
plt.xticks(worldCup.Year, rotation=70)
plt.yticks(worldCup.GoalsScored)
plt.xlabel("Year",fontsize=20)
plt.ylabel("Goals",fontsize=20)
plt.title("Years vs Goals",fontsize=24)
plt.plot(worldCup.Year, worldCup.GoalsScored, color='blue')
plt.show()
#Teams qualifying per year
plt.figure(figsize=(10, 10))
plt.barh(worldCup.Year,worldCup.QualifiedTeams,alpha=0.5) 
plt.ylabel('Year')
plt.xlabel('QualifiedTeams')
plt.yticks(worldCup.Year, rotation=30)
plt.xticks(worldCup.QualifiedTeams)
plt.title('No of teams qualifying per year')
plt.show()

#Attendance per year
plt.figure(figsize=(10, 10))
plt.barh(worldCup.Year,worldCup.Attendance,alpha=0.5,color='green') 
plt.ylabel('Year')
plt.xlabel('Attendance')
plt.yticks(worldCup.Year, rotation=30)
plt.xticks(worldCup.Attendance, rotation=30)
plt.title('Attendance in every year')
plt.show()
#MatchesPlayed per year
plt.figure(figsize=(10, 10))
plt.bar(worldCup.Year,worldCup.MatchesPlayed,width =1.2,alpha=0.5,color='orange') 
plt.xlabel('Year')
plt.ylabel('MatchesPlayed')
plt.xticks(worldCup.Year, rotation=30)
plt.yticks(worldCup.MatchesPlayed)
plt.title('MatchesPlayed')
plt.show()
%matplotlib inline
import matplotlib.pyplot as plt


x=worldCup['Winner'].value_counts()[:]
y =worldCup['Runners-Up'].value_counts()[:]
z = worldCup['Third'].value_counts()[:]
s = worldCup['Fourth'].value_counts()[:]
results = pd.concat([x, y, z,s], axis=1)

results = results.sort_values(by=['Winner', 'Runners-Up', 'Third','Fourth'], ascending=False)
results = results.fillna(value=0)
results 

results.plot(y=['Winner', 'Runners-Up', 'Third','Fourth'], kind="bar", 
                  color =['gold','silver','brown','green'], figsize=(20, 8), width=0.9)

from itertools import cycle, islice

matches.replace(to_replace="Germany FR", value="Germany",inplace=True)
matches = matches.replace(to_replace='rn">', value = "" , regex=True)
goals1 = matches.groupby(['Home Team Name'],as_index=False)[["Home Team Goals"]].sum()
goals2 = matches.groupby(['Away Team Name'],as_index=False)[["Away Team Goals"]].sum()

goals1 =goals1.rename(index=str, columns={"Home Team Name": "country", "Home Team Goals": "goals1"})
goals2 =goals2.rename(index=str, columns={"Away Team Name": "country", "Away Team Goals": "goals2"})


goals =goals1.merge(goals2, left_on='country', right_on='country', how='outer')
goals = goals.fillna(value=0)
goals['total_goals'] = goals['goals1'] + goals['goals2']
goals = goals.drop(['goals1', 'goals2'], axis=1)

goals = goals[goals.total_goals != 0]
goals.sort_values(by=['total_goals'], ascending=False, inplace =True)
goals.reset_index(drop=True,inplace=True)

key = goals['country']
goals = goals.iloc[key.argsort()]

my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(goals)))
plt.figure(figsize=(25, 10))
plt.bar(goals.country,goals.total_goals,alpha=0.5, color=my_colors) 
plt.xlabel('Country')
plt.ylabel('Goals')
plt.xticks(goals.country, rotation=90, size=15)
plt.title('Goals by country')
plt.show()

import wordcloud
plt.figure(figsize=(15,15))
famous_players = players["Player Name"]

famous_players = famous_players.str.replace(' ', '')
counts = famous_players.value_counts().to_dict()


wordcloud = wordcloud.WordCloud(
                          background_color='white',
                          max_words=200,
                          max_font_size=40 
                         ).generate_from_frequencies(counts)

plt.imshow(wordcloud)
plt.title("Players Wordcloud",fontsize=24)
plt.show()

matches.rename(columns={'Home Team Name': 'HomeTeamName', 'Away Team Name': 'AwayTeamName'}, inplace=True)
teams = ['Belgium','England']
belVsEng = matches.loc[matches.HomeTeamName.isin(teams) | matches.AwayTeamName.isin(teams)]

England_faceoff = belVsEng.loc[(belVsEng['HomeTeamName']=='England')| (belVsEng['AwayTeamName']=='England')]




for i in England_faceoff.index:
    if England_faceoff.at[i,'HomeTeamName']=='England':
        England_faceoff.at[i, 'goals'] = England_faceoff.at[i,'Home Team Goals']
        England_faceoff.at[i, 'country'] = England_faceoff.at[i,'HomeTeamName']
    else:
        England_faceoff.at[i, 'goals'] = England_faceoff.at[i,'Away Team Goals']
        England_faceoff.at[i, 'country'] = England_faceoff.at[i,'AwayTeamName']

England_faceoff =England_faceoff.reset_index()


England_faceoff_goals=pd.DataFrame()
England_faceoff_goals=England_faceoff[['Year','country','goals']]
England_faceoff_goals
England_faceoff_Yeargoals = England_faceoff_goals.groupby(['Year','country'], as_index=False).sum()



### Same for Belgium:

Belgium_faceoff = belVsEng.loc[(belVsEng['HomeTeamName']=='Belgium')| (belVsEng['AwayTeamName']=='Belgium')]
for i in Belgium_faceoff.index:
    if Belgium_faceoff.at[i,'HomeTeamName']=='Belgium':
        Belgium_faceoff.at[i, 'goals'] = Belgium_faceoff.at[i,'Home Team Goals']
        Belgium_faceoff.at[i, 'country'] = Belgium_faceoff.at[i,'HomeTeamName']
    else:
        Belgium_faceoff.at[i, 'goals'] = Belgium_faceoff.at[i,'Away Team Goals']
        Belgium_faceoff.at[i, 'country'] = Belgium_faceoff.at[i,'AwayTeamName']

Belgium_faceoff =Belgium_faceoff.reset_index()


Belgium_faceoff_goals=pd.DataFrame()
Belgium_faceoff_goals=Belgium_faceoff[['Year','country','goals']]
Belgium_faceoff_goals.reset_index()
Belgium_faceoff_Yeargoals = Belgium_faceoff_goals.groupby(['Year','country'], as_index=False).sum()



years=[]
years1 =Belgium_faceoff_Yeargoals.Year
years2 =England_faceoff_Yeargoals.Year
years = years1.append(years2)

years_array = pd.Series(years).values
years_array = np.unique(years_array)

years_array_string = years_array.astype(np.str)
print(years_array_string)

plt.figure(figsize=(12, 8))
plt.plot(Belgium_faceoff_Yeargoals.Year, Belgium_faceoff_Yeargoals.goals, '-y', label='Belgium')
plt.plot(England_faceoff_Yeargoals.Year, England_faceoff_Yeargoals.goals,'-r', label='England')
plt.legend(loc='upper left')
plt.xticks(years_array)
plt.show()


matches.rename(columns={'Home Team Name': 'HomeTeamName', 'Away Team Name': 'AwayTeamName'}, inplace=True)
teams = ['France','Croatia']
franvsCro = matches.loc[matches.HomeTeamName.isin(teams) | matches.AwayTeamName.isin(teams)]

France_faceoff = franvsCro.loc[(franvsCro['HomeTeamName']=='France')| (franvsCro['AwayTeamName']=='France')]




for i in France_faceoff.index:
    if France_faceoff.at[i,'HomeTeamName']=='France':
        France_faceoff.at[i, 'goals'] = France_faceoff.at[i,'Home Team Goals']
        France_faceoff.at[i, 'country'] = France_faceoff.at[i,'HomeTeamName']
    else:
        France_faceoff.at[i, 'goals'] = France_faceoff.at[i,'Away Team Goals']
        France_faceoff.at[i, 'country'] = France_faceoff.at[i,'AwayTeamName']

France_faceoff =France_faceoff.reset_index()


France_faceoff_goals=pd.DataFrame()
France_faceoff_goals=France_faceoff[['Year','country','goals']]
##France_faceoff_goals
France_faceoff_Yeargoals = France_faceoff_goals.groupby(['Year','country'], as_index=False).sum()


### Same for Croatia:

Croatia_faceoff = franvsCro.loc[(franvsCro['HomeTeamName']=='Croatia')| (franvsCro['AwayTeamName']=='Croatia')]
for i in Croatia_faceoff.index:
    if Croatia_faceoff.at[i,'HomeTeamName']=='Croatia':
        Croatia_faceoff.at[i, 'goals'] = Croatia_faceoff.at[i,'Home Team Goals']
        Croatia_faceoff.at[i, 'country'] = Croatia_faceoff.at[i,'HomeTeamName']
    else:
        Croatia_faceoff.at[i, 'goals'] = Croatia_faceoff.at[i,'Away Team Goals']
        Croatia_faceoff.at[i, 'country'] = Croatia_faceoff.at[i,'AwayTeamName']

Croatia_faceoff =Croatia_faceoff.reset_index()


Croatia_faceoff_goals=pd.DataFrame()

Croatia_faceoff_goals=Croatia_faceoff[['Year','country','goals']]
Croatia_faceoff_goals.reset_index()
Croatia_faceoff_Yeargoals = Croatia_faceoff_goals.groupby(['Year','country'], as_index=False).sum()

#print(Croatia_faceoff_goals)

years=[]
years1 =France_faceoff_Yeargoals.Year
years2 =Croatia_faceoff_Yeargoals.Year
years = years1.append(years2)

years_array = pd.Series(years).values
years_array = np.unique(years_array)

years_array_string = years_array.astype(np.str)
#print(years_array_string)

plt.figure(figsize=(12, 8))
plt.plot(Croatia_faceoff_Yeargoals.Year, Croatia_faceoff_Yeargoals.goals, '-b', label='Croatia')
plt.plot(France_faceoff_Yeargoals.Year, France_faceoff_Yeargoals.goals,'-r', label='France')
plt.legend(loc='upper left')
plt.xticks(years_array)
plt.show()
