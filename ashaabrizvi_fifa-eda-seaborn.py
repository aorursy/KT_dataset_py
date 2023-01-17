import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100

%matplotlib inline

fifa_20 = pd.read_csv('../input/fifa-20-complete-player-dataset/players_20.csv')
fifa_20.head()
fifa_20.shape # To check no of rows and columns
col = list(fifa_20.columns)  # To print all the columns

print(col)
useless_column = ['dob','sofifa_id','player_url','long_name','body_type','real_face','nation_position','loaned_from','nation_jersey_number']
fifa_20 = fifa_20.drop(useless_column, axis = 1)
fifa_20.head()
fifa_20.shape # To check how many columns did we dropped
fifa_20['BMI'] = fifa_20 ['weight_kg'] / (fifa_20['height_cm'] / 100) ** 2
fifa_20.head()
fifa_20[['short_name','player_positions']]
new_player_position = fifa_20['player_positions'].str.get_dummies(sep=',').add_prefix('Position')

new_player_position.head()
fifa_20 =  pd.concat([fifa_20,new_player_position],axis = 1)
fifa_20.head()
fifa_20 =  fifa_20.drop('player_positions',axis=1)
fifa_20.head()
columns = ['ls','st','rs','lw','lf','cf','rf','rw','lam','cam','ram','lm','lcm','cm','rcm','rm','lwb','ldm', 'cdm','rdm','rwb','lb','lcb','cb','rcb','rb']
fifa_20[columns].head()
for col in columns:

  fifa_20[col]=fifa_20[col].str.split('+',n=1,expand = True)[0]



fifa_20[columns]
fifa_20[columns] = fifa_20[columns].fillna(0)
fifa_20[columns] = fifa_20[columns].astype(int)
fifa_20[columns]
columns = ['dribbling','defending','physic','passing','shooting','pace']
fifa_20[columns]
fifa_20[columns].isnull().sum()
for col in columns:

  fifa_20[col] = fifa_20[col].fillna(fifa_20[col].median())

fifa_20[columns]
fifa_20 = fifa_20.fillna(0)
fifa_20.isnull().sum()
fifa_20.head()
sns.relplot(x='overall',y='value_eur',hue='age',palette = 'viridis',size="BMI", sizes=(15, 200),aspect=2.5,data=fifa_20)

plt.title('Overall Rating v  Value in Euros',fontsize = 20)

plt.xlabel('Overall Rating')

plt.ylabel('Value in Euros')

plt.show()
sns.relplot(x='potential',y='wage_eur',hue='age',palette = 'viridis',size="BMI", sizes=(15, 200),aspect=2.5,data=fifa_20)

plt.title('Potential Rating  v  Wage in Euros',fontsize = 20)

plt.xlabel('Potential')

plt.ylabel('Wage in Euros')

plt.show()
plt.figure(dpi=125)

sns.countplot('preferred_foot',data=fifa_20)

plt.xlabel('Preferred Foot Players')

plt.ylabel('Count')

plt.title('Count of Preferred Foot')

Right,Left=fifa_20.preferred_foot.value_counts()

print('Left Preferred',Left)

print('Right Preferred',Right)

plt.show()
plt.figure(dpi=125)

sns.countplot('international_reputation',data=fifa_20.head(100))

plt.xlabel('International Reputation')

plt.ylabel('Count')

plt.title('Count of International Reputation of Top 100 Players')

plt.show()
plt.figure(dpi=125)

sns.countplot('team_jersey_number',data=fifa_20.head(100))

plt.xlabel('Jersey No.')

plt.ylabel('Count')

plt.xticks(rotation=60)

plt.title('Count of Team Jersey No. in Top 100 Players')

plt.show()
plt.figure(dpi=125)

sns.countplot('team_position',data=fifa_20.head(100))

plt.xlabel('Team Positions')

plt.ylabel('Count')

plt.xticks(rotation=60)

plt.title('Count of Team Positions in Top 100 Players')

plt.show()
plt.figure(dpi=125)

sns.distplot(a=fifa_20['age'],kde=False,bins=20)

plt.axvline(x=np.mean(fifa_20['age']),c='green',label='Mean Age of All Players')

plt.legend()

plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Distribution of Age')

plt.show()
plt.figure(dpi=125)

sns.distplot(a=fifa_20['BMI'],kde=False,bins=20)

plt.axvline(x=np.mean(fifa_20['BMI']),c='green',label='Mean BMI of All Players')

plt.legend()

plt.xlabel('BMI')

plt.ylabel('Count')

plt.title('Distribution of BMI')

plt.show()
plt.figure(figsize=(14,5))

sns.countplot('nationality',data=fifa_20.head(20))

plt.xlabel('Nationality')

plt.ylabel('Count')

plt.xticks(rotation=45)

plt.title('Which Country has Max players in Top 20',fontsize = 20)

plt.show()
plt.figure(figsize=(14,5))

sns.countplot('club',data=fifa_20.head(20))

plt.xlabel('Club')

plt.ylabel('Count')

plt.xticks(rotation=45)

plt.title('Which Club has Max players in Top 20',fontsize = 20)

plt.show()
plt.figure(dpi=125)

sns.countplot('team_jersey_number',data=fifa_20.head(20))

plt.xlabel('Jersey No')

plt.ylabel('Count')

plt.title('Which Jersey Number comes most in Top 20')

plt.show()
plt.figure(dpi=125)

sns.countplot('team_position',data=fifa_20.head(20))

plt.xlabel('Player Position')

plt.ylabel('Count')

plt.title('Which type of player comes most in Top 20')

plt.show()
plt.figure(dpi=125)

sns.countplot('age',data=fifa_20.head(20))

plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Which age belongs to Max players in Top 20')

plt.show()
plt.figure(dpi=125)

sns.countplot('preferred_foot',data=fifa_20.head(20))

plt.xlabel('Preferred Foot')

plt.ylabel('Count')

plt.title('Left Foot v Right Foot in Top 20')

Right,Left=fifa_20.head(20).preferred_foot.value_counts()

print('Left Preferred',Left)

print('Right Preferred',Right)

plt.show()
plt.figure(dpi=125)

x=fifa_20.head(20)['weight_kg']

y=fifa_20.head(20)['pace']



sns.regplot(x,y)

plt.title('Weight v Pace')

plt.xlabel('Weight')

plt.ylabel('Pace')

plt.show()
plt.figure(dpi=125)

x=fifa_20.head(20)['height_cm']

y=fifa_20.head(20)['pace']



sns.regplot(x,y)

plt.title('Height v Pace')

plt.xlabel('Height')

plt.ylabel('Pace')

plt.show()
plt.figure(dpi=125)

x=fifa_20.head(20)['BMI']

y=fifa_20.head(20)['pace']



sns.regplot(x,y)

plt.title('BMI v Pace')

plt.xlabel('BMI')

plt.ylabel('Pace')

plt.show()
plt.figure(dpi=125)

sns.stripplot(x = "potential", y = "short_name", data = fifa_20.head(20))

plt.xlabel('Potential')

plt.ylabel('Player Name')

plt.title('Player\'s Potential')

plt.show()



plt.figure(dpi=125)

sns.stripplot(x = "overall", y = "short_name", data = fifa_20.head(20))

plt.xlabel('Overall Rating')

plt.ylabel('Player Name')

plt.title('Player\'s Overall Rating')

plt.show()
plt.figure(dpi=125)

sns.stripplot(x = "pace", y = "short_name", data = fifa_20.head(20))

plt.xlabel('Pace')

plt.ylabel('Player Name')

plt.title('Player\'s Pace')

plt.show()
plt.figure(dpi=125)

sns.stripplot(x = "passing", y = "short_name", data = fifa_20.head(20))

plt.xlabel('Passing')

plt.ylabel('Player Name')

plt.title('Player\'s Passing Skill')

plt.show()
from wordcloud import WordCloud
plt.subplots(figsize=(20,8))

wordcloud = WordCloud(background_color='white',width=1920,height=1080).generate(" ".join(fifa_20.head(20)['club']))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('cast.png')

plt.show()
plt.subplots(figsize=(20,8))

wordcloud = WordCloud(background_color='white',width=1920,height=1080).generate(" ".join(fifa_20.head(20)['nationality']))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('cast.png')

plt.show()
plt.subplots(figsize=(20,8))

wordcloud = WordCloud(background_color='white',width=1920,height=1080).generate(" ".join(fifa_20.head(10)['short_name']))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('cast.png')

plt.show()