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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')
import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv('/kaggle/input/fifa19/data.csv')
df.head()
df.info()
df.drop(['Unnamed: 0', 'ID', 'Photo', 'Flag','Club Logo'], axis=1,inplace=True)
df['International Reputation'].fillna(1,inplace=True)

df['Weak Foot'].fillna(1,inplace=True)

df['Skill Moves'].fillna(1,inplace=True)

df['Height'].fillna("5'9",inplace=True)

df['Weight'].fillna("198lbs",inplace=True)

df['Crossing'].fillna(df['Crossing'].mean(),inplace=True)

df['Finishing'].fillna(df['Finishing'].mean(),inplace=True)

df['HeadingAccuracy'].fillna(df['HeadingAccuracy'].mean(),inplace=True)

df['ShortPassing'].fillna(df['ShortPassing'].mean(),inplace=True)

df['Volleys'].fillna(df['Volleys'].mean(),inplace=True)

df['Dribbling'].fillna(df['Dribbling'].mean(),inplace=True)

df['FKAccuracy'].fillna(df['FKAccuracy'].mean(),inplace=True)

df['LongPassing'].fillna(df['LongPassing'].mean(),inplace=True)

df['BallControl'].fillna(df['BallControl'].mean(),inplace=True)

df['Acceleration'].fillna(df['Acceleration'].mean(),inplace=True)

df['SprintSpeed'].fillna(df['SprintSpeed'].mean(),inplace=True)

df['Agility'].fillna(df['Agility'].mean(),inplace=True)

df['Balance'].fillna(df['Balance'].mean(),inplace=True)

df['Jumping'].fillna(df['Jumping'].mean(),inplace=True)

df['Stamina'].fillna(df['Stamina'].mean(),inplace=True)

df['Strength'].fillna(df['Strength'].mean(),inplace=True)

df['Position'].fillna('ST', inplace = True)

df['ShotPower'].fillna(df['ShotPower'].mean(),inplace=True)

df['Reactions'].fillna(df['Reactions'].mean(),inplace=True)

df['Preferred Foot'].fillna('Right', inplace = True)

df['Wage'].fillna('€200K', inplace = True)

df['Work Rate'].fillna('Medium/Medium', inplace = True)

df['Marking'].fillna(df['Marking'].mean(), inplace = True)

df['StandingTackle'].fillna(df['StandingTackle'].mean(), inplace = True)

df['SlidingTackle'].fillna(df['SlidingTackle'].mean(), inplace = True)
df.fillna(0, inplace=True)
def extract(Value):

    col = Value.replace('€', '')

    if 'M' in col:

        col = float(col.replace('M', ''))*1000000

    elif 'K' in Value:

        col = float(col.replace('K', ''))*1000

    return float(col)
df['Value']=df['Value'].apply(extract)

df['Wage']=df['Wage'].apply(extract)
plt.figure(figsize=(12,6))

sns.countplot(df['Age'],palette='viridis')

plt.title('Count Of Age')

plt.xlabel('Age')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=(12,6))

df['Nationality'].value_counts().head(50).plot.bar(cmap='inferno')

plt.title('Count Of Nationality')

plt.xlabel('Nationality')

plt.ylabel('Count')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(12,6))

sns.distplot(df['Overall'],color='k',kde=False)

plt.title('Distribution Of Overall')

plt.xlabel('Overall')

plt.show()
plt.figure(figsize=(12,6))

sns.distplot(df['Potential'],kde=False,color='b')

plt.title('Distribution Of Potential')

plt.xlabel('Potential')

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(df['Skill Moves'],palette='bone')

plt.title('Count Of Skill Moves')

plt.xlabel('Skill Moves')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(df['Work Rate'],palette='rocket')

plt.title('Count Of Work Rate')

plt.xlabel('Work Rate')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(df['Position'],palette='icefire_r')

plt.title('Count Of Position')

plt.xlabel('Position')

plt.ylabel('Count')

plt.show()
col=df[['Age','Overall','BallControl','SprintSpeed','Agility','Balance',

        'Acceleration','Stamina']]

sns.pairplot(col)
sns.set(palette='Dark2')

sns.lmplot(x='Acceleration',y='SprintSpeed',data=df,scatter_kws={'color':'cyan','edgecolor':'blue','linewidth':'0.7'},

           line_kws={'color':'black'})
sns.lmplot(x='BallControl',y='Dribbling',data=df,col='Preferred Foot',

           scatter_kws={'color':'cyan','edgecolor':'blue','linewidth':'0.7'},

           line_kws={'color':'black'})
plt.figure(figsize=(9,9))

size=df['International Reputation'].value_counts()

label=['1','2','3','4','5']

colors=plt.cm.Wistia(np.linspace(0, 1, 5))

explode=[0.1,0.1,0.2,0.6,1]

plt.pie(size,labels=label,colors=colors,explode=explode,shadow=True)

plt.title('International Reputation')

plt.show()
plt.figure(figsize=(9,9))

size=df['Weak Foot'].value_counts()

label=['3','2','4','5','1']

colors=plt.cm.Wistia(np.linspace(0, 1, 5))

explode=[0,0.04,0.06,0.2,0.6]

plt.pie(size,labels=label,colors=colors,explode=explode,shadow=True)

plt.title('Weak Foot')

plt.show()
size=df['Preferred Foot'].value_counts()

label=['Right','Left']

colors=plt.cm.Wistia(np.linspace(0, 1, 5))

explode=[0.04,0.04]

fig,ax=plt.subplots(1,2,figsize=(14,7))

_=sns.countplot('Preferred Foot',data=df,ax=ax[0], palette='Wistia')

_=plt.pie(size,labels=label,colors=colors,explode=explode,shadow=True,autopct='%.2f%%')

plt.title('Preferred Foot')

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(df['Height'],palette='bone')

plt.title('Count Of Height')

plt.xlabel('Height')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=(17,6))

sns.countplot(df['Weight'],palette='bone')

plt.title('Count Of Weight')

plt.xlabel('Position')

plt.ylabel('Count')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(12,6))

sns.set_style('whitegrid')

sns.distplot(df['Wage'],color='black')

plt.title('Distribution of Wages')

plt.xticks(rotation=90)

plt.show()
def technical(df):

    return int(round((df[['Crossing','Dribbling','Finishing','HeadingAccuracy','LongShots','Marking',

                          'FKAccuracy','LongPassing','ShortPassing','StandingTackle',

                          'SlidingTackle','BallControl']].mean()).mean()))

def mental(df):

    return int(round((df[['Aggression','Composure','Positioning','Vision',

                          'Interceptions']].mean()).mean()))



def physical(df):

    return int(round((df[['Acceleration','Agility','Balance','Jumping','Stamina',

                          'Strength','Reactions','ShotPower']].mean()).mean()))
df['Technical']=df.apply(technical,axis=1)

df['Mental']=df.apply(mental,axis=1)

df['Physical']=df.apply(physical,axis=1)
player_data=df[['Name','Age','Nationality','Overall','Club','Technical','Mental',

                'Physical','Position','Potential','Wage']]

player_data.head()
df.drop(['Technical','Mental','Physical'],axis=1,inplace=True)
player_data.iloc[player_data.groupby(player_data['Overall'])['Potential'].idxmax()][['Position',

                                                                 'Name', 'Age', 'Club', 'Nationality',

                                                                                      'Technical','Mental','Physical']].head(10)
player_data.iloc[player_data.groupby(player_data['Position'])['Potential'].idxmax()][['Position',

                                                                 'Name', 'Age', 'Club', 'Nationality',

                                                                                      'Technical','Mental','Physical']].head(10)
player_data[player_data['Wage']>340000].sort_values(by='Wage',ascending=False).head(10)
player_data.sort_values('Age',ascending=False).head(10)
player_data.sort_values('Age').head(10)
plt.figure(figsize=(16,12))

sns.heatmap(df[['Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value','Wage', 

                'Special', 'Preferred Foot', 'Weak Foot', 'Skill Moves', 'Work Rate', 'Body Type',

                'Height', 'Weight','Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',

                'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',

       'Acceleration', 'SprintSpeed', 'Agility', 'Balance',

       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',

       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',

       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',

       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']].corr(),linecolor='black',linewidths=0.1)

plt.show()
df['Nationality'].value_counts().head(10)
nation=('England','Germany','Spain','Argentina','France','Brazil','Italy','Columbia','Japan','Netherlands')

df_nation=df.loc[df['Nationality'].isin(nation) & df['Overall']]
plt.figure(figsize=(12,6))

sns.barplot(x=df_nation['Nationality'],y=df_nation['Overall'], palette='Wistia')

plt.title('Position')

plt.show()
nation=('England','Germany','Spain','Argentina','France','Brazil','Italy','Columbia','Japan','Netherlands')

df_nation=df.loc[df['Nationality'].isin(nation) & df['Potential']]



plt.figure(figsize=(12,6))

sns.barplot(x=df_nation['Nationality'],y=df_nation['Potential'], palette='inferno')

plt.title('Nationality vs Potential')

plt.xlabel('Nationality')

plt.ylabel('Potential')

plt.show()
nation=('England','Germany','Spain','Argentina','France','Brazil','Italy','Columbia','Japan','Netherlands')

df_nation=df.loc[df['Nationality'].isin(nation) & df['Overall']]



plt.figure(figsize=(12,6))

sns.lineplot(x=df_nation['Nationality'],y=df_nation['Age'])

plt.title('Nationality vs Age')

plt.xlabel('Nationality')

plt.ylabel('Age')

plt.show()
nation=('England','Germany','Spain','Argentina','France','Brazil','Italy','Columbia','Japan','Netherlands')

player_data_nation=player_data.loc[df['Nationality'].isin(nation) & player_data['Technical']]



plt.figure(figsize=(12,6))

sns.boxplot(x=player_data_nation['Nationality'],y=player_data_nation['Technical'], palette='viridis')

plt.title('Nationality vs Technical Skills')

plt.xlabel('Nationality')

plt.ylabel('Technical Skills')

plt.show()
nation=('England','Germany','Spain','Argentina','France','Brazil','Italy','Columbia','Japan','Netherlands')

player_data_nation=player_data.loc[df['Nationality'].isin(nation) & player_data['Mental']]



plt.figure(figsize=(12,6))

sns.boxplot(x=player_data_nation['Nationality'],y=player_data_nation['Mental'], palette='viridis')

plt.title('Nationality vs Mental Skills')

plt.xlabel('Nationality')

plt.ylabel('Mental Skills')

plt.show()
nation=('England','Germany','Spain','Argentina','France','Brazil','Italy','Columbia','Japan','Netherlands')

player_data_nation=player_data.loc[df['Nationality'].isin(nation) & player_data['Physical']]



plt.figure(figsize=(12,6))

sns.boxplot(x=player_data_nation['Nationality'],y=player_data_nation['Physical'], palette='plasma')

plt.title('Nationality vs Physical Skills')

plt.xlabel('Nationality')

plt.ylabel('Physical Skills')

plt.show()
player_data[player_data['Nationality']=='India'][['Name','Age','Overall',

                                                  'Technical','Mental',

                                                  'Physical','Position']].sort_values(by='Age',ascending=False)
df['Club'].value_counts().head(10)
clubs=('CD Leganés','Newcastle United','Cardiff City','Everton','TSG 1899 Hoffenheim','Southampton',

      'Tottenham Hotspur','Valencia CF','Wolverhampton Wanderers')

df_club=df.loc[df['Club'].isin(clubs) & df['Overall']]



plt.figure(figsize=(14,6))

sns.barplot(x=df_club['Club'],y=df_club['Overall'],palette='twilight')

plt.xticks(rotation=90)

plt.title('Club vs Ovreall')

plt.xlabel('Club')

plt.ylabel('Overall')

plt.show()
clubs=('CD Leganés','Newcastle United','Cardiff City','Everton','TSG 1899 Hoffenheim','Southampton',

      'Tottenham Hotspur','Valencia CF','Wolverhampton Wanderers')

df_club=player_data.loc[player_data['Club'].isin(clubs) & player_data['Age']]



plt.figure(figsize=(14,6))

sns.lineplot(x=df_club['Club'],y=df_club['Age'],palette='plasma')

plt.xticks(rotation=90)

plt.title('Club vs Age')

plt.xlabel('Club')

plt.ylabel('Age')

plt.show()
clubs=('CD Leganés','Newcastle United','Cardiff City','Everton','TSG 1899 Hoffenheim','Southampton',

      'Tottenham Hotspur','Valencia CF','Wolverhampton Wanderers')

df_club=df.loc[df['Club'].isin(clubs) & df['Potential']]



plt.figure(figsize=(14,6))

sns.barplot(x=df_club['Club'],y=df_club['Potential'],palette='twilight')

plt.xticks(rotation=90)

plt.title('Club vs Potential')

plt.xlabel('Club')

plt.ylabel('Potential')

plt.show()
clubs=('CD Leganés','Newcastle United','Cardiff City','Everton','TSG 1899 Hoffenheim','Southampton',

      'Tottenham Hotspur','Valencia CF','Wolverhampton Wanderers')

df_club=player_data.loc[player_data['Club'].isin(clubs) & player_data['Technical']]



plt.figure(figsize=(14,6))

sns.boxplot(x=df_club['Club'],y=df_club['Technical'],palette='twilight')

plt.xticks(rotation=90)

plt.title('Club vs Technical Skills')

plt.xlabel('Club')

plt.ylabel('Technical Skills')

plt.show()
clubs=('CD Leganés','Newcastle United','Cardiff City','Everton','TSG 1899 Hoffenheim','Southampton',

      'Tottenham Hotspur','Valencia CF','Wolverhampton Wanderers')

df_club=player_data.loc[player_data['Club'].isin(clubs) & player_data['Mental']]



plt.figure(figsize=(14,6))

sns.boxplot(x=df_club['Club'],y=df_club['Mental'],palette='plasma')

plt.xticks(rotation=90)

plt.title('Club vs Mental Skills')

plt.xlabel('Club')

plt.ylabel('Mental Skills')

plt.show()
clubs=('CD Leganés','Newcastle United','Cardiff City','Everton','TSG 1899 Hoffenheim','Southampton',

      'Tottenham Hotspur','Valencia CF','Wolverhampton Wanderers')

df_club=player_data.loc[player_data['Club'].isin(clubs) & player_data['Physical']]



plt.figure(figsize=(14,6))

sns.boxplot(x=df_club['Club'],y=df_club['Physical'],palette='plasma')

plt.xticks(rotation=90)

plt.title('Club vs Physical Skills')

plt.xlabel('Club')

plt.ylabel('Physical Skills')

plt.show()
clubs=('CD Leganés','Newcastle United','Cardiff City','Everton','TSG 1899 Hoffenheim','Southampton',

      'Tottenham Hotspur','Valencia CF','Wolverhampton Wanderers')

df_club=player_data.loc[player_data['Club'].isin(clubs) & player_data['Wage']]



plt.figure(figsize=(14,6))

sns.violinplot(x=df_club['Club'],y=df_club['Wage'],palette='coolwarm')

plt.xticks(rotation=90)

plt.title('Club vs Wage')

plt.xlabel('Club')

plt.ylabel('Wage')

plt.show()
player_features = ('Acceleration', 'Aggression', 'Agility', 

                   'Balance', 'BallControl', 'Composure', 

                   'Crossing', 'Dribbling', 'FKAccuracy', 

                   'Finishing', 'GKDiving', 'GKHandling', 

                   'GKKicking', 'GKPositioning', 'GKReflexes', 

                   'HeadingAccuracy', 'Interceptions', 'Jumping', 

                   'LongPassing', 'LongShots', 'Marking', 'Penalties')
from math import pi



idx = 1

plt.figure(figsize=(15,45))

for position_name, features in df.groupby(df['Position'])[player_features].mean().iterrows():

    top_features = dict(features.nlargest(5))

    

    # number of variable

    categories=top_features.keys()

    N = len(categories)



    # We are going to plot the first line of the data frame.

    # But we need to repeat the first value to close the circular graph:

    values = list(top_features.values())

    values += values[:1]



    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)

    angles = [n / float(N) * 2 * pi for n in range(N)]

    angles += angles[:1]



    # Initialise the spider plot

    ax = plt.subplot(9, 3, idx, polar=True)



    # Draw one axe per variable + add labels labels yet

    plt.xticks(angles[:-1], categories, color='grey', size=8)



    # Draw ylabels

    ax.set_rlabel_position(0)

    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)

    plt.ylim(0,100)

    

    plt.subplots_adjust(hspace = 0.5)

    

    # Plot data

    ax.plot(angles, values, linewidth=1, linestyle='solid')



    # Fill area

    ax.fill(angles, values, 'b', alpha=0.1)

    

    plt.title(position_name, size=11, y=1.1)

    

    idx += 1 