# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
players = pd.read_csv('../input/fifa19/data.csv')
# Functions that used to estimate some abilities of the players:

def Attack(players):

    attack_np = np.array(players[['Finishing','FKAccuracy','Penalties','ShotPower','HeadingAccuracy']])

    attack_np_pct = attack_np[0]*.3+attack_np[1]*.2+attack_np[2]*.15+attack_np[3]*.2+attack_np[4]*.15

    res = int(round(attack_np_pct))

    return res

def Speed(players):

    speed_np = np.array(players[['Acceleration','SprintSpeed','Agility','Dribbling']])

    speed_np_pct = speed_np[0] * .3 + speed_np[1] * .3 + speed_np[2] * .1 + speed_np[3] * .3

    res = int(round(speed_np_pct))

    return res

def Defends(players):

    defends_np = np.array(players[['Aggression','Interceptions','StandingTackle','SlidingTackle','Marking']])

    defends_np_pct = defends_np[0] * .15 + defends_np[1] * .15 + defends_np[2] * .30 + defends_np[3] * .30+defends_np[4]*.1

    res = int(round(defends_np_pct))

    return res

def Passing(players):

    passing_np = np.array(players[['ShortPassing','LongPassing','Crossing','Vision']])

    passing_np_pct = passing_np[0] * .30+ passing_np[1] * .30 + passing_np[2] * .30 +passing_np[3]*.1

    res = int(round(passing_np_pct))

    return res

def Technique(players):

    technique_np = np.array(players[['BallControl','Balance','Positioning','Curve']])

    technique_np_pct = technique_np[0] * .3 + technique_np[1] * .3 + technique_np[2] * .2 + technique_np[3]*.2

    res = int(round(technique_np_pct))

    return res

def Goalkeeping(players):

    Goalkeeping_np = np.array(players[['GKDiving', 'GKHandling','GKPositioning', 'GKReflexes']])

    Goalkeeping_np_pct = Goalkeeping_np[0] * .3 + Goalkeeping_np[1] * .3 + Goalkeeping_np[2] * .1 + Goalkeeping_np[3] * .3

    res = int(round(Goalkeeping_np_pct))

    return res
sns.heatmap(players.isnull(),yticklabels=False,cbar=False,cmap='cubehelix')
#Cleaning the Data:

#print(players.isna().sum())

players['ShotPower'].fillna(players['ShotPower'].mean(), inplace = True)

players['Overall'].fillna(players['Overall'].mean(), inplace = True)

players['Potential'].fillna(players['Potential'].mean(), inplace = True)

players['HeadingAccuracy'].fillna(players['HeadingAccuracy'].mean(), inplace = True)

players['Penalties'].fillna(players['Penalties'].mean(), inplace = True)

players['FKAccuracy'].fillna(players['FKAccuracy'].mean(), inplace = True)

players['Finishing'].fillna(players['Finishing'].mean(), inplace = True)

players['ShortPassing'].fillna(players['ShortPassing'].mean(), inplace = True)

players['Volleys'].fillna(players['Volleys'].mean(), inplace = True)

players['Dribbling'].fillna(players['Dribbling'].mean(), inplace = True)

players['SprintSpeed'].fillna(players['SprintSpeed'].mean(), inplace = True)

players['Agility'].fillna(players['Agility'].mean(), inplace = True)

players['Aggression'].fillna(players['Aggression'].mean(), inplace = True)

players['Interceptions'].fillna(players['Interceptions'].mean(), inplace = True)

players['Dribbling'].fillna(players['Dribbling'].mean(), inplace = True)

players['Interceptions'].fillna(players['Interceptions'].mean(), inplace = True)

players['StandingTackle'].fillna(players['StandingTackle'].mean(), inplace = True)

players['SlidingTackle'].fillna(players['SlidingTackle'].mean(), inplace = True)

players['Marking'].fillna(players['Marking'].mean(), inplace = True)

players['Vision'].fillna(players['Vision'].mean(), inplace = True)

players['Crossing'].fillna(players['Crossing'].mean(), inplace = True)

players['LongPassing'].fillna(players['LongPassing'].mean(), inplace = True)

players['Acceleration'].fillna(players['Acceleration'].mean(), inplace = True)

players['Curve'].fillna(players['Curve'].mean(), inplace = True)

players['FKAccuracy'].fillna(players['FKAccuracy'], inplace = True)

players['BallControl'].fillna(players['BallControl'].mean(), inplace = True)

players['Balance'].fillna(players['Balance'].mean(), inplace = True)

players['Positioning'].fillna(players['Positioning'].mean(), inplace = True)

players['GKDiving'].fillna(players['GKDiving'].mean(), inplace = True)

players['GKHandling'].fillna(players['GKHandling'].mean(), inplace = True)

players['GKPositioning'].fillna(players['GKPositioning'].mean(), inplace = True)

players['GKKicking'].fillna(players['GKKicking'].mean(), inplace = True)

players['GKReflexes'].fillna(players['GKReflexes'].mean(), inplace = True)

players['Weight'].fillna('165lbs', inplace = True)

players['Contract Valid Until'].fillna(2019, inplace = True)

players['Height'].fillna("5'11", inplace = True)

players['Loaned From'].fillna('None', inplace = True)

players['Joined'].fillna('None', inplace = True)

players['Jersey Number'].fillna(100, inplace = True)

players['Body Type'].fillna('Normal', inplace = True)

players['Position'].fillna('ST', inplace = True)

players['Club'].fillna('None', inplace = True)

players['Work Rate'].fillna('Medium/ Medium', inplace = True)

players['Skill Moves'].fillna(int(round(players['Skill Moves'].mean())), inplace = True)

players['Weak Foot'].fillna(3, inplace = True)

players['Preferred Foot'].fillna('Right', inplace = True)

players['International Reputation'].fillna(1, inplace = True)

players['Wage'].fillna('100000', inplace = True)
# Adding new columns to players by using the functions:

players['Attacking'] = players.apply(Attack,axis=1)

players['Speed'] = players.apply(Speed,axis=1)

players['Defending'] = players.apply(Defends,axis=1)

players['Passing'] = players.apply(Passing,axis=1)

players['Technique'] = players.apply(Technique,axis=1)

players['GoalKeeping'] = players.apply(Goalkeeping,axis=1)
# Changing str of wage into numeric

players['Wage'] = players['Wage'].str.replace('K','000')

players['Wage'] = players['Wage'].str.replace('€','')

players['Wage'] = pd.to_numeric(players['Wage'])



#Chaning str of value into numeric

players['Value'] = players['Value'].str.replace('K','000')

players['Value'] = players['Value'].str.replace('€','')

players['Value'] = players['Value'].str.replace('.','')

players['Value'] = players['Value'].str.replace('M','000000')

players['Value'] = pd.to_numeric(players['Value'])



#Changing str of weight into numeric

players['Weight'] = players['Weight'].str.replace('lbs','')

players['Weight'] = pd.to_numeric(players['Weight'])

#Players that can be bought with 10M Euros

players_u40 = players[players['Value']<=10000000]['Name']

print("List of players who can be bought with 10M Euros : \n '{}".format(players_u40.head(20)))

#Players who will end their contracts by next year(2020)

players_endcontract = players[players['Contract Valid Until']=='2020']['Name']

print("List of players who will end their contracts by 2020 : \n {}".format(players_endcontract.head(20)))

# Weight Distribution of the players:

plt.style.use('seaborn')

sns.distplot(players['Weight'],hist=20,kde=True,color='red')

plt.title('Distribution of The Weights of The Players')
# Players Value distribution

plt.figure(figsize=(10,5))

plt.style.use('_classic_test')

sns.distplot(players['Value'],bins=20,kde=True,color='green')

plt.title('Distribution of The Players Value')

plt.xlabel('Value')

plt.show()
#Top Striker(According to the Overall Rate) of Season 2019, Messi's Data:

Messi_data = players.iloc[0]

dic_Messi = {'Abilities':['Overall','Attack','Speed','Passing','Technique','Defending'],

       "Rate":Messi_data[['Overall','Attacking','Speed','Passing','Technique','Defending']]}

Messi = pd.DataFrame.from_dict(dic_Messi)

plt.style.use('seaborn-bright')

plt.figure(figsize=(10,5))

sns.barplot(x='Abilities',y='Rate',data=Messi,palette='viridis').set_title("Messi's Abilities Data")
#Top Midfielder(According to the Overall Rate) of Season 2019, Kevin DeBruyne's Data:

KDB_data = players.iloc[4]

dic_KDB = {'Abilities':['Overall','Attack','Speed','Passing','Technique','Defending'],

       "Rate":KDB_data[['Overall','Attacking','Speed','Passing','Technique','Defending']]}

KDB = pd.DataFrame.from_dict(dic_KDB)

plt.style.use('seaborn-bright')

plt.figure(figsize=(10,5))

sns.barplot(x='Abilities',y='Rate',data=Messi,palette='coolwarm').set_title("Kevin DeBruyne's Abilities Data")
#Top Defender(According to the Overall Rate) of Season 2019, Ramos's Data:

Ramos_data = players.iloc[8]

dic_Ramos = {'Abilities':['Overall','Attack','Speed','Passing','Technique','Defending'],

       "Rate":Ramos_data[['Overall','Attacking','Speed','Passing','Technique','Defending']]}

Ramos = pd.DataFrame.from_dict(dic_Ramos)

plt.style.use('seaborn-bright')

plt.figure(figsize=(10,5))

sns.barplot(x='Abilities',y='Rate',data=Ramos,palette='Set2').set_title("Ramos's Abilities Data")
#Top GoalKeeper(According to the Overall Rate) of Season 2019, DeGea's Data:

DeGea_data = players.iloc[3]

dic_DeGea = {'Abilities':['Overall','GKDiving', 'GKHandling','GKKicking', 'GKPositioning', 'GKReflexes'],

       "Rate":DeGea_data[['Overall','GKDiving', 'GKHandling','GKKicking', 'GKPositioning', 'GKReflexes']]}

DeGea = pd.DataFrame.from_dict(dic_DeGea)

plt.style.use('seaborn-bright')

plt.figure(figsize=(10,5))

sns.barplot(x='Abilities',y='Rate',data=DeGea,palette='deep').set_title("De Gea's Abilities Data")

#Number of Players in Each Ages

plt.style.use('ggplot')

plt.figure(figsize=(10,5))

sns.distplot(players['Age'],bins=30,color = 'blue')

plt.title(' Number of players in Each Ages')

plt.xlabel('Age')

plt.ylabel('Number of players')
#Number of players in Each countries

plt.style.use('seaborn-darkgrid')

plt.figure(figsize=(20,10))

sns.countplot(x='Nationality',data=players, palette='bright')

plt.title('Number of Players in Each Countries')

plt.xticks(rotation=90)

plt.xlabel('Nationalities')
#Number of players in Each Jersey Numbers

players['Jersey Number'] = players['Jersey Number'].astype(int)

plt.style.use('ggplot')

plt.figure(figsize=(20,10))

sns.countplot(x='Jersey Number',data= players, palette='dark')

plt.title('Count on  Jersey Numbers')

plt.xticks(rotation=90)

plt.xlabel('Jersey Numbers')
#Top 10(According to Overall Rates) players

plt.figure(figsize=(8,5))

plt.style.use('seaborn-muted')

top10 = players.sort_values('Overall',ascending=False).head(10)

sns.barplot(x='Name',y='Overall',data=top10,palette='deep')

plt.xticks(rotation=90)

plt.ylim(0,100)

plt.title('Top 10 Players Overall Rate')

plt.xlabel('Names')

plt.ylabel('Overall')
#Top 10(According to Overall Rates) players Value Distribution

plt.figure(figsize=(8,5))

plt.style.use('fivethirtyeight')

sns.lineplot(x='Name',y='Value',data=top10,palette='deep')

plt.xticks(rotation=90)

plt.title('Top 10 Overall Players Value')

plt.xlabel('Names')

plt.ylabel('Value')

#Top 10 Most Valued players

plt.figure(figsize=(8,5))

top10_value = players.sort_values('Value',ascending=False).head(10)

sns.barplot(x='Name',y='Value',data=top10_value,palette='deep')

plt.xticks(rotation=90)

plt.title('Top 10 Valued Players')

plt.xlabel('Names')

plt.ylabel('Value')
#Top 10 Most Valude Players Overall Rates

plt.figure(figsize=(8,5))

plt.style.use('ggplot')

sns.lineplot(x='Name',y='Overall',data=top10_value,palette='deep')

plt.xticks(rotation=90)

plt.ylim(0,100)

plt.title('Top 10 Valued Players Overall Rates')

plt.xlabel('Names')

plt.ylabel('Overall')
#Top 10(According its Overall Rates) Clubs

plt.figure(figsize=(10,5))

Club_top_10 = players.groupby('Club')['Overall'].mean().reset_index().sort_values('Overall', ascending=False).head(10)

plt.xticks(rotation=90)

plt.ylim(0,100)

plt.title(' Top 10 Overall Clubs')

sns.barplot(x='Club',y='Overall',data=Club_top_10,palette='deep')
#Top 10 Most Valued Clubs

plt.figure(figsize=(8,5))

plt.style.use('grayscale')

club = players.groupby('Club')['Value'].sum().reset_index().sort_values('Value', ascending=False).head(10)

plt.xticks(rotation=90)

plt.title(' Top 10 Most Valued Clubs')

sns.barplot(x='Club',y='Value',data=club,palette='deep')
#Top 10 National Teams

plt.figure(figsize=(8,5))

plt.style.use('seaborn-bright')

Nation_top_10 = players.groupby('Nationality')['Overall'].max().reset_index().sort_values('Overall', ascending=False).head(10)

plt.xticks(rotation=90)

plt.ylim(0,100)

plt.title(' Top 10 National Teams')

sns.barplot(x='Nationality',y='Overall',data=Nation_top_10,palette='deep')

topclubs_10_lis = []                       # To save the name of top 10 clubs in a list

for k in range(0,10):

    topclubs_10_lis.append(Club_top_10.iloc[k][0])
# Distribution of Top10 Clubs International Reputations

plt.style.use('Solarize_Light2')

Inter_Rep_data = players.loc[(players['Club'].isin(topclubs_10_lis)) & (players['International Reputation'])]

plt.xticks(rotation=90)

sns.violinplot(x='Club',y='International Reputation',data=Inter_Rep_data,palette='deep').set_title('Distribution of an International Reputation of Top10 Clubs')
# Distribution of Top 10 Clubs Skill Moves

skillmovesdata = players.loc[(players['Club'].isin(topclubs_10_lis)) & (players['Skill Moves'])]

plt.xticks(rotation=90)

plt.style.use('Solarize_Light2')

sns.violinplot(x='Club',y='Skill Moves',data=skillmovesdata,palette='deep').set_title('Skill Moves Distribution of Top10 Clubs ')
# Distribution of Top10 Clubs Weights

weightdata = players.loc[(players['Club'].isin(topclubs_10_lis)) & (players['Weight'])]

sns.boxplot(x='Club',y='Weight',data=weightdata,palette='deep').set_title('Weights Distribution of Top10 Clubs')

plt.xticks(rotation=90)

plt.xlabel('Top 10 Clubs')

plt.ylabel('Weights(lbs)')
#Values of Players According to their Preferred Foot

plt.style.use('ggplot')

plt.figure(figsize=(10,10))

sns.lmplot(x='Overall',y='Value',data=players,col='Preferred Foot')

plt.xlabel('Overall')

plt.xlim(0,100)

plt.ylabel('Value')

#Correlation between Long Shots Rate VS Free Kick Accuracy

sns.lmplot(x='LongShots',y='FKAccuracy',data=players,col='Preferred Foot')

plt.xlabel('LongShots')

plt.ylabel('FKAccuracy')

plt.xlim(0,100)

plt.ylim(0,100)



# Correlation Between Ball Control and Dribbling According to the Skill Moves that Players know

sns.lmplot(x='Dribbling',y='BallControl',data=players,col='Skill Moves')

plt.xlabel('Dribbling')

plt.ylabel('BallControl')

plt.xlim(0,100)

plt.ylim(0,100)

sns.lineplot(x='Age',y='Overall',data=players,color='red')

sns.lineplot(x='Age',y='Potential',data=players,color='blue')

plt.style.use('grayscale')

plt.xlabel('Age')

plt.ylabel('Overall')

plt.ylim(0,100)

plt.title('At What Age Do Players Performance and Potential Deteriorate?')
sns.lineplot(x='Height',y='SprintSpeed',data=players)

plt.style.use('seaborn-colorblind')

plt.xlabel('Height')

plt.ylabel('SprintSpeed')

plt.ylim(0,100)

plt.title('Do Heights Affect the SpringSpeed?')
sns.lineplot(x='Weight',y='BallControl',data=players,palette='viridis')

plt.style.use('ggplot')

plt.xticks(rotation=90)

plt.xlabel('Weight')

plt.ylabel('BallControl')

plt.ylim(0,100)

plt.title('Do Weights Affect the BallControl?')
sns.lineplot(x='Curve',y='FKAccuracy',data=players,palette='colorblind')

plt.xticks(rotation=90)

plt.xlabel('Curve')

plt.ylabel('FKAccuracy')

plt.title('Does Curving Ability Affect the Free Kick Accuracy?')
sns.lineplot(x='Strength',y='Interceptions',data=players,palette='Set1')

plt.xticks(rotation=90)

plt.xlabel('Strength')

plt.ylabel('Interception')

plt.xlim(0,100)

plt.ylim(0,100)

plt.title('Does Strength Affect Interception?')
# Heatmap on All the Quantitative Columns

plt.figure(figsize=(15,15))

sns.heatmap(players[['Age', 'Overall', 'Potential', 'Value',

                    'Wage', 'Special', 'International Reputation',

                    'Skill Moves', 'Work Rate', 'Height', 'Weight',

                    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

                    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

                    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

                    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

                    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

                    'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                    'GKKicking', 'GKPositioning', 'GKReflexes']].corr(),annot=True,cmap='YlGnBu')


korean_players = players[players['Nationality'] == 'Korea Republic']



fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(20,10))

plt.style.use('fivethirtyeight')

axes[0].boxplot(korean_players['Overall'])

axes[0].set_title('Korean Players Overall rate')

axes[0].set_ylabel('Overall Rate')





axes[1].hist(korean_players['Age'],bins= 20)

axes[1].set_title('Korean Players age distribution')

axes[1].set_ylabel('Number of players')

axes[1].set_xlabel('Age')



axes[2].boxplot(korean_players['Wage'])

axes[2].set_title('Korean Players Wage distribution')

axes[2].set_ylabel('Euro')
# Best Korean Player :

kor_top_index = korean_players['Overall'].idxmax()

kor_top = players.iloc[kor_top_index]
#Comparison of Korean Best player with Korean Players Average

kor_overall_mean = korean_players['Overall'].mean()

kor_top_overall = kor_top['Overall']



kor_attacking_mean = korean_players['Attacking'].mean()

kor_top_attacking = kor_top['Attacking']



kor_speed_mean = korean_players['Speed'].mean()

kor_top_speed = kor_top['Speed']





dic = {'Overall Rate' : [kor_top_overall,kor_overall_mean],'Name':[kor_top['Name'],'Kor players Avg'],

       'Attacking Ability': [kor_top_attacking,kor_attacking_mean],'Speed': [kor_top_speed,kor_speed_mean]}

kor_top_df = pd.DataFrame.from_dict(dic)
plt.style.use('ggplot')

plt.figure()

sns.barplot(x='Name',y='Overall Rate',data = kor_top_df,hue='Name',palette='rainbow').set_title("Heungmin Son's Overall rate vs Kor Avg Overall rate")

plt.figure()

sns.barplot(x='Name',y='Attacking Ability',data=kor_top_df,hue='Name',palette='rocket').set_title("Heungmin Son's Attack vs Kor Avg Attack")

plt.figure()

sns.barplot(x='Name',y='Speed',data=kor_top_df,hue='Name',palette='deep').set_title("Heungmin Son's Speed vs Kor Avg Speed" )
#Korea's Highest Potential player

kor_future_index = korean_players['Potential'].idxmax()

kor_future = players.iloc[kor_future_index]

print(kor_future)

dic2 = {"Abilities":['Potential','Attack','Speed','Passing','Technique','Defending'],

       "Rate":kor_future[['Potential','Attacking','Speed','Passing','Technique','Defending']]}

kor_future_df = pd.DataFrame.from_dict(dic2)

plt.style.use('Solarize_Light2')

sns.barplot(x='Abilities',y='Rate',data=kor_future_df,palette='rainbow').set_title("KangIn Lee's Data")
# Predicting KangIn Lee's Growth by Linear Regression

from sklearn.model_selection import train_test_split

X = korean_players[['Attacking','Speed','Defending','Passing','Technique','GoalKeeping','Potential']]

y = korean_players['Overall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

predictions = lm.predict(X_test)

coef_kor_players = pd.DataFrame(lm.coef_,index = X.columns,columns=['Coef'])

print(coef_kor_players)

plt.figure()

plt.scatter(y_test,predictions)

plt.title('Will KangIn Lee grow more?')
#Tottenham Hotspur team:

spurs = players[players['Club']=='Tottenham Hotspur']

spurs_value = spurs['Value'].sum()

print('Tottenham HotSpur Total Team Values are €{}'.format(spurs_value))
# Data of Spurs 

plt.figure(figsize=(20,4))

plt.style.use('ggplot')

sns.countplot(x='Nationality',data=spurs,palette='coolwarm').set_title('Spurs Players Nationalities')

plt.figure()

sns.countplot(x='Position',data=spurs,palette='viridis').set_title('Number of Spurs Players In Each Postitions')

plt.figure()

sns.countplot(x='Preferred Foot',data=spurs,hue='Preferred Foot',palette='Set2').set_title('Preferred Foot of Each Spurs Players')

plt.figure()

sns.countplot(x='Age',data = spurs,palette='Set1').set_title('Age Distribution of Tottenham Hotspur')
# Top player(According to the Overall Rates) of the team



Top_spurs_index = spurs['Overall'].idxmax()

Top_spurs = players.iloc[Top_spurs_index]

print("The top player of Tottenham Hotspur is: {}".format(Top_spurs['Name']))

print("Harry Kane's Values are: €{0}".format(Top_spurs['Value']))

dic_kane = {"Abilities":['Overall','Attack','Speed','Passing','Technique'],

      "Rate":Top_spurs[['Overall','Attacking','Speed','Passing','Technique']]}

Top_spurs_df = pd.DataFrame.from_dict(dic_kane)

plt.style.use('seaborn-pastel')

sns.barplot(x='Abilities',y='Rate',data=Top_spurs_df).set_title("Harry Kane's Data")
# The Oldest an the Youngest Players in Tottenham Hotspurs

spurs_youngest_index = spurs['Age'].idxmin()

spurs_youngest = players.iloc[spurs_youngest_index]['Name']

spurs_oldest_index = spurs['Age'].idxmax()

spurs_oldest = players.iloc[spurs_oldest_index]['Name']

print("The oldest player in spurs is {0} and the youngest player is {1}".format(spurs_oldest,spurs_youngest))
# Best Spurs Players in Each Categories(Attack,Speed,Passing,Technique,Defends,GoalKeeping)

spurs_attack_mean = spurs['Attacking'].mean()

spurs_pass_mean = spurs['Passing'].mean()

spurs_speed_mean = spurs['Speed'].mean()

spurs_defends_mean = spurs['Defending'].mean()

spurs_technique_mean = spurs['Technique'].mean()

spurs_goalkeeping_mean = spurs[spurs['Position']=='GK']['GoalKeeping'].mean()



spurs_best_attack = players.iloc[spurs['Attacking'].idxmax()]

spurs_best_speed  = players.iloc[spurs['Speed'].idxmax()]

spurs_best_pass   = players.iloc[spurs['Passing'].idxmax()]

spurs_best_tech   = players.iloc[spurs['Technique'].idxmax()]

spurs_best_defends = players.iloc[spurs['Defending'].idxmax()]

spurs_best_keeper = players.iloc[spurs['GoalKeeping'].idxmax()]

category_dic = {"Categories": ['Attack(H. Kane)','Attack(H. Kane)','Speed(Lucas Moura)','Speed(Lucas Moura)','Pass(C. Eriksen)','Pass(C. Eriksen)','Technique(C. Eriksen)','Technique(C. Eriksen)','Defends(J. Vertonghen)','Defends(J. Vertonghen)','GoalKeeping(H. Lloris)','GoalKeeping(H. Lloris)'],

                "Rate": [spurs_best_attack['Attacking'],spurs_attack_mean,spurs_best_speed['Speed'],spurs_speed_mean,

                     spurs_best_pass['Passing'],spurs_pass_mean,spurs_best_tech['Technique'],spurs_technique_mean,

                     spurs_best_defends['Defending'],spurs_defends_mean,spurs_best_keeper['GoalKeeping'],spurs_goalkeeping_mean],

                "Rate of: ":["Best","Mean","Best","Mean","Best","Mean","Best","Mean",

                         "Best","Mean","Best","Mean"]}



spurs_cat_df = pd.DataFrame.from_dict(category_dic)

plt.figure(figsize=(20,5))

plt.style.use('bmh')

sns.barplot(x='Categories',y='Rate',hue = 'Rate of: ',data=spurs_cat_df,palette='bright').set_title("Best Player's Each Rates VS Spurs Team Avg")

plt.legend(loc='upper right')
# Some Graphs on Tottenham Hotspurs

plt.xticks(rotation=90)

sns.barplot(x='Categories',y='Rate',hue = 'Rate of: ',data=spurs_cat_df,palette='Set2').set_title("Best Players of Each Categories VS Spurs Team Avg")

plt.legend(prop={'size': 10})

plt.figure(figsize=(10,4))

sns.scatterplot(x='Wage',y='Overall',palette='deep',data=spurs).set_title('Do Spurs Paying Their Players Correctly?')

plt.figure(figsize=(10,4))

sns.scatterplot(x='Wage',y='Potential',palette='deep',data=spurs).set_title("Does the Potential Affects The Player's Wage?")



sns.scatterplot(x='Value',y='Wage',data=spurs,palette='coolwarm').set_title("Do Players Get Paid According To Their Value?Value")
#Finding the Best 11 of Tottenham Hotspurs

j=0

lis =[]

for i in spurs['Position']:

    if i in ('LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW'):



            lis.append(1)

    elif i in ('LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',

                                         'CDM', 'RDM'):



        lis.append(2)



    elif i in ('RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB'):

        lis.append(3)

    else:

        lis.append(4)

    j+=1

forward =[]

midfield =[]

defender = []

goalie = []

j=0



for i in lis:



    if i == 1:

        forward.append(j)           # putting the index of the best attcker in a list

    elif i == 2:

        midfield.append(j)          # putting the index of the best attcker in a list

    elif i == 3:

        defender.append(j)          # putting the index of the best attcker in a list

    else:

        goalie.append(j)            

    j+=1



spurs11 = []

max_for = 0



for i in forward:

    index_ = 0

    if(spurs.iloc[i]['Overall'] > max_for):

        max_for = spurs.iloc[i]['Overall']

        max_index = index_

    index_+=1



spurs11.append(spurs.iloc[max_index]['Name'])





for k in range(0,5):

    max_mid = 0

    index = 0

    for i in midfield:



        if(spurs.iloc[i]['Overall'] > max_mid):

            max_mid = spurs.iloc[i]['Overall']

            max_index2 = i

            pop_index = index

        index +=1



    spurs11.append(spurs.iloc[max_index2]['Name'])

    midfield.pop(pop_index)

for k in range(0,4):

    max_def = 0

    index_2 = 0

    for i in defender:



        if(spurs.iloc[i]['Overall'] > max_def):

            max_def = spurs.iloc[i]['Overall']



            max_index3 = i

            pop_index2 = index_2

        index_2 = index_2 + 1

    spurs11.append(spurs.iloc[max_index3]['Name'])

    defender.pop(pop_index2)



spurs11.append(spurs_best_keeper['Name'])

print("Spurs Best 11 are: ", " ,".join(spurs11))
