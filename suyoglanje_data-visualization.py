import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/fifa19/data.csv")
data.head()
data.describe()
data.describe(include='object')
print(data.isnull().sum())
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



data['Weight'].fillna(data['Weight'].mode, inplace = True)



data['Contract Valid Until'].fillna(2019, inplace = True)



data['Height'].fillna(data['Height'].mode, inplace = True)



data['Loaned From'].fillna('None', inplace = True)



data['Joined'].fillna('Jan 1, 2019', inplace = True)



data['Jersey Number'].fillna(0, inplace = True)



data['Body Type'].fillna('Normal', inplace = True)



data['Position'].fillna('ST', inplace = True)



data['Club'].fillna('No Club', inplace = True)



data['Work Rate'].fillna('Medium/ Medium', inplace = True)



data['Skill Moves'].fillna(data['Skill Moves'].median(), inplace = True)



data['Weak Foot'].fillna(3, inplace = True)



data['Preferred Foot'].fillna('Right', inplace = True)



data['International Reputation'].fillna(1, inplace = True)



data['Wage'].fillna(data['Wage'].mode, inplace = True)
data.fillna(0, inplace = True)
plt.figure(figsize=(12,6))



sns.countplot(data['Preferred Foot'],palette = 'Blues')

plt.title('Most Preferred Foot by the Players',fontsize= 22)

plt.show()
labels = ['1','2','3','4','5']



sizes = data['International Reputation'].value_counts()



plt.figure(figsize = (12,6))

explode = [0.1,0.1,0.4,0.5,1]



plt.pie(sizes,labels = labels ,explode = explode,shadow= True)



plt.title('International Reputation of the Players',fontsize = 22)

plt.legend()

plt.show()
labels = ['5','4','3','2','1']



sizes = data['Weak Foot'].value_counts()



plt.figure(figsize = (12,6))

explode = [0.1,0.1,0.2,0.3,0.4]



plt.pie(sizes,labels = labels ,explode = explode,shadow= True)



plt.title('Distribution of Week Foot among Players',fontsize = 22)

plt.legend()

plt.show()
plt.figure(figsize = (15, 6))



ax = sns.countplot('Position',data=data,palette = 'Blues')



ax.set_xlabel(xlabel = 'Positions of Football Players',fontsize = 18)

ax.set_ylabel(ylabel = 'Count of Players',fontsize = 18)

ax.set_title(label = 'Comparison of Positions of Players',fontsize = 22)



plt.show()
plt.figure(figsize = (15, 6))



ax = sns.countplot(x = 'Skill Moves', data = data, palette = 'Blues')



ax.set_title(label = 'Count of players on Basis of their skill moves', fontsize = 22)



ax.set_xlabel(xlabel = 'Number of Skill Moves', fontsize = 18)



ax.set_ylabel(ylabel = 'Count', fontsize = 18)



plt.show()
def wage(value):

    wage = value.replace('€','')

    

    if 'M' in wage:

        wage = float(wage.replace('M',''))*1000000

    elif 'K' in wage:

        wage = float(wage.replace('K',''))*1000

        

    return float(wage)
data['Wage'] = data['Wage'].apply(lambda x : wage(x))



data['Wage'].head()
plt.figure(figsize=(15,6))



sns.lineplot(data = data['Wage'])



plt.title("Wages of Players")



plt.show()
plt.figure(figsize = (15, 6))



sns.countplot(x = 'Work Rate', data = data)



plt.title('Different work rates of the Players Participating in the FIFA 2019', fontsize = 22)



plt.xlabel('Work rates associated with the players', fontsize = 18)



plt.ylabel('count of Players', fontsize = 18)



plt.show()


x = data.Special

plt.figure(figsize = (15, 6))



ax = sns.distplot(x, bins = 58, kde = True)

ax.set_title(label = 'Histogram for the Speciality Scores of the Players', fontsize = 22)

ax.set_xlabel(xlabel = 'Special score range', fontsize = 18)

ax.set_ylabel(ylabel = 'Count of the Players',fontsize = 18)



plt.show()
x = data.Potential

plt.figure(figsize = (15, 6))





ax = sns.distplot(x, bins = 58, kde = False)

ax.set_title(label = 'Histogram of players Potential Scores', fontsize = 20)

ax.set_xlabel(xlabel = "Player\'s Potential Scores", fontsize = 16)

ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)



plt.show()


x = data.Overall

plt.figure(figsize = (15, 6))





ax = sns.distplot(x, bins = 52, kde = True, color = 'r')

ax.set_title(label = 'Histogram of players Overall Scores', fontsize = 22)

ax.set_xlabel(xlabel = "Player\'s Scores", fontsize = 18)

ax.set_ylabel(ylabel = 'Number of players', fontsize = 18)



plt.show()
plt.figure(figsize = (15, 8))



plt.style.use('dark_background')

data['Nationality'].value_counts().head(80).plot.bar(color = 'red', figsize = (20, 7))

plt.title('Different Nations Participating in FIFA 2019', fontsize = 30, fontweight = 20)

plt.xlabel('Name of The Country')

plt.ylabel('count')

plt.show()
x = data.Age

plt.figure(figsize = (15, 6))

ax = sns.distplot(x, bins = 58, kde = False, color = 'r')

ax.set_title(label = 'Histogram of players age', fontsize = 22)

ax.set_xlabel(xlabel = "Player\'s age", fontsize = 18)

ax.set_ylabel(ylabel = 'Number of players', fontsize = 18)



plt.show()
selected_columns = ['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',

                    'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot',

                    'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Height', 'Weight',

                    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

                    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

                    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

                    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

                    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

                    'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                    'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']



data_selected = pd.DataFrame(data, columns = selected_columns)

data_selected.columns
plt.figure(figsize = (30, 20))

sns.heatmap(data_selected[['Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',

                    'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot',

                    'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Height', 'Weight',

                    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

                    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

                    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

                    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

                    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

                    'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                    'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']].corr(), annot = True)



plt.title('Histogram of the Dataset', fontsize = 30)

plt.show()

some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')



data_clubs = data.loc[data['Club'].isin(some_clubs) & data['Overall']]



plt.figure(figsize = (15, 6))

ax = sns.boxplot(x = data_clubs['Club'], y = data_clubs['Overall'])

ax.set_title(label = 'Distribution of Overall Score in Different popular Clubs', fontsize = 22)

ax.set_xlabel(xlabel = 'Some Popular Clubs', fontsize = 16)

ax.set_ylabel(ylabel = 'Overall Score', fontsize = 16)



plt.xticks(rotation = 90)

plt.show()