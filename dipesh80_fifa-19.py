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

import seaborn as sns 

import matplotlib.pyplot as plt

%matplotlib inline
fifa_df = pd.read_csv('../input/fifa19/data.csv')
fifa_df.shape
plt.figure(figsize=(20,10))

sns.heatmap(data=fifa_df.isna())

plt.show()
fifa_df.isna().any()
fifa_df.head()
fifa_df.columns
def county(x):

    return fifa_df[fifa_df['Nationality']==x][['Name','Age','Position','Overall']].head()

county('India')
#fifa_df['Club'].unique()

def club_details(x):

    return fifa_df[fifa_df['Club']==x][['Name','Jersey Number','Position','Overall','Age','Wage','Value','Contract Valid Until']]

club_details('Manchester United')
x = club_details('Manchester United')

x.shape
fifa_df.describe()
null_columns = fifa_df.columns[fifa_df.isnull().any()]

temp = fifa_df[null_columns].isnull().sum()

temp_df = pd.DataFrame(temp)

temp_df.columns = ['sum']

temp_df[temp_df['sum']>3600]

#as we can see that there is only one column who has more than 30% of null records 
# filling the missing value for the continous variables for proper data visualization



fifa_df['ShortPassing'].fillna(fifa_df['ShortPassing'].mean(), inplace = True)

fifa_df['Volleys'].fillna(fifa_df['Volleys'].mean(), inplace = True)

fifa_df['Dribbling'].fillna(fifa_df['Dribbling'].mean(), inplace = True)

fifa_df['Curve'].fillna(fifa_df['Curve'].mean(), inplace = True)

fifa_df['FKAccuracy'].fillna(fifa_df['FKAccuracy'], inplace = True)

fifa_df['LongPassing'].fillna(fifa_df['LongPassing'].mean(), inplace = True)

fifa_df['BallControl'].fillna(fifa_df['BallControl'].mean(), inplace = True)

fifa_df['HeadingAccuracy'].fillna(fifa_df['HeadingAccuracy'].mean(), inplace = True)

fifa_df['Finishing'].fillna(fifa_df['Finishing'].mean(), inplace = True)

fifa_df['Crossing'].fillna(fifa_df['Crossing'].mean(), inplace = True)

fifa_df['Weight'].fillna('200lbs', inplace = True)

fifa_df['Contract Valid Until'].fillna(2019, inplace = True)

fifa_df['Height'].fillna("5'11", inplace = True)

fifa_df['Loaned From'].fillna('None', inplace = True)

fifa_df['Joined'].fillna('Jul 1, 2018', inplace = True)

fifa_df['Jersey Number'].fillna(8, inplace = True)

fifa_df['Body Type'].fillna('Normal', inplace = True)

fifa_df['Position'].fillna('ST', inplace = True)

fifa_df['Club'].fillna('No Club', inplace = True)

fifa_df['Work Rate'].fillna('Medium/ Medium', inplace = True)

fifa_df['Skill Moves'].fillna(fifa_df['Skill Moves'].median(), inplace = True)

fifa_df['Weak Foot'].fillna(3, inplace = True)

fifa_df['Preferred Foot'].fillna('Right', inplace = True)

fifa_df['International Reputation'].fillna(1, inplace = True)

fifa_df['Wage'].fillna('€200K', inplace = True)
fifa_df.isna().any().sum()
fifa_df.fillna(0,inplace=True)
fifa_df.isna().any().sum()
fifa_df.shape
def defending(data):

    return int(round((data[['Marking', 'StandingTackle', 

                               'SlidingTackle']].mean()).mean()))



def general(data):

    return int(round((data[['HeadingAccuracy', 'Dribbling', 'Curve', 

                               'BallControl']].mean()).mean()))



def mental(data):

    return int(round((data[['Aggression', 'Interceptions', 'Positioning', 

                               'Vision','Composure']].mean()).mean()))



def passing(data):

    return int(round((data[['Crossing', 'ShortPassing', 

                               'LongPassing']].mean()).mean()))



def mobility(data):

    return int(round((data[['Acceleration', 'SprintSpeed', 

                               'Agility','Reactions']].mean()).mean()))

def power(data):

    return int(round((data[['Balance', 'Jumping', 'Stamina', 

                               'Strength']].mean()).mean()))



def rating(data):

    return int(round((data[['Potential', 'Overall']].mean()).mean()))



def shooting(data):

    return int(round((data[['Finishing', 'Volleys', 'FKAccuracy', 

                               'ShotPower','LongShots', 'Penalties']].mean()).mean()))
#adding some extras features into fifa_df that we have calcualted based on our analysis

fifa_df['Defending'] = fifa_df.apply(defending, axis = 1)

fifa_df['General'] = fifa_df.apply(general, axis = 1)

fifa_df['Mental'] = fifa_df.apply(mental, axis = 1)

fifa_df['Passing'] = fifa_df.apply(passing, axis = 1)

fifa_df['Mobility'] = fifa_df.apply(mobility, axis = 1)

fifa_df['Power'] = fifa_df.apply(power, axis = 1)

fifa_df['Rating'] = fifa_df.apply(rating, axis = 1)

fifa_df['Shooting'] = fifa_df.apply(shooting, axis = 1)
fifa_df['Nationality']
player = fifa_df[['Name','Age','Club','Nationality','Overall','Defending','General','Mental','Passing','Mobility','Power','Rating','Shooting']]
player.head()
sns.set()

#sns.set_style('Drakgrid')

plt.figure(figsize=(8,5))

sns.countplot(x=fifa_df['Preferred Foot'],palette = 'pink')

plt.title("Most Preferred Foot of Fifa Players")

plt.show()
fifa_df['International Reputation'].value_counts(normalize=True)*100
plt.figure(figsize=(9,9))

theme = plt.get_cmap('copper')

lables = ['1','2','3','4','5']

#sizes = [91, 6, 1, 0.28,0.032]

explode = [0.1, 0.1, 0.2, 0.5, 0.9]

size = fifa_df['International Reputation'].value_counts()

plt.pie(x=size,labels=lables,shadow=True,explode=explode, autopct='%1.1f%%',startangle=30)

plt.title('Internation Reputation of players')

plt.legend()

plt.show()
fifa_df['Weak Foot'].value_counts()
plt.figure(figsize=(9,9))

theme = plt.get_cmap('copper')

lables = ['1','2','3','4','5']

#sizes = [91, 6, 1, 0.28,0.032]

explode = [0,0,0,0.1,0.3]

size = fifa_df['Weak Foot'].value_counts()

plt.pie(x=size,labels=lables,shadow=True,explode=explode, autopct='%1.1f%%',startangle=30)

plt.title('Weak foot of players')

plt.legend()

plt.show()
#sns.barplot(x='Position',data=fifa_df)

plt.figure(figsize=(12,8))

#fifa_df['Position'].value_counts().plot(kind='bar')

#we can also do  by using countplot

sns.countplot(x='Position',data=fifa_df,palette = 'bone')

plt.xlabel('Positions of Players')

plt.ylabel('Count of players on specifc position')

plt.title('Postion Wise comparison')

plt.show()
#now cleaining the weight feature and removing lbs from it

fifa_df['Weight'] = fifa_df['Weight'].apply(lambda x:x.replace('lbs',''))
fifa_df['Weight'] = fifa_df['Weight'].astype(float)
fifa_df['Weight'].head()
def extracting_value(x):

    rm = x.replace('€','')

    if 'M' in rm:

        rm = float(rm.replace('M',''))*1000000

    elif 'K' in x:

        rm = float(rm.replace('K',''))*1000

    return(rm)
fifa_df['Value'] = fifa_df['Value'].apply(lambda x:extracting_value(x))

fifa_df['Wage'] = fifa_df['Wage'].apply(lambda x:extracting_value(x))
fifa_df['Wage'].head()
plt.figure(figsize=(10,5))

sns.distplot(fifa_df['Wage'], color = 'blue')

plt.xlabel('distribution of wages')

plt.ylabel('Count of the players')

plt.show()
plt.figure(figsize=(8,8))

sns.countplot(x='Skill Moves',data=fifa_df,palette = 'pastel')

plt.title('Count on the Basis of Skill Moves')

plt.xlabel('Skill Moves')

plt.ylabel('Count of skill Moves')

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(x='Height',data=fifa_df,palette ='dark')

plt.title("Distribution of the Player Base on Height")

plt.xlabel('Height Cateogry')

plt.ylabel('Count of players base of height distribution')

plt.show()
plt.figure(figsize=(10,8))

sns.distplot(fifa_df['Weight'],color='pink')

plt.title('Disturbution of players Height')

plt.xlabel('Hieght of players')

plt.ylabel('Count of the  players')

plt.show()
plt.figure(figsize=(10,8))

s=sns.countplot(x='Work Rate',data=fifa_df,palette = 'pastel')

plt.title('Work Rate of palyers')

s.set_xticklabels(s.get_xticklabels(),rotation=45)

plt.show()
plt.figure(figsize=(10,8))

fifa_df['Potential'].hist(bins=50)

plt.title('Histogram of players potential')

plt.show()
plt.figure(figsize=(10,8))

sns.distplot(fifa_df['Overall'],kde=False)

plt.title('Distributtion of Overall Scores of Players')

plt.show()
plt.style.use('dark_background')

plt.figure(figsize=(20,7))

fifa_df['Nationality'].value_counts().head(80).plot.bar(color = 'orange')

plt.title('Distribution of players based on Nationality top 80 cuntory',fontsize = 30, fontweight = 20)

plt.xlabel('Name of The Country')

plt.ylabel('count')

plt.show()
plt.figure(figsize=(10,8))

sns.distplot(fifa_df['Age'])
# selecting some of the interesting and important columns from the set of columns in the given dataset



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



data_selected = pd.DataFrame(fifa_df, columns = selected_columns)

data_selected.columns

sns.set_style('whitegrid')

plt.figure(figsize=(20,6))

sns.boxplot(x='Overall',y='Age',hue='Preferred Foot',data=fifa_df)

plt.show()