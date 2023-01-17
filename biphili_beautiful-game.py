# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

plt.style.use('fivethirtyeight')
df=pd.read_csv('../input/fifa19/data.csv')
df.head()
df.info()
print('Rows     :',df.shape[0])

print('Columns  :',df.shape[1])

print('\nFeatures :\n     :',df.columns.tolist())

print('\nMissing values    :',df.isnull().values.sum())

print('\nUnique values :  \n',df.nunique())
df.select_dtypes(exclude=['int','float']).columns
print(df['Position'].unique())

print(df['Photo'].unique())

print(df['Release Clause'].unique())
df.isnull().sum()
#df.fillna(main_df.mean(),inplace=True)
# filling the missing value for the continous variables for proper data visualization



df['ShortPassing'].fillna(df['ShortPassing'].mean(), inplace = True)

df['Volleys'].fillna(df['Volleys'].mean(), inplace = True)

df['Dribbling'].fillna(df['Dribbling'].mean(), inplace = True)

df['Curve'].fillna(df['Curve'].mean(), inplace = True)

df['FKAccuracy'].fillna(df['FKAccuracy'], inplace = True)

df['LongPassing'].fillna(df['LongPassing'].mean(), inplace = True)

df['BallControl'].fillna(df['BallControl'].mean(), inplace = True)

df['HeadingAccuracy'].fillna(df['HeadingAccuracy'].mean(), inplace = True)

df['Finishing'].fillna(df['Finishing'].mean(), inplace = True)

df['Crossing'].fillna(df['Crossing'].mean(), inplace = True)

df['Weight'].fillna('200lbs', inplace = True)

df['Contract Valid Until'].fillna(2019, inplace = True)

df['Height'].fillna("5'11", inplace = True)

df['Loaned From'].fillna('None', inplace = True)

df['Joined'].fillna('Jul 1, 2018', inplace = True)

df['Jersey Number'].fillna(8, inplace = True)

df['Body Type'].fillna('Normal', inplace = True)

df['Position'].fillna('ST', inplace = True)

df['Club'].fillna('No Club', inplace = True)

df['Work Rate'].fillna('Medium/ Medium', inplace = True)

df['Skill Moves'].fillna(df['Skill Moves'].median(), inplace = True)

df['Weak Foot'].fillna(3, inplace = True)

df['Preferred Foot'].fillna('Right', inplace = True)

df['International Reputation'].fillna(1, inplace = True)

df['Wage'].fillna('€200K', inplace = True)
df.describe().T
df=df.drop(columns='Unnamed: 0')
df.head()
df[['Age','Overall','Potential','Jersey Number','Finishing','Stamina']].hist(figsize=(10,8),bins=40,color='r',linewidth='1.5',edgecolor='k')

plt.tight_layout()

plt.show()
#df.groupby('Nationality').mean()
#df['Nationality'].value_counts()
len(df.Nationality.unique())
df['Nationality'].value_counts()[:10]
# Data to plot

England = len(df[df['Nationality'] == 'England'])

Germany = len(df[df['Nationality'] == 'Germany'])

Spain = len(df[df['Nationality'] == 'Spain'])

Argentina = len(df[df['Nationality'] == 'Argentina'])

France = len(df[df['Nationality'] == 'France'])

Brazil = len(df[df['Nationality'] == 'Brazil'])

Italy = len(df[df['Nationality'] == 'Italy'])

Colombia = len(df[df['Nationality'] == 'Colombia'])

Japan = len(df[df['Nationality'] == 'Japan'])

Netherlands = len(df[df['Nationality'] == 'Netherlands'])



labels = 'England','Germany','Spain','Argentina','France','Brazil','Italy','Colombia','Japan','Netherlands'

sizes = [England,Germany,Spain,Argentina,France,Brazil,Italy,Colombia,Japan,Netherlands]

plt.figure(figsize=(6,6))



# Plot

plt.pie(sizes, explode=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05), labels=labels, colors=sns.color_palette("summer"),

autopct='%1.1f%%', shadow=True, startangle=90)

sns.set_context("paper", font_scale=1.2)

plt.title('Ratio of players by different Nationality', fontsize=16)

plt.show()

df.groupby(["Nationality"])["ID"].count().sort_values(ascending=False).head(10).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=1,figsize=(16,8))

plt.title('Player Count by Nationality')

plt.xlabel('Nationality')

plt.ylabel('Count')

plt.ioff()
df.groupby(["Nationality"])["ID"].count().sort_values(ascending=False).tail(50).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=1,figsize=(16,8))

plt.title('Player Count by Nationality')

plt.xlabel('Nationality')

plt.ylabel('Count')
#f,ax=plt.subplots(1,2,figsize=(18,8))

#df['Age'].value_counts().plot.pie(explode=[0,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

#ax[0].set_title('Share of Sources')

#ax[0].set_ylabel('Count')

sns.countplot('Age',data=df,order=df['Age'].value_counts().index)

#ax[1].set_title('Count of Age')

plt.show()
df.groupby(["Nationality"])["Age"].mean().sort_values(ascending=False).head(10).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=1,figsize=(16,8))

plt.title('Countries with Older Player')

plt.xlabel('Nationality')

plt.ylabel('Age')

plt.ioff()
df.groupby(["Nationality"])["Age"].mean().sort_values(ascending=False).tail(10).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=1,figsize=(16,8))

plt.title('Countries with Younger Player')

plt.xlabel('Nationality')

plt.ylabel('Count')

plt.ioff()
df1=df.copy().drop(columns=['ID'])
#Set up plot style

sns.set(style='white',font_scale=2)



#Compute Correlation Matrix

corr=df1.corr()



#Generate mask for upper traingle

mask=np.zeros_like(corr,dtype=np.bool)

mask[np.triu_indices_from(mask)]=True



#Set up the matplotlib figure

f,ax=plt.subplots(figsize=(18,15))

f.suptitle("Correlation Matrix",fontsize=40)



#Generate a custom diverging Colormap 

cmap=sns.diverging_palette(220,10,as_cmap=True)



#Draw the heat map with the mask and correct aspect ratio

sns.heatmap(corr,mask=mask,cmap=cmap,vmax=.3,center=0,square=True,linewidths=0.5,cbar_kws={'shrink':.5})

plt.ioff()

def country(x):

    return df[df['Nationality'] == x][['Name','Overall','Potential','Position']]





# let's check the Indian Players 

country('India')
#print(df['Club'].unique())
def club(x):

    return df[df['Club'] == x][['Name','Jersey Number','Position','Overall','Nationality','Age','Wage',

                                    'Value','Contract Valid Until']]



club('FC Barcelona')
sns.lineplot(df['Age'], df['Stamina'], palette = 'dark')

plt.title('Age vs Rating', fontsize = 20)



plt.show()
df[df['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(10)
f,ax=plt.subplots(1,2,figsize=(18,8))

df['Preferred Foot'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Preferred Foot')

ax[0].set_ylabel('Count')

sns.countplot('Preferred Foot',data=df,ax=ax[1],order=df['Preferred Foot'].value_counts().index)

ax[1].set_title('Count of Preferred Foot')

plt.show()
df[df['Preferred Foot'] == 'Left'][['Name', 'Age', 'Club', 'Nationality']].head(10)
# defining the features of players



player_features = ('Acceleration', 'Aggression', 'Agility', 

                   'Balance', 'BallControl', 'Composure', 

                   'Crossing', 'Dribbling', 'FKAccuracy', 

                   'Finishing', 'GKDiving', 'GKHandling', 

                   'GKKicking', 'GKPositioning', 'GKReflexes', 

                   'HeadingAccuracy', 'Interceptions', 'Jumping', 

                   'LongPassing', 'LongShots', 'Marking', 'Penalties')



# Top four features for every position in football



for i, val in df.groupby(df['Position'])[player_features].mean().iterrows():

    print('Position {}: {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))
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