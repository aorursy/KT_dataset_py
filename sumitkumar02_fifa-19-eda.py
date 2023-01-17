# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import eli5

from eli5.sklearn import PermutationImportance

from collections import Counter

import missingno as msno



import warnings

warnings.filterwarnings('ignore')



sns.set_style('whitegrid')
data=pd.read_csv('../input/data.csv')
data.head()
data.columns
# checking if the data contains any NULL value



data.isnull().sum()
data.info()
data.describe()
#Number of countries available and top 5 countries with highest number of players

print('Total number of countries : {0}'.format(data['Nationality'].nunique()))

print(data['Nationality'].value_counts().head(5))



#European Countries have most players
#Total number of clubs present and top 5 clubs with highest number of players

print('Total number of clubs : {0}'.format(data['Club'].nunique()))

print(data['Club'].value_counts().head(5))
data.columns
#Player with maximum Potential and Overall Performance

print('Maximum Potential : '+str(data.loc[data['Potential'].idxmax()][2]))

print('Maximum Overall Perforamnce : '+str(data.loc[data['Overall'].idxmax()][2]))
pr_cols=['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',

       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',

       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',

       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',

       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',

       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',

       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

i=0

while i < len(pr_cols):

    print('Best {0} : {1}'.format(pr_cols[i],data.loc[data[pr_cols[i]].idxmax()][2]))

    i += 1
# filling the missing value for the continous variables for proper data visualization



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

data['Weight'].fillna('200lbs', inplace = True)

data['Contract Valid Until'].fillna(2019, inplace = True)

data['Height'].fillna("5'11", inplace = True)

data['Loaned From'].fillna('None', inplace = True)

data['Joined'].fillna('Jul 1, 2018', inplace = True)

data['Jersey Number'].fillna(8, inplace = True)

data['Body Type'].fillna('Normal', inplace = True)

data['Position'].fillna('ST', inplace = True)

data['Club'].fillna('No Club', inplace = True)

data['Work Rate'].fillna('Medium/ Medium', inplace = True)

data['Skill Moves'].fillna(data['Skill Moves'].median(), inplace = True)

data['Weak Foot'].fillna(3, inplace = True)

data['Preferred Foot'].fillna('Right', inplace = True)

data['International Reputation'].fillna(1, inplace = True)

data['Wage'].fillna('€200K', inplace = True)
data.fillna(0, inplace = True)
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

# renaming a column

data.rename(columns={'Club Logo':'Club_Logo'}, inplace=True)



# adding these categories to the data



data['Defending'] = data.apply(defending, axis = 1)

data['General'] = data.apply(general, axis = 1)

data['Mental'] = data.apply(mental, axis = 1)

data['Passing'] = data.apply(passing, axis = 1)

data['Mobility'] = data.apply(mobility, axis = 1)

data['Power'] = data.apply(power, axis = 1)

data['Rating'] = data.apply(rating, axis = 1)

data['Shooting'] = data.apply(shooting, axis = 1)
players = data[['Name','Defending','General','Mental','Passing',

                'Mobility','Power','Rating','Shooting','Flag','Age',

                'Nationality', 'Photo', 'Club_Logo', 'Club']]



players.head()
import requests

import random

from math import pi



import matplotlib.image as mpimg

from matplotlib.offsetbox import (OffsetImage,AnnotationBbox)



def details(row, title, image, age, nationality, photo, logo, club):

    

    flag_image = "img_flag.jpg"

    player_image = "img_player.jpg"

    logo_image = "img_club_logo.jpg"

        

    img_flag = requests.get(image).content

    with open(flag_image, 'wb') as handler:

        handler.write(img_flag)

    

    player_img = requests.get(photo).content

    with open(player_image, 'wb') as handler:

        handler.write(player_img)

     

    logo_img = requests.get(logo).content

    with open(logo_image, 'wb') as handler:

        handler.write(logo_img)

        

    r = lambda: random.randint(0,255)

    colorRandom = '#%02X%02X%02X' % (r(),r(),r())

    

    if colorRandom == '#ffffff':colorRandom = '#a5d6a7'

    

    basic_color = '#37474f'

    color_annotate = '#01579b'

    

    img = mpimg.imread(flag_image)

    

    plt.figure(figsize=(15,8))

    categories=list(players)[1:]

    coulumnDontUseGraph = ['Flag', 'Age', 'Nationality', 'Photo', 'Logo', 'Club']

    N = len(categories) - len(coulumnDontUseGraph)

    

    angles = [n / float(N) * 2 * pi for n in range(N)]

    angles += angles[:1]

    

    ax = plt.subplot(111, projection='polar')

    ax.set_theta_offset(pi / 2)

    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color= 'black', size=17)

    ax.set_rlabel_position(0)

    plt.yticks([25,50,75,100], ["25","50","75","100"], color= basic_color, size= 10)

    plt.ylim(0,100)

    

    values = players.loc[row].drop('Name').values.flatten().tolist() 

    valuesDontUseGraph = [image, age, nationality, photo, logo, club]

    values = [e for e in values if e not in (valuesDontUseGraph)]

    values += values[:1]

    

    ax.plot(angles, values, color= basic_color, linewidth=1, linestyle='solid')

    ax.fill(angles, values, color= colorRandom, alpha=0.5)

    axes_coords = [0, 0, 1, 1]

    ax_image = plt.gcf().add_axes(axes_coords,zorder= -1)

    ax_image.imshow(img,alpha=0.5)

    ax_image.axis('off')

    

    ax.annotate('Nationality: ' + nationality.upper(), xy=(10,10), xytext=(103, 138),

                fontsize= 12,

                color = 'white',

                bbox={'facecolor': color_annotate, 'pad': 7})

                      

    ax.annotate('Age: ' + str(age), xy=(10,10), xytext=(43, 180),

                fontsize= 15,

                color = 'white',

                bbox={'facecolor': color_annotate, 'pad': 7})

    

    ax.annotate('Team: ' + club.upper(), xy=(10,10), xytext=(92, 168),

                fontsize= 12,

                color = 'white',

                bbox={'facecolor': color_annotate, 'pad': 7})



    arr_img_player = plt.imread(player_image, format='jpg')



    imagebox_player = OffsetImage(arr_img_player)

    imagebox_player.image.axes = ax

    abPlayer = AnnotationBbox(imagebox_player, (0.5, 0.7),

                        xybox=(313, 223),

                        xycoords='data',

                        boxcoords="offset points"

                        )

    arr_img_logo = plt.imread(logo_image, format='jpg')



    imagebox_logo = OffsetImage(arr_img_logo)

    imagebox_logo.image.axes = ax

    abLogo = AnnotationBbox(imagebox_logo, (0.5, 0.7),

                        xybox=(-320, -226),

                        xycoords='data',

                        boxcoords="offset points"

                        )



    ax.add_artist(abPlayer)

    ax.add_artist(abLogo)



    plt.title(title, size=50, color= basic_color)
# defining a polar graph



def graphPolar(id = 0):

    if 0 <= id < len(data.ID):

        details(row = players.index[id], 

                title = players['Name'][id], 

                age = players['Age'][id], 

                photo = players['Photo'][id],

                nationality = players['Nationality'][id],

                image = players['Flag'][id], 

                logo = players['Club_Logo'][id], 

                club = players['Club'][id])

    else:

        print('The base has 17917 players. You can put positive numbers from 0 to 17917')

graphPolar(0)
graphPolar(1)
# different positions acquired by the players 



plt.figure(figsize = (12, 8))

sns.set(style = 'dark', palette = 'colorblind', color_codes = True)

ax = sns.countplot('Position', data = data, color = 'orange')

ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)

ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)

plt.show()
# Skill Moves of Players



plt.figure(figsize = (7, 8))

ax = sns.countplot(x = 'Skill Moves', data = data, palette = 'pastel')

ax.set_title(label = 'Count of players on Basis of their skill moves', fontsize = 20)

ax.set_xlabel(xlabel = 'Number of Skill Moves', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()

player_features = (

    'Acceleration', 'Aggression', 'Agility', 

    'Balance', 'BallControl', 'Composure', 

    'Crossing', 'Dribbling', 'FKAccuracy', 

    'Finishing', 'GKDiving', 'GKHandling', 

    'GKKicking', 'GKPositioning', 'GKReflexes', 

    'HeadingAccuracy', 'Interceptions', 'Jumping', 

    'LongPassing', 'LongShots', 'Marking', 'Penalties'

)



from math import pi

idx = 1

plt.figure(figsize=(15,45))

for position_name, features in data.groupby(data['Position'])[player_features].mean().iterrows():

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

    ax = plt.subplot(10, 3, idx, polar=True)



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
df=pd.read_csv('../input/data.csv')
#DROP UNNECESSARY VALUES

drop_cols = df.columns[28:54]

df = df.drop(drop_cols, axis = 1)

df = df.drop(['Unnamed: 0','ID','Photo','Flag','Club Logo','Jersey Number','Joined','Special','Loaned From','Body Type', 'Release Clause',

               'Weight','Height','Contract Valid Until','Wage','Value','Club'], axis = 1)

df = df.dropna()

df.head()
#Turn Real Face into a binary indicator variable

def face_to_num(df):

    if (df['Real Face'] == 'Yes'):

        return 1

    else:

        return 0

    

#Turn Preferred Foot into a binary indicator variable

def right_footed(df):

    if (df['Preferred Foot'] == 'Right'):

        return 1

    else:

        return 0



#Create a simplified position varaible to account for all player positions

def simple_position(df):

    if (df['Position'] == 'GK'):

        return 'GK'

    elif ((df['Position'] == 'RB') | (df['Position'] == 'LB') | (df['Position'] == 'CB') | (df['Position'] == 'LCB') | (df['Position'] == 'RCB') | (df['Position'] == 'RWB') | (df['Position'] == 'LWB') ):

        return 'CB'

    elif ((df['Position'] == 'LDM') | (df['Position'] == 'CDM') | (df['Position'] == 'RDM')):

        return 'DM'

    elif ((df['Position'] == 'LM') | (df['Position'] == 'LCM') | (df['Position'] == 'CM') | (df['Position'] == 'RCM') | (df['Position'] == 'RM')):

        return 'MF'

    elif ((df['Position'] == 'LAM') | (df['Position'] == 'CAM') | (df['Position'] == 'RAM') | (df['Position'] == 'LW') | (df['Position'] == 'RW')):

        return 'AM'

    elif ((df['Position'] == 'RS') | (df['Position'] == 'ST') | (df['Position'] == 'LS') | (df['Position'] == 'CF') | (df['Position'] == 'LF') | (df['Position'] == 'RF')):

        return 'ST'

    else:

        return df.Position



#Get a count of Nationalities in the Dataset, make of list of those with over 250 Players (our Major Nations)

nat_counts = df.Nationality.value_counts()

nat_list = nat_counts[nat_counts > 250].index.tolist()



#Replace Nationality with a binary indicator variable for 'Major Nation'

def major_nation(df):

    if (df.Nationality in nat_list):

        return 1

    else:

        return 0



#Create a copy of the original dataframe to avoid indexing errors

df1 = df.copy()



#Apply changes to dataset to create new column

df1['Real_Face'] = df1.apply(face_to_num, axis=1)

df1['Right_Foot'] = df1.apply(right_footed, axis=1)

df1['Simple_Position'] = df1.apply(simple_position,axis = 1)

df1['Major_Nation'] = df1.apply(major_nation,axis = 1)



#Split the Work Rate Column in two

tempwork = df1["Work Rate"].str.split("/ ", n = 1, expand = True) 

#Create new column for first work rate

df1["WorkRate1"]= tempwork[0]   

#Create new column for second work rate

df1["WorkRate2"]= tempwork[1]

#Drop original columns used

df1 = df1.drop(['Work Rate','Preferred Foot','Real Face', 'Position','Nationality'], axis = 1)

df1.head()

#Split ID as a Target value

target = df1.Overall

df2 = df1.drop(['Overall'], axis = 1)



#Splitting into test and train

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df2, target, test_size=0.2)



#One Hot Encoding

X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)

print(X_test.shape,X_train.shape)

print(y_test.shape,y_train.shape)