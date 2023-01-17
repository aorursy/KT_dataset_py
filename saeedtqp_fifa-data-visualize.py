# basic operations



import numpy as np

import pandas as pd



# for visualizations



import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# file path



import os
%time

data = pd.read_csv('../input/fifa19/data.csv')



print(data.shape)
# Cheking the first 5 rows and columns



data.head()
def country(x):

    return data[data['Nationality'] == x][['Name','Overall','Potential','Position']]





# Let's check the iranian players



country('Iran')
# Analyzing Club Data(manchester United)



def club(x):

    

    return data[data['Club']== x][['Name','Jersey Number',

                                 'Position','Overall','Nationality',

                                 'Age','Wage','Wage','Value','Contract Valid Until']]



club('Manchester United')



x = club('Manchester United')

print('Num of players : ',x.shape[0])
data.describe()
# Checking if The data contains any NULL value



data.isnull().sum()
# Data Cleaning



# Filling the missing value for the continous varibles for porper data visulization



data['ShortPassing'].fillna(data['ShortPassing'].mean(),inplace = True)

data['Volleys'].fillna(data['Volleys'].mean(),inplace = True)

data['Dribbling'].fillna(data['Dribbling'].mean(),inplace = True)

data['Curve'].fillna(data['Curve'].mean(),inplace = True)

data['FKAccuracy'].fillna(data['FKAccuracy'].mean(),inplace = True)

data['LongPassing'].fillna(data['LongPassing'].mean(),inplace = True)

data['BallControl'].fillna(data['BallControl'].mean(),inplace = True)

data['HeadingAccuracy'].fillna(data['HeadingAccuracy'].mean(),inplace = True)

data['Finishing'].fillna(data['Finishing'].mean(),inplace = True)

data['Crossing'].fillna(data['Crossing'].mean(),inplace = True)

data['Weight'].fillna('200lbs',inplace = True)

data['Contract Valid Until'].fillna(2019 , inplace = True)

data['Height'].fillna("5'11", inplace = True)

data['Loaned From'].fillna('None',inplace = True)

data['Joined'].fillna('Jul 1, 2018', inplace = True)

data['Jersey Number'].fillna(8 , inplace = True)

data['Body Type'].fillna('Normal',inplace = True)

data['Position'].fillna('ST',inplace = True)

data['Club'].fillna('No Club',inplace = True)

data['Work Rate'].fillna('Medium / Medium',inplace = True)

data['Skill Moves'].fillna(data['Skill Moves'].median(),inplace = True)

data['Weak Foot'].fillna(3,inplace = True)

data['Preferred Foot'].fillna('Right',inplace = True)

data['International Reputation'].fillna(1,inplace = True)

data['Wage'].fillna('€200K',inplace = True)
data.fillna(0,inplace=True)
def defending(data):

    

    return int(round((data[['Marking' , 'StandingTackle',

                           'SlidingTackle']].mean()).mean()))
def general(data):

    return int(round((data[['HeadingAccuracy',

                           'Dribbling','Curve','BallControl']].mean()).mean()))
def mental(data):

    

    return int(round((data[['Aggression','Interceptions',

                            'Positioning','Vision',

                            'Composure']].mean()).mean()))

def passing(data):

    return int(round((data[['Crossing','ShortPassing',

                            'LongPassing']].mean()).mean()))
def mobility(data):

    return int(round((data[['Acceleration','SprintSpeed','Agility',

                           'Reactions']].mean()).mean()))
def power(data):

    

    return int(round((data[['Balance','Jumping','Stamina','Strength']].mean()).mean()))
def rating(data):

    return int(round((data[['Potential','Overall']].mean()).mean()))
def shooting(data):

    return int(round((data[['Finishing','Volleys','FKAccuracy','ShotPower',

                            'LongShots','Penalties']].mean()).mean()))
# Renaming a columns



data.rename(columns = {'Club Logo':'Club_Logo'},inplace = True)
# Adding these categories to the data



data['Defending'] = data.apply(defending , axis = 1)

data['General'] = data.apply(general , axis = 1)

data['Mental'] = data.apply(mental,axis = 1 )

data['Passing'] = data.apply(passing , axis = 1)

data['Mobility'] = data.apply(mobility , axis = 1)

data['Power'] = data.apply(power , axis = 1)

data['Rating'] = data.apply(rating , axis = 1)

data['Shooting'] = data.apply(shooting , axis = 1)

players = data[['Name','Defending','General','Mental','Passing',

               'Mobility','Power','Rating','Shooting','Flag','Age',

               'Nationality','Photo','Club_Logo','Club']]



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
# Defining a polar graph



def graphPolar(id = 0):

    if 0<= id < len(data.ID):

        details(row = players.index[id],

               title = players['Name'][id],

               age = players['Age'][id],

               photo = players['Photo'][id],

               nationality = players['Nationality'][id],

               image = players['Flag'][id],

               logo = players['Club_Logo'][id],

    

                club = players['Club'][id])

    else:

            

            print('The base has 17917 players . you can put positive number from 0 to 17917')
graphPolar(0)
graphPolar(2)
# comparison of preferred foot over the difreent players



plt.rcParams['figure.figsize'] = (10 ,5 )

sns.countplot(data['Preferred Foot'] , palette = 'pink')

plt.title('Most Preferred foot of the Players',fontsize = 20)

plt.show()
# plotting a pie chart to represent share of international repuatation



labels = ['1','2','3','4','5']



sizes = data['International Reputation'].value_counts()

colors = plt.cm.copper(np.linspace(0,1,5))

explode = [0.1 , 0.1 , 0.2 , 0.5 , 0.9]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(sizes , labels = labels , colors = colors , explode = explode , shadow = True)

plt.title('International repuatation for the footbal Players ',fontsize = 20)

plt.legend()

plt.show()

# plotting a pie cart to represent the share of weak foot players



labels = ['5','4','3','2','1']

size = data['Weak Foot'].value_counts()

colors = plt.cm.Wistia(np.linspace(0,1,5))

explode = [0,0,0,0,0.1]



plt.pie(size , labels = labels , colors = colors , explode = explode , shadow = True, startangle = 90)

plt.title('Distribution of Weak foot among players ',fontsize = 25)

plt.legend()

plt.show()
#defining a function for cleaning the wage column



def extract_value_from(Value):

    out = Value.replace('€','')

    if 'M' in out:

        out = float(out.replace('M','')) * 1000000

    elif 'K' in value:

        out = float(out.replace('K',''))*1000

    return float(out)
data['Value'] = data['Value'].apply(lambda x: extract_value_from(x))

data['Wage'] = data['Wage'].apply(lambda x: extract_value_from(x))



data['Wage'].head()