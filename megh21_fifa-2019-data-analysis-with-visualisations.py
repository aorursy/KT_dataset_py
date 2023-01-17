from IPython.display import HTML

HTML('''

<script>

  function code_toggle() {

    if (code_shown){

      $('div.input').hide('500');

      $('#toggleButton').val('Show Code')

    } else {

      $('div.input').show('500');

      $('#toggleButton').val('Hide Code')

    }

    code_shown = !code_shown

  }



  $( document ).ready(function(){

    code_shown=false;

    $('div.input').hide()

  });

</script>

<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/data.csv")
data.shape
data.columns
data.info()
#Creating Seperate Data Frame with Players of Overall Ratings >= 90

df1 = data.query("Overall>=90")
df1.head(10)
#Age Breakdown of Players 

f , axes = plt.subplots(figsize = (7,5))

f.suptitle('Age Breakdown Of Players',fontsize=16 , x = 0.82 , y = 1.1)



plt.subplot(1,2,1)

v1 = plt.hist(df1.Age)

plt.xlabel('Age')

plt.ylabel('Number Of Players')

plt.title('Number of Players Based on Age With Rating>90')



plt.subplot(1,2,2)

v2 = plt.hist(data.Age, color = 'Orange')

plt.xlabel('Age')

plt.ylabel('Number Of Players')

plt.title('Number of Players Based on Age (Full Dataset)')



plt.subplots_adjust(left= None, bottom=None, right=1.5, top=None, wspace=1, hspace=None)
plt.figure(figsize=(10,8))

viz1 = sns.heatmap(df1.corr(), annot = False , cmap = 'Greens',linewidths=4)

plt.title('Heatmap for the Dataset',fontsize=20 , x = 0.5 , y = 1.5)

plt.show()
f , axes = plt.subplots(figsize = (7,5))

plt.subplot(1,2,1)

v1 = sns.scatterplot(x= "Overall" , y= "Potential" , data = df1, legend='brief')

plt.title('Players With Rating>90')

#plt.legend(loc = 'upper left', bbox_to_anchor = (1,1))

plt.subplot(1,2,2)

v2 = sns.scatterplot(x= "Overall" , y= "Potential" , data = data, color = 'Orange', legend='brief')

plt.title('All Players')

plt.subplots_adjust(left= None, bottom=None, right=1.5, top=None, wspace=1, hspace=None)

f.suptitle('Overall vs Potential',fontsize=16 , x = 0.8, y = 1.0)
df1 = data.query("Overall>=90")

#Cleaning the Value and Wage Columns

df1['Value'] = df1['Value'].str.replace('€', '')

df1['Value'] = df1['Value'].str.replace('M', '')

df1['Wage'] = df1['Wage'].str.replace('€', '')

df1['Wage'] = df1['Wage'].str.replace('K', '')



#Setting datatype for Columns

df1.Value = df1.Value.astype('float')

df1.Wage = df1.Wage.astype('int')

df1.Name = df1.Name.astype('category')



f , axes = plt.subplots(figsize = (12,12))



plt.subplot(2,2,1)

df2 = df1.sort_values(['Value'])

v1 = sns.barplot(x = "Name" , y  = 'Value', data = df2 ,order = df2['Name'])

plt.xlabel(' ')

plt.title("Value Of Players Sorted (In Millions)")

plt.xticks(rotation = 90)



plt.subplot(2,2,2)

df2 = df1.sort_values(['Wage'])

v2 = sns.barplot(x = "Name" , y  = 'Wage', data = df2 ,order = df2['Name'])

plt.xlabel(' ')

plt.title("Wage Of Players Sorted (In Thousands)")

plt.xticks(rotation = 90)



plt.subplot(2,2,3)

df2 = df1.sort_values(['Age'])

v3 = sns.barplot(x = "Name" , y  = 'Age', data = df2 ,order = df2['Name'])

plt.title("Age Of Players Sorted")

plt.xticks(rotation = 90)



plt.subplot(2,2,4)

df2 = df1.sort_values(['Special'])

v2 = sns.barplot(x = "Name" , y  = 'Special', data = df2 ,order = df2['Name'])

plt.title("Special Ability Of Players Sorted")

plt.xticks(rotation = 90)



plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

f.suptitle('Barplots for Value, Wage, Age & Special',fontsize=16 , x = 0.5, y = 1)

plt.show()
f , axes = plt.subplots(figsize = (8,4))



plt.subplot(1,2,1)

v1 =df1['Preferred Foot'].value_counts().plot.bar()

plt.xlabel('Preferred Foot')

plt.ylabel('Number Of Players')

plt.title('Breakdown of Players Based on Preferred Foot With Rating>90')





plt.subplot(1,2,2)

v2 =data['Preferred Foot'].value_counts().plot.bar()

plt.xlabel('Preferred Foot')

plt.ylabel('Number Of Players')

plt.title('Breakdown of Players Based on Preferred Foot (Full Dataset)')



plt.subplots_adjust(left= None, bottom=None, right=1.5, top=None, wspace=1, hspace=None)

f.suptitle('Breakdown of Players Based on Preferred Foot',fontsize=16 , x = 0.8, y = 1.1)

plt.show()
#Age Breakdown of Players 

f , axes = plt.subplots(figsize = (7,5))

f.suptitle('Overall Rating Of Players',fontsize=16 , x = 0.82 , y = 1.1)



plt.subplot(1,2,1)

v1 = plt.hist(df1.Overall)

plt.xlabel('Overall Rating')

plt.ylabel('Number Of Players')

plt.title('Number of Players Based on Overall Rating >90')



plt.subplot(1,2,2)

v2 = plt.hist(data.Overall, color = 'Orange')

plt.xlabel('Overall Rating')

plt.ylabel('Number Of Players')

plt.title('Number of Players Based on Overall Rating (Full Dataset)')



plt.subplots_adjust(left= None, bottom=None, right=1.5, top=None, wspace=1, hspace=None)
f , axes = plt.subplots(figsize = (12,6))

f.suptitle('Overall vs Potential vs Age Of Players',fontsize=16 , x = 0.82 , y = 1.1)



plt.subplot(1,2,1)

v1 = sns.lineplot(x = 'Age' , y = 'Overall' , data = data, color = 'grey' , ci = None)

plt.title('Overall vs Age Of Players')



plt.subplot(1,2,2)

v2 = sns.lineplot(x = 'Age' , y = 'Potential' , data = data, color = 'grey', ci = None )

plt.title('Potential vs Age Of Players')



plt.subplots_adjust(left= None, bottom=None, right=1.5, top=None, wspace=0.5, hspace=None)
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



data.fillna(0, inplace = True)

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
graphPolar(2)
graphPolar(5)
graphPolar(6)