# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.
data_path = ('/kaggle/input/BacheloretteDSFinal-Dogu.csv')

df = pd.read_csv(data_path)

#Welcome to my first data science skill-specific project

#I am relatively new to coding; I am currently enrolled in CS 1110!

shape = df.shape

columns = df.columns.tolist()

print("Shape of the data: ", shape)

print("Columns within the dataset: ", columns)
df.head(10)
#Gives the count of each state

df['Hometown'].value_counts()





# destination = df['HomeTown'].value_counts()

# plt.pie(Hometown, labels = Hometown.index)
print('In seasons 11-15, there were {:,} unique contestants. {:,} contestants have appeared in more than one season.'.format(df['Name'].nunique(), len([x for x in df['Name'].value_counts() if x > 1])))
print('In seasons 11-15, there were {:,} unique hometowns. {:,} hometowns have appeared multiple times.'.format(df['Hometown'].nunique(), len([x for x in df['Hometown'].value_counts() if x > 1])))
#Takes a while to load, Heat Map of the US

perstate = df[df['State'] != '']['State'].value_counts().to_dict()



data = [dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = 'Reds',

        reversescale = True,

        locations = list(perstate.keys()),

        locationmode = 'USA-states',

        text = list(perstate.values()),

        z = list(perstate.values()),

        marker = dict(

            line = dict(

                color = 'rgb(255, 255, 255)',

                width = 2)

            ),

        )]



layout = dict(

         title = 'Bachelorette Contestants by State',

         geo = dict(

             scope = 'usa',

             projection = dict(type = 'albers usa'),

             countrycolor = 'rgb(255, 255, 255)',

             showlakes = True,

             lakecolor = 'rgb(255, 255, 255)')

         )



figure = dict(data = data, layout = layout)

iplot(figure)

#simple histogram for all ages; season 11-15



df['Age'].hist(bins=15, color='DarkRed')
#Gives min age

df['Age'].min()
#Gives max age

df['Age'].max()
sort_by_life = df.sort_values('Occupation',ascending=False).dropna()

print(sort_by_life)



df['Girlfriend While on the Show?'].value_counts()

#Shows how many contestants had girlfriends (yes) and how many did not (no)

# libraries

import matplotlib.pyplot as plt

import squarify    # pip install squarify (algorithm for treemap)

 

# If you have 2 lists

squarify.plot(sizes=[138,2], label=["no","yes"], alpha=.7 )

plt.axis('off')

plt.show()

 

df= df.dropna(how ='all')
#Splits the column at all the characters specified and casts all the letters to uppercase

#Needed for Word Cloud

import re 

import itertools as it



def resplit(x):

    return re.split(r'[^A-Za-z]',x.upper())





occupations = np.concatenate(df.Occupation.apply(resplit), axis=0)
#Needed for next cell

' '.join(occupations)

#Make a Word Cloud for Occupations

#Takes a while to load due to large width and height values

#Colors are random but within the red range



# Libraries

from wordcloud import WordCloud

import matplotlib.pyplot as plt

    

# Create a list of word

text=(' '.join(occupations))

wordcloud = WordCloud(width=2000, height=1500, margin=0, background_color= "white", colormap="Reds").generate(text)

 

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()

#Gives the counts for each eye color

df['Eye Color'].value_counts()
#Gives the percentage of each eye color

total_eye_count= df['Eye Color'].value_counts().sum()

partial_eye_count =  df['Eye Color'].value_counts()

for i in partial_eye_count:

    eye_percentage= partial_eye_count/total_eye_count

    eye_percentage= eye_percentage* 100

print(eye_percentage)

#Donut plot for eye color distributions

#Creates a pie chart for the different eye colors

import matplotlib.pyplot as plt

names='Brown','Blue','Green'

size=[85.106383,9.929078,4.964539]

plt.pie(size, labels=names, colors=['sienna','skyblue','lightgreen'])



#Creates a white circle for the center of the plot

#Makes pie chart become a donut plot

my_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()





#Simple Eye Color Counts

df['Eye Color'].value_counts()

#Gives the percentage of each hair color

total_hair_count= df['Hair Color'].value_counts().sum()

partial_hair_count= df['Hair Color'].value_counts()

for i in partial_hair_count:

    hair_percentage= partial_hair_count/total_hair_count

    hair_percentage= hair_percentage* 100

print(hair_percentage)

#Donut Plot for Hair Color Distributions

#Creates pie chart with specific hair colors

import matplotlib.pyplot as plt

names='Brown','Blonde'

size=[94.326241,5.673759]

plt.pie(size, labels=names, colors=['sienna','gold'])



# Creates a white circle for the center of the plot

#Makes pie chart become a donut plot

my_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()

plt.savefig('DonutPlotHair') 

#Finds the mean height in cm, this could have been done with .mean() but the specific steps were shown to demonstrate understanding of the process

totalheight= df['Height (cm)'].sum()

count_of_height = df['Height (cm)'].value_counts().sum()

average = (totalheight)/(count_of_height)

print(average)



#Takes the height in cm and returns it in ft and in

def to_inch(x):

    ft_raw = 0.0328084*x

    ft = int(ft_raw)

    rem = ft_raw-ft

    inches = round(rem*12,2)

    print('The average contestant is',ft,'feet and', inches,'inches tall.')

    

to_inch(180.9)



#Counts the repitition of  heights

df['Height (cm)'].value_counts()



#Gives the largest height value

df['Height (cm)'].max()



#Gives the smallest height value

df['Height (cm)'].min()



#Used to calculate min, max, and median height values

max_height= to_inch(198.12)

print(max_height)

avg_height= to_inch(184.799801980198)

print(avg_height)

min_height = to_inch(170)

print(min_height)

#Gives the Z Scores for Height

df['height_z'] = ((df['Height (cm)']-df['Height (cm)'].mean())/df['Height (cm)'].std()).dropna()

print(df['height_z'].round(2))
#Converts height from cm to ft, in

df['Height (cm)'].dropna().apply(to_inch)
df = df.dropna(how='all')
#The next four cells were supposed to calculate the linear regression of the data. However, I could not get it to work. This would have been done by 

#graphing both the x and y values and then finding the line of best fit 

import re



def resplit(x):

    return re.split(r"[^A-Za-z']",x.upper())



occupations = np.concatenate(df.Occupation.apply(resplit), axis=0)
df = df.dropna(how='all')

df['f_name'] = df.Name.apply(resplit).apply(lambda x: x[0])

df['l_name'] = df.Name.apply(resplit).apply(lambda x: x[-1])

df['Season'] = df['Season'].apply(str)

df['Height (cm)'] = df['Height (cm)'].fillna(df['Height (cm)'].mean())

cat_data = ['Season', 'Age', 'Hometown', 'State', 'College', 'Occupation',

           'Height (cm)', 'Girlfriend While on the Show?', 'Hair Color',

           'Eye Color', 'f_name', 'l_name']

x = pd.get_dummies(df[cat_data])
y = pd.Series(np.random.randint(0,2,(141,)), name='Win_Loss')

#error for regression results here

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs', max_iter=500)

model.fit(x,y)
print(model.coef_)

print(model.intercept_)