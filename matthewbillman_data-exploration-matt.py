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
data_path = ('/kaggle/input/bachelorettedsfinal/BacheloretteDSFinal.csv')

df = pd.read_csv(data_path)
shape = df.shape

columns = df.columns.tolist()

print("Shape of the data: ", shape)

print("Columns within the dataset: ", columns)
df.head(10)
df['Hometown'].value_counts()





# destination = df['HomeTown'].value_counts()

# plt.pie(Hometown, labels = Hometown.index)
print('In seasons 11-15, there were {:,} unique contestants. {:,} contestants have appeared in more than one season.'.format(df['Name'].nunique(), len([x for x in df['Name'].value_counts() if x > 1])))
print('In seasons 11-15, there were {:,} unique hometowns. {:,} hometowns have appeared multiple times.'.format(df['Hometown'].nunique(), len([x for x in df['Hometown'].value_counts() if x > 1])))
df[df['Season']==14]['State'].value_counts()



home = df[df['Season']==15]['State'].value_counts()

plt.pie(home, labels = home.index)
total15= df[df['Season']==15]['State'].value_counts().sum()

fifteen= df[df['Season']==15]['State'].value_counts()



for i in fifteen:

    dist15= fifteen/total15

    dist15= dist15 *100

print(dist15)
#UNDODonut Plot for Season 15 State Distribution

#Creates pie chart with specific colors

import matplotlib.pyplot as plt

names='California','Illinois','Michigan','Georgia','Florida','Texas','Maryland','Alabama','Pennsylvania',"Kentucky",'Tennessee','Massachusetts'

size=[30,13.3,10,10,6.67,6.67,6.67,3.33,3.33,3.33,3.33,3.33]

plt.pie(size, labels=names, colors=['darkred','darksalmon',"red",'mistyrose','firebrick','crimson'])



# Creates a white circle for the center of the plot

my_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()



total14= df[df['Season']==14]['State'].value_counts().sum()

fourteen= df[df['Season']==14]['State'].value_counts()



for i in fourteen:

    dist14= fourteen/total14

    dist14 = dist14 *100

print(dist14)
#UNDODonut Plot for Season 14 State Distribution

#Creates pie chart with specific colors

import matplotlib.pyplot as plt

names='California','Florida','Illinois', 'Massachussets','New York','Georgia','Virginia','New Jersey','Idaho',"Ohio",'Colorado','Minnesota'

size=[25,14.28,14.28,10.71,10.71,3.57,3.57,3.57,3.57,3.57,3.57,3.57]

plt.pie(size, labels=names, colors=['darkred',"red",'firebrick','darksalmon','mistyrose','crimson'])



# Creates a white circle for the center of the plot

my_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()





total13= df[df['Season']==13]['State'].value_counts().sum()

thirteen= df[df['Season']==13]['State'].value_counts()



for i in thirteen:

    dist13= thirteen/total13

    dist13= dist13 *100

print(dist13)
#UNDODonut Plot for Season 13 State Distribution

#Creates pie chart with specific colors

import matplotlib.pyplot as plt

names='Florida','California','Illinois','Texas','Connecticut','Minnesota','Wisconsin','Colorado','Michigan','New York','Georgia','Arkansas','Maryland','Other'

size=[22.5,22.5,9.67,9.67,6.45,3.33,3.33,3.33,3.33,3.33,3.33,3.33,3.33,3.33]

plt.pie(size, labels=names, colors=['darkred',"red",'firebrick','darksalmon','mistyrose','crimson'])



# Creates a white circle for the center of the plot

my_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()
total12= df[df['Season']==12]['State'].value_counts().sum()

twelve= df[df['Season']==12]['State'].value_counts()

for i in twelve:

    dist12= twelve/total12

    dist12 = dist12*100

print(dist12)
#UNDODonut Plot for Season 12 State Distribution

#Creates pie chart with specific colors

import matplotlib.pyplot as plt

names='California','Illinois','Colorado','Other','Indiana','Arizona','Washinton','New York','Texas','New Jersey','Connecticut','Florida','Oklahoma','Ohio','Tennessee'

size=[30.8,11.5,7.7,7.7,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8]

plt.pie(size, labels=names, colors=['darkred',"red",'firebrick','darksalmon','mistyrose','crimson'])



# Creates a white circle for the center of the plot

my_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()
#Finds specific Distribution of Each State

total11= df[df['Season']==11]['State'].value_counts().sum()

eleven= df[df['Season']==11]['State'].value_counts()

for i in eleven:

    dist11= eleven/total11

    dist11= dist11 *100



print(dist11)



#Donut Plot for Season 11 State Distribution

#Creates pie chart with specific colors



# df = pd.DataFrame({'State':['Illinois','Florida','Missouri','Other','New York','Michigan','Tennessee','Kentucky','Ohio','Wisconsin','Virginia','Massachusetts','Colorado','Idaho','Indiana','North Carolina','California','Rhode Island','Kansas','Connecticut','New Jersey','Georgia'], 'val':[11.5,7.7,7.7,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8]})

# ax = df.plot.bar(x='State', y='val', rot=0)



import matplotlib.pyplot as plt

names='Illinois','Florida','Missouri','Other','New York','Michigan','Tennessee','Kentucky','Ohio','Wisconsin','Virginia','Massachusetts','Colorado','Idaho','Indiana','North Carolina','California','Rhode Island','Kansas','Connecticut','New Jersey','Georgia'

size=[11.5,7.7,7.7,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8,3.8]

plt.pie(size, labels=names, colors=['darkred',"red",'firebrick','darksalmon','mistyrose','crimson'])



# Creates a white circle for the center of the plot

my_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()
#NOT COMPLETE: donut plot of all seasons; however, should probably do bar graph for this? The size line is INCORRECT

import matplotlib.pyplot as plt

names='California','Illinois', 'Florida','Georgia','Massachusetts', 'New York','Missouri','Maryland','Texas','Tennessee','New Jersey','Colorado','Alabama','Virginia','Iowa','Kentucky','Ohio','Pennsylvania','Minnesota'

size=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

plt.pie(size, labels=names, colors=['darkred',"red",'firebrick','darksalmon','mistyrose','crimson'])



# Creates a white circle for the center of the plot

my_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()



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

#simple histogram for ages; season 15



df[df['Season']==15]['Age'].hist(bins=5, color='DarkRed')

plt.savefig('S15Histogram')  # saves the current figure



#simple histogram for ages; season 14

df[df['Season']==14]['Age'].hist(bins=5, color='DarkRed')

#simple histogram for ages; season 13

df[df['Season']==13]['Age'].hist(bins=5, color='DarkRed')

#simple histogram for ages; season 12

df[df['Season']==12]['Age'].hist(bins=5, color='DarkRed')
#simple histogram for ages; season 11

df[df['Season']==11]['Age'].hist(bins=5, color='DarkRed')
#simple histogram for all ages; season 11-15



df['Age'].hist(bins=15, color='DarkRed')
df['Age'].min()
df['Age'].max()
sort_by_life = df.sort_values('Occupation',ascending=False).dropna()

print(sort_by_life)



#Gives the most entered occupations NOT COMPLETE

df['Occupation'].value_counts()

df['Girlfriend While on the Show?'].value_counts()

# libraries

import matplotlib.pyplot as plt

import squarify    # pip install squarify (algorithm for treemap)

 

# If you have 2 lists

squarify.plot(sizes=[138,2], label=["no","yes"], alpha=.7 )

plt.axis('off')

plt.show()

 

# # If you have a data frame?

# import pandas as pd

# df = pd.DataFrame({'nb_people':[8,3,4,2], 'group':["group A", "group B", "group C", "group D"] })

# squarify.plot(sizes=df['nb_people'], label=df['group'], alpha=.8 )

# plt.axis('off')

# plt.show()

df = df.dropna(how='all')
import re



def resplit(x):

    return re.split(r"[^A-Za-z']",x.upper())



occupations = np.concatenate(df.Occupation.apply(resplit), axis=0)
#Make a Word Cloud for Occupations NOT COMPLETE



# Libraries

from wordcloud import WordCloud

import matplotlib.pyplot as plt

    

# Create a list of word

text=(' '.join(occupations))

wordcloud = WordCloud(width=1200, height=600, margin=0, background_color= "white", colormap="Reds").generate(text)

 

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()



df['Eye Color'].value_counts()
totaleye= df['Eye Color'].value_counts().sum()

partialeye= df['Eye Color'].value_counts()

for i in partialeye:

    eyeratio= partialeye/totaleye

    eyeratio= eyeratio* 100

print(eyeratio)
#Donut Plot for Eye Color Distributions

#Creates pie chart with specific colors

import matplotlib.pyplot as plt

names='Brown','Blue','Green'

size=[85.106383,9.929078,4.964539]

plt.pie(size, labels=names, colors=['sienna','skyblue','lightgreen'])



# Creates a white circle for the center of the plot

my_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()





df['Eye Color'].value_counts()



totalhair= df['Hair Color'].value_counts().sum()

partialhair= df['Hair Color'].value_counts()

for i in partialhair:

    hairratio= partialhair/totalhair

    hairratio= hairratio* 100

print(hairratio)
#Donut Plot for Hair Color Distributions

#Creates pie chart with specific colors

import matplotlib.pyplot as plt

names='Brown','Blonde'

size=[94.326241,5.673759]

plt.pie(size, labels=names, colors=['sienna','gold'])



# Creates a white circle for the center of the plot

my_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()



#Finds the mean height in cm

totalheight= df['Height (cm)'].sum()

count_of_height = df['Height (cm)'].value_counts().sum()

average = (totalheight)/(count_of_height)

print(average)
df['height_z'] = ((df['Height (cm)']-df['Height (cm)'].mean())/df['Height (cm)'].std()).dropna()


def to_inch(x):

    ft_raw = 0.0328084*x

    ft = int(ft_raw)

    rem = ft_raw-ft

    inches = round(rem*12,2)

    return (ft, inches)









#Converts the mean height in cm to inches

inch = average*.39

#Finds the remainder of the height in inches divided by 12, and converts that to inches

feet2= inch%12

full_inch= feet2*12

rounded_inch= full_inch.round(2)



#Finds the feet as an integer

feet= (inch//12).astype(int)



print('The average contestant is',feet,'feet and', rounded_inch,'inches tall.')
df['Height (cm)'].dropna().apply(to_inch)
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
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs')

model.fit(x,y)