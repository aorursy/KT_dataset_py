# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot

from collections import Counter

import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=20)     

plt.rc('ytick', labelsize=20)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')

df.head()
df.info()


country=df["country"]

country=country.dropna()



country=", ".join(country)

country=country.replace(',, ',', ')





country=country.split(", ")

country= list(Counter(country).items())

country.remove(('Vatican City', 1))

country.remove(('East Germany', 1))

print(country)
max_show_country=country[0:11]

max_show_country = pd.DataFrame(max_show_country) 

max_show_country= max_show_country.sort_values(1)



fig, ax = plt.subplots(1, figsize=(8, 6))

fig.suptitle('Plot of country vs shows')

ax.barh(max_show_country[0],max_show_country[1],color='blue')

plt.grid(b=True, which='major', color='#666666', linestyle='-')



plt.show()
df1=pd.read_csv('/kaggle/input/country-codes/country_code.csv')

df1=df1.drop(columns=['Unnamed: 2'])

df1.head()


country_map = pd.DataFrame(country) 

country_map=country_map.sort_values(1,ascending=False)

location = pd.DataFrame(columns = ['CODE']) 

search_name=df1['COUNTRY']



for i in country_map[0]:

    x=df1[search_name.str.contains(i,case=False)] 

    x['CODE'].replace(' ','')

    location=location.append(x)





print(location)

locations=[]

temp=location['CODE']

for i in temp:

    locations.append(i.replace(' ',''))
data = [dict( type = 'choropleth', locations=locations,z=list(country_map[1]),colorscale = [[0,"rgb(0,128,0)"],[0.50,"rgb(50,205,50)"],

                        [1,"rgb(124,252,0)"]], autocolorscale = False,reversescale = True)]

layout = dict(title = 'Graphical Representation of number of netflix movies and shows worldwide' )   

fig = dict( data=data, layout=layout )

iplot( fig )
genre=df["listed_in"]

genre=", ".join(genre)

genre=genre.replace(',, ',', ')

genre=genre.split(", ")

genre= list(Counter(genre).items())

print(genre)



max_genre=genre[0:11]

max_genre = pd.DataFrame(max_genre) 

max_genre= max_genre.sort_values(1)



plt.figure(figsize=(35,10))

plt.xlabel('number of restaurants')

plt.ylabel('localities')

plt.barh(max_genre[0],max_genre[1], color='red')