# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd

df = pd.read_csv('../input/top-10-highest-grossing-films-19752018/blockbusters.csv')

df.head()
df['Main_Genre'].value_counts()
#df.info()

#df.shape

pd.set_option('display.max_rows',437)
filt=df['imdb_rating']>8.0

df.loc[filt,['title','Main_Genre','imdb_rating','year']]
df.sort_values(by=['worldwide_gross'],ascending=[False])
grp_year=df.groupby(['year'])

grp_year['Main_Genre'].value_counts()
grp_year['Main_Genre'].value_counts().loc[2018]
for i, row in df.iterrows(): #Iterate through each row of dataframe

    gross = df.worldwide_gross[i]

    gross = gross.replace('$','') #Trims $ from the values

    gross = gross.replace(',','') #Trims , from the values.

    df.worldwide_gross[i] = gross
for i, row in df.iterrows():

    gross = df.worldwide_gross[i]

    gross = float(gross)

    gross = gross/1000000

    df.worldwide_gross[i] = int(gross)

df.head()
grp_year['worldwide_gross'].sum()
no_of_movies = df['year'].value_counts()

no_of_movies
#no of romantic movies in a year

no_of_walt_stdio = grp_year['studio'].apply(lambda x : x.str.contains('Walt Disney Pictures').sum())

no_of_walt_stdio
yearwise_walt_studio_release = pd.concat([no_of_movies, no_of_walt_stdio],axis='columns')

yearwise_walt_studio_release
yearwise_walt_studio_release['%ofWaltReleases'] = (yearwise_walt_studio_release['studio']/yearwise_walt_studio_release['year'])*100

yearwise_walt_studio_release


plt.figure(figsize=(15,5))

plt.scatter(x=df['Main_Genre'],y=df['imdb_rating'],s=50,c='green',marker='o',edgecolor='black',linewidth=1)

plt.title("IMDB Rating V/S Genres of Movies")

plt.xlabel("IMDB Rating")

plt.ylabel("Genres of Movies")

plt.xticks(rotation=45)
plt.figure(figsize=(15,5))

plt.title("imdb_rating v/s length")

plt.xlabel("IMDB Rating")

plt.ylabel("Length of Movie")

plt.scatter(x=df['imdb_rating'],y=df['length'],s=100,c='blue',linewidth=1,edgecolor='black')
from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go





trace = go.Box(

    x=df.rating,

    y=df.imdb_rating,

    marker=dict(color='blue')

    )



data=[trace]



layout = go.Layout(

title='Maturity Rating v/s IMDb Ratings of Highest Grossing Films',

    xaxis = dict(title = 'Maturity Rating', gridwidth = 2),

    yaxis = dict(title = 'IMDb Rating', gridwidth = 2),

    hovermode = 'closest',

    paper_bgcolor='rgb(200, 200, 200)',

    plot_bgcolor='rgb(200, 200, 200)'

)





figure=go.Figure(data=data,layout=layout)

init_notebook_mode(connected=True)



iplot(figure)

from collections import Counter

import csv



plt.style.use("fivethirtyeight")



with open('../input/top-10-highest-grossing-films-19752018/blockbusters.csv') as csv_file:

    csv_reader = csv.DictReader(csv_file)

    

    movie_counter=Counter()

    

    for row in csv_reader:

        movie_counter.update(row['Main_Genre'].split(';'))

        

#print(movie_counter)





movie_category=[]

popularity=[] 



for item in movie_counter:

    movie_category.append(item)

    popularity.append(movie_counter[item])

    

print(movie_category)

print(popularity)

    

plt.figure(figsize=(15,10))

plt.barh(movie_category,popularity)

plt.xlabel("Count of movies")

plt.ylabel("Movie Category")

plt.title("Count of movies as Per GENERE till date")



plt.show()
bins = list(df['year'].unique())

bins.sort(reverse=False)

#print(type(bins))

#print(bins)

movies_in_a_year=list(df['year'])

#print(movies_in_a_year)



plt.figure(figsize=(12,10))

plt.hist(movies_in_a_year,bins=bins,edgecolor='black',rwidth=0.8)



plt.xlabel('Year')

plt.ylabel('No Of Movies')

plt.show()



from collections import Counter

import csv



plt.style.use("fivethirtyeight")



with open('../input/top-10-highest-grossing-films-19752018/blockbusters.csv') as csv_file:

    csv_reader = csv.DictReader(csv_file)

    

    studio_count=Counter()

    

    for row in csv_reader:

        studio_count.update(row['studio'].split(';'))





#print(studio_count)

slices =[]

labels =[]



for item in studio_count:

    labels.append(item)

    slices.append(studio_count[item])

        

#print(slices)

#print(labels)



plt.figure(figsize=(20,20))

plt.title("Pie Chart of movies under various studios")

plt.pie(slices,labels=labels,autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

plt.show()