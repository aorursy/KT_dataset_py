

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import cm

import matplotlib

import plotly.graph_objects as go



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('../input/bollywood-box-office-20172020/bollywoodboxoffice_raw.csv')

df.head(5)
#deleting unwanted columns

df = df.drop(['movie_url','movie_director_url'], axis = 1) 

df.columns


columns = ['movie_opening','movie_weekend', 'movie_firstweek', 'movie_total','movie_total_worldwide']



for i , c in enumerate(columns):

    df[c] = df[c].replace({':': '', 'cr': '', ',': '','---':''," ":''}, regex=True) #replacing special characters

    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0) #converting to float



df.dtypes


#getting release date and movie length from movie_movierelease

new = df["movie_movierelease"].str.split('|', n = 1, expand = True)  #splitting the values into two



# making separate release date column from new data frame 

df["release_date"]= new[0] 

  

# making separate movie length column from new data frame 

df["movie_length"]= new[1]



df= df.drop(['movie_movierelease'], axis = 1)
#Repeating the same thing from previous code for other columns

#for column movie_producer

new = df["movie_producer"].str.split(':', n = 1, expand = True)

df["producer"]= new[1] 

df = df.drop(['movie_producer'], axis = 1)
#for column movie_banner

new = df["movie_banner"].str.split(':', n = 1, expand = True)

df["banner"]= new[1] 

df = df.drop(['movie_banner'], axis = 1)
#for column movie actors

new = df["movie_stars"].str.split(':', n = 1, expand = True)

df["actors"]= new[1] 

df = df.drop(['movie_stars'], axis = 1)
#getting day , month and year from release date

df['release_day'] = pd.DatetimeIndex(df['release_date']).day

df['release_month'] = pd.DatetimeIndex(df['release_date']).month

df['release_year'] = pd.DatetimeIndex(df['release_date']).year



look_up = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',

            6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}



df['release_month'] = df['release_month'].apply(lambda x: look_up[x])

#converting movie length column to totaltime in minutes

new = df.movie_length.str.split(n=4 , expand = True)

df['hours'] = new[0]

df['minutes'] = new[2]

df.hours = df.hours.replace({' ': ''}, regex=True)

df.minutes = df.minutes.replace({' ': ''}, regex=True)



df.hours = df.hours.astype(str).astype(int)

df.minutes = df.minutes.fillna(0)

df.minutes = df.minutes.astype(str).astype(int)





df['runtime'] = df['hours'] * 60 + df['minutes']
#keeping only required columns and put them in order

df = df[['movie_name', 'movie_opening', 'movie_weekend', 'movie_firstweek',

       'movie_total', 'movie_total_worldwide', 'movie_genre', 'movie_director',

        'release_date', 'release_day', 'release_month', 'release_year','runtime','producer', 'banner',

       'actors','movie_details']]
df.head()
df.to_csv('bollywood_box_clean.csv', index=False)
df = df.sort_values('movie_opening', ascending = False)



dftop = df.head(15)





sns.set_context("talk")

sns.set_style("darkgrid")

plt.style.use("fivethirtyeight")

plt.figure(figsize = (12,10))

ax = sns.barplot(x = 'movie_opening', y = 'movie_name', data = dftop,palette="dark")

plt.title("Highest first day earning Films", size = 15)

ax.set_ylabel('')

ax.set_xlabel('Crores')
df = df.sort_values('movie_total_worldwide', ascending = False)



dftop = df.head(15)
sns.set_context("talk")

sns.set_style("darkgrid")

plt.style.use("fivethirtyeight")

plt.figure(figsize = (12,10))

ax = sns.barplot(x = 'movie_total_worldwide', y = 'movie_name', data = dftop,palette="dark")

plt.title("Highest grossing Films", size = 15)

ax.set_ylabel('')

ax.set_xlabel('Crores')
df['release_year'] = df.release_year.astype('category')
import plotly.express as px



fig = px.scatter(df, x="runtime", y="movie_total_worldwide", color="release_year",

                 size='movie_total_worldwide')

fig.show()
df.release_date = pd.to_datetime(df.release_date) #convert to datetime

df['day'] =df.release_date.dt.dayofweek #make a new column for day of the week where monday is o , tuesday is 1 and so on.
df.day.value_counts()
df.day = df.day.replace([4, 3, 2], ['Friday','Thursday', 'Tuesday'])
ax = sns.countplot(x="day", data= df)
sns.set_context("talk")

sns.set_style("darkgrid")

plt.style.use("fivethirtyeight")

plt.figure(figsize = (12,10))



ax = sns.barplot(x = 'movie_total_worldwide', y = 'movie_name', hue = 'day',data = df.head(20))