import pandas as pd

import numpy as np
tvshows_df = pd.read_csv('../input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv')
tvshows_df
tvshows_df.drop(columns=['type'],axis=1)
new_df=tvshows_df[tvshows_df.IMDb.notna()]

new_df=new_df[new_df.Age.notna()]

new_df=new_df[new_df['Rotten Tomatoes'].notna()]

new_df.drop(columns='Unnamed: 0')

new_df.reset_index(inplace=True)

new_df=new_df.drop(columns=['index','Unnamed: 0','type'])
new_df
new_df['IMDb'].describe()
new_df[new_df.IMDb==new_df.IMDb.min()]
new_df[new_df.IMDb==new_df.IMDb.max()]
new_df.Age.unique()#To determine unique values
IMDb_rating=np.array(list(new_df.IMDb))*10

RT_rating=np.array(np.char.strip(np.array(list(new_df["Rotten Tomatoes"])),'%'),dtype='float64')

weighted_rating=(IMDb_rating+RT_rating)/2
new_df["Weighted Rating"]=weighted_rating
new_df['Rotten Tomatoes New']=RT_rating
new_df[new_df['Rotten Tomatoes New']==new_df['Rotten Tomatoes New'].min()]
new_df[new_df['Rotten Tomatoes New']==new_df['Rotten Tomatoes New'].max()]
import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline



sns.set_style('darkgrid')

matplotlib.rcParams['font.size'] = 14

matplotlib.rcParams['figure.figsize'] = (9, 5)

matplotlib.rcParams['figure.facecolor'] = '#00000000'
plt.hist(new_df.Age,bins=5)

plt.xlabel("Age Ranges")

plt.ylabel("Relative Frequency")

plt.title("Age Histogram")

plt.show()
fig,ax = plt.subplots()

bp_data=[IMDb_rating,RT_rating]

ax.boxplot(bp_data)

ax.set_xticklabels(["IMDb","Rotten tomatoes"])

plt.ylabel("Relative Ratings(Scale 0-100)")

plt.title("Ratings Boxplot")

plt.show()