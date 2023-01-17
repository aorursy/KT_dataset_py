import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from plotly import __version__

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.go_offline()

init_notebook_mode(connected=True)

%matplotlib inline

from PIL import Image





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



dataset=pd.read_csv("../input/top50spotify2019/top50.csv",encoding='ISO-8859-1')

dataset.head()
dataset.columns=[each.split()[0] if(len(each.split())>=2) else each.replace(".","_") for each in dataset.columns]
print("Are there any Missing Data? :",dataset.isnull().any().any())

print(dataset.isnull().sum())
from wordcloud import WordCloud

plt.style.use('seaborn')

wrds1 = dataset["Artist_Name"].str.split("(").str[0].value_counts().keys()

from matplotlib.colors import LinearSegmentedColormap

colors = ["#000000", "#111111", "#101010", "#121212", "#212121", "#222222"]

cmap = LinearSegmentedColormap.from_list("mycmap", colors)

wc1 = WordCloud(scale=5,max_words=1000,colormap=cmap,background_color="white").generate(" ".join(wrds1))

plt.figure(figsize=(12,18))

plt.imshow(wc1,interpolation="bilinear")

plt.axis("off")

plt.title("Most Featured Artists in the Top 50 list ",color="r",fontsize=30)

plt.show()
print("\n\nDifferent Genres in Dataset:\n")

print("There are {} different values\n".format(len(dataset.Genre.unique())))

print(dataset.Genre.unique())
print(type(dataset['Genre']))

popular_genre = dataset.groupby('Genre').size()

popular_genre = popular_genre.sort_values(ascending=False)

popular_genre

genre_list = dataset['Genre'].values.tolist()

genre_top10 = popular_genre[0:10,]

genre_top10 = genre_top10.sort_values(ascending=True)

genre_top10 = pd.DataFrame(genre_top10, columns = [ 'Number of Songs'])

genre_top10
plt.figure(figsize=(16,8))





ax = sns.barplot(x = genre_top10.index ,y = 'Number of Songs' , data = genre_top10, orient = 'v', palette = sns.color_palette("muted", 20), saturation = 0.8)



plt.title("Top 10 Genres among the top 50 songs of 2019",fontsize=30)

plt.ylabel('Number of Songs', fontsize=25)

plt.xlabel('Genre', fontsize=10)









plt.show
#Horizontal bar plot

Genre_lists=list(dataset['Genre'].unique())

BeatPerMinute=[]

Energy_=[]

share_Dance=[]

Acousticness=[]

#share_trust=[]

for each in Genre_lists:

    region=dataset[dataset['Genre']==each]

    BeatPerMinute.append(sum(region.Beats_Per_Minute)/len(region))

    Energy_.append(sum(region.Energy)/len(region))

    share_Dance.append(sum(region.Danceability)/len(region))

    Acousticness.append(sum(region.Acousticness__)/len(region))

    #share_trust.append(sum(region.Trust)/len(region))

#Visualization

f,ax = plt.subplots(figsize = (9,5))

sns.set_color_codes("pastel")

sns.barplot(x=BeatPerMinute,y=Genre_lists,color='g',label="Beat Per Minute")

sns.barplot(x=Energy_,y=Genre_lists,color='b',label="Energy")

sns.barplot(x=share_Dance,y=Genre_lists,color='c',label="Danceability")

sns.barplot(x=Acousticness,y=Genre_lists,color='y',label="Acousticness")

#sns.barplot(x=share_trust,y=region_lists,color='r',label="Trust")

ax.legend(loc="lower right",frameon = True)

ax.set(xlabel='Added features value', ylabel='Genre',title = "Top Genres Similarity")

plt.show()
sns.catplot(x = "Length_", y = "Genre", kind = "bar" ,palette = "pastel",

            edgecolor = ".6",data = dataset)
dataset.columns

plt.figure(figsize=(10,6))

sns.heatmap(dataset[['Beats_Per_Minute','Energy','Danceability','Liveness','Valence_','Length_','Acousticness__','Speechiness_','Popularity']].corr(),annot=True)
sns.set_style(style='dark')

sns.kdeplot(data=dataset['Danceability'], shade=True)
# Set conditions

D=dataset['Danceability']>=75

Nd=(dataset['Danceability']<75)

# Create DataFrame 

data=[D.sum(),Nd.sum()]

Dance=pd.DataFrame(data,columns=['percent'],

                   index=['Danceable','Non-Danceable'])

Dance
dataset[['Track_Name','Artist_Name','Genre','Danceability']].sort_values(by='Danceability',ascending=False).head(24)