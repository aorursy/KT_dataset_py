import warnings

warnings.filterwarnings("ignore")



# Utils

from os import path



# Data 

import numpy as np

import pandas as pd



# Viz

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



%matplotlib inline

sns.set(style="darkgrid")



print(f"Matplotlib Version : {mpl.__version__}")
data_path = "../input/"
test = pd.read_csv(data_path+"titanic/test.csv")

train = pd.read_csv(data_path+"titanic/train.csv")



# df = pd.read_csv(data_path+'imdb.csv',error_bad_lines=False);

df = train
df
df.columns = [i.lower() for i in df.columns]
df.head(3)
sns.set(style="white", palette="muted", color_codes=True)



f, axes = plt.subplots(3, 2, figsize=(14, 14))

sns.despine(left=True) # hides the border



sns.countplot(x="pclass", data=df, hue="sex", ax=axes[0, 0])

sns.boxplot(x="age",  hue="sex" , data=df, ax=axes[0, 1])

sns.countplot(x="survived", data=df, hue="sex", ax=axes[1, 0])

sns.swarmplot(x='embarked', y='pclass', data=df, ax=axes[1, 1])

sns.swarmplot(x="survived", y="fare", data=df,ax=axes[2, 0])

sns.boxplot(x="embarked", y="survived", data=df,ax=axes[2, 1])

plt.tight_layout()
C, Q, S = len(df[df['embarked']=='C']), len(df[df['embarked']=='Q']), len(df[df['embarked']=='S'])
import folium



iceburg = [41.7666636, -50.2333324]

southampton = [50.909698, -1.404351]

cherbourg = [49.630001, -1.620000]

queenstown = [51.850334, -8.294286]



m = folium.Map(

    location=iceburg,

    tiles='Stamen Terrain',

    zoom_start=3

)



tooltip = 'Click me!'



folium.Marker(southampton, popup='<h3>1. Southampton, 10 April 1912 </h3> <i> Titanic successfully arrives at Southampton shortly after midnight</i>', tooltip=tooltip).add_to(m)

folium.Marker(cherbourg, popup='<h3>2.Cherbourg, 10 April 1912 </h1>', tooltip=tooltip).add_to(m)

folium.Marker(queenstown, popup='<h3>3.Queenstown, 11 April 1912 </h1> ', tooltip=tooltip).add_to(m)

#  https://latitude.to/articles-by-country/general/942/sinking-of-the-rms-titanic

folium.Marker(iceburg, 

              popup='<h3>4. Crash - 15 April 1912 </h3>', 

              tooltip=tooltip,

             icon=folium.Icon(color='red', icon='info-sign')).add_to(m)

folium.PolyLine([southampton, cherbourg], fill_color="red").add_to(m)

folium.PolyLine([cherbourg, queenstown], fill_color="red").add_to(m)

folium.PolyLine([queenstown, iceburg], fill_color="red").add_to(m)

m
plt.rcParams['figure.figsize'] = (7, 7)

wordcloud = WordCloud(stopwords=STOPWORDS,background_color = 'white', width = 1000,  

                      height = 1000, max_words = 200).generate(' '.join(df['name']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Passenger Names',fontsize = 10)

plt.show();