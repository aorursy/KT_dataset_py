# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import seaborn as sns 

import numpy as np
shu =pd.read_csv('../input/top50spotify2019/top50.csv',encoding='ISO-8859-1',index_col=0)
shu.head(5)
from matplotlib import cm
sns.FacetGrid(shu,height=10).map(sns.distplot,'Beats.Per.Minute').add_legend()

plt.title("pdf of Bpm")

plt.ylabel("probablity")

plt.grid()

plt.show()
sns.FacetGrid(shu,height=10).map(sns.distplot,'Energy',bins=np.linspace(0,100,50)).add_legend()

plt.title("pdf of energy")

plt.ylabel("probablity")

plt.grid()

plt.show()
shu['Artist.Name'].value_counts().plot(kind='bar',figsize=(10,8),colormap=cm.get_cmap('ocean'))
sns.FacetGrid(shu,height=10).map(sns.distplot,'Danceability',bins=np.linspace(0,100,50)).add_legend()

plt.title("pdf of danceabilty")

plt.ylabel("probablity")

plt.grid()

plt.show()
wordcloud=WordCloud(width=1000,height=600,max_font_size=200,max_words=150,background_color='black').generate("".join(shu['Artist.Name']))

plt.figure(figsize=[10,10])

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis("off")

plt.show()
wordcloud=WordCloud(width=1000,height=600,max_font_size=200,max_words=150,background_color='black').generate("".join(shu['Track.Name']))

plt.figure(figsize=[10,10])

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis("off")

plt.show()
shu.info()
#bit-variante

sns.set_style("whitegrid")

sns.FacetGrid(shu,height=6).map(plt.scatter,'Beats.Per.Minute','Popularity').add_legend()

plt.show()
sns.set_style("whitegrid")

sns.FacetGrid(shu,height=6).map(plt.scatter,'Energy','Popularity').add_legend()

plt.show()
sns.set_style("whitegrid")

sns.FacetGrid(shu,height=5).map(plt.scatter,'Danceability','Popularity').add_legend()

plt.show()
shu.isnull().sum()
shu.rename(columns={'Track.Name':'Track_Name','Artist.Name':'Artist_Name','Beats.Per.Minute':'Beats_Per_Minute','Loudness..dB..':'Loudness_dB','Valence.':'Valence','Length.':'Length','Acousticness..':'Acousticness','Speechiness.':'Speechiness'},inplace=True)
shu.head()
data= shu.groupby('Artist_Name')
data.first()
data.get_group('Ed Sheeran')
data_1 = shu.groupby('Popularity')
data_1.first().max()
data_2 = shu.groupby(['Genre','Popularity'])
data_2.first()
tag="Shawn Mendes"

shu['relevent']=shu["Artist_Name"].fillna("").apply(lambda x:1 if tag.lower() in x.lower() else 0)

small = shu[shu['relevent']==1]

small[["Track_Name","Genre","Popularity"]]
tag="Ed Sheeran"

shu['relevent']=shu["Artist_Name"].fillna("").apply(lambda x:1 if tag.lower() in x.lower() else 0)

small = shu[shu['relevent']==1]

small[["Track_Name","Genre","Popularity"]]
small =shu.sort_values("Beats_Per_Minute",ascending=True)

small =small[small['Energy']!=""]

small[["Track_Name","Beats_Per_Minute"]][:20]
small =shu.sort_values("Energy",ascending=True)

small =small[small['Beats_Per_Minute']!=""]

small[["Track_Name","Energy"]][:20]