# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud,STOPWORDS



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/star-wars-survey-data/star_wars.csv", encoding="ISO-8859-1")

df.head()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
#codes with eiv from Tanmay Pandey @tanmaypandey

from PIL import Image

mask = np.array(Image.open("../input/star-wars-movie-scripts/wordcloud_masks/yoda.png"))
evi=open('../input/star-wars-movie-scripts/SW_EpisodeVI.txt','r')
evi=evi.read()
stop_words=set(STOPWORDS)

evi_wc=WordCloud(width=800,height=500,mask=mask,random_state=21, max_font_size=110,stopwords=stop_words).generate(evi)
fig=plt.figure(figsize=(16,8))

plt.imshow(evi_wc)
categorical_cols = [cname for cname in df.columns if

                    df[cname].nunique() < 10 and 

                    df[cname].dtype == "object"]





# Select numerical columns

numerical_cols = [cname for cname in df.columns if 

                df[cname].dtype in ['int64', 'float64']]
print(numerical_cols)
sns.scatterplot(x='RespondentID',y='Gender',data=df)
sns.countplot(df["RespondentID"])

plt.xticks(rotation=45)

plt.show()
sns.countplot(df["Gender"])

plt.xticks(rotation=45)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS0zpt4W3eBqaw1QCNVR8jwJKmlbyXjFciavTE5cEu4Mfvva-00&s',width=400,height=400)
sns.countplot(df["Age"])

plt.xticks(rotation=45)

plt.show()
episodeIV = pd.read_csv('../input/star-wars-movie-scripts/SW_EpisodeIV.txt', delim_whitespace=True, names=["index","character","dialogue"] ,header = None)

episodeV = pd.read_csv('../input/star-wars-movie-scripts/SW_EpisodeV.txt', delim_whitespace=True, names=["index","character","dialogue"] ,header = None)

episodeVI = pd.read_csv('../input/star-wars-movie-scripts/SW_EpisodeVI.txt', delim_whitespace=True, names=["index","character","dialogue"] ,header = None)
episodeV.head()
episodeIV["episode"]="A New Hope"

episodeV["episode"]="The Empire Strikes Back"

episodeVI["episode"]="Return of The Jedi"

data=pd.concat([episodeIV,episodeV,episodeVI],axis=0,ignore_index=True)
vader=data[data.character=="VADER"]
from PIL import Image

wave_mask_vader= np.array(Image.open("../input/star-wars-movie-scripts/wordcloud_masks/vader.png"))
plt.subplots(figsize=(10,10))

stopwords= set(STOPWORDS)

wordcloud = WordCloud(mask=wave_mask_vader,background_color="black",colormap="gray" ,contour_width=2, contour_color="gray",

                      width=950,

                          height=950

                         ).generate(" ".join(vader.dialogue))



plt.imshow(wordcloud ,interpolation='bilinear')

plt.axis('off')

plt.savefig('graph.png')



plt.show()
yoda=data[data.character=="YODA"]
wave_mask_yoda = np.array(Image.open("../input/star-wars-movie-scripts/wordcloud_masks/yoda.png"))
plt.subplots(figsize=(10,10))

stopwords= set(STOPWORDS)

wordcloud = WordCloud(mask=wave_mask_yoda,background_color="white",colormap="gray" ,contour_width=2, contour_color="gray",

                      width=950,

                          height=950

                         ).generate(" ".join(yoda.dialogue))



plt.imshow(wordcloud ,interpolation='bilinear')

plt.axis('off')

plt.savefig('graph.png')



plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRRDF4pUdmKtnRNzdBNwj_1ctZtD3UvxbWy6InbCYcvo5zjptuWVg&s',width=400,height=400)
sns.countplot(df["Education"])

plt.xticks(rotation=45)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcScV4pFozAvPaCMTLdnFGIbKJrympui7psnF0gNDicxTg-1PB1lQw&s',width=400,height=400)