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



#Read csv files.

df = pd.read_csv('/kaggle/input/kojima-tweets-en/kojima_tweets_en.csv')



#Created dataframe and sort by Favourites.

tweets_df = df.sort_values(by='Favourites', ascending=False)



#Get needs columns.

tweets_df = tweets_df[['Tweet', 'Retweets', 'Created Date', 'Favourites']]
tweets_df.head()
!pip install wordcloud
text_list = tweets_df['Tweet'].to_list()

text = " ".join(text_list)

text
from wordcloud import WordCloud

import matplotlib.pyplot as plt 

 

stop_words = ['https','co', 'デススト', 'デスストでつながれ']  

 

wordcloud = WordCloud(

    width=900, height=600,   # default width=400, height=200

    background_color="white",   # default=”black”

    stopwords=set(stop_words),

    max_words=200,   # default=200

    min_font_size=4,   #default=4

    collocations = False   #default = True

    ).generate(text)

 

plt.figure(figsize=(15,12))

plt.imshow(wordcloud)

plt.axis("off")

plt.savefig("word_cloud.png")

plt.show()