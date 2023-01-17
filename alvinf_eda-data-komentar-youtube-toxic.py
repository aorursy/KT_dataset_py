# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

from PIL import Image



from nltk.tokenize import word_tokenize

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import matplotlib.pyplot as plt

from matplotlib import pyplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/data-komentar-video-youtube-toxic-ericko-lim/dfyutup-cleanedfixversion3.csv')

df.head()
df = df[df['Comments'].notnull()]

df.head()
df['word_count'] = df['Comments'].apply(lambda x: len(str(x).split(" ")))

df[['Comments','word_count']].head()
df['char_count'] = df['Comments'].str.len() #spasi termasuk

df[['Comments','char_count']].head()
freq = pd.Series(' '.join(df['Comments']).split()).value_counts()[:30]

freq
top_like = df[['Author','likeCount']].sort_values(by='likeCount', ascending=False).head(15)

top_like
authorCom=df[['Author','likeCount']].groupby("Author")

plt.figure(figsize=(15,10))

authorKomeng=authorCom.mean().sort_values(by="likeCount",ascending=False)["likeCount"].head(35).plot.bar()



plt.xticks(rotation=90)

plt.xlabel("Author")

plt.ylabel("JumlahLike")

plt.show()
df[['likeCount','Comments','Author']].sort_values(by='likeCount', ascending=False).head(15)
ymask = np.array(Image.open("../input/logoyoutube/youtube-logo.png"))



text=df['Comments']

text1=str(text)

type(text1)

wordcloud=WordCloud(background_color="white", mask=ymask).generate(text1)



plt.figure(figsize=(15,10))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis("off")

plt.show()