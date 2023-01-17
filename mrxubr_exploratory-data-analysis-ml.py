import pandas as pd

import seaborn as sns

#df = pd.read_csv('../../dataset/training/train_lyrics_1000.csv', usecols=range(7))



#df.head()# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv('../input/lyrics-generation/train_lyrics_1000.csv', usecols=range(7))

df.head()
from matplotlib import pyplot as plt

%matplotlib inline
blue = '#5A6FFA'

green = '#FFFF00'
happy, sad = sum(df.loc[:, 'mood'] == 'happy'), sum(df.loc[:, 'mood'] == 'sad')

print(happy, sad)
from matplotlib import rcParams

rcParams['font.size'] = 18



piechart = plt.pie(

    (happy, sad),

    labels=('happy','sad'),

    shadow=True,

    colors=(green, blue),

    explode=(0,0.15), # space between slices 

    startangle=90,    # rotate conter-clockwise by 90 degrees

    autopct='%1.1f%%',# display fraction as percentages

)



plt.axis('equal')   

plt.tight_layout()

#plt.savefig('./images/pie_happy_sadn.eps', dpi=300)
import numpy as np

import seaborn as sns



sns.set_style('whitegrid');



plt.hist(df['year'], bins=np.arange(1900, 2020,5),color = "green", rwidth = 20)

plt.xlabel('year')

plt.ylabel('count')

plt.xlim([df['year'].min()-5, df['year'].max()+5])

plt.tight_layout()

#plt.savefig('./images/histo_yearn.eps', dpi=300)
import warnings

warnings.filterwarnings('ignore')

sns.pairplot(data = df ,hue = 'genre' ,size = 8 ,diag_kind = 'kde',palette="husl")

plt.legend(loc='lower center',  ncol=3)
!pip install wordcloud
gclass = df.groupby(['genre', 'mood']).size().unstack()



print(gclass)







fig = plt.figure(figsize=(10,4))



sns.set(style="white")



pos = np.arange(1,13)



# absolute values

plt.subplot(121)

plt.bar(pos, gclass.values[:,0], label='happy', color=green)

plt.bar(pos, gclass.values[:,1], bottom=gclass.values[:,0], label='sad', color=blue)

plt.xticks(pos+0.5, gclass.index, rotation='vertical')

plt.ylabel("Count")

plt.xlabel("")

plt.legend(loc='upper left')



plt.gca().yaxis.grid(True) 



# relative values



# normalize

gclass = (gclass.T / gclass.T.sum()).T 



plt.subplot(122)

plt.bar(pos, gclass.values[:,0], label='happy', color=green)

plt.bar(pos, gclass.values[:,1], bottom=gclass.values[:,0], label='sad', color=blue)

plt.xticks(pos+0.5, gclass.index, rotation='vertical')

plt.ylabel('Fraction')

plt.axhline(y=0.5, xmin=0, linewidth=2, color='black', alpha=0.5)

plt.xlabel('')

plt.tight_layout()

#plt.savefig('./images/bar_genre_moodn.eps', dpi=300)
bins = np.arange(1960,2011,10)

happy_bins, b = np.histogram(df.loc[df.loc[:,'mood']=='happy', 'year'], bins=bins)

sad_bins, b = np.histogram(df.loc[df.loc[:,'mood']=='sad', 'year'], bins=bins)

year_bins, b = np.histogram(df.loc[:, 'year'], bins=bins)



fig = plt.figure(figsize=(10,4))



sns.set(style="white")



pos = np.arange(1,6)

labels = ['%s-%s' %(i, i+10) for i in np.arange(1960,2011,10)]



# absolute values

plt.subplot(121)

plt.bar(pos, happy_bins, label='happy', color=green)

plt.bar(pos, sad_bins, bottom=happy_bins, color=blue, label='sad')

plt.xticks(pos, labels, rotation=30)

plt.ylabel("Count")

plt.xlabel("")

plt.legend(loc='upper left')



plt.gca().yaxis.grid(True) 



# relative values



# normalize

happy_bins = happy_bins / year_bins

sad_bins = sad_bins / year_bins



plt.subplot(122)

plt.bar(pos, happy_bins, color=green)

plt.bar(pos, sad_bins, bottom=happy_bins, color=blue, label='sad')

plt.xticks(pos, labels, rotation='30')

plt.ylabel("Fraction")

plt.axhline(y=0.5, xmin=0, linewidth=2, color='black', alpha=0.5)

plt.xlabel("")

plt.tight_layout()

#plt.savefig('./images/bar_year_moodn.eps', dpi=300)
%matplotlib inline
from scipy.misc import imread

from wordcloud import WordCloud

import matplotlib.pyplot as plt

from sklearn.feature_extraction import text 

from wordcloud import WordCloud, STOPWORDS

twitter_mask = imread('../input/twitter/twitter_mask.png', flatten=True)





happy_words = ' '.join(df.loc[df['mood']=='happy', 'lyrics']).encode().decode("utf-8", "replace")



happy_wordcloud = WordCloud( 

                      font_path='../input/font-style/CabinSketch-Bold.ttf',

                      stopwords=STOPWORDS,

                      background_color='white',

                      width=1800,

                      height=1400,

                      mask=twitter_mask

            ).generate(happy_words)



plt.figure( figsize=(20,10), facecolor='k')

plt.imshow(happy_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
sad_words = ' '.join(df.loc[df['mood']=='sad', 'lyrics']).encode().decode("utf-8", "replace")



sad_wordcloud = WordCloud( 

                      font_path='../input/font-style/CabinSketch-Bold.ttf',

                      stopwords=STOPWORDS,

                      background_color='white',

                      width=1800,

                      height=1400

            ).generate(sad_words)



plt.figure( figsize=(20,10), facecolor='k')

plt.imshow(sad_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
words = ' '.join(df.loc[:, 'lyrics']).encode().decode("utf-8", "replace")



wordcloud = WordCloud( 

                      font_path='../input/font-style/flux architect bold.ttf',

                      stopwords=STOPWORDS,

                      background_color='white',

                      width=1800,

                      height=1400

            ).generate(words)









plt.figure( figsize=(20,10), facecolor='k')

plt.imshow(wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
