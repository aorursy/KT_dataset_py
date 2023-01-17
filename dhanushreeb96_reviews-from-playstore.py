# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ruzzle= pd.read_csv('../input/Ruzzle.csv')
ruzzle=ruzzle.rename(columns={'Most Relevant reviews for ruzzle': 'reviews'})
ruzzle
gre= pd.read_csv('../input/GRE.csv', encoding='cp1252')
gre=gre.rename(columns={'Most Relevant Reviews for GRE Vocab Flashcards': 'reviews'})
gre
impeng= pd.read_csv('../input/Improve English.csv', encoding='cp1252')
impeng=impeng.rename(columns={'Most Relavant': 'reviews'})
impeng
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
stopwords=set(STOPWORDS)
stopwords.update(['app','keep','second','this','much','wa','even','Im','also'])
def make_wordcloud(text, title=None):
    wordcloud=WordCloud(stopwords= stopwords, background_color="white", random_state=1,min_font_size=6,collocations=False).generate(text)
    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()
ruzzlerev=" ".join(phrase for phrase in ruzzle.reviews)
grerev=" ".join(phrase for phrase in gre.reviews)
impengrev=" ".join(phrase for phrase in impeng.reviews)
make_wordcloud(ruzzlerev)
make_wordcloud(grerev)
make_wordcloud(impengrev)