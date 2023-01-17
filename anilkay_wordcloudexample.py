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
data=pd.read_csv("/kaggle/input/nyt-comments/ArticlesMarch2018.csv")

data.head()
snippet=data["snippet"]
text=""

for snip in snippet:

    text=text+" "+snip
print(text[0:1000])
from wordcloud import STOPWORDS

type(STOPWORDS)

STOPWORDS.add("one")

STOPWORDS.add("say")

STOPWORDS.add("said")

STOPWORDS.add("will")

STOPWORDS.add("well")

STOPWORDS.add("will")
%matplotlib inline

from wordcloud import WordCloud

wordcloud = WordCloud(max_font_size=40,stopwords=STOPWORDS).generate(text.lower())

import matplotlib.pyplot as plt

plt.figure(figsize=(20,40))

plt.imshow(wordcloud)