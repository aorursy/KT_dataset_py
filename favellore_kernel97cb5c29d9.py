# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from wordcloud import WordCloud, STOPWORDS



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Texas Last Statement - CSV.csv", encoding="latin1")
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")
df.head()
sns.countplot(x='Race',data=df,palette='viridis')
#Bar chart of Age

df.Age.value_counts().head(25).plot.bar()
#line plot of Age

df.Age.value_counts().sort_index().plot.line()
df.groupby(['CountyOfConviction','Age']).NumberVictim.sum().nlargest(10).plot(kind='barh')
lastword = df["LastStatement"]

wordcloud_q = WordCloud(

                          background_color='white',

                          stopwords=set(STOPWORDS),

                          max_words=250,

                          max_font_size=40, 

                          random_state=1705

                         ).generate(str(lastword))
def cloud_plot(wordcloud):

    fig = plt.figure(1, figsize=(20,15))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
cloud_plot(wordcloud_q)