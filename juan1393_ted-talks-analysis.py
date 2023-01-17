%matplotlib inline

import numpy as np

import pandas as pd

from wordcloud import WordCloud

import matplotlib.pyplot as plt
df = pd.read_csv('../input/ted_main.csv')
df.head()
df.info()
df[df.isnull().any(axis=1)]
df['published_date'] = pd.to_datetime(df['published_date'], unit='s')
published_date_year = df.groupby(df['published_date'].dt.year)['published_date'].count()
bar_labels = published_date_year.keys()

x_pos = list(range(len(published_date_year)))



plt.bar(x_pos,

        # using the data from the mean_values

        published_date_year, 

        # aligned in the center

        align='center',

        # with color

        color='#FFC222')



plt.ylabel('Count')

plt.xticks(x_pos, bar_labels, rotation='vertical')

plt.title('Number of talks per year')



plt.show()
most_viewed = df.sort_values(by='views', ascending=False)

most_viewed
wordcloud = WordCloud(max_font_size=40).generate(' '.join(df['tags']))

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
df.corr()