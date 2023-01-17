# inspired by https://www.kaggle.com/nikitaromanov/d/egrinstein/20-years-of-games/a-quick-review-of-the-ign-reviews/notebook
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

%matplotlib inline
vgs = pd.read_csv('../input/vgsales.csv')
vgs.info()
vgs.head()
table_sales = pd.pivot_table(vgs,values=['Global_Sales'],index=['Year'],columns=['Genre'],aggfunc='max',margins=False)



plt.figure(figsize=(19,16))

sns.heatmap(table_sales['Global_Sales'],linewidths=.5,annot=True,vmin=0.01,cmap='PuBu')

plt.title('Max Global_Sales of games')
def top(df, n = 1, column = 'Global_Sales'):

    return df.sort_values(by=column)[-n:]
vgs.groupby(['Year'], group_keys=False).apply(top)[['Year', 'Name', 'Global_Sales', 'Genre', 'Platform', 'Publisher']]
vgs.groupby(['Name'])['Global_Sales'].sum().sort_values(ascending=False)[:40]
table_count = pd.pivot_table(vgs,values=['Global_Sales'],index=['Year'],columns=['Genre'],aggfunc='count',margins=False)



plt.figure(figsize=(19,16))

sns.heatmap(table_count['Global_Sales'],linewidths=.5,annot=True,fmt='2.0f',vmin=0)

plt.title('Count of games')
most_pub = vgs.groupby('Publisher').Global_Sales.sum()

most_pub.sort_values(ascending=False)[:20]



table_publisher = pd.pivot_table(vgs[vgs.Publisher.isin(most_pub.sort_values(ascending=False)[:20].index)],values=['Global_Sales'],index=['Year'],columns=['Publisher'],aggfunc='sum',margins=False)





plt.figure(figsize=(19,16))

sns.heatmap(table_publisher['Global_Sales'],linewidths=.5,annot=True,vmin=0.01,cmap='PuBu')

plt.title('Sum Publisher Global_sales of games')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from PIL import Image



stopwords = set(STOPWORDS)



for x in vgs.Genre.unique():

    wc = WordCloud(background_color="white", max_words=2000, 

                   stopwords=stopwords, max_font_size=40, random_state=42)

    wc.generate(vgs.Name[vgs.Genre == x].to_string())

    plt.imshow(wc)

    plt.title(x)

    plt.axis("off")

    plt.show()