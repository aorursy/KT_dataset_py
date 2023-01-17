# inspired by https://www.kaggle.com/nikitaromanov/d/egrinstein/20-years-of-games/a-quick-review-of-the-ign-reviews/notebook
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import plotly

%matplotlib inline

from bokeh.models import HoverTool, ColumnDataSource

from bokeh.palettes import Viridis6

from bokeh.plotting import figure, show, output_notebook



N = 4000

x = np.random.random(size=N) * 100

y = np.random.random(size=N) * 100

radii = np.random.random(size=N) * 1.5

colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(np.floor(50+2*x), np.floor(30+2*y))]

output_notebook()
TOOLS="resize,crosshair,pan,wheel_zoom,box_zoom,reset,tap,previewsave,box_select,poly_select,lasso_select"



p = figure(tools=TOOLS)

p.scatter(x,y, radius=radii, fill_color=colors, fill_alpha=0.6, line_color=None)

#hover = p.select_one(HoverTool)

#hover.point_policy = "follow_mouse"

#hover.tooltips = [

#    ("Radius", "@radii"),

#    ("(Long, Lat)", "($x, $y)"),

#]



show(p)
vgs = pd.read_csv('../input/vgsales.csv')
vgs.info()
vgs.head()


plt.figure(figsize=(19,16)) 

sns.set_context("notebook", font_scale=1.1)

sns.set_style("ticks")



sns.lmplot('Year', 'Global_Sales', 

           data=vgs, 

           fit_reg=False, 

           #dropna=True,

           hue="Genre",  

           scatter_kws={"marker": "D", 

                        "s": 100})

plt.title('Histogram of Genres sales over years')

plt.xlabel('Global Sales')

plt.ylabel('Time')

#plt.figure(figsize=(19,16)) 

#table_sales = pd.pivot_table(vgs,values=['Global_Sales'],index=['Genre'],columns=['Platform'],aggfunc='sum',margins=False)

ts= vgs.groupby('Genre')

ts.info()

ts.head()

#sns.set_context("notebook", font_scale=1.1)

#sns.set_style("ticks")



#sns.stripplot('Global_Sales', 'Genre', data=table_sales, jitter=True, hue="Platform")

#plt.title('Histogram of Genres sales')

#plt.xlabel('Global Sales')

#plt.ylabel('Time')
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