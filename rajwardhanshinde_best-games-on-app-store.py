!pip install bubbly
import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import plotly.express as px

py.init_notebook_mode(connected=True)



from bokeh.io import output_notebook, output_file, show, push_notebook

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource, CategoricalColorMapper, HoverTool

from bokeh.models.widgets import Tabs, Panel

output_notebook()



from IPython.html.widgets import interact



from bubbly.bubbly import bubbleplot

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print('Data:-')

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv', parse_dates=['Original Release Date', 'Current Version Release Date'])

df.head()
print(f"There are {df.shape[0]} rows and {df.shape[1]} columns in dataset")
df.info()
df.describe()
'''Here we are extracting the apps having at least 200 reviews and selecting our primary genre as games'''

df = df.loc[(df['User Rating Count'] > 200) & (df['Primary Genre']=='Games')]
'''A Function To Plot Pie Plot using Plotly'''



def pie_plot(cnt_srs, colors, title):

    labels=cnt_srs.index

    values=cnt_srs.values

    trace = go.Pie(labels=labels, 

                   values=values, 

                   title=title, 

                   hoverinfo='percent+value', 

                   textinfo='percent',

                   textposition='inside',

                   hole=0.7,

                   showlegend=True,

                   marker=dict(colors=colors,

                               line=dict(color='#000000',

                                         width=2),

                              )

                  )

    return trace
py.iplot([pie_plot(df['Genres'].value_counts().sort_values(ascending=False).head(10), ['cyan'], 'Genres')])
'''Converting Size in MB's'''

df['Size'] = round(df['Size']/1000000)

fig = bubbleplot(dataset=df, x_column='Average User Rating', y_column='User Rating Count', size_column='Size', bubble_column='Genres',

                 color_column='Genres', x_title='Avg Rating', y_title='Ratings Count', title='Ratings vs Rating_Count', x_logscale=False, 

                 y_logscale=True,

                 scale_bubble=3, height=650)

py.iplot(fig)
df['Release Year'] = df['Original Release Date'].dt.year



fig, ax = plt.subplots(1, 2, figsize=(15, 8))

sns.lineplot(x='Release Year', y='Price', data=df, palette='Wistia', ax=ax[0])

ax[0].set_title('Release Year vs Price')



sns.lineplot(x='Release Year', y='Size', data=df, palette='Wistia', ax=ax[1])

ax[1].set_title('Relase Year vs Size')

plt.tight_layout()

plt.show()
df.dropna(inplace=True)

data = df.set_index('Release Year')

y = data.loc[2008].Size

x = data.loc[2008].Price

data = data[['Name', 'Price', 'Size']]

output_notebook()
source = ColumnDataSource(data={

    "x": data.loc[2009].Price,

    "y": data.loc[2009].Size,

    "Name": data.loc[2009].Name,

    "Price": data.loc[2009].Price,

    "Size": data.loc[2009].Size

})

hover = HoverTool(tooltips=[('Name', '@Name'), ('Price', '@Price'), ('Size', '@Size')])

plot = figure(title='Evolution Of Games', x_axis_label='Price', y_axis_label='Size', tools=[hover, 'crosshair', 'pan', 'box_zoom'])

plot.circle('x', 'y' , source=source, hover_color='red')



def update(x_axis, y_axis, year=2009):   

    c1=x_axis

    c2=y_axis

    new_data={

        "x":data.loc[year, c1],

        "y":data.loc[year, c2],

        "Name":data.loc[year].Name,

        "Price": data.loc[year].Price,

        "Size": data.loc[year].Size

     }

    source.data = new_data

    plot.xaxis.axis_label=c1

    plot.yaxis.axis_label=c2

    push_notebook()



show(plot, notebook_handle=True)
'''Toggle year from here It will show up when u will fork the kernel and run this cell'''

interact(update, x_axis=['Price'], y_axis=['Size'], year=(2009, 2019))
paid = df[df['Price']>0]

free = df[df['Price']==0]

fig, ax = plt.subplots(1, 2, figsize=(15,8))

sns.countplot(data=paid, y='Average User Rating', ax=ax[0], palette='plasma')

ax[0].set_title('Paid Games')

ax[0].set_xlim([0, 1000])



sns.countplot(data=free, y='Average User Rating', ax=ax[1], palette='viridis')

ax[1].set_title('Free Games')

ax[1].set_xlim([0,1000])

plt.tight_layout();

plt.show()
py.iplot([pie_plot(df['Age Rating'].value_counts(), ['cyan', 'gold', 'red'], 'Age Rating')])
price = df.sort_values(by='Price', ascending=False)[['Name', 'Price', 'Average User Rating', 'Size', 'Icon URL']].head(10)

price.iloc[:, 0:-1]
import urllib.request

from PIL import Image



plt.figure(figsize=(6,3))

plt.subplot(121)

image = Image.open(urllib.request.urlopen(price.iloc[0,-1]))

plt.imshow(image)

plt.axis('off')



plt.subplot(122)

image = Image.open(urllib.request.urlopen(price.iloc[1,-1]))

plt.imshow(image)

plt.axis('off')



plt.show()
review = df.sort_values(by='User Rating Count', ascending=False)[['Name', 'Price', 'Average User Rating', 'Size', 'User Rating Count', 'Icon URL']].head(10)

review.iloc[:, 0:-1]
plt.figure(figsize=(6,3))

plt.subplot(131)

image = Image.open(urllib.request.urlopen(review.iloc[0,-1]))

plt.imshow(image)

plt.axis('off')



plt.subplot(132)

image = Image.open(urllib.request.urlopen(review.iloc[1,-1]))

plt.imshow(image)

plt.axis('off')



plt.subplot(133)

image = Image.open(urllib.request.urlopen(review.iloc[2,-1]))

plt.imshow(image)

plt.axis('off')



plt.show()
best = df.sort_values(by=['Average User Rating', 'User Rating Count'], ascending=False)[['Name', 'Average User Rating', 'User Rating Count', 'Size', 

                                                                                         'Price', 'Icon URL']].head(10)

best.iloc[:, 0:-1]
plt.figure(figsize=(5,5))

image = Image.open(urllib.request.urlopen(best.iloc[0, -1]))

plt.axis('off')

plt.imshow(image)

plt.show()