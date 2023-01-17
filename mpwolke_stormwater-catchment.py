#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSkpCtrcNezrDL1J4nr5QmPG9uMTiSkVakZVH-wscsXElxoE1z2zw&s',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTRD7Tw9T38FfJyA2b9Xj0usIdjnFOPRLbnzhwZRZafPc2aWvcjTg&s',width=400,height=400)
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadscatchcsv/catch.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'catch.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
df.dtypes
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSaY-EzPvk6LCFk2STaOVodvDeYuzsVZ3HUNWDlXFc96ywYFTEo&s',width=400,height=400)
print("The number of nulls in each column are \n", df.isna().sum())
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTUAWCOiapvlFeGSoCzBZezYG8ygCIyraf8rMojXYzdv9swrT-j&s',width=400,height=400)
sns.distplot(df["OBJECTID"].apply(lambda x: x**4))

plt.show()
fig, ax =plt.subplots(figsize=(8,6))

sns.scatterplot(x='OBJECTID', y='SS_CATCH', data=df)

plt.xticks(rotation=90)

plt.yticks(rotation=45)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS8qmW_lUjCLvc_Y2vLCIKiyG2YOGRkWufWJEjRkQvbbOmR-S0Ujw&s',width=400,height=400)
sns.regplot(x=df['OBJECTID'], y=df['SS_CATCH'])
sns.regplot(x=df['OBJECTID'], y=df['SS_CATCH1'])
sns.lmplot(x="OBJECTID", y="SS_CATCH", hue="SS_CATCH1", data=df)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTog8rGP_d4vW3v-mKHKxyK5CrdzNcJuprB2b2UbP8R-V8LfZS4cA&s',width=400,height=400)
plt.figure(figsize=(10,10))

plt.title('STORM CATCHMENT')

ax=sns.heatmap(df.corr(),

               linewidth=2.6,

               annot=True,

               center=1)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ_OIaIvB4BkWYBqnmSr1GYV6SoHReZ6hIqmeS1RRBsGEcYkRkTTA&s',width=400,height=400)
#Codes from Andre Sionek @andresionek

import plotly.express as px



# Grouping it by Genre and track

plot_data = df.groupby(['OBJECTID', 'SS_CATCH'], as_index=False).SS_CATCH1.sum()



fig = px.bar(plot_data, x='OBJECTID', y='SS_CATCH1', color='SS_CATCH')

fig.update_layout(

    title_text='STORM CATCHMENT',

    height=500, width=1000)

fig.show()
#Codes from Andre Sionek @andresionek

import plotly.express as px



# Grouping it by Genre and track

plot_data = df.groupby(['OBJECTID', 'SS_CATCH'], as_index=False).SOE.sum()



fig = px.line_polar(plot_data, theta='OBJECTID', r='SOE', color='SS_CATCH')

fig.update_layout(

    title_text='STORM CATCHMENT',

    height=500, width=1000)

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSV-Tfm4d3uXm7ec1k3P0iyo5DMgy33g9kG8UodsD0CHeVhIZKW&s',width=400,height=400)
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('OBJECTID').size()/df['SS_CATCH'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.NAME)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="white").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.SUB_NUM)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRQ8-RX3N1W1kyVNOQMgPzE_MWUREYZIZUFvdAf7AYWqcL0AKSD&s',width=400,height=400)