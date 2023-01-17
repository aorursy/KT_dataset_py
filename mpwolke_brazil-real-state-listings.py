#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRDL_3FzBXNhTKkHNJ0uZjssO63ngHQ4qWLYy-BEJ_xRXbU3Vcc',width=400,height=400)
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
df= pd.read_json('/kaggle/input/brazil-real-estate-listings/properati-real-estate-listings-brazil/datapackage.json',orient='columns',lines=True)
df.head()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.title)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
nRowsRead = 1000 # specify 'None' if want to read whole file

# properati_br_2016_11_01_properties_rent.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/brazil-real-estate-listings/properati-real-estate-listings-brazil/data/properati_br_2016_11_01_properties_rent.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'properati_br_2016_11_01_properties_rent.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head()
plt.figure(figsize=(12,4))

sns.countplot(hue=df1['property_type'], x=df1['price'])

plt.show()
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf1 = df1.groupby('property_type').size()/df1['place_name'].count()*100

labels = lowerdf1.index

values = lowerdf1.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
plt.figure(figsize=(8,4))

sns.scatterplot(x='property_type',y='rooms',data=df1)

plt.xticks(rotation=90)

plt.yticks(rotation=45)

plt.show()
nRowsRead = 1000 # specify 'None' if want to read whole file

# properati_br_2016_11_01_properties_rent.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df2 = pd.read_csv('/kaggle/input/brazil-real-estate-listings/properati-real-estate-listings-brazil/original/properati-BR-2016-11-01-properties-sell.csv', delimiter=',', nrows = nRowsRead)

df2.dataframeName = 'properati-BR-2016-11-01-properties-sell.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df2.plot(subplots=True, figsize=(10, 10), sharex=False, sharey=False)

plt.show()
df2.head()
sns.countplot(df2["property_type"])

plt.xticks(rotation=90)

plt.show()
labels1=df2.property_type.value_counts().index

sizes1=df2.property_type.value_counts().values

plt.figure(figsize=(11,11))

plt.pie(sizes1,labels=labels1,autopct="%1.1f%%")

plt.title("Property Type",size=15)

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df2.description)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()