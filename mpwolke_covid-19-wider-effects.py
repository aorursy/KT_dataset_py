# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

import plotly.graph_objects as go

import plotly.offline as py



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadssecondarycsv/secondary.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'secondary.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head()
# checking dataset



print ("Rows     : " ,df.shape[0])

print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n" ,df.columns.tolist())

print ("\nMissing values :  ", df.isnull().sum().values.sum())

print ("\nUnique values :  \n",df.nunique())
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
ax = df['Indicator'].value_counts().plot.barh(figsize=(18, 10))

ax.set_title('Indicator Distribution', size=18)

ax.set_ylabel('Indicator', size=14)

ax.set_xlabel('Count', size=14)
ax = df['Drivers_of_COVID19_Impact'].value_counts().plot.barh(figsize=(14, 6), color='r')

ax.set_title('Drivers of COVID19 Impact Distribution', size=18)

ax.set_ylabel('Drivers_of_COVID19_Impact', size=14)

ax.set_xlabel('Count', size=14)
ax = df['Block'].value_counts().plot.barh(figsize=(14, 6), color='g')

ax.set_title('Thematic Blocks Distribution', size=18)

ax.set_ylabel('Block', size=14)

ax.set_xlabel('Count', size=14)
sns.countplot(x="Pillar",data=df,palette="GnBu_d",edgecolor="black")

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.title("Pillars of Covid-19 Impacts")

# changing the font size

sns.set(font_scale=1)
ax = sns.countplot(x = 'Justification',data=df,order=[' No elections taking place in 2020', 'Post-Covid-19 Benin had a 42.2% povertty rate while pre-Covid-19 was 41.1%.', 'Unemployment rate for men is calculated on a yearly basis, thus the latest data is from 2019 which is 2.13% ', ' Insufficient data regarding IDPs for the COVID-19 pandemic. The latest IDPs report was indicated in 2018.'])

for p in ax.patches:

        ax.annotate(format(p.get_height()), (p.get_x()+0.2, p.get_height()))

plt.xticks(rotation=45)
fig = px.bar(df, x= "Entry_Date", y= "Drivers_of_COVID19_Impact", color_discrete_sequence=['crimson'], title='Drivers of COVID19 Impacts')

fig.show()
fig = px.bar(df, x= "Entry_Date", y= "Block", color_discrete_sequence=['#27F1E7'], title='Thematic Blocks of Covid-19 Impacts')

fig.show()
fig = px.bar(df, x= "Entry_Date", y= "Indicator", color_discrete_sequence=['#fcb103'], title='Indicators of Covid-19 Impacts')

fig.show()
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('Indicator').size()/df['Indicator_Type'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values,marker_colors = px.colors.sequential.speed, hole=.6)])

fig.show()
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('Justification').size()/df['Country'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values,marker_colors = px.colors.sequential.speed, hole=.6)])

fig.show()
fig = px.bar(df, x= "Value Date", y= "Indicator", color_discrete_sequence=['#ca03fc'], title='Indicators of Covid-19 Impacts')

fig.show()
fig = go.Figure(data=[go.Bar(

            x=df['Entry_Date'][0:10], y=df['Indicator'][0:10],

            text=df['Indicator'][0:10],

            textposition='auto',

            marker_color='black'



        )])

fig.update_layout(

    title='Indicators of COVID19 Impact',

    xaxis_title="Entry_Date",

    yaxis_title="Indicators of COVID19 Impact",

)

fig.show()
fig = go.Figure(data=[go.Scatter(

    x=df['Entry_Date'][0:10],

    y=df['Block'][0:10],

    mode='markers',

    marker=dict(

        color=[145, 140, 135, 130, 125, 120,115,110,105,100],

        size=[100, 90, 70, 60, 60, 60,50,50,40,35],

        showscale=True

        )

)])

fig.update_layout(

    title='Thematic Blocks of Covid-19 Effects',

    xaxis_title="Entry_Date",

    yaxis_title="Block",

)

fig.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Indicator)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='GnBu', background_color="white").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Source)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set1', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Block)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set2', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Justification)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Pillar)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='GnBu', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()