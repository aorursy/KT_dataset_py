# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/wall-art-sales/Wall Art sales - Sheet1.csv', encoding='ISO-8859-2')

df.head(10)
df.isnull().sum()
cnt_srs = df['Link'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Blues',

        reversescale = True

    ),

)



layout = dict(

    title='Link Distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Link")
cnt_srs = df['Brand'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Reds',

        reversescale = True

    ),

)



layout = dict(

    title='Brand Distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Brand")
cnt_srs = df['Shipping'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Greens',

        reversescale = True

    ),

)



layout = dict(

    title='Shipping Distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Shipping")
cnt_srs = df['Discount'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Purples',

        reversescale = True

    ),

)



layout = dict(

    title='Discount Distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Discount")
# Count Plot

plt.style.use("classic")

plt.figure(figsize=(10, 8))

sns.countplot(df['Brand'], palette='Accent_r')

plt.xlabel("Brand")

plt.ylabel("Count")

plt.title("Brand")

plt.xticks(rotation=45, fontsize=8)

plt.show()
sns.countplot(x="Shipping",data=df,palette="GnBu_d",edgecolor="black")

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.title("Shipping Delivery Time")

# changing the font size

sns.set(font_scale=1)
#Code from Gabriel Preda

#plt.style.use('dark_background')

def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set2')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count("Discount", "Discount", df,4)
ax = df['Brand'].value_counts().plot.barh(figsize=(14, 6))

ax.set_title('Brand Distribution', size=18)

ax.set_ylabel('Brand', size=14)

ax.set_xlabel('Count', size=14)
ax = df['Shipping'].value_counts().plot.barh(figsize=(14, 6), color='r')

ax.set_title('Shipping Distribution', size=18)

ax.set_ylabel('Shipping', size=14)

ax.set_xlabel('Count', size=14)
fig = px.bar(df[['Brand','Shipping']].sort_values('Shipping', ascending=False), 

                        y = "Shipping", x= "Brand", color='Shipping', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(title_text="Brands & Shipping Delivery Time")



fig.show()
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('Brand').size()/df['Shipping'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values,marker_colors = px.colors.sequential.speed, hole=.6)])

fig.show()
fig = px.bar(df, x= "Brand", y= "Shipping", color_discrete_sequence=['crimson'], title='Brand & Shipping Delivery Time')

fig.show()
ax = sns.countplot(x = 'Link',data=df,order=['Copper Metal Decorative Wall Art', 'Gold Metal Decorative With Led Wall Art', 'Silver Metal Decorative Wall Art', 'Brown Metal Mesh Leav Wall Decor With Led'])

for p in ax.patches:

        ax.annotate(format(p.get_height()), (p.get_x()+0.2, p.get_height()))

plt.xticks(rotation=45) 
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Brand)

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

text = " ".join(str(each) for each in df.Link)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='GnBu', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()