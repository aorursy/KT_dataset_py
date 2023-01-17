#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRWMDpZwTgGdC31X4ct_Z7-ncRgVNYBGz-R9f-llUKyOOiOJwoX&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/magdalena-colombia-data/Magdalena_Data/Magdalena_Data/Outputs/Calibration dataset/Pesquerias/Series de tiempo - Pesquerias 1993-2010.xlsx')

df.head()
df = df.rename(columns={'Latitud (N)':'lat', 'longitud (O)': 'lon', 'Nombe científico': 'nombe', 'Genéro taxonómico': 'genero', 'Orden taxonómico': 'orden'})
import plotly.express as px

plot_map=df.groupby(['año','localidad']).sum().reset_index()

names=plot_map['localidad']

values=plot_map['biomasa (Toneladas)']

fig = px.choropleth(plot_map, locations=names,

                    locationmode='country names',

                    color=values)

fig.update_layout(title="Biomass")

fig.show()
dfcorr=df.corr()

dfcorr
plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=False,cmap='summer')

plt.show()
corr = df.corr(method='pearson')

sns.heatmap(corr)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.especie)

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

text = " ".join(str(each) for each in df.genero)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="green").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.orden)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="red").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
import plotly.offline as py

cnt_srs = df['orden'].value_counts().head()

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

    title='Species Distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="orden")
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT0nY-IfWVr_Yv7Z346DE36SF-cLXMem1DSOxTddhL5UhPr-scX&usqp=CAU',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRHD_s_lAusoGRFpJ6m1K5oUQV4r7g3-N2X3y2UMpNKBUIfYX05&usqp=CAU',width=400,height=400)