#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSWXrNubZBo8HvGb7U3ZolUrVjYJ2S7TN-fY4hBUDiy1OyXpfe6&usqp=CAU',width=400,height=400)
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
df = pd.read_csv('../input/feriados-sao-paulo-2015/feriados_sao_paulo_2015.csv', encoding='ISO-8859-2')

df.head()
px.histogram(df, x='Dia', color='Feriado')
fig = px.bar(df, 

             x='Feriado', y='Dia', color_discrete_sequence=['#D63230'],

             title='Holidays in São Paulo City', text='Feriado')

fig.show()
# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("São Paulo City Holidays")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=df.index, y=df['Feriado'])



# Add label for vertical axis

plt.ylabel("SP  City Holidays")
sns.countplot(df["Feriado"])

plt.xticks(rotation=90)

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Feriado)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set2', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()