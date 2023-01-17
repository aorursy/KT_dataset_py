# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed
df = pd.read_csv('../input/zomato.csv', index_col=0, encoding="ISO-8859-1")

df1 = pd.read_excel('../input/Country-Code.xlsx')

df.head()
df3 = pd.merge(df, df1, on=('Country Code')) #Соединение 2 таблиц по коду страны

print(df3['Country'].value_counts().head(5)) #Статистика по странам

df3['City'].value_counts().head(5) #Статистика по городам
df.groupby('Cuisines')['Aggregate rating'].mean().plot(kind='box', figsize = (20, 7)) #Построение графика зависимости рейтинга от количества видов кухни, подаваемых в ресторане
df.groupby('Price range')['Aggregate rating'].mean().plot(kind = 'bar', figsize = (20, 7))
print(df[df['Has Table booking'] == 'No']['Aggregate rating'].mean()) #Средний рейтинг ресторанов без возможности бронирования столиков

print(df[df['Has Table booking'] == 'Yes']['Aggregate rating'].mean()) #Почти то же самое, только с данной возможностью
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()

import plotly.graph_objs as go





plot_data = [dict(

    type='scattergeo',

    lon = df['Longitude'],

    lat = df['Latitude'],

    text = df['Restaurant Name'],

    mode = 'markers',

    marker = dict(

        cmin = 0,

        color = df['Rating color']

    )

    

)]

fig = go.Figure(data=plot_data)

iplot(fig)
numeric = ['Price range', 'Votes', 'Aggregate rating']

sns.heatmap(df[numeric].corr(method='spearman'));