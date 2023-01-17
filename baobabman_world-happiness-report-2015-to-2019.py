!pip install chart_studio
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import chart_studio.plotly as py

from chart_studio.plotly import plot, iplot

from plotly.offline import iplot
df_2015 = pd.read_csv('../input/world-happiness/2015.csv')

df_2016 = pd.read_csv('../input/world-happiness/2016.csv')

df_2017 = pd.read_csv('../input/world-happiness/2017.csv')

df_2018 = pd.read_csv('../input/world-happiness/2018.csv')

df_2019 = pd.read_csv('../input/world-happiness/2019.csv')
df_2015
df_2019
sns.distplot(df_2019['Score']) # seaborn.distplot() : 데이터의 전체 분포도를 그림
corrmat=df_2019.corr() # df.corr() : 데이터 상관관계 매트릭스를 생성

mask = np.zeros_like(corrmat, dtype=np.bool) # np.zeros_like() : 주어진 행렬과 같은 크기의 0으로 채워진 행렬을 반환

mask[np.triu_indices_from(mask)]=True # np.triu_indices_from() : 행렬의 상단 삼각형을 반환

plt.figure(figsize=(10, 10))

sns.heatmap(corrmat, annot=True, vmax=.8, square=True, cmap="PiYG", center=0, mask=mask) # seaborn.heatmap() : 히트맵 열지도를 그림
data = dict(type='choropleth', locations=df_2019['Country or region'], locationmode='country names', z=df_2019['Overall rank'], text=df_2019['Country or region'], colorbar={'title':'Happiness Rank'})

layout = dict(title='Global Happiness 2019', geo=dict(showframe=False))

choromap3 = go.Figure(data=[data], layout=layout)

iplot(choromap3) # plotly.offline.iplot() : 지도에 데이터를 표시함
kcj_df=df_2019[df_2019['Country or region'].str.contains('South Korea|China|Japan')]

kcj_df
sns.scatterplot(x='Country or region', y='Score', data=kcj_df)