# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/college-basketball-dataset/cbb.csv', index_col = 0)
df.head()
df.shape
df.info()
df.describe()
df['WIN_RATIO'] = df['W'] / df['G']
df['Eff_DIFF'] = df['ADJOE'] - df['ADJDE']
df.head(10)
df.describe()
df.sample(20)
px.scatter(df, x='WIN_RATIO', y='Eff_DIFF',trendline='ols',color='W')
px.scatter(df, x='WAB', y='Eff_DIFF', trendline='ols',color='W')
px.scatter(df, x='WAB', y='W',trendline='ols')
px.scatter(df, x='WAB', y='ADJOE', trendline='ols', color='WAB')
px.scatter(df, x='WAB', y='ADJDE', trendline='ols', color='WAB')
#px.scatter(df, x='ADJOE', y='W',trendline='ols' ,color='ADJDE')
px.scatter(df, x='ADJDE', y='W',trendline='ols',color='ADJOE')
px.scatter(df, x='Eff_DIFF', y= 'W',color='ADJDE', trendline='ols')
df['POSTSEASON'] = df['POSTSEASON'].fillna('Did not make tourney')
df['POSTSEASON'].sample(20)
px.histogram(df, x='POSTSEASON', color ='W')
px.scatter(df,x='ADJDE', y='ADJOE',color='POSTSEASON')
px.scatter(df,x='Eff_DIFF', y='W',color='POSTSEASON')
place_ADJOE = df[['ADJOE', 'POSTSEASON']]

place_ADJOE.sort_values(by='ADJOE', ascending=False).head(20)
place_ADJDE = df[['ADJDE', 'POSTSEASON']]

place_ADJDE.sort_values(by='ADJDE', ascending=True).head(20)
px.scatter(df, x='ADJOE', y='ADJ_T', trendline="ols", color='W')
px.scatter(df, x='ADJ_T', y='ADJDE', trendline="ols", color ='W')