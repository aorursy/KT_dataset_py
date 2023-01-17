import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
df.head()
df.shape
df.info()
len(df.Platform.unique())
df[df.Year <= 1980]
df.pivot_table(index='Platform', columns='Genre', values='Global_Sales', aggfunc='sum').dropna()
df.pivot_table(index='Publisher', values='Global_Sales', aggfunc='sum').sort_values(by='Global_Sales', ascending=False)
df.pivot_table(index=['Platform'], values='Global_Sales', aggfunc='sum').sort_values(by='Global_Sales', ascending=False).head()
# Global sales drop after 2010...little to no data after 2016
df.pivot_table(index=['Year'], values='Global_Sales', aggfunc='sum').sort_values(by='Year', ascending=False)
len(set(df[df['Year'] == 2015]['Name']))
# confirmed num of game sales reported quickly dropped off, peaking 2009-2010
len(set(df[df['Year']==2010]['Name']))
# dropping rows after 2010
df = df[df['Year']<=2010]
# pivot again without dropped data, using groupby instead of pivot table
df.groupby('Platform').agg({'Global_Sales':'sum'}).sort_values('Global_Sales', ascending=False).head()
# best sales by platform genre
df.groupby(['Platform','Genre']).agg({'Global_Sales':'sum'}).sort_values('Global_Sales', ascending=False).head()
# best years for sales are 2006-2010
df.groupby('Year').agg({'Global_Sales':'sum'}).sort_values('Global_Sales', ascending=False).head()
# best selling games of all time
df_games = df.groupby('Name').agg({'Global_Sales':'sum'})
df_games.sort_values('Global_Sales', ascending=False).head()
# make new column to find num of games per year
df_year = df.groupby('Year').agg({'Global_Sales':'sum'})
df_year.sort_values('Global_Sales', ascending=False).head()
# vg = sns.load_dataset(df)
sns.jointplot(x='Global_Sales', y='Year', data=df)
sns.distplot(df['Year'], kde=False, bins=30, color='red')
# only correlations are country sales to global sales
dfc = df.corr()
sns.heatmap(dfc, cmap='coolwarm') 
plt.style.use('ggplot')
df.plot.area(alpha=0.4)
plt.show()

df.plot.scatter(x='Global_Sales',y='Year', cmap='seismic')
plt.show()
df['Year'].hist()
games_per_year = df.groupby('Year')['Name'].nunique()
games_per_year
ax = sns.scatterplot(x=df.Year, y=df.Publisher,
                     hue=games_per_year, size=games_per_year, sizes=(20, 400), legend=False)
