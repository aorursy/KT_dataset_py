#Importing libraries and Reading the file 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



df = pd.read_csv("../input/videogamesales/vgsales.csv")

print('Ready!')
df.columns=["rank","name","platform","year","genre","publisher","na","eu","jp","rest","gb"]
df.head()
df.info()
# A game named 'Imagine: Makeup Artist' had an incorrect year value. Changed it to 2009.

df.year = df.year.replace(2020.0, 2009.0)

df.year.max()
df['year'] = df['year'].fillna(2009.0)

df['year'].isna().sum()
#convert type of column to int

df['year'] = df['year'].astype('int')
#drop 2017 rows as they are just a total of 3 entries. 

df.drop(df[df['year']==2017.0].index, axis=0, inplace=True)
df['publisher'] = df['publisher'].fillna('Unknown Publisher')

df['publisher'].isna().sum()
df.info()
plt.bar(data=df.head(10), x='name', height='gb', color = 'green')

plt.xticks(rotation=90)

plt.title('Top 10 selling games of all time')

plt.show()



top_10_na=df.sort_values('na', ascending=0).head(10)

plt.bar(data=top_10_na, x='name', height='na', color='orange')

plt.xticks(rotation=90)

plt.title('Top 10 selling games of all time in North America')

plt.show()



top_10_eu=df.sort_values('eu', ascending=0).head(10)

plt.bar(data=top_10_eu, x='name', height='eu', color='blue')

plt.xticks(rotation=90)

plt.title('Top 10 selling games of all time in Europe')

plt.show()



top_10_jp=df.sort_values('jp', ascending=0).head(10)

plt.bar(data=top_10_jp, x='name', height='jp', color='red')

plt.xticks(rotation=90)

plt.title('Top 10 selling games of all time in Japan')

plt.show()



top_10_rest=df.sort_values('rest', ascending=0).head(10)

plt.bar(data=top_10_rest, x='name', height='rest', color='purple')

plt.xticks(rotation=90)

plt.title('Top 10 selling games of in the rest of the world')

plt.show()
df.groupby('platform').agg({'gb': sum}).head(10).sort_values(by='gb', ascending=False)
df.groupby('platform').agg({'na': sum}).head(5).sort_values(by='na', ascending=False)
df.groupby('platform').agg({'eu': sum}).head(5).sort_values(by='eu', ascending=False)
df.groupby('platform').agg({'jp': sum}).head(5).sort_values(by='jp', ascending=False)
df.groupby('platform').agg({'rest': sum}).head(5).sort_values(by='rest', ascending=False)
df.groupby('genre').agg({'gb': sum}).head(5).sort_values(by='gb', ascending=False)
df.groupby('genre').agg({'na': sum}).head(5).sort_values(by='na', ascending=False)
df.groupby('genre').agg({'eu': sum}).head(5).sort_values(by='eu', ascending=False)
df.groupby('genre').agg({'jp': sum}).head(5).sort_values(by='jp', ascending=False)
df.groupby('genre').agg({'rest': sum}).head(5).sort_values(by='rest', ascending=False)
yearlytop_sort = df[['year', 'name', 'platform','gb']].sort_values(by=['year', 'gb'], ascending=[True, False])

yearlytop = yearlytop_sort.groupby('year')

yearlytop = yearlytop.head(1).reset_index(drop=1)

yearlytop



#How do I remove the index ranks? Anyone?
platform_agg = df.groupby(['year', 'platform']).agg({'gb' : sum})

top_platform = platform_agg['gb'].groupby(level=0, group_keys=False)

top_platform_final = top_platform.apply(lambda x: x.sort_values(ascending=False).head(1))

# top_platform_final

top_platform.nlargest(1)
genre_agg = df.groupby(['year', 'genre']).agg({'gb' : sum})

top_genre = genre_agg['gb'].groupby(level=0, group_keys=False)

# top_genre_final = top_genre.apply(lambda x: x.sort_values(ascending=False).head(1))

top_genre.nlargest(1)
total_genre_sales = df.groupby('genre')['gb'].sum()

genre_labels = ['Action', 'Adventure', 'Fighting', 'Misc', 'Platform', 'Puzzle', 'Racing', 'Role-Playing', 'Shooter', 'Simulation', 'Sports', 'Strategy']

fig = plt.figure(figsize =(10, 7)) 

plt.pie(total_genre_sales, labels=genre_labels, shadow=True, autopct='%1.1f%%')

plt.show()
total_genre_sales

#Tabular repreentation of the output
yearly_sales = df.groupby('year').agg({'gb':sum})

yearly_sales = yearly_sales.sort_values(by='year', ascending=1).reset_index()



fig = plt.figure(figsize =(9, 6)) 

plt.bar(data=yearly_sales, x='year', height='gb')

plt.xticks(rotation=90)

plt.title('Yearly sales since 1980')

plt.show()
top_publisher = df[['publisher', 'gb']]

top_publisher = top_publisher.groupby('publisher')['gb'].sum()

fig = plt.figure(figsize =(9, 8)) 

top_publisher.nlargest(5).plot.bar(x='publisher', y='gb')

plt.title('Count of games sold annually')

plt.show()
yearly_game = df.groupby('year')['name'].count().reset_index()
fig = plt.figure(figsize =(9, 8)) 

plt.bar(yearly_game['year'], yearly_game['name'])

plt.xticks(rotation=90)

plt.title('Count of games sold annually')

plt.show()