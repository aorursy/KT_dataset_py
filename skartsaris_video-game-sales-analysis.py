



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



plt.rcParams["figure.figsize"] = [20, 5]



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df  = pd.read_csv('../input/vgsales/vgsales.csv')

df.head()
df.isnull().sum() 
null_df = df[df.isnull().any(axis=1)]

null_df
df.dropna(axis = 0, inplace = True)

null_df = df[df.isnull().any(axis=1)]
zero_sales = df[['Global_Sales','JP_Sales','EU_Sales','NA_Sales']].loc[df['JP_Sales'] == 0.00].sum()

zero_sales
zero_sales = df[['Global_Sales','JP_Sales','EU_Sales','NA_Sales']].loc[df['NA_Sales'] == 0.00]

zero_sales.sum()
zero_sales = df[['Global_Sales','JP_Sales','EU_Sales','NA_Sales']].loc[df['EU_Sales'] == 0.00]

zero_sales.sum()
zero_sales = df[['Global_Sales','JP_Sales','EU_Sales','NA_Sales']].loc[df['Global_Sales'] == 0.00]

zero_sales.sum()


df['Year'] = df['Year'].apply(lambda x: int(x))

df.head()
1 - 0.21885813148788927
df[['Platform', 'Global_Sales']].groupby(['Platform']).sum().sort_values(by = 'Global_Sales', ascending = False).head(5)
df[['Name', 'Global_Sales','Platform']].groupby(['Name']).sum().sort_values(by = 'Global_Sales', ascending = False).head(10)


games_10_sales = df[df['Global_Sales'] > 10].sort_values(by = 'Global_Sales', ascending = False).head(15)



fig = px.bar(games_10_sales, x="Year", y="Global_Sales", color="Name", title="Games with more than 10 million sales", width=1300, height=500)

fig.show()
games_10_sales.groupby(['Platform','Year','Name']).sum('Global_Sales').sort_values(by = 'Global_Sales',ascending = False).head(5)
df[['Rank','Genre','Global_Sales']].groupby(['Genre']).sum().sort_values(by = 'Global_Sales',ascending = False).head(5)
df[df['Genre'] == 'Action'].sort_values(by = 'Global_Sales', ascending = False).head(15)
#Create a new dataframe with the European Sales dropping games not released in Europe

EU_only_sales = df[['EU_Sales','Genre','Platform','Publisher','Name']]

EU_only_sales = EU_only_sales[EU_only_sales['EU_Sales'] > 0]

EU_only_sales.groupby([ 'Genre', 'Platform', 'Publisher', 'Name']).sum('EU_Sales').sort_values(by = 'EU_Sales', ascending = False).head(10)
#Create a new dataframe with the Japanese Sales dropping games not released in Japan

Jp_only_sales = df[['JP_Sales','Genre','Platform','Publisher','Name', 'Rank']]

Jp_only_sales = Jp_only_sales[Jp_only_sales['JP_Sales'] > 0]

Jp_only_sales.groupby(['Rank',  'Genre', 'Platform', 'Publisher', 'Name']).sum('JP_Sales').sort_values(by = 'JP_Sales', ascending = False).head(10)
#Create a new dataframe with the North America Sales dropping games not released in North America

NA_only_sales = df[['NA_Sales','Genre','Platform','Publisher','Name', 'Rank']]

NA_only_sales = NA_only_sales[NA_only_sales['NA_Sales'] > 0]

NA_only_sales.groupby(['Rank',  'Genre', 'Platform', 'Publisher', 'Name']).sum('NA_Sales').sort_values(by = 'NA_Sales', ascending = False).head(10)
y = df['Global_Sales']

x = df['Year']

plt.bar(x , y)

plt.ylabel('Global sales in millions')

plt.grid()

plt.title('Sales developement through the years')



df[df['Year'].isin(['1985','1989','1996','2006'])].sort_values(by= 'Global_Sales', ascending = False).head(4)
#Group Global_Sales into Genres and then plot global sales

Global_Sales_df = df[['Platform','Global_Sales']]

Global_Sales_df.groupby(['Platform']).mean('Global_Sales').plot.bar()

plt.grid()

plt.title('Platform sales')

plt.ylabel('Global sales in millions')


sony_vs_microsoft = df[['Name','Platform','Publisher','Global_Sales']]

sony_vs_microsoft = sony_vs_microsoft[sony_vs_microsoft['Platform'].isin(['PS2', 'XB','PS3','PS','XOne','X360'])].sort_values(by = 'Global_Sales', ascending = False).head(50)

grossing_games  = sony_vs_microsoft[sony_vs_microsoft['Name'].isin(['Gran Turismo','Gran Turismo 2','Gran Turismo 3: A-Spec','Gran Turismo 4','Gran Turismo 5', 'Uncharted 2: Among Thieves',

'Uncharted 3: Drake\'s Deception','Gears of War 2', 'Halo 2', 'Halo 3', 'Halo 4'])].groupby(['Platform','Name']).sum('Global_Sales')

grossing_games



sales_df = df[['Year','Global_Sales','Publisher']].sort_values(by = 'Global_Sales', ascending= False).head(100)



plt.figure(figsize=(20,10))

sns.lineplot(data = sales_df, x = sales_df['Year'], y = sales_df['Global_Sales'], hue='Publisher')

plt.ylabel('Global sales in millions')

plt.title("Historical global sales by publisher")
Top_Publishers = df[df['Publisher'].str.contains('Nintendo')==False]

Top_Publishers = Top_Publishers[Top_Publishers['Global_Sales'] >= 4]

Top_Publishers = Top_Publishers[['Publisher', 'Global_Sales']]

Top_Publishers.groupby('Publisher').sum().plot.bar()

plt.ylabel('Global sales in millions')

plt.title("Global sales by publisher")
plt.figure(figsize=(20,10))

plt.title("Global Sales by Region")

sns.lineplot(data = df[['Global_Sales','Year']], y = df['Global_Sales'], x = df['Year'], label = 'Global Sales' )

sns.lineplot(data = df[['JP_Sales','Year']], y = df['JP_Sales'], x = df['Year'], label = 'Japan Sales' )

sns.lineplot(data = df[['EU_Sales','Year']], y = df['EU_Sales'], x = df['Year'], label = 'EU Sales')

sns.lineplot(data = df[['NA_Sales','Year']], y = df['NA_Sales'], x = df['Year'], label = 'NA Sales' )

plt.xlabel("Years")

plt.ylabel('Global sales in millions')
plt.title("The gaming industry in decline")



sns.barplot(x = df['Year'], y = df['Global_Sales'])

plt.ylabel('Global sales in millions')