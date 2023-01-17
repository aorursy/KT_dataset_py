import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("vgsales.csv")
df.head(10)
df.info()
df = df.dropna()
df['Year'] = df['Year'].astype(int)
df.head()
years = df['Year'].unique()
#All years in ascending order
years.sort()
# Game counts for each year. Year in ascending order
gamecounts = df.groupby('Year').count()['Rank']
yearwise_gamecount = pd.DataFrame(columns=['Year','GamesThatYear'])
yearwise_gamecount['Year'] = years
yearwise_gamecount['GamesThatYear'] = gamecounts.values
plt.figure(figsize=(12,6))
ax = sns.barplot(x='Year',y='GamesThatYear',data=yearwise_gamecount)
ax.set_xticklabels(years,rotation=45)
plt.title("No of Games Released every Year")
yearwise_globalsales = pd.DataFrame(columns=['Year','GlobalSalesThatYear'])
# Global sales for each year. Year in ascending order
globalsales = df.groupby('Year').sum()['Global_Sales']
yearwise_globalsales['Year'] = years
yearwise_globalsales['GlobalSalesThatYear'] = globalsales.values
plt.figure(figsize=(12,6))
ax = sns.barplot(x='Year',y='GlobalSalesThatYear',data=yearwise_globalsales)
ax.set_xticklabels(years,rotation=45)
plt.title("Global Sales every Year")

publisher_wise_gamecount = df.groupby('Publisher').count()
m = publisher_wise_gamecount['Rank'].mean()
publisher_wise_gamecount = publisher_wise_gamecount[publisher_wise_gamecount['Rank'] >= m ].reset_index()[['Publisher','Rank']]
publisher_wise_gamecount.columns = ['Publisher','GamesByPublisher']
publisher_wise_gamecount.sort_values('GamesByPublisher',ascending=False).head(10)

platform_wise_gamecount = df.groupby('Platform').count().reset_index()
platform_wise_gamecount = platform_wise_gamecount.sort_values(by='Rank',ascending=False).head(5)[['Platform','Rank']]
platform_wise_gamecount.columns = ['Platform','GamesOnThatPlatform']
platform_wise_gamecount
df1 = df.groupby(['Year','Platform']).count().reset_index()
ds = df1[df1['Platform'] == 'DS'][['Year','Rank']]
ps2 = df1[df1['Platform'] == 'PS2'][['Year','Rank']]
ps3 = df1[df1['Platform'] == 'PS3'][['Year','Rank']]
wii = df1[df1['Platform'] == 'Wii'][['Year','Rank']]
x360 = df1[df1['Platform'] == 'X360'][['Year','Rank']]
ds.columns = [['Year','DS']]
ps2.columns = [['Year','PS2']]
ps3.columns = [['Year','PS3']]
wii.columns = [['Year','Wii']]
x360.columns = [['Year','X360']]
plt.plot(ds['Year'],ds['DS'])
plt.plot(ps2['Year'],ps2['PS2'])
plt.plot(ps3['Year'],ps3['PS3'])
plt.plot(wii['Year'],wii['Wii'])
plt.plot(x360['Year'],x360['X360'])
plt.legend()
plt.xlabel("Year")
plt.ylabel("No. of Games on platform")
plt.title("Year-wise game distribution across Top 5 platforms")

genre_wise_sales = df.groupby('Genre').sum().reset_index()[['Genre','Global_Sales']].sort_values('Global_Sales')
genre_wise_sales
plt.figure(figsize=(12,6))
sns.barplot(x='Genre',y='Global_Sales',data=genre_wise_sales)
sales_by_genre = df.groupby(['Genre']).sum().loc[:, 'NA_Sales':'Global_Sales']
plt.figure(figsize=(8, 10))
sns.set(font_scale=1.0)
sns.heatmap(sales_by_genre.loc[:, 'NA_Sales':'Other_Sales'], annot=True, fmt = '.1f',cmap=sns.cubehelix_palette(8))
plt.title("Which Area prefers which Genre?")

publisher_wise_sales = df.groupby('Publisher').sum().reset_index()[['Publisher','Global_Sales']].sort_values('Global_Sales',ascending=False).head(5)
publisher_wise_sales
df1 = df.groupby(['Publisher','Genre']).sum().reset_index()[['Publisher','Genre','Global_Sales']]
nintendo_genrewise_sales = df1[df1['Publisher'] == 'Nintendo'][['Genre','Global_Sales']]
ea_genrewise_sales = df1[df1['Publisher'] == 'Electronic Arts'][['Genre','Global_Sales']]
activision_genrewise_sales = df1[df1['Publisher'] == 'Activision'][['Genre','Global_Sales']]
sony_genrewise_sales = df1[df1['Publisher'] == 'Sony Computer Entertainment'][['Genre','Global_Sales']]
ubisoft_genrewise_sales = df1[df1['Publisher'] == 'Ubisoft'][['Genre','Global_Sales']]

all_genres = df['Genre'].unique()
all_genres.sort()
all_genres = all_genres.tolist()
new_df = pd.DataFrame(columns=all_genres,index=publisher_wise_sales['Publisher'])
new_df.loc['Nintendo'][:] = nintendo_genrewise_sales.T.loc['Global_Sales']
new_df.loc['Electronic Arts'][:] = ea_genrewise_sales.T.loc['Global_Sales']
new_df.loc['Activision'][:] = activision_genrewise_sales.T.loc['Global_Sales']
new_df.loc['Sony Computer Entertainment'][:] = sony_genrewise_sales.T.loc['Global_Sales']
new_df.loc['Ubisoft'][:] = ubisoft_genrewise_sales.T.loc['Global_Sales']
new_df

N = 5
plt.figure(figsize=(10,6))
actionSales = new_df['Action'].values
adventureSales = new_df['Adventure'].values
fightingSales = new_df['Fighting'].values
miscSales = new_df['Misc'].values
platformSales = new_df['Platform'].values
puzzleSales = new_df['Puzzle'].values
racingSales = new_df['Racing'].values
roleplayingSales = new_df['Role-Playing'].values
shooterSales = new_df['Shooter'].values
simulationSales = new_df['Simulation'].values
sportsSales = new_df['Sports'].values
strategySales = new_df['Strategy'].values
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, actionSales, width,color='#EF5350')
p2 = plt.bar(ind, adventureSales, width,bottom=actionSales,color='#EC407A')
p3 = plt.bar(ind, fightingSales, width,bottom=adventureSales,color='#AB47BC')
p4 = plt.bar(ind, miscSales, width,bottom=fightingSales,color='#42A5F5')
p5 = plt.bar(ind, platformSales, width,bottom=miscSales,color='#26C6DA')
p6 = plt.bar(ind, puzzleSales, width,bottom=platformSales,color='#26A69A')
p7 = plt.bar(ind, racingSales, width,bottom=puzzleSales,color='#66BB6A')
p8 = plt.bar(ind, roleplayingSales, width,bottom=racingSales,color='#D4E157')
p9 = plt.bar(ind, shooterSales, width,bottom=roleplayingSales,color='#FFEE58')
p10 = plt.bar(ind, simulationSales, width,bottom=shooterSales,color="#FFA726")
p11 = plt.bar(ind, sportsSales, width,bottom=simulationSales,color="#8D6E63")
p12 = plt.bar(ind, strategySales, width,bottom=sportsSales,color="#BDBDBD")

plt.ylabel('Sales')
plt.title('Sales by Genre and Publisher')
plt.xticks(ind, ('Nintendo', 'EA', 'Activision', 'Sony', 'Ubisoft'))
#plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0],p3[0],p4[0],p5[0],p6[0],p7[0],p8[0],p9[0],p10[0],p11[0],p12[0]), 
           ('Action', 'Adventure','Fighting','Misc','Platform','Puzzle','Racing','Role-Playing','Shooter','Simulation','Sports','Strategy'))

plt.show()
