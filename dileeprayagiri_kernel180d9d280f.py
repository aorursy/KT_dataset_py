import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("dark")

df = pd.read_csv('../input/vgsales.csv')
df.head(10)
df = df.dropna()

df = df[df.Year < 2018.0]

df.isnull().sum()
df.info()
####Invalid Years were dropped so expect prior 2018 years in the data####

print("Max Year Value: ", df['Year'].max())

print("Number of games: ", len(df))

print("Number of publishers: ", df['Publisher'].nunique())

print("Number of platforms: ", df['Platform'].nunique())

print("Number of genres: ", df['Genre'].nunique())
###Barplots for Global, NA, EU and Japan sales by Year#####



y = df.groupby(['Year']).sum()

y = y['Global_Sales']

x = y.index.astype(int)



y1 = df.groupby(['Year']).sum()

y1 = y1['NA_Sales']

x1 = y1.index.astype(int)

y1

y2 = df.groupby(['Year']).sum()

y2 = y2['EU_Sales']

x2 = y2.index.astype(int)



y3 = df.groupby(['Year']).sum()

y3 = y3['JP_Sales']

x3 = y3.index.astype(int)



y4 = df.groupby(['Year']).sum()

y4 = y4['Other_Sales']

x4 = y4.index.astype(int)



plt.figure(figsize = (10,5))

ax = sns.barplot(x = x, y = y)

ax.set_title("Games Sales in Millions ($)")

ax.set_xlabel("Year", fontsize=10)

ax.set_ylabel("Millions ($)", fontsize = 10)

ax.set_xticklabels(labels = x, fontsize=10, rotation=90)

plt.show()



fig = plt.figure(figsize = (12,8))

ax1 = fig.add_subplot(221)

sns.barplot(x = x1,y= y1)

ax1.set_title("North America Sales in Millions ($)")

ax1.set_xlabel("Year", fontsize=10)

ax1.set_ylabel("Millions ($)", fontsize = 10)

ax1.set_xticklabels(labels = x1, fontsize=6, rotation=90)





ax2 = fig.add_subplot(222)

sns.barplot(x = x2,y= y2)

ax2.set_title("Europe Sales in Millions ($)")

ax2.set_xlabel("Year", fontsize=10)

ax2.set_ylabel("Millions ($)", fontsize = 10)

ax2.set_xticklabels(labels = x2, fontsize=6, rotation=90)



ax3 = fig.add_subplot(223)

sns.barplot(x = x3,y= y3)

ax3.set_title("Japan Sales in Millions ($)")

ax3.set_xlabel("Year", fontsize=10)

ax3.set_ylabel("Millions ($)", fontsize = 10)

ax3.set_xticklabels(labels = x3, fontsize=6, rotation=90)



ax4 = fig.add_subplot(224)

sns.barplot(x = x4,y= y4)

ax4.set_title("Other Sales in Millions ($)")

ax4.set_xlabel("Year", fontsize=10)

ax4.set_ylabel("Millions ($)", fontsize = 10)

ax4.set_xticklabels(labels = x4, fontsize=6, rotation=90)



plt.tight_layout()
#######Yearly Global sales by Genre and by Publisher (Sub-Plotted)

table = df.pivot_table('Global_Sales', index='Genre', columns='Year', aggfunc='sum')

genres = table.idxmax()

sales = table.max()

years = table.columns.astype(int)

data = pd.concat([genres, sales], axis=1)

data.columns = ['Genre', 'Global Sales']



table1 = df.pivot_table('Global_Sales', index='Publisher', columns='Year', aggfunc='sum')

publishers = table1.idxmax()

sales1 = table1.max()

years1 = table1.columns.astype(int)

data1 = pd.concat([publishers, sales1], axis=1)

data1.columns = ['Publisher', 'Global Sales']



fig1 = plt.figure(figsize = (12,8))

ax = fig1.add_subplot(121)

ax = sns.pointplot(y = 'Global Sales', x = years, hue='Genre', data=data, size=15)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Global Sales Per Year', fontsize=16)

ax.set_title(label='Highest Genre Revenue in $ Millions Per Year', fontsize=12)

ax.set_xticklabels(labels = years, fontsize=8, rotation=90)



ax1 = fig1.add_subplot(122)

ax1 = sns.pointplot(y = 'Global Sales', x = years1, hue='Publisher', data=data1, size=15)

ax1.set_xlabel(xlabel='Year', fontsize=16)

ax1.set_ylabel(ylabel='Global Sales Per Year', fontsize=16)

ax1.set_title(label='Highest Publisher Revenue in $ Millions Per Year', fontsize=12)

ax1.set_xticklabels(labels = years1, fontsize=8, rotation=90)

plt.tight_layout()

plt.show();
####Barplot for Top 5 Platform Games Released####

data = df.groupby(['Platform']).count().iloc[:,0]

data = pd.DataFrame(data.sort_values(ascending=False))[0:5]

platforms = data.index

data.columns = ['Releases']

colors = sns.color_palette("GnBu_d", len(data))

plt.figure(figsize=(12,8))

ax = sns.barplot(y = platforms , x = 'Releases', data=data, orient='h', palette=colors)

ax.set_xlabel(xlabel='Number of Releases', fontsize=16)

ax.set_ylabel(ylabel='Platform', fontsize=16)

ax.set_title(label='Top 5 Total Platform Games Released', fontsize=20)

ax.set_yticklabels(labels = platforms, fontsize=14)

plt.show();
####Barplot for Total Revenue Per Year in $ Millions of Top 5 Platforms####

top5 = ['DS', 'PS2', 'PS3', 'Wii', 'X360']

table = df.pivot_table('Global_Sales', columns='Platform', index='Year', aggfunc='sum')

data = [table[i] for i in top5]

data = np.array(data)

data = pd.DataFrame(np.reshape(data, (5, 38)))

years = table.index.astype(int)



plt.figure(figsize=(12,8))

ax = sns.heatmap(data)

ax.set_xticklabels(labels = years, fontsize=12, rotation=50)

ax.set_yticklabels(labels = top5[::-1], fontsize=14, rotation=0)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Platform', fontsize=16)

ax.set_title(label='Total Revenue Per Year in $ Millions of Top 5 Platforms', fontsize=20)

plt.show();
####Jointplot for Wii_Sales####

Wii_sales = df[df['Platform'] == 'Wii']





sns.jointplot("Year", "Global_Sales",data=Wii_sales, color='green')
####regplot for the same Wii_sales Dataset###

ax=sns.regplot(x='Year', y='Global_Sales', data=Wii_sales)

ax.set_title('Wii_Global sales', fontsize=20, Color = 'green')
####Pair Plot for Genre Distribution by Market###

Market_Pair = df.drop(['Rank', 'Other_Sales', 'Global_Sales'], axis=1)

sns.pairplot(Market_Pair,hue='Genre')
####Violin Plot for Yearly Global Sales####

plt.figure(figsize=(15,10))

ax = sns.violinplot("Year", "Global_Sales", data=df )

ax.set_xlabel("Year", fontsize=10)

ax.set_ylabel("Millions ($)", fontsize = 12)

ax.set_xticklabels(labels = x, fontsize=12, rotation=90)