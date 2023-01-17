import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("dark")

sns.despine()
df = pd.read_csv("../input/vgsales.csv")

df.head(1)
year_data = df['Year']

print("Max Year Value: ", year_data.max())
max_entry = year_data.idxmax()

max_entry = df.iloc[max_entry]

pd.DataFrame(max_entry).T
df['Year'] = df['Year'].replace(2020.0, 2009.0)

print("Max Year Value: ", year_data.max())
print("Number of games: ", len(df))

publishers = df['Publisher'].unique()

print("Number of publishers: ", len(publishers))

platforms = df['Platform'].unique()

print("Number of platforms: ", len(platforms))

genres = df['Genre'].unique()

print("Number of genres: ", len(genres))
print(df.isnull().sum())

df = df.dropna()
y = df.groupby(['Year']).sum()

y = y['Global_Sales']

x = y.index.astype(int)



plt.figure(figsize=(12,8))

ax = sns.barplot(y = y, x = x)

ax.set_xlabel(xlabel='$ Millions', fontsize=16)

ax.set_xticklabels(labels = x, fontsize=12, rotation=50)

ax.set_ylabel(ylabel='Year', fontsize=16)

ax.set_title(label='Game Sales in $ Millions Per Year', fontsize=20)

plt.show();
x = df.groupby(['Year']).count()

x = x['Global_Sales']

y = x.index.astype(int)



plt.figure(figsize=(12,8))

colors = sns.color_palette("muted")

ax = sns.barplot(y = y, x = x, orient='h', palette=colors)

ax.set_xlabel(xlabel='Number of releases', fontsize=16)

ax.set_ylabel(ylabel='Year', fontsize=16)

ax.set_title(label='Game Releases Per Year', fontsize=20)

plt.show();
table = df.pivot_table('Global_Sales', index='Publisher', columns='Year', aggfunc='sum')

publishers = table.idxmax()

sales = table.max()

years = table.columns.astype(int)

data = pd.concat([publishers, sales], axis=1)

data.columns = ['Publisher', 'Global Sales']



plt.figure(figsize=(12,8))

ax = sns.pointplot(y = 'Global Sales', x = years, hue='Publisher', data=data, size=15)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Global Sales Per Year', fontsize=16)

ax.set_title(label='Highest Publisher Revenue in $ Millions Per Year', fontsize=20)

ax.set_xticklabels(labels = years, fontsize=12, rotation=50)

plt.show();
table = df.pivot_table('Global_Sales', index='Genre', columns='Year', aggfunc='sum')

genres = table.idxmax()

sales = table.max()

years = table.columns.astype(int)

data = pd.concat([genres, sales], axis=1)

data.columns = ['Genre', 'Global Sales']



plt.figure(figsize=(12,8))

ax = sns.pointplot(y = 'Global Sales', x = years, hue='Genre', data=data, size=15, palette='Dark2')

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Global Sales Per Year', fontsize=16)

ax.set_title(label='Highest Genre Revenue in $ Millions Per Year', fontsize=20)

ax.set_xticklabels(labels = years, fontsize=12, rotation=50)

plt.show();
table = df.pivot_table('Global_Sales', index='Name', columns='Year')

table.columns = table.columns.astype(int)

games = table.idxmax()

sales = table.max()

years = table.columns

data = pd.concat([games, sales], axis=1)

data.columns = ['Game', 'Global Sales']





colors = sns.color_palette("GnBu_d", len(years))

plt.figure(figsize=(12,8))

ax = sns.barplot(y = years , x = 'Global Sales', data=data, orient='h', palette=colors)

ax.set_xlabel(xlabel='Global Sales Per Year', fontsize=16)

ax.set_ylabel(ylabel='Year', fontsize=16)

ax.set_title(label='Highest Revenue Per Game in $ Millions Per Year', fontsize=20)

plt.show();

data
table = df.pivot_table('Global_Sales', index='Platform', columns='Year', aggfunc='sum')

platforms = table.idxmax()

sales = table.max()

years = table.columns.astype(int)

data = pd.concat([platforms, sales], axis=1)

data.columns = ['Platform', 'Global Sales']



plt.figure(figsize=(12,8))

ax = sns.pointplot(y = 'Global Sales', x = years, hue='Platform', data=data, size=15)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Global Sales Per Year', fontsize=16)

ax.set_title(label='Highest Total Platform Revenue in $ Millions Per Year', fontsize=20)

ax.set_xticklabels(labels = years, fontsize=12, rotation=50)

plt.show();
data = df.groupby(['Publisher']).count().iloc[:,0]

data = pd.DataFrame(data.sort_values(ascending=False))[0:10]

publishers = data.index

data.columns = ['Releases']



colors = sns.color_palette("spring", len(data))

plt.figure(figsize=(12,8))

ax = sns.barplot(y = publishers , x = 'Releases', data=data, orient='h', palette=colors)

ax.set_xlabel(xlabel='Number of Releases', fontsize=16)

ax.set_ylabel(ylabel='Publisher', fontsize=16)

ax.set_title(label='Top 10 Total Publisher Games Released', fontsize=20)

ax.set_yticklabels(labels = publishers, fontsize=14)

plt.show();
data = df.groupby(['Publisher']).sum()['Global_Sales']

data = pd.DataFrame(data.sort_values(ascending=False))[0:10]

publishers = data.index

data.columns = ['Global Sales']



colors = sns.color_palette("cool", len(data))

plt.figure(figsize=(12,8))

ax = sns.barplot(y = publishers , x = 'Global Sales', data=data, orient='h', palette=colors)

ax.set_xlabel(xlabel='Revenue in $ Millions', fontsize=16)

ax.set_ylabel(ylabel='Publisher', fontsize=16)

ax.set_title(label='Top 10 Total Publisher Game Revenue', fontsize=20)

ax.set_yticklabels(labels = publishers, fontsize=14)

plt.show();
top10 = ['Electronic Arts', 'Activision', 'Namco Bandai Games', 'Ubisoft', 'Konami Digital Entertainment', 'THQ', 'Nintendo', 'Sony Computer Entertainment', 'Sega', 'Take-Two Interactive']

table = df.pivot_table('Global_Sales', columns='Publisher', index='Year', aggfunc='sum')

data = [table[i] for i in top10]

data = np.array(data)

data = pd.DataFrame(np.reshape(data, (10, 38)))

years = table.index.astype(int)



plt.figure(figsize=(12,8))

ax = sns.heatmap(data)

ax.set_xticklabels(labels = years, fontsize=12, rotation=50)

ax.set_yticklabels(labels = top10[::-1], fontsize=14, rotation=0)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Publisher', fontsize=16)

ax.set_title(label='Total Revenue Per Year in $ Millions of Top 10 Publishers', fontsize=20)

plt.show();
top10 = ['Electronic Arts', 'Activision', 'Namco Bandai Games', 'Ubisoft', 'Konami Digital Entertainment', 'THQ', 'Nintendo', 'Sony Computer Entertainment', 'Sega', 'Take-Two Interactive']

table = df.pivot_table('Global_Sales', columns='Publisher', index='Year', aggfunc='mean')

data = [table[i] for i in top10]

data = np.array(data)

data = pd.DataFrame(np.reshape(data, (10, 38)))

years = table.index.astype(int)



plt.figure(figsize=(12,8))

ax = sns.heatmap(data, cmap='viridis')

ax.set_xticklabels(labels = years, fontsize=12, rotation=50)

ax.set_yticklabels(labels = top10[::-1], fontsize=14, rotation=0)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Publisher', fontsize=16)

ax.set_title(label='Average Revenue Per Game in $ Millions of Top 10 Publishers', fontsize=20)

plt.show();
top10 = ['Electronic Arts', 'Activision', 'Namco Bandai Games', 'Ubisoft', 'Konami Digital Entertainment', 'THQ', 'Nintendo', 'Sony Computer Entertainment', 'Sega', 'Take-Two Interactive']

table = df.pivot_table('Global_Sales', columns='Publisher', index='Year', aggfunc='count')

data = [table[i] for i in top10]

data = np.array(data)

data = pd.DataFrame(np.reshape(data, (10, 38)))

years = table.index.astype(int)



plt.figure(figsize=(12,8))

ax = sns.heatmap(data, cmap='terrain')

ax.set_xticklabels(labels = years, fontsize=12, rotation=50)

ax.set_yticklabels(labels = top10[::-1], fontsize=14, rotation=0)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Publisher', fontsize=16)

ax.set_title(label='Number of Game Releases Per Year by Top 10 Publishers', fontsize=20)

plt.show();
genres = ['Action', 'Adventure', 'Fighting', 'Misc', 'Platform', 'Puzzle', 'Racing', 'Role-Playing', 'Shooter', 'Simulation', 'Sports', 'Strategy']

table = df.pivot_table('Global_Sales', columns='Genre', index='Year', aggfunc='sum')

data = [table[i] for i in genres]

data = np.array(data)

data = pd.DataFrame(np.reshape(data, (12, 38)))

years = table.index.astype(int)



plt.figure(figsize=(12,8))

ax = sns.heatmap(data, cmap='plasma')

ax.set_xticklabels(labels = years, fontsize=12, rotation=50)

ax.set_yticklabels(labels = genres[::-1], fontsize=14, rotation=0)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Genre', fontsize=16)

ax.set_title(label='Total Revenue Per Genre Per Year in $ Millions', fontsize=20)

plt.show();
rel = df.groupby(['Genre']).count().iloc[:,0]

rel = pd.DataFrame(rel.sort_values(ascending=False))

genres = rel.index

rel.columns = ['Releases']



colors = sns.color_palette("summer", len(rel))

plt.figure(figsize=(12,8))

ax = sns.barplot(y = genres , x = 'Releases', data=rel, orient='h', palette=colors)

ax.set_xlabel(xlabel='Number of Releases', fontsize=16)

ax.set_ylabel(ylabel='Genre', fontsize=16)

ax.set_title(label='Genres by Total Number of Games Released', fontsize=20)

ax.set_yticklabels(labels = genres, fontsize=14)

plt.show();
rev = df.groupby(['Genre']).sum()['Global_Sales']

rev = pd.DataFrame(rev.sort_values(ascending=False))

genres = rev.index

rev.columns = ['Revenue']



colors = sns.color_palette('Set3', len(rev))

plt.figure(figsize=(12,8))

ax = sns.barplot(y = genres , x = 'Revenue', data=rev, orient='h', palette=colors)

ax.set_xlabel(xlabel='Revenue in $ Millions', fontsize=16)

ax.set_ylabel(ylabel='Genre', fontsize=16)

ax.set_title(label='Genres by Total Revenue Generated in $ Millions', fontsize=20)

ax.set_yticklabels(labels = genres, fontsize=14)

plt.show();
data = pd.concat([rev, rel], axis=1)

data = pd.DataFrame(data['Revenue'] / data['Releases'])

data.columns = ['Revenue Per Release']

data = data.sort_values(by='Revenue Per Release',ascending=False)

genres = data.index



colors = sns.color_palette("CMRmap", len(data))

plt.figure(figsize=(12,8))

ax = sns.barplot(y = genres , x = 'Revenue Per Release', data=data, orient='h', palette=colors)

ax.set_xlabel(xlabel='Revenue per Release in $ Millions', fontsize=16)

ax.set_ylabel(ylabel='Genre', fontsize=16)

ax.set_title(label='Revenue Per Release Generated in $ Millions Per Genre', fontsize=20)

ax.set_yticklabels(labels = genres, fontsize=14)

plt.show();
data = pd.concat([df['Name'][0:10], df['Global_Sales'][0:10]], axis=1)



plt.figure(figsize=(12,8))

colors = sns.color_palette("gist_earth", len(data))

ax = sns.barplot(y = 'Name' , x = 'Global_Sales', data=data, orient='h', palette=colors)

ax.set_xlabel(xlabel='Revenue in $ Millions', fontsize=16)

ax.set_ylabel(ylabel='Name', fontsize=16)

ax.set_title(label='Top 10 Games by Revenue Generated in $ Millions', fontsize=20)

ax.set_yticklabels(labels = games, fontsize=14)

plt.show();
data = df.sum()

data = pd.DataFrame([data['EU_Sales'], data['JP_Sales'], data['NA_Sales'], data['Other_Sales']])

regions = ['European Union', 'Japan', 'North America', 'Other']

data.index = regions

data.columns = ['Revenue']

data = data.sort_values(by='Revenue', ascending=False)



plt.figure(figsize=(12,8))

colors = sns.color_palette("Dark2", len(data))

ax = sns.barplot(y = regions , x = 'Revenue', data=data, orient='h', palette=colors)

ax.set_xlabel(xlabel='Revenue in $ Millions', fontsize=16)

ax.set_ylabel(ylabel='Region', fontsize=16)

ax.set_title(label='Total Revenue Generated in $ Millions by Region', fontsize=20)

ax.set_yticklabels(labels = regions, fontsize=14)

plt.show();
data = df

data = pd.DataFrame([data['EU_Sales'], data['JP_Sales'], data['NA_Sales'], data['Other_Sales']]).T

regions = ['European Union', 'Japan', 'North America', 'Other']

q = data.quantile(0.90)

data = data[data < q]

plt.figure(figsize=(12,8))



colors = sns.color_palette("Set1", len(data))

ax = sns.boxplot(data=data, orient='h', palette=colors)

ax.set_xlabel(xlabel='Revenue per Game in $ Millions', fontsize=16)

ax.set_ylabel(ylabel='Region', fontsize=16)

ax.set_title(label='Distribution of Sales Per Game in $ Millions Per Region', fontsize=20)

ax.set_yticklabels(labels=regions, fontsize=14)

plt.show()
table = df.pivot_table('Global_Sales', columns='Year', index='Name')

q = table.quantile(0.90)

data = table[table < q]

years = table.columns.astype(int)



plt.figure(figsize=(12,8))

ax = sns.boxplot(data=data)

ax.set_xticklabels(labels=years, fontsize=12, rotation=50)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Revenue Per Game in $ Millions', fontsize=16)

ax.set_title(label='Distribution of Revenue Per Game by Year in $ Millions', fontsize=20)

plt.show()

plt.show()
EU = df.pivot_table('EU_Sales', columns='Name', index='Year', aggfunc='sum').sum(axis=1)

NA = df.pivot_table('NA_Sales', columns='Name', index='Year', aggfunc='sum').sum(axis=1)

JP = df.pivot_table('JP_Sales', columns='Name', index='Year', aggfunc='sum').sum(axis=1)

Other = df.pivot_table('Other_Sales', columns='Name', index='Year', aggfunc='sum').sum(axis=1)

years = Other.index.astype(int)

regions = ['European Union','Japan','North America','Other']



plt.figure(figsize=(12,8))

ax = sns.pointplot(x=years, y=EU, color='mediumslateblue', scale=0.7)

ax = sns.pointplot(x=years, y=NA, color='cornflowerblue', scale=0.7)

ax = sns.pointplot(x=years, y=JP, color='orchid', scale=0.7)

ax = sns.pointplot(x=years, y=Other, color='thistle', scale=0.7)

ax.set_xticklabels(labels=years, fontsize=12, rotation=50)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Revenue in $ Millions', fontsize=16)

ax.set_title(label='Distribution of Total Revenue Per Region by Year in $ Millions', fontsize=20)

ax.legend(handles=ax.lines[::len(years)+1], labels=regions, fontsize=18)

plt.show();
lizt = ['Electronic Arts', 'Activision', 'Namco Bandai Games', 'Ubisoft', 'Konami Digital Entertainment', 'THQ', 'Nintendo', 'Sony Computer Entertainment', 'Sega', 'Take-Two Interactive']

data = df.pivot_table('Global_Sales', columns='Publisher', index='Year', aggfunc='sum')

data = [[data[i] for i in lizt]]

data = np.array(data)

data = np.reshape(data, (10, 38))

data = pd.DataFrame(data)

data.index = lizt

ind = ['1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987',

                '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',

                '1996', '1997', '1998', '1999', '2000', '2001', '2002',

                '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',

                '2011', '2012', '2013', '2014', '2015', '2016', '2017']

data.columns = ind

data = data.fillna(0)

totals = pd.DataFrame(data.sum()).T

totals = pd.DataFrame(totals)

x1980 = data.iloc[:,0]/totals['1980'].values

x1981 = data.iloc[:,1]/totals['1981'].values

x1982 = data.iloc[:,2]/totals['1982'].values

x1983 = data.iloc[:,3]/totals['1983'].values

x1984 = data.iloc[:,4]/totals['1984'].values

x1985 = data.iloc[:,5]/totals['1985'].values

x1986 = data.iloc[:,6]/totals['1986'].values

x1987 = data.iloc[:,7]/totals['1987'].values

x1988 = data.iloc[:,8]/totals['1988'].values

x1989 = data.iloc[:,9]/totals['1989'].values

x1990 = data.iloc[:,10]/totals['1990'].values

x1991 = data.iloc[:,11]/totals['1991'].values

x1992 = data.iloc[:,12]/totals['1992'].values

x1993 = data.iloc[:,13]/totals['1993'].values

x1994 = data.iloc[:,14]/totals['1994'].values

x1995 = data.iloc[:,15]/totals['1995'].values

x1996 = data.iloc[:,16]/totals['1996'].values

x1997 = data.iloc[:,17]/totals['1997'].values

x1998 = data.iloc[:,18]/totals['1998'].values

x1999 = data.iloc[:,19]/totals['1999'].values

x2000 = data.iloc[:,20]/totals['2000'].values

x2001 = data.iloc[:,21]/totals['2001'].values

x2002 = data.iloc[:,22]/totals['2002'].values

x2003 = data.iloc[:,23]/totals['2003'].values

x2004 = data.iloc[:,24]/totals['2004'].values

x2005 = data.iloc[:,25]/totals['2005'].values

x2006 = data.iloc[:,26]/totals['2006'].values

x2007 = data.iloc[:,27]/totals['2007'].values

x2008 = data.iloc[:,28]/totals['2008'].values

x2009 = data.iloc[:,29]/totals['2009'].values

x2010 = data.iloc[:,30]/totals['2010'].values

x2011 = data.iloc[:,31]/totals['2011'].values

x2012 = data.iloc[:,32]/totals['2012'].values

x2013 = data.iloc[:,33]/totals['2013'].values

x2014 = data.iloc[:,34]/totals['2014'].values

x2015 = data.iloc[:,35]/totals['2015'].values

x2016 = data.iloc[:,36]/totals['2016'].values

x2017 = data.iloc[:,37]/totals['2017'].values



years = [x1980,x1981,x1982,x1983,x1984,x1985,x1986,x1987,x1988,x1989,x1990,x1991,x1992,x1993,x1994,x1995,x1996,x1997,x1998,x1999,x2000,x2001,x2002,x2003,x2004,x2005,x2006,x2007,x2008,x2009,x2010,x2011,x2012,x2013,x2014,x2015,x2016,x2017]

years = pd.concat(years, axis=1)

years.columns = ind

lizt = ['Electronic Arts', 'Activision', 'Nintendo', 'Sony Computer Entertainment', 'Ubisoft']

liztr = lizt[::-1]

plt.figure(figsize=(12,8))

ax = sns.pointplot(x=ind, y=years.loc['Electronic Arts'], color='brown', scale=0.7)

ax = sns.pointplot(x=ind, y=years.loc['Activision'], color='darkorange', scale=0.7)

ax = sns.pointplot(x=ind, y=years.loc['Nintendo'], color='darkolivegreen', scale=0.7)

ax = sns.pointplot(x=ind, y=years.loc['Sony Computer Entertainment'], color='darkred', scale=0.7)

ax = sns.pointplot(x=ind, y=years.loc['Ubisoft'], color='yellowgreen', scale=0.7)

ax.set_xticklabels(labels = ind, fontsize=12, rotation=50)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Publisher', fontsize=16)

ax.set_title(label='Market Share of Top 5 Publishers Per Year', fontsize=20)

ax.legend(handles=ax.lines[::len(ind)+1], labels=liztr, fontsize=18)

plt.show();
lizt = ['Action', 'Adventure', 'Fighting', 'Misc', 'Platform', 'Puzzle', 'Racing', 'Role-Playing', 'Shooter', 'Simulation', 'Sports', 'Strategy']

data = df.pivot_table('Global_Sales', columns='Genre', index='Year', aggfunc='sum')

data = [[data[i] for i in lizt]]

data = np.array(data)

data = np.reshape(data, (12, 38))

data = pd.DataFrame(data)

data.index = lizt

ind = ['1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987',

                '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',

                '1996', '1997', '1998', '1999', '2000', '2001', '2002',

                '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',

                '2011', '2012', '2013', '2014', '2015', '2016', '2017']

data.columns = ind

data = data.fillna(0)

totals = pd.DataFrame(data.sum()).T

totals = pd.DataFrame(totals)

x1980 = data.iloc[:,0]/totals['1980'].values

x1981 = data.iloc[:,1]/totals['1981'].values

x1982 = data.iloc[:,2]/totals['1982'].values

x1983 = data.iloc[:,3]/totals['1983'].values

x1984 = data.iloc[:,4]/totals['1984'].values

x1985 = data.iloc[:,5]/totals['1985'].values

x1986 = data.iloc[:,6]/totals['1986'].values

x1987 = data.iloc[:,7]/totals['1987'].values

x1988 = data.iloc[:,8]/totals['1988'].values

x1989 = data.iloc[:,9]/totals['1989'].values

x1990 = data.iloc[:,10]/totals['1990'].values

x1991 = data.iloc[:,11]/totals['1991'].values

x1992 = data.iloc[:,12]/totals['1992'].values

x1993 = data.iloc[:,13]/totals['1993'].values

x1994 = data.iloc[:,14]/totals['1994'].values

x1995 = data.iloc[:,15]/totals['1995'].values

x1996 = data.iloc[:,16]/totals['1996'].values

x1997 = data.iloc[:,17]/totals['1997'].values

x1998 = data.iloc[:,18]/totals['1998'].values

x1999 = data.iloc[:,19]/totals['1999'].values

x2000 = data.iloc[:,20]/totals['2000'].values

x2001 = data.iloc[:,21]/totals['2001'].values

x2002 = data.iloc[:,22]/totals['2002'].values

x2003 = data.iloc[:,23]/totals['2003'].values

x2004 = data.iloc[:,24]/totals['2004'].values

x2005 = data.iloc[:,25]/totals['2005'].values

x2006 = data.iloc[:,26]/totals['2006'].values

x2007 = data.iloc[:,27]/totals['2007'].values

x2008 = data.iloc[:,28]/totals['2008'].values

x2009 = data.iloc[:,29]/totals['2009'].values

x2010 = data.iloc[:,30]/totals['2010'].values

x2011 = data.iloc[:,31]/totals['2011'].values

x2012 = data.iloc[:,32]/totals['2012'].values

x2013 = data.iloc[:,33]/totals['2013'].values

x2014 = data.iloc[:,34]/totals['2014'].values

x2015 = data.iloc[:,35]/totals['2015'].values

x2016 = data.iloc[:,36]/totals['2016'].values

x2017 = data.iloc[:,37]/totals['2017'].values



years = [x1980,x1981,x1982,x1983,x1984,x1985,x1986,x1987,x1988,x1989,x1990,x1991,x1992,x1993,x1994,x1995,x1996,x1997,x1998,x1999,x2000,x2001,x2002,x2003,x2004,x2005,x2006,x2007,x2008,x2009,x2010,x2011,x2012,x2013,x2014,x2015,x2016,x2017]

years = pd.concat(years, axis=1)

years.columns = ind

data.columns = ind

liztr = lizt[::-1]

lizt = ['Electronic Arts', 'Activision', 'Nintendo', 'Sony Computer Entertainment', 'Ubisoft']

plt.figure(figsize=(12,8))

ax = sns.pointplot(x=ind, y=years.loc['Platform'], color='crimson', scale=0.7)

ax = sns.pointplot(x=ind, y=years.loc['Shooter'], color='slateblue', scale=0.7)

ax = sns.pointplot(x=ind, y=years.loc['Role-Playing'], color='orange', scale=0.7)

ax.set_xticklabels(labels = ind, fontsize=12, rotation=50)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Genre', fontsize=16)

ax.set_title(label='Market Share Per Genre Per Year', fontsize=20)

ax.legend(handles=ax.lines[::len(ind)+1], labels=liztr, fontsize=18)

plt.show();
def turn_off_labels(ax, first=True):

    if first == False:

        x_axis = ax.axes.get_xaxis()

        x_label = x_axis.get_label()

        x_label.set_visible(False)

        y_axis = ax.axes.get_yaxis()

        y_axis.set_visible(False)

    else:

        x_axis = ax.axes.get_xaxis()

        x_label = x_axis.get_label()

        x_label.set_visible(False)
EU = df.pivot_table('EU_Sales', columns='Publisher', aggfunc='sum').T

EU = EU.sort_values(by='EU_Sales', ascending=False).iloc[0:3]

EU_publishers = EU.index



JP = df.pivot_table('JP_Sales', columns='Publisher', aggfunc='sum').T

JP = JP.sort_values(by='JP_Sales', ascending=False).iloc[0:3]

JP_publishers = JP.index



NA = df.pivot_table('NA_Sales', columns='Publisher', aggfunc='sum').T

NA = NA.sort_values(by='NA_Sales', ascending=False).iloc[0:3]

NA_publishers = NA.index



Other = df.pivot_table('Other_Sales', columns='Publisher', aggfunc='sum').T

Other = Other.sort_values(by='Other_Sales', ascending=False).iloc[0:3]

Other_publishers = Other.index



colors =  {'Nintendo':sns.xkcd_rgb["teal blue"], 'Electronic Arts':sns.xkcd_rgb["denim blue"], 'Activision':sns.xkcd_rgb["dark lime green"], 'Namco Bandai Games':sns.xkcd_rgb["pumpkin"], 'Konami Digital Entertainment':sns.xkcd_rgb["burnt umber"], 'Sony Computer Entertainment':sns.xkcd_rgb["yellow orange"]}

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(1,4,1)

ax1.set_xticklabels(labels = EU_publishers, rotation=90, size=14)

turn_off_labels(ax1)

sns.barplot(x=EU_publishers, y=EU['EU_Sales'], palette=colors)

plt.title('European Union', size=18)

plt.ylabel('Revenue in $ Millions', size=16)



ax2 = fig.add_subplot(1,4,2)

ax2.set_xticklabels(labels = JP_publishers, rotation=90, size=14)

turn_off_labels(ax2, first=False)

sns.barplot(x=JP_publishers, y=JP['JP_Sales'], palette=colors)

plt.title('Japan', size=18)



ax3 = fig.add_subplot(1,4,3)

ax3.set_xticklabels(labels = NA_publishers, rotation=90, size=14)

turn_off_labels(ax3, first=False)

sns.barplot(x=NA_publishers, y=NA['NA_Sales'], palette=colors)

plt.title('North America', size=18)



ax4 = fig.add_subplot(1,4,4)

ax4.set_xticklabels(labels = Other_publishers, rotation=90, size=14)

turn_off_labels(ax4, first=False)

sns.barplot(x=Other_publishers, y=Other['Other_Sales'], palette=colors)

plt.title('Other', size=18)

plt.suptitle('Top 3 Publishers by Revenue Per Region', size=22)

plt.show();
EU = df.pivot_table('EU_Sales', columns='Name', aggfunc='sum').T

EU = EU.sort_values(by='EU_Sales', ascending=False).iloc[0:3]

EU_games = EU.index



JP = df.pivot_table('JP_Sales', columns='Name', aggfunc='sum').T

JP = JP.sort_values(by='JP_Sales', ascending=False).iloc[0:3]

JP_games = JP.index



NA = df.pivot_table('NA_Sales', columns='Name', aggfunc='sum').T

NA = NA.sort_values(by='NA_Sales', ascending=False).iloc[0:3]

NA_games = NA.index



Other = df.pivot_table('Other_Sales', columns='Name', aggfunc='sum').T

Other = Other.sort_values(by='Other_Sales', ascending=False).iloc[0:3]

Other_games = Other.index



colors =  {'Wii Sports':"salmon", 'Grand Theft Auto V':"mediumseagreen", 'Mario Kart Wii':"lightskyblue", 'Pokemon Red/Pokemon Blue':"lightslategray", 'Pokemon Gold/Pokemon Silver':"cornflowerblue", 'Super Mario Bros.':"plum", 'Duck Hunt':"pink", 'Grand Theft Auto: San Andreas':"seagreen"}

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(1,4,1)

ax1.set_xticklabels(labels = EU_games, rotation=90, size=14)

turn_off_labels(ax1)

sns.barplot(x=EU_games, y=EU['EU_Sales'], palette=colors)

plt.title('European Union', size=18)

plt.ylabel('Revenue in $ Millions', size=16)



ax2 = fig.add_subplot(1,4,2)

ax2.set_xticklabels(labels = JP_games, rotation=90, size=14)

turn_off_labels(ax2, first=False)

sns.barplot(x=JP_games, y=JP['JP_Sales'], palette=colors)

plt.title('Japan', size=18)



ax3 = fig.add_subplot(1,4,3)

ax3.set_xticklabels(labels = NA_games, rotation=90, size=14)

turn_off_labels(ax3, first=False)

sns.barplot(x=NA_games, y=NA['NA_Sales'], palette=colors)

plt.title('North America', size=18)



ax4 = fig.add_subplot(1,4,4)

ax4.set_xticklabels(labels = Other_games, rotation=90, size=14)

turn_off_labels(ax4, first=False)

sns.barplot(x=Other_games, y=Other['Other_Sales'], palette=colors)

plt.title('Other', size=18)

plt.suptitle('Top 3 Games by Revenue Per Region', size=22)

plt.show();
EU = df.pivot_table('EU_Sales', columns='Genre', aggfunc='sum').T

EU = EU.sort_values(by='EU_Sales', ascending=False).iloc[0:3]

EU_genres = EU.index



JP = df.pivot_table('JP_Sales', columns='Genre', aggfunc='sum').T

JP = JP.sort_values(by='JP_Sales', ascending=False).iloc[0:3]

JP_genres = JP.index



NA = df.pivot_table('NA_Sales', columns='Genre', aggfunc='sum').T

NA = NA.sort_values(by='NA_Sales', ascending=False).iloc[0:3]

NA_genres = NA.index



Other = df.pivot_table('Other_Sales', columns='Genre', aggfunc='sum').T

Other = Other.sort_values(by='Other_Sales', ascending=False).iloc[0:3]

Other_genres = Other.index



colors =  {'Action':"orchid", 'Sports':"mediumslateblue", 'Shooter':"cornflowerblue", 'Role-Playing':"steelblue"}

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(1,4,1)

ax1.set_xticklabels(labels = EU_genres, rotation=90, size=14)

turn_off_labels(ax1)

sns.barplot(x=EU_genres, y=EU['EU_Sales'], palette=colors)

plt.title('European Union', size=18)

plt.ylabel('Revenue in $ Millions', size=16)



ax2 = fig.add_subplot(1,4,2, sharey=ax1)

ax2.set_xticklabels(labels = JP_genres, rotation=90, size=14)

turn_off_labels(ax2, first=False)

sns.barplot(x=JP_genres, y=JP['JP_Sales'], palette=colors)

plt.title('Japan', size=18)



ax3 = fig.add_subplot(1,4,3, sharey=ax1)

ax3.set_xticklabels(labels = NA_genres, rotation=90, size=14)

turn_off_labels(ax3, first=False)

sns.barplot(x=NA_genres, y=NA['NA_Sales'], palette=colors)

plt.title('North America', size=18)



ax4 = fig.add_subplot(1,4,4, sharey=ax1)

ax4.set_xticklabels(labels = Other_genres, rotation=90, size=14)

turn_off_labels(ax4, first=False)

sns.barplot(x=Other_genres, y=Other['Other_Sales'], palette=colors)

plt.title('Other', size=18)

plt.suptitle('Top 3 Genres by Revenue Per Region', size=22)

plt.show();
EU = df.pivot_table('EU_Sales', columns='Year', aggfunc='sum').T

EU = EU.sort_values(by='EU_Sales', ascending=False).iloc[0:3]

EU_years = EU.index.astype(int)



JP = df.pivot_table('JP_Sales', columns='Year', aggfunc='sum').T

JP = JP.sort_values(by='JP_Sales', ascending=False).iloc[0:3]

JP_years = JP.index.astype(int)



NA = df.pivot_table('NA_Sales', columns='Year', aggfunc='sum').T

NA = NA.sort_values(by='NA_Sales', ascending=False).iloc[0:3]

NA_years = NA.index.astype(int)



Other = df.pivot_table('Other_Sales', columns='Year', aggfunc='sum').T

Other = Other.sort_values(by='Other_Sales', ascending=False).iloc[0:3]

Other_years = Other.index.astype(int)



colors =  {2006.0:"yellowgreen", 2007.0:"mediumaquamarine", 2008.0:"cornflowerblue", 2009.0:"steelblue", 2010.0:"darkseagreen"}

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(1,4,1)

ax1.set_xticklabels(labels = EU_years, rotation=90, size=14)

turn_off_labels(ax1)

sns.barplot(x=EU_years, y=EU['EU_Sales'], palette=colors)

plt.title('European Union', size=18)

plt.ylabel('Revenue in $ Millions', size=16)



ax2 = fig.add_subplot(1,4,2, sharey=ax1)

ax2.set_xticklabels(labels = JP_years, rotation=90, size=14)

turn_off_labels(ax2, first=False)

sns.barplot(x=JP_years, y=JP['JP_Sales'], palette=colors)

plt.title('Japan', size=18)



ax3 = fig.add_subplot(1,4,3, sharey=ax1)

ax3.set_xticklabels(labels = NA_years, rotation=90, size=14)

turn_off_labels(ax3, first=False)

sns.barplot(x=NA_years, y=NA['NA_Sales'], palette=colors)

plt.title('North America', size=18)



ax4 = fig.add_subplot(1,4,4, sharey=ax1)

ax4.set_xticklabels(labels = Other.index, rotation=90, size=14)

turn_off_labels(ax4, first=False)

sns.barplot(x=Other_years, y=Other['Other_Sales'], palette=colors)

plt.title('Other', size=18)

plt.suptitle('Top 3 Years by Revenue Per Region', size=22)

plt.show();
EU = df.pivot_table('EU_Sales', columns='Platform', aggfunc='sum').T

EU = EU.sort_values(by='EU_Sales', ascending=False).iloc[0:3]

EU_plats = EU.index



JP = df.pivot_table('JP_Sales', columns='Platform', aggfunc='sum').T

JP = JP.sort_values(by='JP_Sales', ascending=False).iloc[0:3]

JP_plats = JP.index



NA = df.pivot_table('NA_Sales', columns='Platform', aggfunc='sum').T

NA = NA.sort_values(by='NA_Sales', ascending=False).iloc[0:3]

NA_plats = NA.index



Other = df.pivot_table('Other_Sales', columns='Platform', aggfunc='sum').T

Other = Other.sort_values(by='Other_Sales', ascending=False).iloc[0:3]

Other_plats = Other.index



colors =  {'PS':"goldenrod", 'PS2':"maroon", 'PS3':"lightsalmon", 'DS':"coral", 'X360':"peachpuff", 'Wii':"darkorange"}

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(1,4,1)

ax1.set_xticklabels(labels = EU_plats, rotation=90, size=14)

turn_off_labels(ax1, first=False)

sns.barplot(x=EU_plats, y=EU['EU_Sales'], palette=colors)

plt.title('European Union', size=18)

plt.ylabel('Revenue in $ Millions', size=16)



ax2 = fig.add_subplot(1,4,2, sharey=ax1)

ax2.set_xticklabels(labels = JP_plats, rotation=90, size=14)

turn_off_labels(ax2, first=False)

sns.barplot(x=JP_plats, y=JP['JP_Sales'], palette=colors)

plt.title('Japan', size=18)



ax3 = fig.add_subplot(1,4,3, sharey=ax1)

ax3.set_xticklabels(labels = NA_plats, rotation=90, size=14)

turn_off_labels(ax3, first=False)

sns.barplot(x=NA_plats, y=NA['NA_Sales'], palette=colors)

plt.title('North America', size=18)



ax4 = fig.add_subplot(1,4,4, sharey=ax1)

ax4.set_xticklabels(labels = Other_plats, rotation=90, size=14)

turn_off_labels(ax4, first=False)

sns.barplot(x=Other_plats, y=Other['Other_Sales'], palette=colors)

plt.title('Other', size=18)

plt.suptitle('Top 3 Platforms by Revenue Per Region', size=22)

plt.show();
rpy = df['Global_Sales'] / (2017 - df['Year'])

ms = df['Global_Sales'] / df['Global_Sales'].sum()

df['RevPG'] = rpy

df['MktShr'] = ms



data1 = df.pivot_table('MktShr', columns='Publisher', index='Global_Sales')

data2 = df.pivot_table('RevPG', columns='Publisher', index='Global_Sales')



EA1 = data1['Electronic Arts']

Act1 = data1['Activision']

Ubi1 = data1['Ubisoft']

Nint1 = data1['Nintendo']

Sony1 = data1['Sony Computer Entertainment']

Tktwo1 = data1['Take-Two Interactive']

Namc1 = data1['Namco Bandai Games']

Kona1 = data1['Konami Digital Entertainment']

THQ1 = data1['THQ']

Seg1 = data1['Sega']



lizt1 = [EA1, Act1, Ubi1, Nint1, Sony1, Tktwo1, Namc1, Kona1, THQ1, Seg1]

data1 = pd.concat(lizt1, ignore_index=True, axis=1)

data1.columns = ['Electronic Arts', 'Activision', 'Ubisoft', 'Nintendo', 'Sony Computer Entertainment', 'Take-Two Interactive', 'Namco Bandai Games', 'Konami Digital Entertainment', 'THQ', 'Sega']

data1.index = range(0, len((data1)))



EA2 = data2['Electronic Arts']

Act2 = data2['Activision']

Ubi2 = data2['Ubisoft']

Nint2 = data2['Nintendo']

Sony2 = data2['Sony Computer Entertainment']

Tktwo2 = data2['Take-Two Interactive']

Namc2 = data2['Namco Bandai Games']

Kona2 = data2['Konami Digital Entertainment']

THQ2 = data2['THQ']

Seg2 = data2['Sega']



lizt2 = [EA2, Act2, Ubi2, Nint2, Sony2, Tktwo2, Namc2, Kona2, THQ2, Seg2]

data2 = pd.concat(lizt2, ignore_index=True, axis=1)

data2.columns = ['Electronic Arts', 'Activision', 'Ubisoft', 'Nintendo', 'Sony Computer Entertainment', 'Take-Two Interactive', 'Namco Bandai Games', 'Konami Digital Entertainment', 'THQ', 'Sega']

data2.index = range(0, len((data1)))



fig = plt.figure(figsize=(12,8))

ax = sns.swarmplot(x=np.log(data1['Activision']), y=np.log(data2['Activision']), label='Activision')

ax = sns.swarmplot(x=np.log(data1['Ubisoft']), y=np.log(data2['Ubisoft']), label='Ubisoft')

ax = sns.swarmplot(x=np.log(data1['Nintendo']), y=np.log(data2['Nintendo']), label='Nintendo')

ax = sns.swarmplot(x=np.log(data1['Take-Two Interactive']), y=np.log(data2['Take-Two Interactive']), label='Take=Two Interactive')

ax = sns.swarmplot(x=np.log(data1['Sony Computer Entertainment']), y=np.log(data2['Sony Computer Entertainment']), label='Sony Computer Entertainment')

ax = sns.swarmplot(x=np.log(data1['Electronic Arts']), y=np.log(data2['Electronic Arts']), label='Electronic Arts')

ax = sns.swarmplot(x=np.log(data1['Namco Bandai Games']), y=np.log(data2['Namco Bandai Games']), label='Namco Bandai Games')

ax = sns.swarmplot(x=np.log(data1['Konami Digital Entertainment']), y=np.log(data2['Konami Digital Entertainment']), label='Konami Digital Entertainment')

ax = sns.swarmplot(x=np.log(data1['THQ']), y=np.log(data2['THQ']), label='THQ')

ax = sns.swarmplot(x=np.log(data1['Sega']), y=np.log(data2['Sega']), label='Sega')







ax.set_xlabel(xlabel='Game Revenue Per Year', fontsize=16)

ax.set_ylabel(ylabel='Game Market Share', fontsize=16)

ax.set_title(label='Revenue Per Year Versus Share of Total Revenue for All Games by Top 10 Publishers', fontsize=20)



plt.tick_params(axis='x', which='both', bottom='off',

                top='off', labelbottom='off')

plt.show();
act = data1['Activision']

EA = data1['Electronic Arts']

ubi = data1['Ubisoft']

nint = data1['Nintendo']

sony = data1['Sony Computer Entertainment']

tktwo = data1['Take-Two Interactive']

Namc = data1['Namco Bandai Games']

Kona = data1['Konami Digital Entertainment']

THQ = data1['THQ']

Seg = data1['Sega']



act = np.log(act)

EA = np.log(EA)

ubi = np.log(ubi)

nint = np.log(nint)

sony = np.log(sony)

tktwo = np.log(tktwo)

Namc = np.log(Namc)

Kona = np.log(Kona)

THQ = np.log(THQ)

Seg = np.log(Seg)



min1 = nint.min() - 1

max1 = nint.max() + 1



fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(5,2,1)



sns.swarmplot(y=act, color='cornflowerblue')

plt.title('Activision', size=12)

ax1.set_ylim([min1,max1])

y_axis = ax1.axes.get_yaxis()

y_axis.set_visible(False)



ax2 = fig.add_subplot(5,2,2)

sns.swarmplot(y=EA, color='salmon')

plt.title('Electronic Arts', size=12)

ax2.set_ylim([min1,max1])

y_axis = ax2.axes.get_yaxis()

y_axis.set_visible(False)



ax3 = fig.add_subplot(5,2,3)

sns.swarmplot(y=nint, color='lightblue')

plt.title('Nintendo', size=12)

ax3.set_ylim([min1,max1])

y_axis = ax3.axes.get_yaxis()

y_axis.set_visible(False)



ax4 = fig.add_subplot(5,2,4)

sns.swarmplot(y=ubi, color='thistle')

plt.title('Ubisoft', size=12)

ax4.set_ylim([min1,max1])

y_axis = ax4.axes.get_yaxis()

y_axis.set_visible(False)



ax5 = fig.add_subplot(5,2,5)

sns.swarmplot(y=sony, color='orchid')

plt.title('Sony Computer Entertainment', size=12)

ax5.set_ylim([min1,max1])

y_axis = ax5.axes.get_yaxis()

y_axis.set_visible(False)



ax6 = fig.add_subplot(5,2,6)

sns.swarmplot(y=tktwo, color='mediumseagreen')

ax6.set_ylim([min1,max1])

y_axis = ax6.axes.get_yaxis()

y_axis.set_visible(False)

plt.title('Take-Two Interactive', size=12)



ax7 = fig.add_subplot(5,2,7)

sns.swarmplot(y=Namc, color='mediumaquamarine')

ax7.set_ylim([min1,max1])

y_axis = ax7.axes.get_yaxis()

y_axis.set_visible(False)

plt.title('Namco Bandai Games', size=12)



ax8 = fig.add_subplot(5,2,8)

sns.swarmplot(y=Kona, color='yellowgreen')

ax8.set_ylim([min1,max1])

y_axis = ax8.axes.get_yaxis()

y_axis.set_visible(False)

plt.title('Konami Digital Entertainment', size=12)



ax9 = fig.add_subplot(5,2,9)

sns.swarmplot(y=THQ, color='coral')

ax9.set_ylim([min1,max1])

y_axis = ax9.axes.get_yaxis()

y_axis.set_visible(False)

plt.title('THQ', size=12)



ax10 = fig.add_subplot(5,2,10)

sns.swarmplot(y=Seg, color='mediumslateblue')

ax10.set_ylim([min1,max1])

y_axis = ax10.axes.get_yaxis()

y_axis.set_visible(False)

plt.title('Sega', size=12)

plt.suptitle('Distribution of Market Share For Top 10 Publishers', size=22, x=0.5, y=0.94)

plt.subplots_adjust(wspace=0)

plt.show();

data2 = data2.dropna()

act = data2['Activision']

EA = data2['Electronic Arts']

ubi = data2['Ubisoft']

nint = data2['Nintendo']

sony = data2['Sony Computer Entertainment']

tktwo = data2['Take-Two Interactive']



act = np.log(act)

EA = np.log(EA)

ubi = np.log(ubi)

nint = np.log(nint)

sony = np.log(sony)

tktwo = np.log(tktwo)



fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(3,2,1)

sns.distplot(act, kde=False, color=sns.xkcd_rgb["jungle green"])

y_axis = ax1.axes.get_yaxis()

y_axis.set_visible(False)



ax2 = fig.add_subplot(3,2,2)

sns.distplot(EA, kde=False, color=sns.xkcd_rgb["dark blue green"])

y_axis = ax2.axes.get_yaxis()

y_axis.set_visible(False)



ax3 = fig.add_subplot(3,2,3)

sns.distplot(ubi, kde=False, color=sns.xkcd_rgb["purple blue"])

y_axis = ax3.axes.get_yaxis()

y_axis.set_visible(False)



ax4 = fig.add_subplot(3,2,4)

sns.distplot(nint, kde=False, color=sns.xkcd_rgb["fuchsia"])

y_axis = ax4.axes.get_yaxis()

y_axis.set_visible(False)



ax5 = fig.add_subplot(3,2,5)

sns.distplot(sony, kde=False, color=sns.xkcd_rgb["marine blue"])

y_axis = ax5.axes.get_yaxis()

y_axis.set_visible(False)



ax6 = fig.add_subplot(3,2,6)

sns.distplot(tktwo, kde=False, color=sns.xkcd_rgb["azure"])

y_axis = ax6.axes.get_yaxis()

y_axis.set_visible(False)



plt.suptitle('Log Scale Distribution of Revenue Per Year Per Game For Top 6 Publishers', size=22, x=0.5, y=0.94)

plt.show();
act = data2['Activision']

EA = data2['Electronic Arts']

ubi = data2['Ubisoft']

nint = data2['Nintendo']

sony = data2['Sony Computer Entertainment']

tktwo = data2['Take-Two Interactive']



act = np.log(act)

EA = np.log(EA)

ubi = np.log(ubi)

nint = np.log(nint)

sony = np.log(sony)

tktwo = np.log(tktwo)



fig = plt.figure(figsize=(12,12))

sns.distplot(act, kde=False, label='Activision', hist_kws={"histtype": "step", "linewidth": 3,

                  "alpha": 1, "color": sns.xkcd_rgb["azure"]})



sns.distplot(EA, kde=False, label='Electronic Arts', hist_kws={"histtype": "step", "linewidth": 3,

                  "alpha": 1, "color": sns.xkcd_rgb["dark blue green"]})





sns.distplot(nint, kde=False, label='Nintendo', hist_kws={"histtype": "step", "linewidth": 3,

                  "alpha": 1, "color": sns.xkcd_rgb["fuchsia"]})



plt.suptitle('Log Scale Distribution of Revenue Per Year For Top 3 Publishers Per Game Release', size=22, x=0.5, y=0.94)

plt.xlabel('Publisher', size=16)

plt.legend(prop={'size':26}, loc=2)

plt.show();