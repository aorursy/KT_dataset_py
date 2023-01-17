#importing libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#reading the data

vg = pd.read_csv('../input/vgsales.csv')
vg.info()
vg.head()
vg.tail()
vg.describe(include = 'all')
# Checking for NaN's
vg.isnull().sum()
# Deleting the entries with NaN's
vg.dropna(subset = ['Year','Publisher'], how = 'any', axis = 0, inplace = True)

# Setting the year as int
vg.Year = vg.Year.astype(int)

#Checking the Df info again
vg.info()
# Fixing one registry, the game came out in 2009, and the data go up to early 2017
## After a quick filter and google search, i found out that the game came out in '09
## vg.loc[vg.Year == 2020]

vg.Year.replace(2020, 2009, inplace= True)
# Consolidating the data in another DataFrame

'''This dataframe contains all the sales data ordered by platform and global sales. It already meets my specification of knowing
what genre does best at each platform. But its tedious to read and takes too long to absorb the information'''

cons = vg.groupby(['Platform','Genre'], as_index = True).sum()
cons.drop(['Rank','Year'], axis= 1, inplace = True)
cons.sort_values(by = ['Platform','Global_Sales'], ascending = False, inplace = True)
cons.head(24)
# Gathering the same data as the above DF, but in a more 'plot friendly shape'

cons2 = vg.groupby(['Platform','Genre'], as_index = False).sum()
cons2.drop(['Rank','Year'], axis= 1, inplace = True)
cons2.sort_values(by = ['Platform','Global_Sales'], ascending = False, inplace = True)

# Filtering the games that sold more than 4 mil. copies, to generate a cleaner plot
cons2.loc[cons2.Global_Sales >= 4].sort_values(by='Global_Sales', ascending=False)
# Setting the colors used in the plots below, fell free to try any other color palette

#palette=sns.diverging_palette(128,240,79,32,1,n=12)
#palette= sns.color_palette("YlGnBu", 12)

clrz = ['#d70909',"#f26522",'#0000ff','#FFE00E','#a864a8','#790000','#005826','#00bff3','#636363','#8dc63f','#440e62','#ec008c']
# Ploting all the data from the DF above

with sns.plotting_context('notebook'):
    
    sns.catplot(data = cons2.loc[cons2.Global_Sales >= 4].sort_values(by='Global_Sales', ascending=False),
                x= 'Platform', y= 'Global_Sales', hue= 'Genre', ci = None, kind= 'swarm', 
                dodge = False, alpha = .8, aspect=2.5, marker = 'o', palette=sns.color_palette(clrz))
    
    sns.despine(left= True, bottom=True)
    plt.title('Average Game Sales By Genre and Platform (in milion units)')
    plt.grid(axis='both',which='major')
    plt.show()

# Plotting a better graph

#This is an overlayed plot
f, ax = plt.subplots(ncols=1,nrows=1,sharey=True,figsize=(15, 10),dpi = 300)

with sns.plotting_context('notebook'):
    
    g = sns.barplot(data=cons2.loc[cons2.Global_Sales >= 4].sort_values(by='Global_Sales', ascending=False), x='Platform',
                y='Global_Sales', hue ='Genre', ci=None, dodge=False, alpha= .15,palette=sns.color_palette(clrz), ax=ax)
       
    g.set_xticklabels('')
    g.set_xlabel('')  
    
    ax2 = ax # Using the same axis (The ax.twinx method upsets the Y axis, and is a pain to realign later)
    
    g = sns.stripplot(data = cons2.loc[cons2.Global_Sales >= 4].sort_values(by='Global_Sales', ascending=False),
                x= 'Platform', y= 'Global_Sales', hue= 'Genre', palette=sns.color_palette(clrz),  
                dodge = True, alpha = 1, marker = 'D',ax=ax2  )
    sns.despine(left= True, bottom=True)
    plt.title('Average Game Sales By Genre and Platform (in milion units)')
    plt.grid(axis='both',which='major')
    plt.legend(ncol=2, frameon=True, loc='upper right')
    plt.show()

# Sales by genre and region plot

regional = vg.groupby('Genre').sum().sort_values('Global_Sales', ascending = False).drop(['Rank','Year'], axis=1)
regional.plot(kind = 'bar', figsize = (15,8), rot= 0, fontsize = 12, grid= True, width=0.8)
plt.title('Sales by genre and region (in milion units)')
plt.show()
print()
regional
NA = vg.sort_values('NA_Sales',ascending = False).head(10)
EU = vg.sort_values('EU_Sales',ascending = False).head(10)
JP = vg.sort_values('JP_Sales',ascending = False).head(10)
Other = vg.sort_values('Other_Sales',ascending = False).head(10)
Global = vg.sort_values('Global_Sales',ascending = False).head(10)

#top10 = pd.concat([NA,EU,JP,Other,Global], axis = 0, ignore_index = True)
NA
NAg = NA.groupby('Genre').sum().drop(['Rank','Year'], axis = 1).sort_values(by='NA_Sales', ascending = False).reset_index()
sns.barplot(data=NAg, y='Genre',x='NA_Sales')
plt.grid(axis='both',which='major')
plt.title('Number of sales, top 10 games sold in North America by genre')
plt.xlabel('Sales in milions units')
plt.show()
print()
NAg
EU
EUg = EU.groupby('Genre').sum().drop(['Rank','Year'], axis = 1).sort_values(by='EU_Sales', ascending = False).reset_index()
sns.barplot(data=EUg, y='Genre',x='EU_Sales')
plt.grid(axis='both',which='major')
plt.title('Number of sales, top 10 games sold in Europe by genre')
plt.xlabel('Sales in milions units')
plt.show()
print()
EUg
JP
JPg = JP.groupby('Genre').sum().drop(['Rank','Year'], axis = 1).sort_values(by='JP_Sales', ascending = False).reset_index()
sns.barplot(data=JPg, y='Genre',x='JP_Sales')
plt.grid(axis='both',which='major')
plt.title('Number of sales, top 10 games sold in Japan by genre')
plt.xlabel('Sales in milions units')
plt.show()
print()
JPg
Other
Otherg = Other.groupby('Genre').sum().drop(['Rank','Year'], axis = 1).sort_values(by='Other_Sales', ascending = False).reset_index()
sns.barplot(data=Otherg, y='Genre',x='Other_Sales')
plt.grid(axis='both',which='major')
plt.title('Number of sales, top 10 games sold in Other regions by genre')
plt.xlabel('Sales in milions units')
plt.show()
print()
Otherg
Global
Globalg = Global.groupby('Genre').sum().drop(['Rank','Year'], axis = 1).sort_values(by='Global_Sales', ascending = False).reset_index()
sns.barplot(data=Globalg, y='Genre',x='Global_Sales')
plt.grid(axis='both',which='major')
plt.title('Number of sales, top 10 games sold Globally by genre')
plt.xlabel('Sales in milions units')
plt.show()
print()
Globalg
pub = vg.groupby('Publisher').sum().sort_values('Global_Sales', ascending = False).drop(['Rank','Year'], axis=1)

f = plt.figure(figsize=(15, 10),dpi = 300)
sns.barplot(data= pub.reset_index().head(20), x= 'Global_Sales', y='Publisher')
plt.grid(axis='both')
plt.title('Number of Sales by Publisher (In million units)')
plt.show()
# 'massaging' the data to generate the plot below
pub2 = pub.sort_values('Global_Sales', ascending = False).reset_index().head(20).melt(
    id_vars='Publisher', value_vars=pub.columns, var_name= 'Region',value_name='Sales')

# Replacing the region tags
pub2.Region.replace(to_replace='NA_Sales',value='North America',inplace=True)
pub2.Region.replace(to_replace='EU_Sales',value='Europe',inplace=True)
pub2.Region.replace(to_replace='JP_Sales',value='Japan',inplace=True)
pub2.Region.replace(to_replace='Other_Sales',value='Other Regions',inplace=True)
pub2.Region.replace(to_replace='Global_Sales',value='Global Sales',inplace=True)

# Generating the plot
with sns.plotting_context('notebook'):    
    c = sns.catplot(data= pub2, y='Publisher', x='Sales', col='Region', col_wrap = 3, kind='bar',sharex=False,)
    c.fig.set_dpi(150)
    c.set_titles('Number of Sales in {col_name}')

#Wrangling the data to plot the info
GenPub = vg.groupby(['Publisher','Genre']).sum().sort_values('Global_Sales', ascending = False).drop(['Rank','Year'], axis=1).reset_index()

GenPub.rename({'NA_Sales':'North America','EU_Sales':'Europe','JP_Sales':'Japan','Other_Sales':'Other Regions','Global_Sales':'Global'},
              axis= 'columns', inplace= True)

GenPub.groupby(['Publisher','Genre'],as_index=False).sum().sort_values('Global', ascending = False, inplace = True)

GenPub = GenPub.melt(id_vars=['Publisher','Genre'], var_name='Region', value_name='Sales')
# Top 10 Publishers by genre
sns.set_context('notebook')

f, ax = plt.subplots(figsize=(10, 7),dpi = 200,)
f = sns.barplot(data= GenPub.loc[GenPub.Sales >= 60], x='Sales', y='Publisher', hue='Genre', estimator=max, errwidth=0) 
ax.set_xticks([50,150,250,350,450],minor=True)
plt.legend(loc=4)
plt.title('Top 15 publishers by genre')
plt.xlabel('Sales in milion units')
plt.grid(which='both', axis='both', alpha=0.3)

# Overlay dots at the bars
#sns.stripplot(data= GenPub.loc[GenPub.Sales >= 60], x='Sales', y='Publisher', hue='Genre', ax=ax, dodge = True)

# Gathering the data
msold = vg.groupby(['Year','Name','Platform'], axis= 0, as_index=True).sum().sort_values(by=['Global_Sales'], ascending=False)#.reset_index()
msold.head()
# Rearanging the data
msold1 = msold.sort_values(by='Year',kind='mergesort').reset_index()
msold1.head()
msold1.loc[msold1.Year == 1981].head(1)
ano = 1980
most_sold = pd.DataFrame()
for val in msold1.iterrows():
    most_sold = msold1.loc[msold1.Year == ano].head(1).append(most_sold)
    #print(x)
    ano+= 1
    val = ano
    if (ano == 2017):
        break
# Most sold games by year
most_sold.sort_values(by='Year',ascending = True)
sby = vg.groupby('Year').sum().drop(['Rank','NA_Sales','EU_Sales','JP_Sales','Other_Sales'], axis=1).reset_index()
f = plt.figure(figsize=(15, 10),dpi = 300)
sns.barplot(data=sby,y='Global_Sales', x='Year', estimator = max)
plt.xticks(rotation=90)
plt.ylabel('Global Sales in Milion Units')
plt.title('Game Sales by Year')
plt.grid(axis='both')
plt.show()
# Gathering the data and ordering by Year and global sales

gby = vg.groupby(['Year','Genre'], as_index= False).sum().drop(['Rank','NA_Sales','EU_Sales','JP_Sales','Other_Sales'], axis=1)
gby.sort_values(by=['Year','Global_Sales'], ascending=[True,False], inplace= True)
# This is kinda hard to see, but i think its the best plot that i could came up with, 
#please feel free to modify and represent the information in some other way!

f = sns.catplot(data=gby, y='Global_Sales', x='Year', hue = 'Genre', orient='v', kind= 'strip', aspect=2, height=8, 
                dodge=False, marker='D', palette=clrz)
plt.grid(axis='both',which='both')
plt.xticks(rotation=45)
f.ax.set_yticks([10,30,50,70,90,110,130], minor=True)
plt.ylabel('Global Sales in Milion Units')
plt.title('Yearly Global Sales by Genre', fontdict={'fontsize':15})
plt.show()
# 6 Most sold consoles overall
cons = vg.groupby(['Platform'], as_index=False).sum().drop(['Rank','Year','NA_Sales','EU_Sales','JP_Sales','Other_Sales'], axis=1).sort_values('Global_Sales', ascending=False).head(6)
cons
# 7th and 8th gen consoles
cons2 = vg.groupby(['Platform'], as_index=False).sum().drop(['Rank','Year','NA_Sales','EU_Sales','JP_Sales','Other_Sales'], axis=1).sort_values('Global_Sales', ascending=False)

cons2.loc[(cons2.Platform == 'X360') | (cons2.Platform == 'PS3') | (cons2.Platform == 'Wii') | 
         (cons2.Platform == 'PS4') | (cons2.Platform == 'XOne') | (cons2.Platform == 'WiiU') | (cons2.Platform == 'DS') |
        (cons2.Platform == 'PSP') | (cons2.Platform == '3DS') | (cons2.Platform == 'PSV')]
temp = vg.groupby(['Platform','Year'], as_index= False).sum().drop('Rank', axis=1)
cons12 = temp.loc[(temp.Platform == 'PS') | (temp.Platform == 'PS2') | (temp.Platform == 'PS3') | (temp.Platform == 'PS4') | 
        (temp.Platform == 'X360') | (temp.Platform == 'XOne') | (temp.Platform == 'Wii') | (temp.Platform == 'WiiU') | 
        (temp.Platform == 'DS') | (temp.Platform == 'PSP') | (temp.Platform == '3DS') | (temp.Platform == 'PSV')]
cons12.describe()
cons12.loc[cons12.Year == 1985]
cons12.loc[cons12.Platform == 'DS']
cons12.loc[(cons12['Platform'] == 'DS') & (cons12['Year'] == 1985)]
cons12.loc[(cons12['Platform'] == 'DS') & (cons12['Year'] == 2014)]
# Acessing the 2014 register of the DS platform, through the slice 17:18, and then on the 4th column, assigning the 0.02 value
cons12.iloc[17:18,(4)] = 0.02
# Acessing the 2014 register of the DS platform, through the slice 17:18, and then on the last column, updating the value to 0.04
cons12.iloc[17:18,(-1)] = 0.04
# Deleting the 1985 entry
cons12.drop(labels=25, axis=0, inplace= True)

# And on the last line, the updated values
cons12.head(17)
# Most sold consoles from 6th, 7th and 8th generations
g = sns.catplot(data= cons12, x='Year', y='Global_Sales', col='Platform', col_wrap=2, kind='bar', 
            sharex=False, sharey= True, aspect= 2.5, height = 4.5, estimator= max)
race = vg.loc[(vg.Genre == 'Racing') & (vg.Year < 2000) & (vg.Year > 1989)]
race.reset_index(inplace=True, drop=True)
race


