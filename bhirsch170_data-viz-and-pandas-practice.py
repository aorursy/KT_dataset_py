#Import packages and read in file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

ign_csv=pd.read_csv('../input/ign.csv')
ign_csv.head()
#Rename score and score_phrase columns

ign_csv.rename(columns={ ign_csv.columns[5]: "rating" }, inplace=True)
ign_csv.rename(columns={ ign_csv.columns[1]: "opinion" }, inplace=True)


#Drop unecessary columns
ign_csv=ign_csv.drop(columns=['Unnamed: 0','url'])

print(ign_csv.columns)


#Check column types

ign_csv.info()
#Format release_month column
ign_csv['release_month']=ign_csv['release_month'].apply(lambda x: '{0:0>2}'.format(x))
ign_csv['release_month']=pd.to_numeric(ign_csv['release_month'])
#Check numeric values for outliers
fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4,figsize=(18,4))

sns.boxplot(x=ign_csv['release_year'],ax=ax1)
sns.boxplot(x=ign_csv['release_month'],ax=ax2)
sns.boxplot(x=ign_csv['release_day'],ax=ax3)
sns.boxplot(x=ign_csv['rating'],ax=ax4)
#Fix outlier in release year
print(ign_csv[ign_csv['release_year']<1996])

ign_csv.at[516, 'release_year'] = 2012

print(ign_csv.loc[516,:])
sns.boxplot(x=ign_csv['release_year'])

#Show unique genres 
print(ign_csv['genre'].unique())

#Use , delimiter to split genre's with more than one genre into two columns
ign_csv[['genre','genre2']] = ign_csv['genre'].str.split(',',expand=True)
ign_csv.rename(columns={ign_csv.columns[4]:'genre1'},inplace=True)
#Closer look at the 59 unique platforms in the data set
print(ign_csv['platform'].nunique())
print(ign_csv['platform'].unique())




#create dictionary mapping platforms to manufacturers
x=     {'PlayStation Vita':'Sony', 'iPad':'Mobile', 'Xbox 360':'Microsoft', 'PlayStation 3':'Sony',
       'Macintosh':'Apple', 'PC':'Computer', 'iPhone':'Mobile', 'Nintendo DS':'Nintendo', 'Nintendo 3DS':'Nintendo',
       'Android':'Mobile', 'Wii':'Nintendo', 'PlayStation 4':'Sony', 'Wii U':'Nintendo', 'Linux':'Computer',
       'PlayStation Portable':'Sony', 'PlayStation':'Sony', 'Nintendo 64':'Nintendo', 'Saturn':'Sega',
       'Lynx':'Atari', 'Game Boy':'Nintendo', 'Game Boy Color':'Nintendo', 'NeoGeo Pocket Color':'Other',
       'Game.Com':'Tiger', 'Dreamcast':'Sega', 'Dreamcast VMU':'Sega', 'WonderSwan':'Bandai', 'Arcade':'Arcade',
       'Nintendo 64DD':'Nintendo', 'PlayStation 2':'Sony', 'WonderSwan Color':'Bandai',
       'Game Boy Advance':'Nintendo', 'Xbox':'Microsoft', 'GameCube':'Nintendo', 'DVD / HD Video Game':'Other',
       'Wireless':'Computer', 'Pocket PC':'Mobile', 'N-Gage':'Mobile', 'NES':'Nintendo', 'iPod':'Mobile', 'Genesis':'Sega',
       'TurboGrafx-16':'Other', 'Super NES':'Nintendo', 'NeoGeo':'Other', 'Master System':'Sega',
       'Atari 5200':'Atari', 'TurboGrafx-CD':'Other', 'Atari 2600':'Atari', 'Sega 32X':'Sega', 'Vectrex':'Bandai',
       'Commodore 64/128':'Computer', 'Sega CD':'Sega', 'Nintendo DSi':'Nintendo', 'Windows Phone':'Mobile',
       'Web Games':'Computer', 'Xbox One':'Microsoft', 'Windows Surface':'Mobile', 'Ouya':'Other',
       'New Nintendo 3DS':'Nintendo', 'SteamOS':'Valve'}

#map keys to new column: platform_owner
ign_csv['platform_owner']=ign_csv['platform'].map(x)



#Change relevant columns to categories for faster processing
for col in ['opinion', 'platform', 'genre1', 'genre2','platform_owner']:
    ign_csv[col] = ign_csv[col].astype('category')
    

#Order opinion categories based on verbiage used    
ign_csv['opinion']=ign_csv['opinion'].astype(pd.api.types.CategoricalDtype (categories= ['Masterpiece','Amazing', 'Great','Good','Okay','Mediocre','Bad','Awful','Painful','Unbearable','Disaster'], ordered=True))
#Check for null values
print(ign_csv.info())
print(ign_csv.isnull().sum())
ign_csv[ign_csv['genre1'].isnull()]

#Recreate dataframe with no null values in genre1
ign_csv=ign_csv[pd.notnull(ign_csv['genre1'])]

#check if nulls have been dropped
print(ign_csv.isnull().sum())
#Making new table of average rating associated with each 'opinion' word to give better context

ign_opinions=pd.DataFrame(ign_csv.groupby('opinion')['rating'].mean())
ign_opinions.reset_index(inplace=True)

#Plotting chart comparing average rating by opinion word
plt.figure(figsize=(12,4))
ax=sns.barplot(x=ign_opinions['opinion'],y=ign_opinions['rating'])
ax.set(ylabel='Average Rating',xlabel='Reviewer Opinion')
plt.show()
ign_opinions_freq=pd.DataFrame(ign_csv.groupby('opinion')['rating'].count())
ign_opinions_freq.reset_index(inplace=True)


#Plotting chart comparing average rating by opinion word
plt.figure(figsize=(12,4))
ax= sns.barplot(x='opinion',y='rating',data=ign_opinions_freq)
ax.set(ylabel='Number of Reviews',xlabel='Reviewer Opinion')
plt.show()

#Create Dataframe to showcase top rated genres

#Remove weird genre
ign_csv=ign_csv[ign_csv['genre1']!='Hardware']

ign_genre_rating=pd.DataFrame(ign_csv.groupby('genre1')['rating'].mean().sort_values(ascending=False).head(10))

ign_genre_rating.reset_index(inplace=True)

ign_genre_rating['genre1'].cat.remove_unused_categories(inplace=True)

#Order genres by rating
order=ign_genre_rating.sort_values('rating',ascending=False)
order=order['genre1'].tolist()

#Create dataframe to compare average overall ratings over time
ign_timeline=pd.DataFrame(ign_csv.groupby('release_year')['rating'].mean())

ign_timeline.reset_index(inplace=True)

ign_timeline.release_year=pd.to_datetime(ign_timeline.release_year,format='%Y')





#Plot two charts for average overall ratings over time and top rated genres 
fig, (ax1,ax2)=plt.subplots(nrows=2,ncols=1,figsize=(18,8))

sns.lineplot(x='release_year',y='rating',data=ign_timeline,ax=ax1)

sns.barplot(x='rating',y='genre1', data=ign_genre_rating,order=order,ax=ax2)


#Set custom labels

ax1.set(ylabel='Average Overall Rating',xlabel='Year')
ax2.set(ylabel='Genre',xlabel='Average Rating')
plt.show()



#Creating dataframes to plot most reviewed platforms from each decade

#The 90s
decade_1=(ign_csv['release_year']>=1996) & (ign_csv['release_year']<=1999)

the90s=ign_csv[decade_1]

the90s=pd.DataFrame(the90s.groupby('platform')['rating'].count().sort_values(ascending=False))

the90s.reset_index(inplace=True)

the90s.dropna(inplace=True)

the90s['platform'].cat.remove_unused_categories(inplace=True)

the90s=the90s.head(15)

#The 2000s
decade_2=(ign_csv['release_year']>=2000) & (ign_csv['release_year']<=2009)

the2000s=ign_csv[decade_2]

the2000s=pd.DataFrame(the2000s.groupby('platform')['rating'].count().sort_values(ascending=False))

the2000s.reset_index(inplace=True)

the2000s.dropna(inplace=True)

the2000s['platform'].cat.remove_unused_categories(inplace=True)

the2000s=the2000s.head(15)

#The 2010s
decade_3=(ign_csv['release_year']>=2010) & (ign_csv['release_year']<=2016)

the2010s=ign_csv[decade_3]

the2010s=pd.DataFrame(the2010s.groupby('platform')['rating'].count().sort_values(ascending=False))

the2010s.reset_index(inplace=True)

the2010s.dropna(inplace=True)

the2010s['platform'].cat.remove_unused_categories(inplace=True)

the2010s=the2010s.head(15)

#Order bars by rating
order1=the90s.sort_values('rating',ascending=False)
order1=order1['platform'].tolist()

order2=the2000s.sort_values('rating',ascending=False)
order2=order2['platform'].tolist()

order3=the2010s.sort_values('rating',ascending=False)
order3=order3['platform'].tolist()





#Plot all 3 decades

fig, (ax1,ax2,ax3)=plt.subplots(nrows=3,ncols=1,figsize=(28,23))


sns.barplot('platform','rating',data=the90s,order=order1,ax=ax1)
ax1.set(ylabel='Number of Reviews',title='Most Reviewed Platforms 1996-1999')

sns.barplot('platform','rating',data=the2000s,order=order2,ax=ax2)
ax2.set(ylabel='Number of Reviews',title='Most Reviewed Platforms 2000-2010')

sns.barplot('platform','rating',data=the2010s,order=order3,ax=ax3)
ax3.set(ylabel='Number of Reviews',title='Most Reviewed Platforms 2010-2016')

plt.show()
#Create release date column
ign_csv['release_date']=ign_csv['release_year'].map(str)+'-'+ign_csv['release_month'].map(str)+'-'+ign_csv['release_day'].map(str)
ign_csv['release_date']= pd.to_datetime(ign_csv['release_date'],format='%Y-%m-%d')

ign_csv['day_of_week'] = ign_csv['release_date'].dt.day_name()

#Drop duplicates to prevent multiple release date counts from titles shipped on multiple platforms
ign_tuesday= ign_csv.drop_duplicates(subset='title')

#Create dataframe for graphing the dates
ign_tuesday=ign_tuesday[['release_year','day_of_week']].copy()

ign_tuesday=ign_tuesday.groupby(['release_year','day_of_week'])['day_of_week'].count()

ign_tuesday=pd.DataFrame(ign_tuesday)

ign_tuesday.columns=['day_count']

ign_tuesday.reset_index(inplace=True)

ign_tuesday.columns=['year','Weekday','day_count']

#Format year for the x axis
ign_tuesday.year=pd.to_datetime(ign_tuesday.year,format='%Y')


#Plot multi line chart
fig, ax = plt.subplots(figsize=(15,5))

sns.lineplot(x='year',y='day_count',hue='Weekday',data=ign_tuesday)

ax.set(title='Release Day Trends 1996-2016', ylabel='Number of Releases', xlabel='Year',)

ax.legend(loc='upper left')

plt.show()









ign_xp=ign_csv.copy()

ign_xp=ign_xp.groupby(['release_year','platform_owner'])['title'].count()

ign_xp=pd.DataFrame(ign_xp)

ign_xp=ign_xp.reset_index()

xpn=(ign_xp['platform_owner']=='Sony') | (ign_xp['platform_owner']=='Microsoft') | (ign_xp['platform_owner']=='Nintendo')

ign_xpn=ign_xp[xpn]

ign_xpn.columns=['Year','Company','Number of Reviews']

ign_xpn['Company'].cat.remove_unused_categories(inplace=True)

Year=pd.to_datetime(ign_xpn.Year,format='%Y')


#Plot multi line chart
fig, ax = plt.subplots(figsize=(15,5))

sns.lineplot(x=Year,y='Number of Reviews',hue='Company',data=ign_xpn)

ax.set(title='Number of Reviews on Company Platform')

ax.legend(loc='upper left')

plt.show()
