import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

events=pd.read_csv("../input/athlete_events.csv")
region=pd.read_csv("../input/noc_regions.csv")
events.head()
events.shape
events.describe()
region.info()
events.info()
region.describe()
region.head()
data=pd.merge(events,region ,on='NOC',how='left')
data
data.isnull().sum()
plt.figure(figsize=(20,10))
sns.heatmap(data.isnull(),cbar='False')
print("total unique no of olympic player: ",len(data['ID'].unique()))
print("total no of female olympic player: ",len(data[data.Sex=='F']))
print("total no of male olympic player: ",len(data[data.Sex=='M']))
sns.countplot(data.Sex)
plt.title('Male vs female',color='orange',size=17)
plt.show()
print(data.Age.max(),data.Age.mean(),data.Age.min())
data.loc[(data['Age']==data.Age.min())] #min age person 
data.loc[(data['Age']==data.Age.max())]
plt.figure(figsize=(20,10))
sns.boxplot('Year','Age',data=data)
print(data.Height.max(),data.Height.mean(),data.Height.min())
data.loc[(data['Height']==data.Height.max())].head(1)
data.loc[(data['Height']==data.Height.min())].head(1)
data.loc[(data['Height']==data.Height.min())].tail(1)
plt.figure(figsize=(20,10))
sns.scatterplot('Year','Height',data=data)
corr=data[['Age','Height','Weight']].corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr,annot=True,cmap="Greens")
def medal(medal): 
        x = data[data.Medal == medal].region.value_counts().head(10)
        x.plot(kind='bar',figsize=(15,10))
        plt.title(medal + ' medals',size=15)
        plt.show()
        
medal('Gold') 

medal('Silver')

medal('Bronze') #function calling
def medal_by_sex(medal): 
        x = data[data.Medal == medal].Sex.value_counts().head(10)
        x.plot(kind='bar',figsize=(15,10))
        plt.title(medal + ' medals',size=15)
        print(x)
        plt.show()
        
medal_by_sex('Gold')        
medal_by_sex('Silver') 
medal_by_sex('Bronze') 
womens_in_summer_olympic=data[(data.Sex=='F')&(data.Season=='Summer')]
plt.figure(figsize=(30,10))
sns.boxenplot(x='Year',data=womens_in_summer_olympic,order='Year',scale="linear",palette='Greens')
womens_in_winter_olympic=data[(data.Sex=='F')&(data.Season=='Winter')]
plt.figure(figsize=(30,10))
sns.countplot(x='Year',data=womens_in_winter_olympic,palette='PuBu')
womens_in_olympicgames=data[(data.Sex=='F')]
plt.figure(figsize=(30,10))
sns.countplot(x='Year',data=womens_in_olympicgames,palette='Oranges')
#gold medel for women
women_gold_medal=data[((data.Sex=='F')&(data.Medal=='Gold'))]
plt.figure(figsize=(30,10))
sns.countplot(x='Year',data=women_gold_medal,palette='PuRd')
mens_in_summer_olympic=data[(data.Sex=='M')&(data.Season=='Summer')]
plt.figure(figsize=(30,10))
sns.countplot(x='Year',data=mens_in_summer_olympic,palette='Greens')
mens_in_winter_olympic=data[(data.Sex=='M')&(data.Season=='Winter')]
plt.figure(figsize=(30,10))
sns.countplot(x='Year',data=mens_in_winter_olympic,palette='PuBu')
mens_in_olympicgames=data[(data.Sex=='M')]
plt.figure(figsize=(30,10))
sns.countplot(x='Year',data=mens_in_olympicgames,palette='PuBuGn')
#gold medel for men
men_gold_medal=data[((data.Sex=='M')&(data.Medal=='Gold'))]
plt.figure(figsize=(30,10))
sns.countplot(x='Year',data=men_gold_medal,palette='OrRd')
#top 10 sports in olympic 
top10_sports=((data.Sport).value_counts()).head(10)
plt.figure(figsize=(25,15))
plt.title('Sports',size=15)
top10_sports.plot(kind='bar')

sports_1896=data[(data.Year==1896)].Sport.unique().tolist()
print('total sports:',len(sports_1896))
print('sports in 1896:')
sports_1896
#most popular sports in 1896
most_sports_1896=data[(data.Year==1896)].Sport.value_counts()
plt.figure(figsize=(25,15))
plt.title('Popular sports in 1896',size=15)
most_sports_1896.plot(kind='bar')
sports_2016=data[(data.Year==2016)].Sport.unique().tolist()
print('total sports:',len(sports_2016))
print('sports in 2016:')
sports_2016
#most popular sports in 2016(top 10)
most_sports_2016=data[(data.Year==2016)].Sport.value_counts().head(10)
plt.figure(figsize=(25,15))
plt.title('Popular sports in 2016',size=15)
most_sports_2016.plot(kind='bar')
#women popular sports
women_popular_sports=womens_in_olympicgames.Sport.value_counts().head(10)
plt.figure(figsize=(20,10))
women_popular_sports.plot(kind='bar')

#mens popular sports
mens_popular_sports=mens_in_olympicgames.Sport.value_counts().head(10)
plt.figure(figsize=(20,10))
mens_popular_sports.plot(kind='bar')
#top 10 country
top10_country=data.region.value_counts().head(10)
plt.figure(figsize=(15,5))
top10_country.plot(kind='bar')
usa_region = data[data.region == 'USA']
usa_region.head()
usa_medals = usa_region.Medal.value_counts()
plt.figure(figsize=(15,5))
usa_medals.plot(kind='bar');
gold_medals_usa = usa_region[(usa_region.Medal == 'Gold')]
gold_medals_usa.Event.value_counts().reset_index(name='Medal').head(10)
basketball=usa_region[(usa_region.Event == "Basketball Men's Basketball") & (usa_region.Medal == 'Gold')]
basketball.head()
print('Average Age: ', basketball.Age.mean())
print('Average Height: ', basketball.Height.mean())
print('Min Height: ', basketball.Height.min())
print('Max Height: ', basketball.Height.max())

plt.figure(figsize=(20,10))
sns.violinplot('Year','Age',data=basketball ,color='green')
#Correlation Basketball players
basketball_corr=basketball.corr()
plt.figure(figsize=(20,10))
sns.heatmap(basketball_corr,annot=True,lw=.8,cmap='Blues')
goldmedals=data[data.Medal=='Gold']
goldmedals.head()
goldmedals.isnull().sum()
goldmedals['ID'][goldmedals['Age']>55].count()
masterDisciplines=goldmedals['Sport'][goldmedals['Age']>55]
plt.figure(figsize=(20, 10))
sns.countplot(masterDisciplines)
plt.title('Gold Medals for Athletes Age Over 50')
#top 5 gold country
goldmedel_regionwise=goldmedals.region.value_counts().reset_index(name='Medal').head()
goldmedel_regionwise
plt.figure(figsize=(20, 10))
g=sns.catplot(x="index", y="Medal",hue="index",height=6, kind="bar", palette="muted", data=goldmedel_regionwise)
g.set_ylabels("Top 5 countries")
g.set_xlabels("Number of Medals")
plt.title('Medals per Country')
plt.title('Medals per Country')
goldmedals.head()
notnullmedals=goldmedals[(goldmedals['Height'].notnull()) & (goldmedals['Weight'].notnull())]
notnullmedals
plt.figure(figsize=(25,15))
ax=sns.jointplot(x='Height',y='Weight',data=notnullmedals)
plt.title("Height vs Weight in Olympic Medals")
part = mens_in_olympicgames.groupby('Year')['Sex'].value_counts()
plt.figure(figsize=(20, 10))
part.loc[:,'M'].plot()
plt.title('Variation of Male Athletes over time')
part = womens_in_olympicgames.groupby('Year')['Sex'].value_counts()
plt.figure(figsize=(20, 10))
part.loc[:,'F'].plot()
plt.title('Variation of Female Athletes over time')
plt.figure(figsize=(20, 10))
sns.pointplot('Year', 'Weight', data=mens_in_olympicgames)
plt.title('Variation of Weight for Male Athletes over time')
plt.figure(figsize=(20, 10))
sns.pointplot('Year', 'Weight', data=womens_in_olympicgames)
plt.title('Variation of Weight for Female Athletes over time')
