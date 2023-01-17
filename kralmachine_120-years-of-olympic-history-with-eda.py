import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



data=pd.read_csv('../input/athlete_events.csv')

#The fist five rows in Data

data.head()
#The last five rows in Data show that

data.tail()
#If you show more rows the first you should write it.

data.head(10)
data.sample()
data.sample(frac=0.01)
#Data show info.So I want to see data info's

data.info()

#We see all of columns in Data.They have some features
#Data describe

data.describe() #but I don't want ID,Year columns so,

data[['Age','Height','Weight']].describe()
data.dtypes
sns.boxenplot(x="Height", y="Weight",

              color="b",

              scale="linear", data=data)

plt.show()
# Show the joint distribution using kernel density estimation

g = sns.jointplot(data.Age, data.Height, kind="kde", height=7, space=0)

plt.show()
#I wanto to seee columns so,

data.columns
#data's shape

data.shape
#Data's corr value but I dont want ID,Year columns because overfitting

data.corr()

data[['Age','Height','Weight']].corr()
#Data types in all because I want to see columns variable values

data.dtypes
#Data sizes.its rows x columns :)

data.size
#data sum null values

data.isnull().sum()

#You can see some columns null.Bu when we execute ml algo. it don't work because null
#So we execute Age,Height,Weight use Imputer

from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean') #use mean or median 

Ages=data[['Age']].values

Ages=imputer.fit_transform(Ages[:,0:1])

data.Age=Ages

data.Age.isnull().sum()

##########################################################################

Heights=data[['Height']].values

Heights=imputer.fit_transform(Heights[:,0:1])

data.Height=Heights

data.Height.isnull().sum()

##########################################################################

Weights=data[['Weight']].values

Weights=imputer.fit_transform(Weights[:,0:1])

data.Weight=Weights

data.Weight.isnull().sum()
#Re show null value

data.isnull().sum()

#That's good job.But mead has some null values.It does not matter.
#First column is Sex.now I make analysis this one.

data.Sex.unique() #M,F, non null

data.Sex.value_counts() #M=196594 , F=74522 values

sns.countplot(data.Sex)

plt.title('Sex Values in Data',size=15,color='Blue')

plt.show()
#So, I want to see sum female,male for every year and show plot

sum_year_female=[]

sum_year_male=[]

all_year_unique=data.Year.unique()

data.groupby('Year')['Sex'].value_counts()



for year in all_year_unique:

    sum_year_male.append(data[(data['Year']==year) & (data['Sex']=='M')].Sex.count())

    sum_year_female.append(data[(data['Year']==year) & (data['Sex']=='F')].Sex.count())



all_year_unique=pd.DataFrame(all_year_unique,columns=['Year'])   

sum_year_male=pd.DataFrame(sum_year_male,columns=['Sex'])

sum_year_female=pd.DataFrame(sum_year_female,columns=['Sex'])



year_female=pd.concat([all_year_unique,sum_year_female],axis=1)

year_male=pd.concat([all_year_unique,sum_year_male],axis=1)



year_female=year_female.sort_index() 

year_male=year_male.sort_index()

#we need to sort out our data so that it appears more smoothly

#Now I have sum of year every Sex in Data so I will draw plotting



f,ax1=plt.subplots(figsize=(20,10))

sns.pointplot(x=data.Year.unique(),y=year_male.Sex,color='lime',alpha=0.8)

sns.pointplot(x=data.Year.unique(),y=year_female.Sex,color='red',alpha=0.8)

plt.text(10,0.5,'ALL YEAR FOR EVERY FEMALE',color='red',fontsize = 15,style = 'italic')

plt.text(25,0.5,'ALL YEAR FOR EVERY MALE',color='lime',fontsize = 15,style = 'italic')

plt.xlabel('YEAR',fontsize = 15,color='blue')

plt.xticks(rotation=90)

plt.ylabel('SUM OF GENDERS',fontsize = 15,color='blue')

plt.title('MALE VS FEMALE',fontsize = 20,color='blue')

plt.grid()

plt.show()
ages=data['Age'].unique()

ages=pd.DataFrame(ages,columns=['Ages'])

ages=ages.sort_index()

#We second columns is Age.I will analysis

sum_of_ages=[]

for age in data.Age.unique():

    sum_of_ages.append(sum(data[data['Age']==age].Age.value_counts()))



sum_of_ages=pd.DataFrame(sum_of_ages,columns=['Ages_Count']) 



sum_of_ages[:10]

ages[:10]



data_age=pd.concat([ages[:10],sum_of_ages[:10]],axis=1) #We concat age,age counts in Data because while show them in barplot



data_age

#We count the ages just unique

plt.figure(figsize=(10,10))

sns.barplot(x=data_age.Ages,y=data_age.Ages_Count)

plt.xlabel('Age')

plt.ylabel('Age Values')

plt.title('Age Vs Values')

plt.show()
#Now, I will show Team counts in every year

plt.figure(figsize=(10,10))

sns.countplot(x=data.Team[:300])

plt.xlabel('Team')

plt.ylabel('Counts')

plt.xticks(rotation=90)

plt.title('Sum of Team Counts')

plt.show()
#We took the first 50 teams in this section. 

#We made an inquiry between the teams. 

#This is to process according to both the team and the sex in the questionnaire.

sum_of_female_team=[]

sum_of_male_team=[]

for team in data.Team.unique()[:50]:

    sum_of_female_team.append(data[(data['Team']==team) & (data['Sex']=='F')].Sex.count())

    sum_of_male_team.append(data[(data['Team']==team) & (data['Sex']=='M')].Sex.count())



team_of_300=pd.DataFrame(data.Team.unique()[:300],columns=['Team'])

sum_of_male_team=pd.DataFrame(sum_of_male_team,columns=['Male'])

sum_of_female_team=pd.DataFrame(sum_of_female_team,columns=['Female']) 
#In this section, the analysis obtained at the top is transformed into a graph.

all_data=pd.concat([sum_of_female_team,sum_of_male_team],axis=1)



f,ax=plt.subplots(figsize=(10,10))

sns.barplot(y=data.Team.unique()[:50],x=all_data.Male,color='green',alpha=0.5,label='Male')

sns.barplot(y=data.Team.unique()[:50],x=all_data.Female,color='red',alpha=0.7,label='Female')

ax.legend(loc='lower right',frameon=True)

ax.set(ylabel='Team',xlabel='Rate of Gender',title='Male vs Female')

plt.show()

    

   
data.Games.unique()

all_of_games=data.Games.value_counts()

colors = ['grey','blue','red','yellow','green','brown','lime','pink','orange','purple']

explode = [0,0,0,0,0,0,0,0,0,0]

plt.figure(figsize = (7,7))

plt.pie(all_of_games.values[:10], explode=explode, labels=all_of_games.index[:10], colors=colors, autopct='%1.1f%%')

plt.title('Games',color = 'blue',fontsize = 15)

plt.show()
#In this section you can see how much time has been played and what has been played on the data.

games_type=[]

value_games=data.Games.str.split()

for d in value_games:

    games_type.append(d[1])



games_type=pd.DataFrame(games_type,columns=['Games_Type'])

data['Game_Type']=games_type

types_game=data.Game_Type.value_counts()

colors = ['red','yellow']

explode = [0,0]

plt.figure(figsize = (7,7))

plt.pie(types_game.values, explode=explode, labels=types_game.index, colors=colors, autopct='%1.1f%%')

plt.title('Games Type',color = 'blue',fontsize = 15)

plt.show()
#This section is about how many times the cities are played.

plt.figure(figsize=(10,10))

ax=sns.barplot(x=data.City.value_counts().index,y=data.City.value_counts().values,palette=sns.cubehelix_palette(len(data.City.value_counts().index)))

plt.xlabel('City')

plt.ylabel('Rates')

plt.xticks(rotation=90)

plt.title('Most Common City of Rates',fontsize=15,color='b')

plt.show()

#As you can see there is a lot of proportions in some places here. The most important thing to note here is the graphical situation.
#In this section we will perform the weight and height operations. 

#Here we will determine the average aspect ratios of women and men and plot them.

avg_height_female=data[data['Sex']=='F'].Height.mean()

avg_height_male=data[data['Sex']=='M'].Height.mean()

avg_weight_female=data[data['Sex']=='F'].Weight.mean()

avg_weight_male=data[data['Sex']=='M'].Weight.mean()

#new feature for height and weight but avg

gamer_feature_height_female=['Short' if height<avg_height_female else 'Tall' for height in data[data['Sex']=='F'].Height]

gamer_feature_height_male=['Short' if height<avg_height_male else 'Tall' for height in data[data['Sex']=='M'].Height]

gamer_feature_weight_female=['Weak' if weight<avg_weight_female else 'Normal Fat' for weight in data[data['Sex']=='F'].Weight]

gamer_feature_weight_male=['Weak' if weight<avg_weight_male else 'Normal Fat' for weight in data[data['Sex']=='M'].Weight]
data.Sport.value_counts()

#The most popular 20 sports

sns.barplot(y=data.Sport.value_counts().index[:20],x=data.Sport.value_counts().values[:20])

plt.xlabel('Rates')

plt.ylabel('Sports')

plt.title('Sport of Rates',fontsize=15,color='b')

plt.show()
#20 non-popular sports

non_popular=data.Sport.value_counts()

non_popular=non_popular.sort_index()

sort_data_index=sorted(non_popular.index)

sort_data_values=sorted(non_popular.values)

sns.barplot(y=sort_data_index[:20],x=sort_data_values[:20])

plt.xlabel('Rates')

plt.ylabel('Sports')

plt.title('Sport of Rates',fontsize=15,color='b')

plt.show()

sum_of_medal=[]

for team in data.Team.unique():

    sum_of_medal.append(data[data['Team']==team].Medal.count())
#In this episode, the medals won by his country team are opened.

all_team=pd.DataFrame(data.Team.unique(),columns=['Team'])

sum_of_medal=pd.DataFrame(sum_of_medal,columns=['Medal'])



medal_of_team=pd.concat([all_team,sum_of_medal],axis=1)

medal_of_team=medal_of_team.sort_values(by='Medal', ascending=False)

plt.figure(figsize=(7,7))

plt.xticks(rotation=90)

sns.barplot(x=medal_of_team.Team[:20],y=medal_of_team.Medal[:20])

plt.show()
#This section shows the medals that the Chinese team has won from the very beginning to this time.

sum_of_medal_just_china=[]

sum_of_medal_just_usa=[]

sum_of_medal_just_turkey=[]



for year in data.Year.unique():

    sum_of_medal_just_china.append(data[(data['Year']==year)&(data['Team']=='China')].Medal.count())

    sum_of_medal_just_usa.append(data[(data['Year']==year)&(data['Team']=='United States')].Medal.count())

    sum_of_medal_just_turkey.append(data[(data['Year']==year)&(data['Team']=='Turkey')].Medal.count())

    

sum_of_medal_just_china=pd.DataFrame(sum_of_medal_just_china,columns=['China_Medal_Count'])

sum_of_medal_just_usa=pd.DataFrame(sum_of_medal_just_usa,columns=['USA_Medal_Count'])

sum_of_medal_just_turkey=pd.DataFrame(sum_of_medal_just_turkey,columns=['Turkey_Medal_Count'])



year_unique=pd.DataFrame(data.Year.unique(),columns=['Year'])



all_data_medal_china=pd.concat([year_unique,sum_of_medal_just_china],axis=1)

sum_of_medal_just_usa=pd.concat([year_unique,sum_of_medal_just_usa],axis=1)

sum_of_medal_just_turkey=pd.concat([year_unique,sum_of_medal_just_turkey],axis=1)



all_data_medal_usa=sum_of_medal_just_usa.sort_values(by='USA_Medal_Count',ascending='True')

all_data_medal_turkey=sum_of_medal_just_turkey.sort_values(by='Turkey_Medal_Count',ascending='True')

all_data_medal_china=all_data_medal_china.sort_values(by='China_Medal_Count',ascending='True')

all_data_medal_china

#Drawings will be made in this section. Also they show the medals they won every year.

f,ax1=plt.subplots(figsize=(20,10))

sns.pointplot(x=all_data_medal_usa.Year,y=all_data_medal_usa.USA_Medal_Count,color='lime',alpha=0.8)

sns.pointplot(x=all_data_medal_turkey.Year,y=all_data_medal_turkey.Turkey_Medal_Count,color='black',alpha=0.8)

sns.pointplot(x=all_data_medal_china.Year,y=all_data_medal_china.China_Medal_Count,color='red',alpha=0.8)

plt.text(5,15,'China Medal Count',color='red',fontsize = 20,style = 'italic')

plt.text(5,35,'Turkey Medal Count',color='black',fontsize = 20,style = 'italic')

plt.text(5,55,'USA Medal Count',color='lime',fontsize = 20,style = 'italic')



plt.xlabel('Year',fontsize = 15,color='blue')

plt.xticks(rotation=90)

plt.ylabel('Medal Count',fontsize = 15,color='blue')

plt.title('Medal Count per Year',fontsize = 15,color='blue')

plt.grid()
#Turkey win medal all of time

sum_of_medal_of_turkey=[]

for medal in data.Medal.unique():

    sum_of_medal_of_turkey.append(data[(data['Team']=='Turkey')&(data['Medal']==medal)].Medal.count())



medal_unique=pd.DataFrame(data.Medal.unique(),columns=['Medal'])

sum_of_medal_of_turkey=pd.DataFrame(sum_of_medal_of_turkey,columns=['Medal_of_Turkey'])

all_state_rows=pd.concat([medal_unique,sum_of_medal_of_turkey],axis=1)

all_state_rows

#drawing graph in all every one

plt.figure(figsize=(7,7))

sns.barplot(x=all_state_rows.Medal_of_Turkey,y=all_state_rows.Medal)

plt.xlabel('Medal Count')

plt.ylabel('Medal of Turkey')

plt.title('Medal of Turkey Counts',fontsize=15,color='blue')
#In this section, necessary actions are taken for event processing. 

#In addition, all documents are scanned.

event_data=data.Event.value_counts()

plt.figure(figsize=(7,7))

sns.barplot(x=event_data.index[:20],y=event_data.values[:20])

plt.xlabel('Event Name')

plt.ylabel('Count')

plt.title('Event Name vs Count',fontsize=15,color='blue')

plt.xticks(rotation=90)

plt.show()
#In this section, all the teams are played and the football event that they won is started.

sum_of_event_all_team=[]

for team in data.Team.unique():

    sum_of_event_all_team.append(data[(data['Event']=='Football Men\'s Football')&(data['Team']==team)].Medal.count())



sum_of_event_all_team=pd.DataFrame(sum_of_event_all_team,columns=['Medal_Count_Event'])

all_team=pd.DataFrame(data.Team.unique(),columns=['Team'])

all_data_event_team=pd.concat([all_team,sum_of_event_all_team],axis=1)

all_data_event_team_sorted=all_data_event_team.sort_values(by='Medal_Count_Event',ascending='False')



#Drawings will be made in this section. Also they show the medals they won every year.

f,ax1=plt.subplots(figsize=(20,10))

sns.pointplot(x=all_data_event_team_sorted.Team[1130:1184],y=all_data_event_team_sorted.Medal_Count_Event[1130:1184],color='lime',alpha=0.8)

plt.text(5,15,'All Team event Medal Count',color='red',fontsize = 20,style = 'italic')

plt.xlabel('Team',fontsize = 15,color='blue')

plt.xticks(rotation=90)

plt.ylabel('Medal Count',fontsize = 15,color='blue')

plt.title('Medal Count vs All Team',fontsize = 15,color='blue')

plt.grid()
#data.drop(['Game_Type'],axis=1,inplace=True)

#data.head()

#sns.distplot(data['Height'])

#sns.distplot(data['Weight'])
#Number of Male athletes have increased from 2500 to 7500 per summer game since 1896. 

#Female athletes have steep increase in numbers from 2000 in 1980 games to 6000 athletes in 2016 games. 

#The number of women athletes at the Olympic Games is approaching 50 per cent. Since 2012, 

#women have participated in every Olympic sport at the Games. 

#All new sports to be included in the Games must contain women’s events. 

#The IOC has increased the number of women’s events on the Olympic programme, in collaboration with the IFs and the organising committees.

#Number of athletes in Winter Games are small compared to summer games as expected. 

#The difference in male and female athlete numbers is less compared to Summer Games.



data.Sport.unique()

data.head()

sum_of_male_athletess_summer=[]

sum_of_female_athletess_summer=[]

sum_of_male_athletess_winter=[]

sum_of_female_athletess_winter=[]

for year in data.Year.unique():

    sum_of_male_athletess_summer.append(data[(data['Season']=='Summer')&(data['Sex']=='M')&(data['Sport']=='Athletics')&(data['Year']==year)].ID.count())

    sum_of_female_athletess_summer.append(data[(data['Season']=='Summer')&(data['Sex']=='F')&(data['Sport']=='Athletics')&(data['Year']==year)].ID.count())

    

    sum_of_male_athletess_winter.append(data[(data['Season']=='Winter')&(data['Sex']=='M')&(data['Sport']=='Athletics')&(data['Year']==year)].ID.count())

    sum_of_female_athletess_winter.append(data[(data['Season']=='Winter')&(data['Sex']=='F')&(data['Sport']=='Athletics')&(data['Year']==year)].ID.count())

    

year_unique=pd.DataFrame(data.Year.unique(),columns=['Year'])

sum_of_male_athletess_summer=pd.DataFrame(sum_of_male_athletess_summer,columns=['AthSumSumM'])

sum_of_female_athletess_summer=pd.DataFrame(sum_of_female_athletess_summer,columns=['AthSumSumF'])

######################################################################################################

sum_of_male_athletess_winter=pd.DataFrame(sum_of_male_athletess_winter,columns=['AthWinM'])

sum_of_female_athletess_winter=pd.DataFrame(sum_of_female_athletess_winter,columns=['AthWinF'])



all_data_summer=pd.concat([year_unique,sum_of_male_athletess_summer],axis=1)

all_data_summer=pd.concat([all_data_summer,sum_of_female_athletess_summer],axis=1)



all_data_winter=pd.concat([year_unique,sum_of_male_athletess_winter],axis=1)

all_data_winter=pd.concat([all_data_winter,sum_of_female_athletess_winter],axis=1)



all_data_summer_sorted=all_data_summer.sort_values(by='Year',ascending='False')

all_data_winter_sorted=all_data_winter.sort_values(by='Year',ascending='False')



years=['1994','1998','2002','2006','2010','2014']

all_data_summer_sorted=  all_data_summer.sort_values(by='AthSumSumM',ascending='True')

all_data_summer_sorted=all_data_summer_sorted[6:] 

all_data_summer_sorted=all_data_summer_sorted.sort_values(by='Year',ascending='True')

all_data_summer_sorted



#Drawings will be made in this section. Also they show the medals they won every year.

f,ax1=plt.subplots(figsize=(20,10))

sns.pointplot(x=all_data_summer_sorted.Year,y=all_data_summer_sorted.AthSumSumM,color='lime',alpha=0.8)

sns.pointplot(x=all_data_summer_sorted.Year,y=all_data_summer_sorted.AthSumSumF,color='red',alpha=0.8)

plt.text(5,15,'All Team event Atheletics Count',color='red',fontsize = 20,style = 'italic')

plt.xlabel('Team',fontsize = 15,color='blue')

plt.xticks(rotation=90)

plt.ylabel('Atheletics Count',fontsize = 15,color='blue')

plt.title('Summer Atheletics Count vs Per Year',fontsize = 15,color='blue')

plt.grid()
#All of medal in data show them every country and city.

#There are Gold,Silver,Bronze medals some countries.

plt.subplot(3,1,1)

gold = data[data.Medal == "Gold"].Team.value_counts().head(5)

gold.plot(kind='bar',rot=0,figsize=(20, 10))

plt.ylabel("Gold Medal")



plt.subplot(3,1,2)

silver = data[data.Medal == "Silver"].Team.value_counts().head(5)

silver.plot(kind='bar',rot=0,figsize=(20, 10))

plt.ylabel("Silver Medal")



plt.subplot(3,1,3)

bronze = data[data.Medal == "Bronze"].Team.value_counts().head(5)

bronze.plot(kind='bar',rot=0,figsize=(20, 10))

plt.ylabel("Bronze Medal")



plt.show()
data.City.unique()
#Now we're going to analyze London.

dr=data[(data['City']=='London')]

dr
counter_silvermedal=[]

counter_goldmedal=[]

counter_bronzemedal=[]

for sport in dr.Sport.unique():

    counter_bronzemedal.append(len(dr[(dr['Medal']=='Bronze')&(dr['Sport']==sport)]))

    counter_silvermedal.append(len(dr[(dr['Medal']=='Silver')&(dr['Sport']==sport)]))

    counter_goldmedal.append(len(dr[(dr['Medal']=='Gold')&(dr['Sport']==sport)]))

    
plt.subplot(3,1,1)

sns.barplot(x=dr.Sport.unique(),y=counter_bronzemedal,color='red')

plt.xticks(rotation=90)



plt.subplot(3,1,2)

sns.barplot(x=dr.Sport.unique(),y=counter_silvermedal,color='blue')

plt.xticks(rotation=90)



plt.subplot(3,1,3)

sns.barplot(x=dr.Sport.unique(),y=counter_goldmedal,color='gray')

plt.xticks(rotation=90)



fig, ax = plt.gcf(), plt.gca()

fig.set_size_inches(10, 10)

plt.tight_layout()



plt.show()
age_mean=[]

for year in dr.Year.unique():

    age_mean.append(sum(dr[dr['Year']==year].Age)/len(dr[dr['Year']==year].Age))
age_mean
dr[dr['Age']==12]

#oungest player
dr[dr['Age']==84]

#oungest player
dr[np.logical_and(dr['Age']<84,dr[dr['Age']==12])]

#oungest player