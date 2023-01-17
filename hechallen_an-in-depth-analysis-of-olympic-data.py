import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
data=pd.read_csv('../input/athlete_events.csv')
noc=pd.read_csv('../input/noc_regions.csv')
data=pd.merge(data,noc,on='NOC',how='left')
print(data.head())
print(data.info())
import missingno as msno
msno.matrix(data)
data_summer=data[data.Season=='Summer']
data_summer['age_dec']=data_summer.Age.map(lambda Age: 10*(Age//10))
data_summer['ones']=np.ones(len(data_summer))
data_summer['height_dec']=data_summer.Height.map(lambda Height: 10*(Height//10))
data_summer['weight_dec']=data_summer.Weight.map(lambda Weight: 10*(Weight//10))
data_city_year=data_summer.pivot_table('ones',index='Year',columns='City')
city_sort=data_city_year.sum().sort_values(ascending=False)
print(city_sort)
data_name_year=data_summer.groupby(['Year','Name'])['Medal'].value_counts()
name_gold10=data_name_year.loc[:,:,'Gold'].sort_values(ascending=False).head(10)
print(name_gold10)
data_region_year=data_summer.pivot_table('ones',index='Year',columns='region')
region_times10=data_region_year.sum().sort_values(ascending=False).head(10)
print(region_times10)
data_sport_year=data_summer.groupby(['Year','Sport'])['Medal'].value_counts()
sport_gold10=data_sport_year.loc[:,:,'Gold'].sort_values(ascending=False).head(10)
print(sport_gold10)
data_sport=data_summer.groupby('Sport')['Medal'].value_counts()
sport_gold10_=data_sport.loc[:,'Gold'].sort_values(ascending=False).head(10)
print(sport_gold10_)
data_sport_region=data_summer.groupby(['region','Sport'])['Medal'].value_counts()
region_gold10=data_sport_region.loc[:,:,'Gold'].sort_values(ascending=False).head(20)
print(region_gold10)
data_region=data_summer.groupby('region')['Medal'].value_counts()
region_gold10_=data_region.loc[:,'Gold'].sort_values(ascending=False).head(10)
print(region_gold10_)
data_year_sex=data_summer.groupby('Year')['Sex'].value_counts()
fig,ax=plt.subplots(2,1,figsize=(15,8))
fig.subplots_adjust(hspace=0.5)
data_year_sex.loc[:,'M'].plot(kind='bar',ax=ax[0])
ax[0].set_title('men participating in the Olympics',size=15);
data_year_sex.loc[:,'F'].plot(kind='bar',ax=ax[1])
ax[1].set_title('female participating in the Olympics',size=15);
data_year_age=data_summer.pivot_table(index='Year',columns='age_dec',aggfunc={'ones':sum})
fig,ax=plt.subplots(figsize=(14,8))
fig.subplots_adjust(hspace=0.5)
data_year_age.plot(ax=ax,style='-s')
ax.legend(['age_dec:10','age_dec:20','age_dec:30','age_dec:40','age_dec:50','age_dec:60','age_dec:70','age_dec:80','age_dec:90'])
ax.set_title('aged 10-90 participating in the Olympics',size=15);
fig,ax=plt.subplots(figsize=(14,8))
data_year_medal=data_summer.pivot_table(index='Year',columns='Medal',aggfunc={'ones':sum})
data_year_medal.plot(ax=ax,style='-s')
ax.legend(['Gold','Silver','Bronze'])
ax.set_title('medals in the Olympics',size=15);
plt.figure(figsize=(14,8))
age_list=np.arange(10,60,10)
sns.countplot(x='age_dec',hue='Medal',
	data=data_summer[data_summer.age_dec.isin(age_list)])
plt.figure(figsize=(14,8))
height_list=np.arange(140,220,10)
sns.countplot(x='height_dec',hue='Medal',
	data=data_summer[data_summer.height_dec.isin(height_list)])
plt.figure(figsize=(14,8))
weight_list=np.arange(30,150,10)
sns.countplot(x='weight_dec',hue='Medal',
	data=data_summer[data_summer.weight_dec.isin(weight_list)])
plt.figure(figsize=(14,8))
sport_names=['Athletics','Swimming','Rowing','Gymnastics','Fencing',
'Hockey','Football','Sailing','Cycling','Wrestling']
sns.countplot(x='Sport',hue='Medal',
	data=data_summer[data_summer.Sport.isin(sport_names)])
plt.figure(figsize=(14,8))
region_names=['USA','Russia','Germany','UK','Italy','France','Hungary',
'Australia','Sweden','China']
sns.countplot(x='region',hue='Medal',
	data=data_summer[data_summer.region.isin(region_names)])
plt.figure(figsize=(14,8))
age_list=np.arange(10,70,10)
sns.countplot(x='age_dec',hue='Sex',
	data=data_summer[data_summer.age_dec.isin(age_list)])
plt.figure(figsize=(14,8))
height_list=np.arange(140,220,10)
sns.countplot(x='height_dec',hue='Sex',
	data=data_summer[data_summer.height_dec.isin(height_list)])
plt.figure(figsize=(14,8))
weight_list=np.arange(30,140,10)
sns.countplot(x='weight_dec',hue='Sex',
	data=data_summer[data_summer.weight_dec.isin(weight_list)])
plt.figure(figsize=(14,8))
sns.countplot(x='Sport',hue='Sex',
	data=data_summer[data_summer.Sport.isin(sport_names)])
plt.figure(figsize=(14,8))
sns.countplot(x='region',hue='Sex',
	data=data_summer[data_summer.region.isin(region_names)])
plt.figure(figsize=(14,8))
sns.countplot(x='Medal',hue='Sex',data=data_summer)
data_drop=data_summer[['Sex','Age','Height','Weight','Medal']].dropna()
sns.pairplot(data_drop,hue='Medal')
sns.pairplot(data_drop,hue='Sex')
plt.figure(figsize=(14,8))
age_list=np.arange(10,80,10)
sns.violinplot(x='age_dec',y='Height',hue='Sex',data=data_summer[data_summer.age_dec.isin(age_list)],split=True,inner='quartitle')
plt.figure(figsize=(14,8))
age_list=np.arange(10,80,10)
sns.violinplot(x='age_dec',y='Weight',hue='Sex',data=data_summer[data_summer.age_dec.isin(age_list)],split=True,inner='quartitle')
plt.figure(figsize=(14,8))
age_list=np.arange(10,70,10)
sns.boxenplot(x='age_dec',y='Height',hue='Medal',data=data_summer[data_summer.age_dec.isin(age_list)])
plt.figure(figsize=(14,8))
age_list=np.arange(10,70,10)
sns.boxenplot(x='age_dec',y='Weight',hue='Medal',data=data_summer[data_summer.age_dec.isin(age_list)])
data_china=data_summer[data_summer.region=='China']
fig,ax=plt.subplots(figsize=(14,10))
data_china.groupby('Year')['ones'].sum().plot(kind='barh',ax=ax)
ax.set_title('Chinese athletes in the Olympics',size=15);
fig,ax=plt.subplots(figsize=(14,10))
data_china_medal=data_china.pivot_table(index='Year',columns='Medal',aggfunc={'ones':sum})
data_china_medal.plot(kind='barh',ax=ax)
ax.legend(['Gold','Silver','Bronze'])
ax.set_title('Chinese medals in the Olympics',size=15);
fig,ax=plt.subplots(figsize=(14,8))
data_china_medal=data_china.pivot_table(index='Year',columns='Sex',aggfunc={'ones':sum})
data_china_medal.plot(ax=ax,style='-s')
ax.legend(['Female','Men'])
ax.set_title('Chinese men and female in the Olympics',size=15);
plt.figure(figsize=(14,8))
sns.distplot(data_china.Age.dropna())
sns.kdeplot(data_china.Age.dropna())
plt.figure(figsize=(14,8))
sns.distplot(data_china.Height.dropna())
sns.kdeplot(data_china.Height.dropna())
plt.figure(figsize=(14,8))
sns.distplot(data_china.Weight.dropna())
sns.kdeplot(data_china.Weight.dropna())
fig,ax=plt.subplots(figsize=(14,8))
sns.barplot('Year','Age',data=data_china,palette='cool',capsize=.5,ax=ax)
fig,ax=plt.subplots(figsize=(14,8))
sns.barplot('Year','Height',data=data_china,palette='rainbow',capsize=.5,ax=ax)
fig,ax=plt.subplots(figsize=(14,8))
sns.barplot('Year','Weight',data=data_china,palette='Spectral',capsize=.5,ax=ax)
data_sport_china=data_china.groupby('Sport')['Medal'].value_counts()
sport_china_gold10=data_sport_china.loc[:,'Gold'].sort_values(ascending=False).head(10)
print(sport_china_gold10)
plt.figure(figsize=(14,8))
sport_names=['Diving','Table Tennis','Gymnastics','Volleyball','Weightlifting','Badminton','Shooting','Swimming','Athletics','Judo']
sns.countplot(x='Sport',hue='Medal',data=data_china[data_china.Sport.isin(sport_names)])
plt.figure(figsize=(14,8))
sport_names=['Diving','Table Tennis','Gymnastics','Volleyball','Weightlifting','Badminton','Shooting','Swimming','Athletics','Judo']
sns.countplot(x='Sport',hue='Sex',data=data_china[data_china.Sport.isin(sport_names)])
