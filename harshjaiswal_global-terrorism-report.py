# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import seaborn as sns

terrorism_df=pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')
terrorism_df.rename(columns ={'iyear':'Year','imonth':'Month','country_txt':'Country','iday':'Day','region_txt':'Region','provstate':'State','city':'City','success':'Success','attacktype1_txt':'Attacktype','targtype1_txt':'Target','natlty1_txt':'Nationality','gname':'Group_Name','weaptype1_txt':'Weapon_type',},inplace=True);
pd.set_option('display.max_columns', None)
terrorism_df=terrorism_df[['Year','Month','Country','Day','Region','State','City','Success','Attacktype','Target','Nationality','Group_Name','Weapon_type','latitude','longitude','motive',]]
terrorism_df

terrorism_df.isnull().sum()


terrorism_df['State'].fillna('Unknown', inplace=True)
terrorism_df['City'].fillna('Unknown',inplace=True)
terrorism_df['Nationality'].fillna('Unknown',inplace=True)
terrorism_df

sns.set_style('whitegrid')
terrorism_df.sort_values('Year',ascending=True,inplace=True)
plt.figure(figsize=(15,6))
plt.xlabel('Years')
sns.distplot(terrorism_df.Year,bins=10,color="b").set_title('Terrorism over the Years');

success=terrorism_df.Success.value_counts()
successful_percentage=(success[1]/success.sum())*100
successful_percentage=round(successful_percentage,2)
unsuccessful_percentage=100-successful_percentage
unsuccessful_percentage=round(unsuccessful_percentage,2)
overallattempts=[successful_percentage,unsuccessful_percentage]
label=['Succesful Terrorist attack','Unsuccessful Terrorist attack']
plt.figure(figsize=(15,6))
sns.set_style('darkgrid')
plt.title('Percentage of Successful attacks vs Non successful attacks')
plt.pie(overallattempts, colors=['skyblue','Gray'],shadow=True,labels=label, autopct='%1.1f%%');


Group_name=terrorism_df.Group_Name.value_counts().head(20)
Group_name=Group_name[1:]
plt.figure(figsize=(15,6))
sns.barplot(Group_name,Group_name.index,palette = "BuPu_r").set_title('Most active Terrorist groups');

plt.figure(figsize=(20,6))
sns.countplot('Attacktype',data=terrorism_df,palette='twilight',order=terrorism_df['Attacktype'].value_counts().index).set_title("Types of attacks")
plt.xticks(rotation=90);
country=terrorism_df.Country.value_counts().head(20)
plt.figure(figsize=(15,6))
sns.barplot(country,country.index,palette='viridis').set_title("Top 20 countries  most affected countries");
city=terrorism_df.City.value_counts()
city=city[1:]
plt.figure(figsize=(13,7))
city=city.head(20)
sns.barplot(city.index,city,palette='cubehelix').set_title('Top 20  most affected cities');
plt.xticks(rotation=90); 
sns.set_style('darkgrid')
plt.figure(figsize=(15,6))
sns.countplot('Target',data=terrorism_df,order=terrorism_df['Target'].value_counts().index,palette='mako').set_title('Most common Targets');
plt.xticks(rotation=90);
Region=terrorism_df.Region.value_counts()
plt.figure(figsize=(15,6))
sns.barplot(Region,Region.index,palette='gist_gray').set_title('Most effected regions');

regional_trend=pd.crosstab(terrorism_df.Year,terrorism_df.Region)
regional_trend.plot(color=sns.color_palette('Set2'));
fig=plt.gcf()
fig.set_size_inches(18,6)

group_type_attacks=pd.crosstab(terrorism_df.Region,terrorism_df.Attacktype)
group_type_attacks
plt.figure(figsize=(15,6))
sns.heatmap(group_type_attacks,fmt="d",linewidths=.5, cmap='BrBG').set_title("Region Vs Attacks");
plt.figure(figsize=(15,6))
sns.countplot('Weapon_type',data=terrorism_df,palette='twilight',order=terrorism_df['Weapon_type'].value_counts().index).set_title("Weapons Used")
plt.xticks(rotation=90);
Grouped=terrorism_df.groupby(terrorism_df.Country)
India=Grouped.get_group('India')
India 
time=India.Year.value_counts()
time=time.sort_index()
plt.figure(figsize=(15,6))
sns.lineplot(time.index,time).set_title('Terrorism over the years in India');

Indiasuccess=India.Success.value_counts()
Succesful_perctange_attacks=Indiasuccess[1]/sum(Indiasuccess)*100
Succesful_perctange_attacks=round(Succesful_perctange_attacks,2)
Failures_percentage_attacks=100-Succesful_perctange_attacks
attacks_india=[Succesful_perctange_attacks,Failures_percentage_attacks]
label_india=['Succesful attack%','Unsuccesful attack%']
plt.figure(figsize=(15,6))
plt.pie(attacks_india, colors=['green','Gray'],shadow=True,labels=label, autopct='%1.1f%%');

plt.figure(figsize=(15,6))
plt.title('Terrorist Groups with their attack types')
Indian_terrorist_groups=sns.countplot(India.Group_Name,data=India,order=India['Group_Name'].value_counts()[1:11].index,palette='viridis',hue=India.Attacktype).legend(loc='right', bbox_to_anchor=(1.25, 0.5), ncol=1)
plt.xticks(rotation=90);

plt.figure(figsize=(15,6))
commonTagetandTypes=pd.crosstab(India.Target,India.Attacktype)
sns.heatmap(commonTagetandTypes,fmt="d",linewidths=.5, cmap='inferno').set_title('Common tagets vs Common attack types');
figcity=India.City.value_counts().head(10).index
figstate=India.State.value_counts().head(10).index
fig,axes=plt.subplots(1,2,figsize=(15,10))
plt.tight_layout(pad=5)
sns.countplot('City',data=India,palette='twilight',order=India.City.value_counts().head(10).index,ax=axes[0]).set_xticklabels(figcity,rotation=90);
sns.countplot('State',data=India,palette='winter',order=India.State.value_counts().head(10).index,ax=axes[1]).set_xticklabels(figstate,rotation=90);
middlegroup=terrorism_df.groupby('Region')
MiddleEastAfrica=middlegroup.get_group('Middle East & North Africa')
MiddleEastAfrica
MiddleEastAfrica.isnull().sum()

MiddleEastAfrica['motive'].fillna('Unknown',inplace=True);
middleeastcontribution=terrorism_df.Region.value_counts().head(1)/terrorism_df.Region.value_counts().sum()*100
middleeastcontribution=round(middleeastcontribution,2).tolist()
restoftheworld=100-middleeastcontribution[0]
middleeastcontribution.append(restoftheworld)
label_middle=['Middle east contribution%','Rest of the world%']
plt.figure(figsize=(15,6))
plt.title('Terrorism in middle east vs rest of the globe')
plt.pie(middleeastcontribution, colors=['green','Gray'],shadow=True,labels=label_middle, autopct='%1.1f%%');
Terrorism_over_years=MiddleEastAfrica.Year.value_counts()
Terrorism_over_years=Terrorism_over_years.sort_index()
plt.figure(figsize=(15,7))
sns.set_style('darkgrid')
sns.lineplot(Terrorism_over_years.index,Terrorism_over_years).set_title('Terrorism Trend in the region of Middle East and Africa');
plt.figure(figsize=(15,6))
sns.countplot(MiddleEastAfrica.Country,order=MiddleEastAfrica['Country'].value_counts()[:10].index,data=MiddleEastAfrica,palette='Set1',hue=MiddleEastAfrica.Attacktype).set_title('Countires with common attack types')
plt.xticks(rotation=90);
plt.figure(figsize=(15,6))
sns.countplot(MiddleEastAfrica.City,data=MiddleEastAfrica,order=MiddleEastAfrica['City'].value_counts()[:10].index,palette='Accent');
plt.xticks(rotation=90);

AttackXTarget=pd.crosstab(MiddleEastAfrica.Attacktype,MiddleEastAfrica.Target)
AttackXTarget
plt.figure(figsize=(15,6))
sns.heatmap(AttackXTarget,fmt='d',linewidths=0.5,cmap='hot_r',linecolor='black').set_title('Common attack types on common target');