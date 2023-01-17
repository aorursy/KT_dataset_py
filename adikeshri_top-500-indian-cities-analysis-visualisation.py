# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/cities_r2.csv')

data.sample(10)
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

fig=plt.gcf()

fig.set_size_inches(18.5, 10.5)

data=data.sort_values('state_name')

sns.countplot(data['state_name'],palette='Set2',order=data['state_name'].value_counts().index)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('State',fontsize=20)

plt.ylabel('No. of cities with population more than 100K',fontsize=20)

plt.title('No. of cities in each state',fontsize=20)

plt.tight_layout()

plt.show()
data=data.sort_values('population_total',ascending=False)

top20=data.head(20)
fig=plt.gcf()

fig.set_size_inches(18.5, 10.5)

plt.bar(top20['name_of_city'],top20['population_total'],color=['maroon','black','green','purple'],alpha=0.5)

plt.xticks(rotation=90,fontsize=15)

plt.yticks([12000000,10000000,8000000,6000000,4000000,2000000],['12M','10M','8M','6M','4M','2M'],fontsize=15)

plt.xlabel('City',fontsize=20)

plt.ylabel('Population',fontsize=20)

plt.title('Top 20 cities by population',fontsize=20)

plt.tight_layout()

plt.show()
import warnings

warnings.filterwarnings("ignore")

fig=plt.gcf()

fig.set_size_inches(18.5, 10.5)

sns.distplot(data['effective_literacy_rate_total'],color='black',hist=False,label='Total literacy rate distribution')

sns.distplot(data['effective_literacy_rate_male'],color='blue',hist=False,label='Male literacy rate distribution')

sns.distplot(data['effective_literacy_rate_female'],color='maroon',hist=False,label='Female literacy rate distribution')

plt.xticks(rotation=90,fontsize=15)

plt.xlabel('Literacy rate',fontsize=15)

plt.title('Literacy rate distribution of top 500 Indians cities',fontsize=15)

plt.legend(loc=4,fontsize=15)

plt.show()
data['location']=data['location'].str.split()

data.sample(10)
name=np.array(top20['name_of_city'])

pop=np.array(top20['population_total'])

import folium

India=folium.Map(location=[20.5937,78.9629],width=900, height=500,zoom_start=4,tiles='cartodbpositron')

count=0

for i in top20['location']:

    for x in i:

        a=x.split(',')

        z=[float(a[0]),float(a[1])]

        folium.Marker(location=z,icon=folium.Icon(color='red'),popup=name[count]+'\nPopulation: '+str(round(pop[count]/1000000,3))+'M\nPopulation rank: '+str(count+1)).add_to(India)

        count+=1

display(India)
fig=plt.gcf()

fig.set_size_inches(18.5, 10.5)

plt.stem(top20['name_of_city'],top20['population_total'],label='Total Population')

plt.bar(top20['name_of_city'],top20['population_male'],label='Male Population',color='blue',alpha=0.4)

plt.bar(top20['name_of_city'],top20['population_female'],label='Female Population',color='red',alpha=0.4)

plt.xticks(rotation=90,fontsize=15)

plt.yticks([12000000,10000000,8000000,6000000,4000000,2000000],['12M','10M','8M','6M','4M','2M'],fontsize=15)

plt.xlabel('Literacy rate',fontsize=15)

plt.title('Male vs Females in top 20 cities by population',fontsize=20)

plt.legend(loc=5,fontsize=15)

plt.show()
fig=plt.gcf()

fig.set_size_inches(18.5, 10.5)

plt.scatter(data['effective_literacy_rate_total'],data['sex_ratio'],color='grey',label='Total',alpha=0.5)

plt.scatter(data['effective_literacy_rate_male'],data['sex_ratio'],label='Male')

plt.scatter(data['effective_literacy_rate_female'],data['sex_ratio'],color='maroon',label='Female')

plt.xlabel('Literacy rate',fontsize=15)

plt.ylabel('Sex Ration(Females per 1000 males)',fontsize=15)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.legend(loc=0,fontsize=15)

plt.title('Sex ratio vs Literacy rate',fontsize=20)

plt.tight_layout()

plt.show()
fig=plt.gcf()

fig.set_size_inches(18.5, 10.5)

plt.scatter(data['effective_literacy_rate_total'],data['sex_ratio'],color='grey',label='Total',alpha=0.5)

m,c=np.polyfit(data['effective_literacy_rate_total'],data['sex_ratio'],deg=1)

plt.plot(data['effective_literacy_rate_total'],m*data['effective_literacy_rate_total']+c,color='grey',label='Total plot')

plt.scatter(data['effective_literacy_rate_male'],data['sex_ratio'],label='Male')

m,c=np.polyfit(data['effective_literacy_rate_male'],data['sex_ratio'],deg=1)

plt.plot(data['effective_literacy_rate_male'],m*data['effective_literacy_rate_male']+c,label='Male plot')

plt.scatter(data['effective_literacy_rate_female'],data['sex_ratio'],color='maroon',label='Female')

m,c=np.polyfit(data['effective_literacy_rate_female'],data['sex_ratio'],deg=1)

plt.plot(data['effective_literacy_rate_female'],m*data['effective_literacy_rate_female']+c,color='maroon',label='Female plot')

plt.xlabel('Literacy rate',fontsize=15)

plt.ylabel('Sex Ration(Females per 1000 males)',fontsize=15)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.legend(loc=0,fontsize=15)

plt.title('Sex ratio vs Literacy rate',fontsize=20)

plt.tight_layout()

plt.show()
data=data.sort_values('effective_literacy_rate_total',ascending=False)

top20_literacy=data.head(20)

data=data.sort_values('sex_ratio',ascending=False)

top20_sexratio=data.head(20)
name=np.array(top20_literacy['name_of_city'])

lit=np.array(top20_literacy['effective_literacy_rate_total'])

count=0

map=folium.Map(location=[20.5937,78.9629],width=900, height=500,zoom_start=4,tiles='cartodbpositron')

for i in top20_literacy['location']:

    for x in i:

        a=x.split(',')

        z=[float(a[0]),float(a[1])]

        folium.Marker(location=z,icon=folium.Icon(color='red'),popup=name[count]+'\Literacy: '+str(lit[count])+'%\nLiteracy rank: '+str(count+1)).add_to(map)

        count+=1

display(map)
name=np.array(top20_sexratio['name_of_city'])

s=np.array(top20_sexratio['sex_ratio'])

count=0

map1=folium.Map(location=[20.5937,78.9629],width=900, height=500,zoom_start=4,tiles='cartodbpositron')

for i in top20_sexratio['location']:

    for x in i:

        a=x.split(',')

        z=[float(a[0]),float(a[1])]

        folium.Marker(location=z,icon=folium.Icon(color='red'),popup=name[count]+'\nSex Ratio(F/M): '+str(s[count]/1000)+'\nSex ratio rank: '+str(count+1)).add_to(map1)

        count+=1

display(map1)
data=data.sort_values('total_graduates',ascending=False)

top20_graduates=data.head(20)
fig=plt.gcf()

fig.set_size_inches(18.5, 10.5)

plt.stem(top20_graduates['name_of_city'],top20_graduates['total_graduates'],label='Total graduates')

plt.bar(top20_graduates['name_of_city'],top20_graduates['male_graduates'],label='Male graduates',color='blue',alpha=0.4)

plt.bar(top20_graduates['name_of_city'],top20_graduates['female_graduates'],label='Female graduates',color='red',alpha=0.4)

plt.xticks(rotation=90,fontsize=15)

plt.yticks([2000000,1500000,1000000,500000],['2M','1.5M','1M','0.5M'],fontsize=15)

plt.xlabel('City',fontsize=15)

plt.ylabel('Graduates',fontsize=15)

plt.title('Graduates distribution in top 20 cities(by most number of graduates)',fontsize=20)

plt.legend(loc=5,fontsize=15)

x=range(20)

y=np.array(top20_graduates['total_graduates'])

plt.plot(x,y)

plt.show()