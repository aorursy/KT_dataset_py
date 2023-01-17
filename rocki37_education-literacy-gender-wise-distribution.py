#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
#Loading data
indiancities=pd.read_csv('../input/cities_r2.csv')
indiancities.head()
#plot Statewise-Population
colors=['red','blue','green','yellow','orange']
group_by_population=indiancities[['state_name','population_total']].groupby('state_name').sum()
group_by_population.sort_values('population_total',ascending=False,inplace=True)
group_by_population.plot(kind='bar',color=colors,legend=False,figsize=(8,10))
plt.xlabel('State')
plt.ylabel('Population')
plt.show()
#PLotting male and female population wise 
color=['red','blue','green','yellow','orange']
group_by_male_population=indiancities[['state_name','population_male']].groupby('state_name').sum()
group_by_male_population.sort_values('population_male',ascending=False,inplace=True)
group_by_female_population=indiancities[['state_name','population_female']].groupby('state_name').sum()
group_by_female_population.sort_values('population_female',ascending=False,inplace=True)

fig,axes=plt.subplots(nrows=2,ncols=1)
plt.subplots_adjust(wspace=0.7, hspace=1.3);
group_by_male_population.plot(ax=axes[0],kind='bar',color=colors,legend=False,figsize=(8,8))
plt.title('Male Population wise States')
plt.xlabel('State')
plt.ylabel('Population')

group_by_female_population.plot(ax=axes[1],kind='bar',color=colors,legend=False,figsize=(8,8))
plt.title('Female Population wise States')
plt.xlabel('State')
plt.ylabel('Population')
plt.show()

#Literates distribution
color=['red','blue','green','yellow','orange']
fig,axes=plt.subplots(nrows=2,ncols=1,figsize=(7,7))
#plt.subplots_adjust(wspace=2, hspace=2);
plt.title('Top Five States with maximum literate population')
group_by_literate_population=indiancities[['state_name','literates_total']].groupby('state_name').sum()
group_by_literate_population.sort_values('literates_total',ascending=False,inplace=True)
group_by_literate_population.head().plot.pie(ax=axes[0],subplots=True,legend=False,autopct='%.2f')
#plt.title('Top Five States with maximum literate population')





def findPercent(data,numerator,denominator):
    result=[]
    for num,den in zip(data[numerator],data[denominator]):
        #print  (num/float(den))*100
        result.append((num/float(den))*100)
    return result

statewise_literatecomparison=indiancities[['state_name','population_total','literates_total']].groupby('state_name').sum()
statewise_literatecomparison.sort_values('population_total',ascending=False,inplace=True)
statewise_literatecomparison['Percent Literate']=findPercent(statewise_literatecomparison,'literates_total','population_total')
statewise_literatecomparison.sort_values('Percent Literate',ascending=False,inplace=True)
statewise_literatecomparison['Percent Literate'].head().plot.pie(subplots=True,legend=False,autopct='%.2f')
plt.title('|Top Five States with maximum percent literate population|')
