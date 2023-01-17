# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data17= pd.read_csv('../input/2017.csv')

data16= pd.read_csv('../input/2016.csv')

data15= pd.read_csv('../input/2015.csv')
data17.head()
data17.info()
data17= data17.rename(columns={'Happiness.Rank':'Happiness_Rank','Happiness.Score':'Happiness_Score','Whisker.high':'Whisker_high','Whisker.low':'Whisker_low'

                              ,'Economy..GDP.per.Capita.':'EconomyGDPperCapita','Health..Life.Expectancy.':'HealthLifeExpectancy',

                              'Trust..Government.Corruption.':'TrustGovernmentCorruption','Dystopia.Residual':'DystopiaResidual'})
f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(data17.corr(),annot= True, linewidths=0.5,fmt= '.1f',ax=ax)

plt.show()
#Top 50 Countries 

data17n=data17[0:51]
#Worst 50 Countries

data17a=data17[104:155]

data17x=data17a[::-1]
plt.figure(figsize=(15,15))

sns.barplot(x=data17n.Country,y=data17n.Happiness_Score)

plt.xticks(rotation= 90)

plt.xlabel('Country')

plt.ylabel('Happinees Score')

plt.title('The Happinest 50 Country in The World')
plt.figure(figsize=(15,15))

sns.barplot(x=data17x.Country,y=data17x.Happiness_Score)

plt.xticks(rotation= 90)

plt.xlabel('Country')

plt.ylabel('Happinees Score')

plt.title('The Most Unhappy 50 Countries in The World')
f,ax = plt.subplots(figsize=(12,20))



normalizehappines=data17n.Happiness_Score/max(data17n.Happiness_Score)

normalizeeconamic=data17n.EconomyGDPperCapita/max(data17n.EconomyGDPperCapita)

normalizehealth=data17n.HealthLifeExpectancy/max(data17n.HealthLifeExpectancy)

sns.barplot(x=normalizehappines,y=data17n.Country,color='lime',alpha = 0.3,label='Happiness_Score')

sns.barplot(x=normalizeeconamic,y=data17n.Country,color='orange',alpha = 0.6,label='EconomyGDPperCapita')

sns.barplot(x=normalizehealth,y=data17n.Country,color='grey',alpha = 0.4,label='HealthLifeExpectancy')



ax.legend(loc='lower right',frameon = True)     

ax.set(xlabel='', ylabel='Country',title = "Percentage of Country According to Various parameter ")





f,ax = plt.subplots(figsize=(12,20))



normalizehappines2=data17x.Happiness_Score/max(data17x.Happiness_Score)

normalizeeconamic2=data17x.EconomyGDPperCapita/max(data17x.EconomyGDPperCapita)

normalizehealth2=data17x.HealthLifeExpectancy/max(data17x.HealthLifeExpectancy)

sns.barplot(x=normalizehappines2,y=data17x.Country,color='lime',alpha = 0.3,label='Happiness_Score')

sns.barplot(x=normalizeeconamic2,y=data17x.Country,color='orange',alpha = 0.6,label='EconomyGDPperCapita')

sns.barplot(x=normalizehealth2,y=data17x.Country,color='grey',alpha = 0.4,label='HealthLifeExpectancy')



ax.legend(loc='lower right',frameon = True)     

ax.set(xlabel='', ylabel='Country',title = "Percentage of Country According to Various parameter ")
f,ax = plt.subplots(figsize=(12,20))



normalizeeconamic2=data17x.EconomyGDPperCapita/max(data17x.EconomyGDPperCapita)

sns.barplot(x=data17x.TrustGovernmentCorruption,y=data17x.Country,color='lime',alpha = 0.6,label='TrustGovernmentCorruption')

sns.barplot(x=normalizeeconamic2,y=data17x.Country,color='orange',alpha = 0.4,label='EconomyGDPperCapita')

sns.barplot(x=data17x.HealthLifeExpectancy,y=data17x.Country,color='red',alpha = 0.4,label='HealthLifeExpectancy')



ax.legend(loc='lower right',frameon = True)     

ax.set(xlabel='', ylabel='Country',title = "Percentage of Country According to Various parameter ")




#visualize

f, ax1 = plt.subplots(figsize=(20,10))

sns.pointplot(x='Country',y='HealthLifeExpectancy',data=data17n,color='black',alpha=0.8,)

sns.pointplot(x='Country',y='Freedom',data=data17n,color='red',alpha=0.8)



plt.text(40,0.9,'Health Life Expectancy',color='black',fontsize = 17,style = 'italic')

plt.text(40,0.8,'Freedom',color='red',fontsize = 17,style = 'italic')

plt.xticks(rotation= 90)



plt.xlabel('Countries',fontsize = 15,color='black',)

plt.ylabel('Values',fontsize = 15,color='black')



plt.title('HealthLifeExpectancy Vs Freedom (Top 50 Countries)',fontsize = 20,color='black')

plt.grid()







#visualize

f, ax1 = plt.subplots(figsize=(20,10))

sns.pointplot(x='Country',y='HealthLifeExpectancy',data=data17n,color='purple',alpha=0.8)

sns.pointplot(x='Country',y='Family',data=data17n,color='grey',alpha=0.8)



plt.text(40,0.9,'Health Life Expectancy',color='purple',fontsize = 17,style = 'italic')

plt.text(40,0.8,'Family',color='grey',fontsize = 17,style = 'italic')

plt.xticks(rotation= 90)



plt.xlabel('Countries',fontsize = 15,color='Black')

plt.ylabel('Values',fontsize = 15,color='Black')



plt.title('HealthLifeExpectancy Vs Family (Top 50 Countries)',fontsize = 20,color='black')

plt.grid()



#visualize

f, ax1 = plt.subplots(figsize=(20,10))

sns.pointplot(x='Country',y='HealthLifeExpectancy',data=data17x,color='purple',alpha=0.8)

sns.pointplot(x='Country',y='Family',data=data17x,color='grey',alpha=0.8)



plt.text(40,1.4,'Health Life Expectancy',color='purple',fontsize = 17,style = 'italic')

plt.text(40,1.3,'Family',color='grey',fontsize = 17,style = 'italic')

plt.xticks(rotation= 90)



plt.xlabel('Countries',fontsize = 15,color='Black')

plt.ylabel('Values',fontsize = 15,color='Black')



plt.title('HealthLifeExpectancy Vs Family(Worst 50 Countries)',fontsize = 20,color='black')

plt.grid()





#visualize

f, ax1 = plt.subplots(figsize=(20,10))

sns.pointplot(x='Country',y='HealthLifeExpectancy',data=data17x,color='black',alpha=0.8,)

sns.pointplot(x='Country',y='Freedom',data=data17x,color='red',alpha=0.8)



plt.text(5,0.7,'Health Life Expectancy',color='black',fontsize = 17,style = 'italic')

plt.text(5,0.6,'Freedom',color='red',fontsize = 17,style = 'italic')

plt.xticks(rotation= 90)



plt.xlabel('Countries',fontsize = 15,color='black',)

plt.ylabel('Values',fontsize = 15,color='black')



plt.title('HealthLifeExpectancy Vs Freedom (Top 50 Countries)',fontsize = 20,color='black')

plt.grid()



g = sns.jointplot(data17n.HealthLifeExpectancy, data17n.Family, kind="hex", height=7,color="pink")

plt.savefig('graph.png')

plt.show()
g = sns.jointplot(data17x.HealthLifeExpectancy, data17x.Family, kind="reg", height=7,color="red")

plt.savefig('graph.png')

plt.show()