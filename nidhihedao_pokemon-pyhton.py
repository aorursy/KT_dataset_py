#load libraries
import pandas as pd #data preprocessing
import matplotlib.pyplot as plt
import seaborn as sns #visualization tool
import numpy as np #linear algebra
data= pd.read_csv('../input/pokemon.csv') #read data
data.head()
data.info()
#Line plot
data.Speed.plot(kind='line',color='g',label='Speed',linewidth=1,alpha=0.5,grid=True,linestyle=':')
data.Defense.plot(kind='line',color='r',label='Defense',linewidth=1,alpha=0.5,grid=True,linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('xaxis')
plt.ylabel('yaxis')
plt.title('Line plot')
data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
ax=data.Speed.plot.line(figsize=(12,8),fontsize=16,color="red",linewidth=1)
ax.set_title("Speed",fontsize="16")
#scatter plot
data.plot(kind='scatter',x='Attack',y='Defense',alpha=0.5,color='g')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.title('Scatter plot')
#Histogram
data.Speed.plot(kind='hist',bins=50,figsize=(15,15))
#Box plot
data.boxplot(column='Attack',by='Legendary')
#subplots of histogram
fig,axes=plt.subplots(nrows=1,ncols=2)
data.plot(kind='hist',y='Attack',bins=50,range=(0,250),normed=True,ax=axes[0])
data.plot(kind='hist',y='Attack',bins=50,range=(0,250),normed=True,ax=axes[1],cumulative=True)
#subplots of bar plot
fig,axarr = plt.subplots(2,1,figsize=(10,8))
data.HP.head(20).value_counts().plot.bar(ax=axarr[0],fontsize=10)
axarr[0].set_title("HP",fontsize=10)

data.Defense.head(20).value_counts().plot.bar(ax=axarr[1],fontsize=10)
axarr[1].set_title("Defense",fontsize=10)
#subplots of histogram
fig,axarr = plt.subplots(2,1,figsize=(10,8))
data.HP.value_counts().plot.hist(ax=axarr[0],fontsize=10,bins=25)
axarr[0].set_title("HP",fontsize=10)

data.Defense.value_counts().plot.hist(ax=axarr[1],fontsize=10,bins=25)
axarr[1].set_title("Defense",fontsize=10)
#count plot
sns.countplot(data['HP'])
#dist plot
sns.distplot(data['Attack'],bins=25,kde=False)
#kde
sns.kdeplot(data.query('Attack<200').Attack)
#joint plot
sns.jointplot(x="Attack",y="Defense",data=data)
g=sns.FacetGrid(data,col="Legendary")
g.map(sns.kdeplot,"Attack")
g=sns.FacetGrid(data,row="Legendary",col="Generation")
g.map(sns.kdeplot,"Defense")
#pair plot
sns.pairplot(data[['HP','Attack','Defense']])
sns.lmplot(x="Attack",y="Defense",hue="Legendary",markers=["o","*"],data=data,fit_reg=False)
