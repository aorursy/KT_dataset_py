import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
data=pd.read_csv("../input/athlete_events.csv")
data.head()
data.info()
data.describe(include='all')
data.isnull().sum(axis=0)
plt.figure(figsize=(20,10))
sns.countplot(data['Sport'])
plt.xticks(rotation=90)
plt.title("Number of particiption in each Sport in olympics for 120 Year : 1896 to 2016")
plt.figure(figsize=(20,10))
sns.countplot(data['Sport'],hue='Games',data=data)
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1, 1))
data.loc[:,['Year','Age']].groupby(['Year']).mean().plot(figsize=(20,10))
plt.title("Age Distribution in Olympic Sports irrespective to Type of sport for 120Yr ")
data.loc[:,['Year','Weight']].groupby(['Year']).mean().plot(figsize=(20,10))
plt.title("Weight Distribution in Olympic Sports irrespective to Type of sport for 120Yr ")
data.loc[:,['Year','Height']].groupby(['Year']).mean().plot(figsize=(20,10))
plt.title("Height Distribution in Olympic Sports irrespective to Type of sport for 120Yr ")
data1=data.dropna()
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
a=data1.loc[:,['Year','Sport','Weight']].groupby(['Sport','Year'])['Weight'].mean().reset_index()
a=a.groupby(['Year','Sport'])['Weight'].mean().unstack().T
a['Total']=a.sum(axis=1)
a=a.sort_values(by='Total',ascending=0)
a.drop('Total',axis=1,inplace=True)
a.T.plot(color=colors,marker='o')
fig=plt.gcf()
fig.set_size_inches(30,15)
plt.legend(bbox_to_anchor=(1, 1))
plt.show()

from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
a=data1.loc[:,['Year','Sport','Height']].groupby(['Sport','Year'])['Height'].mean().reset_index()
a=a.groupby(['Year','Sport'])['Height'].mean().unstack().T
a['Total']=a.sum(axis=1)
a=a.sort_values(by='Total',ascending=0)
a.drop('Total',axis=1,inplace=True)
a.T.plot(color=colors,marker='o')
fig=plt.gcf()
fig.set_size_inches(30,15)
plt.legend(bbox_to_anchor=(1, 1))
plt.show()
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
a=data1.loc[:,['Year','Sport','Age']].groupby(['Sport','Year'])['Age'].mean().reset_index()
a=a.groupby(['Year','Sport'])['Age'].mean().unstack().T
a['Total']=a.sum(axis=1)
a=a.sort_values(by='Total',ascending=0)
a.drop('Total',axis=1,inplace=True)
a.T.plot(color=colors,marker='o')
fig=plt.gcf()
fig.set_size_inches(30,15)
plt.legend(bbox_to_anchor=(1, 1))
plt.show()
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
a=data1.loc[:,['Year','Games','Age']].groupby(['Games','Year'])['Age'].mean().reset_index()
a=a.groupby(['Year','Games'])['Age'].mean().unstack().T
a['Total']=a.sum(axis=1)
a=a.sort_values(by='Total',ascending=0)
a.drop('Total',axis=1,inplace=True)
a.T.plot(color=colors,marker='o')
fig=plt.gcf()
fig.set_size_inches(30,15)
plt.legend(bbox_to_anchor=(1, 1))
plt.show()
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
a=data1.loc[:,['Year','Games','Height']].groupby(['Games','Year'])['Height'].mean().reset_index()
a=a.groupby(['Year','Games'])['Height'].mean().unstack().T
a['Total']=a.sum(axis=1)
a=a.sort_values(by='Total',ascending=0)
a.drop('Total',axis=1,inplace=True)
a.T.plot(color=colors,marker='o')
fig=plt.gcf()
fig.set_size_inches(30,15)
plt.legend(bbox_to_anchor=(1, 1))
plt.show()
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
a=data1.loc[:,['Year','Games','Weight']].groupby(['Games','Year'])['Weight'].mean().reset_index()
a=a.groupby(['Year','Games'])['Weight'].mean().unstack().T
a['Total']=a.sum(axis=1)
a=a.sort_values(by='Total',ascending=0)
a.drop('Total',axis=1,inplace=True)
a.T.plot(color=colors,marker='o')
fig=plt.gcf()
fig.set_size_inches(30,15)
plt.legend(bbox_to_anchor=(1, 1))
plt.show()
data.loc[:,['Games','Height']].groupby(['Games']).mean().reset_index('Games')
data.loc[:,['Games','Height']].groupby(['Games']).mean().reset_index().plot(figsize=(20,10))

data.loc[:,['Games','Weight']].groupby(['Games']).mean().reset_index()
data.loc[:,['Games','Weight']].groupby(['Games']).mean().reset_index().plot(figsize=(20,10))

data.loc[:,['Games','Age']].groupby(['Games']).mean().reset_index()
data.loc[:,['Games','Age']].groupby(['Games']).mean().reset_index().plot(figsize=(20,10))

data.loc[:,['Sport','Age']].groupby(['Sport']).mean().reset_index('Sport')
data.loc[:,['Sport','Age']].groupby(['Sport']).mean().reset_index('Sport').plot(figsize=(20,10))


data.loc[:,['Sport','Weight']].groupby(['Sport']).mean().reset_index('Sport')
data.loc[:,['Sport','Weight']].groupby(['Sport']).mean().reset_index('Sport').plot(figsize=(20,10))


data.loc[:,['Sport','Height']].groupby(['Sport']).mean().reset_index('Sport')
data.loc[:,['Sport','Height']].groupby(['Sport']).mean().reset_index('Sport').plot(figsize=(20,10))


plt.figure(figsize=(30,15))
sns.countplot(data1['Medal'],hue="Sport",data=data1)
plt.legend(bbox_to_anchor=(1,1))
plt.title("Medels based on Sport")

plt.figure(figsize=(20,90))
sns.countplot(data1['Medal'],hue="Team",data=data1)
plt.legend(bbox_to_anchor=(1,1))
plt.title("Medels based on Team")

plt.figure(figsize=(50,90))
sns.countplot(data1['Team'],hue="Medal",data=data1)
plt.legend(bbox_to_anchor=(1,1))
plt.xticks(rotation=90)
plt.title("Medels based on Team")

plt.figure(figsize=(20,90))
sns.countplot(data1['Medal'],hue="Event",data=data1)
plt.legend(bbox_to_anchor=(1,1))
plt.title("Medels based on Team")

