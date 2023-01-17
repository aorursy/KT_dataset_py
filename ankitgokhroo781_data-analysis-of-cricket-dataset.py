# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns




# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/Teams.csv')
data
data.info
data.sort_values(ascending='Rank',axis=0,inplace=True,by='Rank')


data
data.sort_values(ascending='Year',axis=0,inplace=True,by='Year')
data


data.plot.bar(x='Teams',y='Rank')
data.plot.barh(x='Teams',y='Rank')
plt.xlabel('Rank')
plt.title('teams vs rank')
data.plot.barh(x='Teams',y='Points')
plt.xlabel('Points')
plt.title('Teams vs points')
data_2014=data[data['Year']==2014]
data_2014
data_2014.plot.barh(x='Teams',y='Rank')
plt.xlabel('Rank')
plt.title('2014')

data_2015=data[data['Year']==2015]
data_2015
data_2015.plot.barh(x='Teams',y='Rank')
plt.xlabel('Rank')
plt.title('2015')

data_2016=data[data['Year']==2016]
data_2016
data_2016.plot.barh(x='Teams',y='Rank')
plt.xlabel('Rank')
plt.title('2016')
data_2017=data[data['Year']==2017]
data_2017
data_2017.plot.barh(x='Teams',y='Rank')
plt.xlabel('Rank')
plt.title('2017')
data_2014=data[data['Year']==2014]
data_2014
data_2014.plot.bar(x='Teams',y='Points')
plt.ylabel('Points')
plt.title('2014')
data_riders=data[data['Teams']=='Riders']
data_riders
data_riders.plot.bar(x='Year',y='Rank')
plt.xlabel('Year')
plt.ylabel('Rank')
plt.title('riders')
data_royals=data[data['Teams']=='Royals']
data_royals
data_royals.plot.bar(x='Year',y='Rank')
plt.xlabel('Year')
plt.ylabel('Rank')
plt.title('Royals')
data_kings=data[data['Teams']=='Kings']
data_kings
data_kings.plot.bar(x='Year',y='Rank')
plt.xlabel('Year')
plt.ylabel('Rank')
plt.title('Kings')
data_devils=data[data['Teams']=='Devils']
data_devils
data_devils.plot.bar(x='Year',y='Rank')
plt.xlabel('Year')
plt.ylabel('Rank')

plt.title('devils')
data
sns.pairplot(data,hue='Teams')



sns.distplot(data['Points'])

sns.pairplot(data_2014,hue='Teams')

sns.pairplot(data_riders,hue='Teams')


sns.pairplot(data_kings,hue='Teams')

sns.pairplot(data_devils,hue='Teams')



sns.pairplot(data_royals,hue='Teams')


sns.pairplot(data_2014,hue='Teams')



sns.pairplot(data_2015,hue='Teams')




sns.pairplot(data_2016,hue='Teams')




sns.pairplot(data_2017,hue='Teams')
sns.heatmap(data.corr(),cmap='coolwarm',annot=True)
sns.heatmap(data_2014.corr(),cmap='coolwarm',annot=True)
sns.heatmap(data_riders.corr(),cmap='coolwarm',annot=True)
sns.lmplot(x='Teams',y='Rank',data=data_2014)
sns.lmplot(x='Teams',y='Rank',data=data_2015)
sns.lmplot(x='Teams',y='Rank',data=data_2016)
sns.lmplot(x='Teams',y='Rank',data=data_2017)
sns.lmplot(x='Points',y='Rank',data=data_riders)
sns.lmplot(x='Points',y='Rank',data=data_kings)
sns.lmplot(x='Points',y='Rank',data=data_royals)
sns.lmplot(x='Points',y='Rank',data=data_devils)
