# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.head()
data.info()
data.shape
total = data.isnull().sum().sort_values(ascending=False)

percent = ((data.isnull().sum())*100)/data.isnull().count().sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)

missing_data.head(40)
data.describe()
data.corr().style.background_gradient(cmap='coolwarm')
import matplotlib.pyplot as plt

import seaborn as sns
f,ax=plt.subplots(1,2,figsize=(18,8))

data['neighbourhood_group'].value_counts().plot.pie(explode=[0,0.1,0,0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Share of Neighbourhood')

ax[0].set_ylabel('Neighbourhood Share')

sns.countplot('neighbourhood_group',data=data,ax=ax[1])

ax[1].set_title('Share of Neighbourhood')

plt.show()
sns.scatterplot(data.longitude,data.latitude,hue=data.neighbourhood_group)

plt.ioff()
import folium 

from folium.plugins import HeatMap

m=folium.Map([40.7128,-74.0060],zoom_start=11)

HeatMap(data[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)
sub_6=data[data.price<300]

viz_4=sub_6.plot(kind='scatter',x='longitude',y='latitude',label='avaibility_365',c='price',cmap=plt.get_cmap('jet'),colorbar=True,alpha=0.4,figsize=(10,10))

viz_4.legend()

plt.ioff()