# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from collections import Counter 
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
vgdata =pd.read_csv('../input/videogamesales/vgsales.csv')
vgdata.info()
vgdata.head()
vgdata.dropna(how="any", inplace=True)
vgdata.info()
vgdata.Year = vgdata.Year.astype(int)
vgdata.info()
vgdata.Platform.head(20)
platform_count = Counter(vgdata.Platform)        
most_platform=platform_count.most_common(20)
platform_name,count = zip(*most_platform) #Unzip (tuple) container values(key and values) into separate tuples (platform_name and count)
platform_name,count = list(platform_name),list(count) #Convert tuples into lists
# Visualise the data

plt.figure(figsize=(15,10))
ax=sns.barplot( x = platform_name, y = count, palette = 'rocket')
plt.xlabel('Platform')
plt.ylabel('Frequency')
plt.title('Most common 20 of Platform')
plt.show()
# Filter the data between 2010 and 2016

first_filter=vgdata.Year>2009
second_filter=vgdata.Year<2017
new_vgdata1=vgdata[first_filter&second_filter]
#Visualization of filtered data[2010-2016]

sns.catplot(x="Year",y="NA_Sales",kind="point",
            data=new_vgdata1,
            hue = "Platform",
            palette='Set1',
            ci = None,
            edgecolor=None,
            height=8.27, 
            aspect=11.7/8.27)
plt.show()
# Create dataframe that counts the Genre for every year

data1=vgdata[['Year','Genre']]
data1=data1.set_index('Year')
data2010=[]


data2010.append([sum(data1.loc[2010].Genre=='Shooter'),sum(data1.loc[2010].Genre=='Sports'), sum(data1.loc[2010].Genre=='Action'),sum(data1.loc[2010].Genre=='Role-Playing')])
data2010.append([sum(data1.loc[2011].Genre=='Shooter'),sum(data1.loc[2011].Genre=='Sports'), sum(data1.loc[2011].Genre=='Action'),sum(data1.loc[2011].Genre=='Role-Playing')])
data2010.append([sum(data1.loc[2012].Genre=='Shooter'),sum(data1.loc[2012].Genre=='Sports'), sum(data1.loc[2012].Genre=='Action'),sum(data1.loc[2012].Genre=='Role-Playing')])
data2010.append([sum(data1.loc[2013].Genre=='Shooter'),sum(data1.loc[2013].Genre=='Sports'), sum(data1.loc[2013].Genre=='Action'),sum(data1.loc[2013].Genre=='Role-Playing')])
data2010.append([sum(data1.loc[2014].Genre=='Shooter'),sum(data1.loc[2014].Genre=='Sports'), sum(data1.loc[2014].Genre=='Action'),sum(data1.loc[2014].Genre=='Role-Playing')])
data2010.append([sum(data1.loc[2015].Genre=='Shooter'),sum(data1.loc[2015].Genre=='Sports'), sum(data1.loc[2015].Genre=='Action'),sum(data1.loc[2015].Genre=='Role-Playing')])
data2010.append([sum(data1.loc[2016].Genre=='Shooter'),sum(data1.loc[2016].Genre=='Sports'), sum(data1.loc[2016].Genre=='Action'),sum(data1.loc[2016].Genre=='Role-Playing')])


df=pd.DataFrame(data2010,columns = ['Shooter' , 'Sports', 'Action','Role-Playing'])
df['Year']=[2010,2011,2012,2013,2014,2015,2016]
df
# Plot the number of every genre in year 2010 until 2016

f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='Year',y='Action',data=df,color='lime',alpha=0.7)
sns.pointplot(x='Year',y='Shooter',data=df,color='red',alpha=0.7)
sns.pointplot(x='Year',y='Sports',data=df,color='blue',alpha=0.7)
sns.pointplot(x='Year',y='Role-Playing',data=df,color='orange',alpha=0.7)


plt.xlabel('Years',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.text(5.7,240,'Action',color='lime',fontsize = 15,style = 'italic')
plt.text(5.7,230,'Shooter',color='red',fontsize = 15,style = 'italic')
plt.text(5.7,220,'Sports',color='blue',fontsize = 15,style = 'italic')
plt.text(5.7,210,'Role-Playing',color='orange',fontsize = 15,style = 'italic')
plt.grid()