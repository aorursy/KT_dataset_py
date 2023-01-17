# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
a=pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')
# This is my first analysis ; So we will start firt by analysing How much different regions we have and their corresponding country
a.head()
a['Region'].value_counts()  # Hence we have these regions in our dataset :)
# Here's the visualisation of these Regions

plt.figure(figsize=(17,6))

order = a['Region'].value_counts(ascending=False).index

region=sns.countplot(x='Region',data=a,order=order)

region.set_title('Different Regions in the World')
# Now we have viewed different regions in the world , let's see the visulation of different regions one by one

a[(a['Region']=='Asia')]['Country'].value_counts()
b=a[ (a['Country']=='India') & (a['Year']<2020) ]

b.head(1)
plt.figure(figsize=(15,6))

p=sns.lineplot(x=b['Year'],y=b['AvgTemperature'])

p.set_title('Year Vs Avg Temprature')
b['City'].unique()
plt.figure(figsize=(15,6))

sns.pointplot(x=b['Year'],y=b['AvgTemperature'],hue=b['City'])

plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)

# We can also plot the graph for individual city by setting particular city name as hue :)
plt.figure(figsize=(40,10))

i=sns.FacetGrid(data=b,col='City',col_wrap= 2, height= 4, aspect= 3, margin_titles=True)

i.map(sns.pointplot,'Year','AvgTemperature')
plt.figure(figsize=(40,10))

i=sns.FacetGrid(data=b,col='City',margin_titles=True)

i.map(sns.distplot,'AvgTemperature')