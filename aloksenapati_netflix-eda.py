# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import cufflinks as cf

import plotly.offline

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Reading the dataset

data=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles_nov_2019.csv')
data.shape
data.columns
data.head()
mov=data.query("type=='Movie'")

mov['min']=mov['duration'].str.split(' ',expand=True)[0]

mov['min']=mov['min'].astype(int)

mov['hr']=mov['min']/60
top20run=mov.sort_values(by='hr',ascending=False).head(20)

plt.figure(figsize=(10,7))

sns.barplot(data=top20run,y='title',x='hr',hue='country',dodge=False)

plt.legend(loc='lower right')

plt.title('Top 10 movies by Run Time')

plt.xlabel('Hours')

plt.ylabel('Movie name')

plt.show()
sns.set(style="darkgrid", palette="pastel", color_codes=True)

plt.figure(figsize=(5,10))

sns.countplot(y='director',data=data,order = data['director'].value_counts().head(20).index)

plt.show()
indcast=[]

plt.rcParams['axes.facecolor']='cyan'

sns.set()

ind=data.query('country=="India"')

for i in ind['cast']:

    indcast.append(i)

newls=[]

for i in indcast:

    newls.append(str(i).split(',')[0])

inddf=pd.DataFrame(newls,columns=['name'])

ind_df=inddf.drop(inddf.query('name=="nan"').index)

ind_df['name'].value_counts().head(20).iplot(kind="bar")



us=data[data['country'].str.contains('United States',na=False)]

uscast=[]

for i in us['cast']:

    uscast.append(i)

newls1=[]

for i in uscast:

    newls1.append(str(i).split(',')[0])

    

usdf=pd.DataFrame(newls1,columns=['name'])

us_df=usdf.drop(usdf.query('name=="nan"').index)

us_df['name'].value_counts().head(20).iplot(kind="bar")