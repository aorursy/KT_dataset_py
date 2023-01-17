# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data.shape
data = pd.read_csv("../input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv")
data.head()
data.info()

data.isnull().sum()
data.drop(["Unnamed: 0",'type','Rotten Tomatoes'],axis=1,inplace=True)
data.dropna(inplace=True)
data.shape
import seaborn as sm
corr=data[["Netflix",'Hulu','Prime Video',"Disney+"]].corr()
sm.heatmap(corr)
data.sum()["Netflix":"Disney+"]
data.sum()["Netflix":"Disney+"].plot(kind='pie',autopct='%1.1f%%',title='pie chart for number of shows/movies in Nwtflix,Hulu,Prime,Diseny')
Age_=data.groupby('Age').sum()[['Netflix','Hulu','Prime Video','Disney+']]
Age_.plot(kind='bar')
colors=['r','b','g','y']
data.Age.value_counts().plot(kind='bar',edgecolor='black',color=colors,title='count of content by age')
Age_.Netflix.plot(kind='bar',title='count_of_Netflix_content_by_age')
Age_.Hulu.plot(kind='bar',title='count_of_Hulu_content_by_age')
Age_['Prime Video'].plot(kind='bar',title='count_of_Prime Video_content_by_age')
Age_['Disney+'].plot(kind='bar',title='count_of_Disney+_content_by_age')
ax=sm.catplot(x='Year',kind='count',data=data,orient="h",height=30,aspect=1,)
ax.fig.suptitle('Number of TV series / movies per year')
ax.fig.autofmt_xdate()
yearswise=data.groupby(["Year"])['Netflix','Hulu','Prime Video','Disney+'].sum()
yearswise.Netflix.plot(title='number of shows/movies in Netflix per year')

yearswise.Hulu.plot(title='number of shows/movies in Hulu per year')
yearswise['Prime Video'].plot(title='number of shows/movies in prime videos per year',color='y')
yearswise['Disney+'].plot(title='number of shows/movies in Desiney+ per year',color='r')

