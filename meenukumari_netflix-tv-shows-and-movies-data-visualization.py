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
import numpy as np  # linear algebra

import pandas as pd  # data processing

import matplotlib.pyplot as plt

import seaborn as sns  #import seaborn as sns





#loading the dataset



path1="/kaggle/input/netflix-shows/netflix_titles.csv"

netflix=pd.read_csv(path1)    



#first 5 rows

netflix.head()

#Shape of the dataset

netflix.shape  
#columns of the dataset

netflix.columns
netflix.count()
#Check for null values in each column

netflix.isnull().sum()
#Check the number of unique values in each columns

netflix.nunique()
#Check for duplicate values

netflix.duplicated().sum()
#Make a copy of the dataset

netflixC=netflix.copy()

netflixC.shape
#Drop null values

netflixC=netflixC.dropna()

netflixC.shape
#First 5 values

netflixC.head(5)
#Convert datetime format

#Add three columns day_added,month_added and year_added 



netflixC["date_added"]=pd.to_datetime(netflixC["date_added"])

netflixC["day_added"]=netflixC["date_added"].dt.day

netflixC["month_added"]=netflixC["date_added"].dt.month

netflixC["year_added"]=netflixC["date_added"].dt.year

netflixC["year_added"].astype(int);

netflixC["day_added"].astype(int);
netflixC.head()
## DATA VISUALISATION

netflixC.type.nunique()
#Type: TV Shows and MOvies

#Plot the Type column



sns.countplot(netflix["type"])

fig = plt.gcf()

fig.set_size_inches(5,5)

plt.title('Type')
netflixC.rating.nunique()
sns.countplot(netflix["rating"])

sns.countplot(netflix['rating']).set_xticklabels(sns.countplot(netflix['rating']).get_xticklabels(), rotation=45, ha="right")

fig = plt.gcf()

fig.set_size_inches(12,10)

plt.title('Rating')
#Relation between TYPE and RATING

sns.countplot(x='rating',hue='type',data=netflix)



fig=plt.gcf()

fig.set_size_inches(12,6)

plt.title('Relation between Type and Rating')

plt.show()
# Pie Chart for the TYPE: TV shows and Rating

labels = ['Movie', 'TV show']

size = netflix['type'].value_counts()

colors = plt.cm.Wistia(np.linspace(0, 1, 2))

explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (8, 8)

plt.pie(size,labels=labels, colors = colors, explode = explode, shadow = True, startangle = 90)

plt.title('Distribution of Type', fontsize = 25)

plt.legend()

plt.show()
# Pie Chart for the TYPE: TV shows and Rating

labels = ['Movie', 'TV show']

size = netflix['type'].value_counts()

colors = plt.cm.Wistia(np.linspace(0, 1, 2))

explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (8, 8)

plt.pie(size,labels=labels, colors = colors, explode = explode, shadow = True, startangle = 90)

plt.title('Distribution of Type', fontsize = 25)

plt.legend()

plt.show()
#Data Visualization using Word Clouds

from wordcloud import WordCloud,STOPWORDS



#For Countries



plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1900,

                          height=1060

                         ).generate(" ".join(netflixC.country))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('country.png')

plt.show()
#For Directors



plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(netflixC.director))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('director.png')

plt.show()
#For Cast

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1900,

                          height=1060

                         ).generate(" ".join(netflixC.cast))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('cast.png')

plt.show()

#For Categories

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(netflixC.listed_in))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('category.png')

plt.show()
