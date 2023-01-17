# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#importing the libraries

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

% matplotlib inline

#import the data

data = pd.read_csv('../input/Netflix Shows.csv', encoding='cp437')
pd.set_option('display.max_columns',5000)
#checking the data fields

data.head()
#how many shows telecast each year

year=data.groupby("release year")['title'].count().reset_index().sort_values(by='release year',ascending=False).reset_index(drop=True)

year.columns=['release_year','No_of_release_showss']

year
year.plot.scatter(x="release_year",y="No_of_release_showss",s=100,figsize=(12,8),c="blue",alpha=0.5)
plt.figure(figsize=(10,10))

sns.barplot(x="release_year",y="No_of_release_showss",data=year)

plt.xticks(rotation=45)

#lets see which  rating are given to TV shows

rating=data.groupby("rating")['title'].count().reset_index().sort_values(by='rating',ascending=False).reset_index(drop=True)

rating
plt.figure(figsize=(10,10))

sns.barplot(x='rating',y='title',data=rating)

plt.xticks(rotation=45)

#lets see how many user rating scores are given 

rating_score=data.groupby("user rating score")['title'].count().reset_index().sort_values(by='user rating score',ascending=False).reset_index(drop=True)

rating_score
plt.figure(figsize=(12,12))

sns.barplot(x='user rating score',y='title',data=rating_score)

plt.xticks(rotation=45)
#lets check which rating have more value

plt.figure(figsize=(12,12))

sns.barplot(x='rating',y='ratingDescription',data=data)

plt.xticks(rotation=45)

data_year=data.groupby(['release year', 'ratingDescription'])['title'].count().reset_index().sort_values(by='release year',ascending=False).reset_index(drop=True)

data_year
area = np.array(data_year['ratingDescription'])
data_year['ratingDescription'].unique()
data_col = {110:'blue',90:'green',70:'red',42:'orange',41:'white',60:'black',10:'black',35:'yellow',100:'gray',124:'pink',80:'green'}
a=[]

for i in data_year['ratingDescription']:

    if i in data_col.keys():

        a.append(data_col[i])



a
data_year.plot.scatter(x="release year",y="title",s=area*4,c=a,alpha=0.5,figsize=(12,8))
from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          width=1200,

                          height=1000

                         ).generate(" ".join(data['title']))





plt.imshow(wordcloud)

plt.axis('off')

plt.show()