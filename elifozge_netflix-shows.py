# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

from wordcloud import WordCloud, STOPWORDS



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
shows=pd.read_csv('../input/netflix-shows/Netflix Shows.csv',encoding = "ISO-8859-1")

shows.head()
shows.drop(['ratingDescription'],axis=1,inplace=True)

shows.rename(columns={'ratingLevel': 'rating_level','release year':'release_year','user rating score':'rating_score','user rating size':'rating_size'},inplace=True)
shows.sort_values(by=['rating_score'],ascending=False,inplace=True)

shows.head(5)
shows = shows.drop_duplicates(keep="first").reset_index(drop=True)

shows.head()
shows.info()
print('There are {} different shows.\nAnd these shows in this years: {} \nThere are {} types of ratings.\nThese are: {}.\n  '.format(len(shows['title']),shows['release_year'].unique(),len(shows['rating'].unique()),shows['rating'].unique(),))
f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(shows.corr(), annot=True,linecolor="red",ax=ax,cmap='coolwarm')

plt.show()
plt.figure(figsize=(15,10))

sns.countplot(x='release_year',data=shows,order=shows['release_year'].value_counts().index)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(10,10))

sns.countplot(x='rating',data=shows,order=shows['rating'].value_counts().index,palette='Paired')

plt.show()
shows_new=shows.dropna()

rating_list=list(shows_new['rating'].unique())

rating_score_mean=[]

rating_count=[]



for i in rating_list:

    x=shows_new[shows_new['rating']==i]

    score_mean=sum(x.rating_score)/len(x)

    rating_score_mean.append(score_mean)

    rating_count.append(len(x))

    

    

data = pd.DataFrame({'rating':rating_list,'mean_score':rating_score_mean,'rating_count':rating_count})

sorted_data = data.sort_values(by='mean_score',ascending=False).reset_index(drop=True)

sorted_data.head()
trace1 = go.Scatter(x=sorted_data.rating,

                    y=sorted_data.mean_score,

                    mode="lines+markers",

                    name="scores",

                    text="scores"

                   )

trace2 = go.Scatter(x=sorted_data.rating,

                    y=sorted_data.rating_count,

                    mode="lines+markers",

                    name="counts",

                    text="counts"

                   )

trace3 = go.Scatter(x=sorted_data.rating,

                    y=sorted_data.mean_score/sorted_data.rating_count,

                    mode="lines+markers",

                    name="score/count",

                    text="score/count"

                   )





layout= go.Layout(dict(title="Mean Scores for Each Rating"))

fig=dict(data=[trace1,trace2,trace3],layout=layout)

iplot(fig)