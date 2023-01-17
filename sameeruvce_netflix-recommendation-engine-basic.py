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
import pandas as pd

df = pd.read_csv("../input/netflix_titles.csv")
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import cufflinks as cf

df.head()

df.describe()
df["date_added"]= pd.to_datetime(df["date_added"])

df['year_added']=df["date_added"].dt.year

df['month_added']=df['date_added'].dt.month

df.head()
df['season_count']=df.apply(lambda x:x['duration'].split(" ")[0] if 'Season'  in x['duration'] else "",axis=1) #apply to each column

df['duration']=df.apply(lambda x:x['duration'].split(" ")[0] if 'Season' not in x['duration'] else "", axis=1)

df.head()



from wordcloud import WordCloud , STOPWORDS , ImageColorGenerator

plt.rcParams['figure.figsize']=(13,13)

wordcloud=WordCloud(stopwords=STOPWORDS,background_color='black',width=1000,height=1000,max_words=121).generate(''.join(df['title']))

#Wordcloud function is used to make the cloud, remove stopwords and generate from df{title}

plt.imshow(wordcloud)

plt.axis('off')

plt.title('MOST POPULAR WORDS IN TITLE')

plt.show()
import plotly.express as px

#x=count(df['type'] if df[type]==)

sr=pd.Series(df['type'])

x=sr.value_counts()

labels = 'Movies','TV Shows'

fig1, ax1 = plt.subplots()

ax1.pie(x,labels=labels,autopct='%1.1f%%')

plt.show()

x