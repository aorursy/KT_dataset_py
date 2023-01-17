# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('../input/winemag-data-130k-v2.csv')
df.head()
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
df2 = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
df2

duplicate_bool=df.duplicated()
duplicate=df.loc[duplicate_bool == True]
print(duplicate)
df.info()
plt.figure(1,figsize=[8,8])
sns.countplot('country',data=df,order=pd.value_counts(df['country']).iloc[:5].index)
plt.figure(2,figsize=[8,8])
sns.countplot('province',data=df,order=pd.value_counts(df['province']).iloc[:5].index)
plt.figure(3,figsize=[8,8])
sns.countplot('winery',data=df,order=pd.value_counts(df['winery']).iloc[:5].index)
plt.figure(4,figsize=[8,8])
sns.countplot('variety',data=df,order=pd.value_counts(df['variety']).iloc[:5].index)
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df['description']))

print(wordcloud)
fig = plt.figure(figsize = (8, 8),facecolor = None)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
plt.figure(1)
sns.boxplot(x=df['points'],color="blue")
plt.figure(2)
sns.boxplot(x=df['price'],color="blue")
plt.figure(1)
df['points'].value_counts().sort_index().plot.line()
plt.title("Distribution of points")
plt.xlabel("points")
plt.ylabel("Values")
plt.figure(2)
df['price'].value_counts().sort_index().plot.line()
plt.title("Distribution of price")
plt.xlabel("price")
plt.ylabel("Values")
a=df[df['price']<200]
a['price'].fillna(a['price'].mean())
a.head()
sns.jointplot(x="price", y="points", data=a)
plt.figure(1)
sns.barplot(y="variety",x="points",data=df,order=pd.value_counts(df['variety']).iloc[:5].index)
plt.figure(2)
sns.barplot(y='winery',x='points',data=df,order=pd.value_counts(df['winery']).iloc[:5].index)
plt.figure(3)
sns.barplot(y="variety",x="price",data=df,order=pd.value_counts(df['variety']).iloc[:5].index)
plt.figure(4)
sns.barplot(y='winery',x='price',data=df,order=pd.value_counts(df['winery']).iloc[:5].index)
b=df[df['country'].isin(['US','France','Canada','Spain'])]
b.head()
g=sns.FacetGrid(b, col="country")
g=g.map(sns.kdeplot, "price", color="r")
g = sns.FacetGrid(b, col = "country")
g.map(sns.kdeplot, "points", color="r")
df1=df.groupby(['winery','designation']).size()
df1.sort_values(ascending=False)
df2=df[df.variety.isin(df.variety.value_counts()[:5].index)]#top 5 variety values
df3=df[df.winery.isin(df.winery.value_counts()[:5].index)]#top 5 winery values
plt.figure(1)
sns.violinplot(x="points",y="variety",data=df2)
plt.figure(2)
sns.violinplot(x="points",y="winery",data=df3)
#df.reset_index(inplace=True)
a=df.pivot_table(values='price',index=['country','province'])
a.sort_values(by=['price'],ascending=False)