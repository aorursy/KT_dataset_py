# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from pandas.tools.plotting import scatter_matrix

import seaborn as sns

from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import pandas as pd

df = pd.read_csv('../input/zomato.csv')
df.head()
df.shape
df.isna().sum()
#df = df.dropna(axis=0, subset=['rate'])

#df.isna()
df[df['rate'].isna()]
#df['book_table'][df['rate'].isna()].value_counts()
df = df.dropna(axis=0, subset=['rate'])

df.dtypes
df.rate.astype(str,inplace=True)

df['rate'] = [x[:-2] for x in df['rate']]
#df['rate'].fillna("0/5",inplace=True)

df['dish_liked'].fillna("None",inplace=True)

df['cuisines'].fillna("None",inplace=True)

df['approx_cost(for two people)'].fillna("0.0",inplace=True)

df['rest_type'].fillna("None",inplace=True)

df['location'].fillna("None",inplace=True)
#df.isna().sum()
df['rate'] = pd.to_numeric(df['rate'],errors='coerce')

df['votes'] = pd.to_numeric(df['votes'],errors='coerce')

df.rename(columns={'approx_cost(for two people)': 'avg_cost'},inplace = True)

df.rename(columns={'listed_in(type)': 'type'},inplace = True)

df.rename(columns={'listed_in(city)': 'city'},inplace = True)

df['avg_cost'] = pd.to_numeric(df['avg_cost'],errors='coerce')

df.plot(x='rate',y='votes',kind='scatter',title='Relation between Rating and Vote')

df.plot(x='avg_cost',y='votes',kind='scatter',title='Relation between Cost and Vote')
df.isna().sum()
df.drop(['url','address','reviews_list'],axis=1,inplace=True)
df.head()
#df.drop(['production_companies','cast'],axis=1,inplace=True)

df.drop_duplicates(keep='first',inplace=True)
df['rate'].dtype
df.type.value_counts()
df.hist(figsize=(8,8))
plt.subplots(1,2,figsize=(8,4))

plt.subplot(1,2,1)

sns.countplot(df['online_order'])

plt.subplot(1,2,2)

sns.countplot(df['book_table'])

plt.tight_layout()
#plt.figure(figsize=(20,5))

#sns.countplot(df['location'],palette='ocean_r',order=df['location'].value_counts().index)

#plt.xticks(rotation=90)
"""

plt.figure(figsize=(15,5))

sns.distplot(df['avg_cost'])

plt.xticks(rotation=90)

sns.distplot(df['avg_cost'])

"""

#df.rate.dtype
plt.figure(figsize=(15,5))

rest_type=df['rest_type'].value_counts()[:20]

sns.barplot(rest_type.index,rest_type,palette='Pastel1')

plt.xlabel('Count')

plt.title("Most popular cuisines of Bangalore")

plt.xticks(rotation=90)
plt.figure(figsize=(15,5))

sns.countplot(df['city'],palette='Purples',order = df['city'].value_counts().index)

plt.xticks(rotation=90)
plt.figure(figsize=(5,5))

sns.countplot(df['type'],palette='Accent',order = df['type'].value_counts().index)

plt.xticks(rotation=90)
plt.figure(figsize=(10,5))

sns.countplot(df['rate'],palette='Oranges')

plt.xticks(rotation=90)
plt.figure(figsize=(7,4))

cuisines=df['cuisines'].value_counts()[:10]

sns.barplot(cuisines.index,cuisines)

plt.xlabel('Count')

plt.title("Most popular cuisines of Bangalore")

plt.xticks(rotation=90)
plt.figure(figsize=(3,3))

df.groupby('book_table')['rate'].mean().plot.bar()

plt.ylabel('Average rating')
plt.figure(figsize=(6,3))

df.groupby('online_order')['rate'].mean().plot.bar()

plt.ylabel('Average rating')
#df.groupby('name')['rate'].mean()
#df.query('online_order=="Yes"').query('book_table=="Yes"')
#plt.figure(figsize=(7,7))

#Rest_locations=df['location'].value_counts()[:15]

#sns.barplot(Rest_locations.index,Rest_locations,palette='Blues')

#plt.xticks(rotation=90)
rest_params = df.groupby(by='type', as_index=False).mean().sort_values(by='rate',ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))

sns.barplot(x='type', y='avg_cost', data=rest_params, ax=ax,palette='Greens')

ax2 = ax.twinx()

sns.lineplot(x='type', y='rate', data=rest_params, ax=ax2, sort=False)

ax.tick_params(axis='x', labelrotation=90)

ax.xaxis.set_label_text("")



xs = np.arange(0,10,1)

ys = rest_params['rate']

for x,y in zip(xs,ys):

    label = "{:.2f}".format(y)

    plt.annotate(label, # this is the text

                 (x,y), # this is the point to label

                 textcoords="offset points", # how to position the text

                 xytext=(0,10), # distance from text to points (x,y)

                 ha='center', # horizontal alignment can be left, right or center

                 color='black')

    

ax.set_title('Average Cost and Rating of Restaurants by Type', size=14)

plt.tight_layout()
"""

df['dish_liked']=df['dish_liked'].apply(lambda x : x.split(',') if type(x)==str else [''])

df['dish_liked'].value_counts()

#rest=df['rest_type'].value_counts()[:9].index

"""
def produce_wordcloud(rest):

    

    plt.figure(figsize=(20,30))

    for i,r in enumerate(rest):

        plt.subplot(3,3,i+1)

        corpus=df[df['rest_type']==r]['dish_liked'].values.tolist()

        corpus=','.join(x  for list_words in corpus for x in list_words)

        wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,

                      width=1500, height=1500).generate(corpus)

        plt.imshow(wordcloud)

        plt.title(r)

        plt.axis("off")

        
numeric_vars = ['avg_cost','rate','votes']

plt.figure(figsize = [8, 5])

sns.heatmap(df[numeric_vars].corr(), annot = True, fmt = '.3f',cmap = 'vlag_r', center = 0)

plt.show()
df['dish_liked']=df['dish_liked'].apply(lambda x : x.split(',') if type(x)==str else [''])

df['dish_liked']

rest=df['rest_type'].value_counts()[:9].index

rest

produce_wordcloud(rest) 
print(rest)
numeric_vars = ['avg_cost','rate','votes']

plt.figure(figsize = [8, 5])

sns.heatmap(df[numeric_vars].corr(), annot = True, fmt = '.3f',cmap = 'vlag_r', center = 0)

plt.show()