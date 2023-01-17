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
import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/zomato-bangalore-restaurants/zomato.csv')
rows = df.shape[0]

columns = df.shape[1]

print("Rows : "+str(rows)+" Columns :  "+str(columns))
df.head()
df['name'].value_counts()[:10]
df[df['name'] == 'Petoo'][:5]
plt.figure(figsize=(7,7))

df_temp = df['name'].value_counts()[:10]

import seaborn as sns

import matplotlib.pyplot as plt

sns.barplot(x = df_temp , y = df_temp.index)
df.columns
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

df_temp = df['online_order'].value_counts()

plt.title('Overall Online Order Service')

plt.pie(df_temp, labels=df_temp.index)

plt.show()
plt.figure(figsize=(5,5))

popular_restaurant = df['name'].value_counts()[:10].index

df_popular = df[df['name'].isin(popular_restaurant)]

df_temp = df_popular['online_order'].value_counts()

plt.pie(df_temp, labels=df_temp.index)

plt.title('Top 10 restaurants')

plt.show()
df.columns
df['rate'][:10]
#There are some null entries, let's handle it first!

df['rate'].isna().sum()
df['rate'] = df['rate'].fillna('0')
df['rate'].isna().sum()
rating = list(df['rate'])

rating = [i.split('/')[0] for i in rating]

total_length = len(rating)

for i in range(total_length):

    if(rating[i] == 'NEW' or rating[i] == '-'):

        rating[i] = '0'

rating= [ float(x) for x in rating ]
df['final_ratings'] = rating
df.head()
df_temp = df[['name','final_ratings']]
df_temp.sort_values('final_ratings',ascending = False)[:10]
df_temp = df_temp.drop_duplicates()

df_temp.sort_values('final_ratings',ascending = False)[:15]
sns.distplot(df_temp['final_ratings'],bins = 30)
df.columns
df['rest_type'].unique()
plt.figure(figsize=(7,7))

df_temp = df['rest_type'].value_counts()[:10]

sns.barplot(x = df_temp,y = df_temp.index)

plt.title('Busy developers!')
df.columns
plt.figure(figsize=(7,7))

rest=df['location'].value_counts()[:10]

sns.barplot(rest,rest.index)
areas = []

cuisines = []

areas = list(df['location'].value_counts()[:10].index)

for i in areas:

    df_temp = df[df['location'] == i]

    cuisines.append((str(df_temp['cuisines'].value_counts()[:1].index[0])))



df_temp =pd.DataFrame()

df_temp['areas'] = areas

df_temp['food'] = cuisines

df_temp

df.columns
df.head()
df_temp = df[df['rest_type'] == 'Casual Dining']

df_temp.head()
words = list(df_temp['dish_liked'])

word_cloud = []

for i in words:

    if(type(i) == str):

        temp = i.split(',')

        for i in temp:

            i=" ".join(i.split())

            word_cloud.append(i)
import matplotlib.pyplot as plt

from wordcloud import WordCloud

#convert list to string and generate

unique_string=(" ").join(word_cloud)

wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis("off")

plt.savefig("your_file_name"+".png", bbox_inches='tight')

plt.show()

plt.close()
'''

'Quick Bites',

 'Casual Dining',

 'Cafe',

 'Delivery',

 'Dessert Parlor',

 'Takeaway, Delivery',

 'Casual Dining, Bar',

 'Bakery',

 'Beverage Shop',

 'Bar'

'''

import matplotlib.pyplot as plt

rest_type = list(df['rest_type'].value_counts()[:10].index)

for i in rest_type:

    

    df_temp = df[df['rest_type'] == i]

    words = list(df_temp['dish_liked'])

    word_cloud = []

    for i in words:

        if(type(i) == str):

            temp = i.split(',')

            for i in temp:

                i=" ".join(i.split())

                word_cloud.append(i)

    unique_string=(" ").join(word_cloud)

    wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)

    plt.figure(figsize=(15,8))

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.savefig("your_file_name"+".png", bbox_inches='tight')

    plt.show()

    plt.close()
rest_type = list(df['rest_type'].unique())

len(rest_type)
df.columns
df['reviews_list'].head()
# Code is referenced from SHAHULES Kaggle Notebook

from tqdm import tqdm

all_ratings = []



for name,ratings in tqdm(zip(df['name'],df['reviews_list'])):

    ratings = eval(ratings)

    for score, doc in ratings:

        if score:

            score = score.strip("Rated").strip()

            doc = doc.strip('RATED').strip()

            score = float(score)

            all_ratings.append([name,score, doc])
all_ratings
rating_df=pd.DataFrame(all_ratings,columns=['name','rating','review'])
rating_df[:10]
rating_df['name'].value_counts()[:10].index
import matplotlib.pyplot as plt

rest_type = list(rating_df['name'].value_counts()[:1].index)

reviews = []

wordcloud = []

for i in rest_type:

    df_temp = rating_df[rating_df['name'] == i]

    reviews = list(df_temp['review'])



print(len(reviews))



for i in reviews:

    temp = i.split(' ')

    for i in temp:

        wordcloud.append(i)

print(len(wordcloud))



print(wordcloud)

    
from nltk import word_tokenize

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

temp = []

for i in wordcloud:

    if(i not in stop):

        temp.append(i)

len(temp)
unique_string=(" ").join(temp)

wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis("off")

plt.savefig("your_file_name"+".png", bbox_inches='tight')

plt.show()

plt.close()
''' 

'Hammered', 'Mast Kalandar', 'Truffles', 'Onesta', 'Crawl Street',

       'Brooks and Bonds Brewery', 'Stoner', 'Cafe Azzure', 'Cafe @ Elanza',

       'Smally's Resto Cafe'],

'''

import matplotlib.pyplot as plt

rest_type = list(rating_df['name'].value_counts()[:10].index)

for i in rest_type:

    reviews = []

    wordcloud = []



    df_temp = rating_df[rating_df['name'] == i]

    reviews = list(df_temp['review'])

    for i in reviews:

        temp = i.split(' ')

        for i in temp:

            wordcloud.append(i)

    temp = []

    for i in wordcloud:

        if(i not in stop):

            temp.append(i)

    unique_string=(" ").join(temp)

    wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)

    plt.figure(figsize=(15,8))

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.savefig("your_file_name"+".png", bbox_inches='tight')

    plt.show()

    plt.close()

    

        

    

    