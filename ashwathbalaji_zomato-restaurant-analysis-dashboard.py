import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/dataset/ZomatoProfiling.csv')

df.head()
df = df[df['res_name']=='The Roastery Coffee House']

df.head(2)
df.shape
mr = (df.isna().sum()/len(df))*100

pd.DataFrame(mr.sort_values(ascending=False) , columns=['Missing Ratio'])
df.fillna('NA',inplace=True)
df['date'] = pd.to_datetime(df['date'])



df['year'] = df['date'].dt.year

df['month'] = df['date'].dt.month

df['time'] = df['date'].dt.time
df['rating'] = df['rating'].apply(lambda x: x.split(' ')[1])
df['user_reviews'] = df['rev_count'].apply(lambda x: x.split(',')[0])

df['user_reviews'] = df['user_reviews'].apply(lambda x: x.split(' ')[0])
df['user_followers']= df['rev_count'][df['rev_count']!='NA'].apply(lambda x: x.split(',')[1])

df['user_followers'] = df['user_followers'].fillna('NA')



df['user_followers'] = df['user_followers'][df['user_followers']!='NA'].apply(lambda x: x.split(' ')[1])
df.drop('rev_count',1,inplace=True)
df['rating'] = df['rating'].astype(float)

df['user_reviews'][df['user_reviews']=='NA'] = np.NaN

df['user_reviews'].fillna(0 ,inplace=True)

df['user_reviews'] = df['user_reviews'].astype(int)
df.head()
df.head(2)
sns.barplot(df['month'][df['year']==2019] , df['user_reviews'] )

plt.xlabel('Month')

plt.ylabel('Total Reviews')

plt.title('Total Reviews in 2019')

plt.xticks([0,1,2],labels=['January','Feburary','March'])

plt.show()
sns.barplot(df['month'][df['year']==2019] , df['rating'] )

plt.xlabel('Month')

plt.ylabel('Average Rating')

plt.title('Average Rating in 2019')

plt.xticks([0,1,2],labels=['January','Feburary','March'])

plt.show()
df[df['rating']==5.0].count()[1]
df[df['rating']<4].count()[1]
df.resample('1m',on='date').size().plot()

plt.grid(True)

plt.title('Reviews per Month')

plt.xlabel('Month')

plt.ylabel('Reviews')
df['text'][df['text']=='NA'] = np.NaN
from wordcloud import WordCloud

ip_string=' '.join(df['text'].str.replace('RATED',' ').dropna().to_list())



wc=WordCloud(background_color='white').generate(ip_string.lower())

plt.imshow(wc)

plt.show()
from IPython.display import Image 

Image("/kaggle/input/dashboard/ZomatoRestaurantProfiling.JPG")