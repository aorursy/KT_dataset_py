import os
import gc
import time
import re
from tqdm.notebook import tqdm as tqdm

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle

import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from wordcloud import WordCloud, ImageColorGenerator

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

%matplotlib inline
us_df = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv')
category_json = pd.read_json('/kaggle/input/youtube-new/US_category_id.json')
us_df.head()
us_df.drop(['video_id','thumbnail_link',
            'comments_disabled','ratings_disabled',
            'video_error_or_removed','description'],axis=1,inplace=True)

category_json = category_json['items']
category_json[0]
column_names = ['category_id','category']
catid_df = pd.DataFrame(columns=column_names)
for data in category_json:
    category_id = data['id']
    category = data['snippet']['title']
    catid_df = catid_df.append({'category_id': category_id,
                                'category': category}, ignore_index=True)
    
catid_df.head()
catid_df['category_id'] = catid_df['category_id'].astype(int)
catid_df.set_index('category_id',inplace=True)
catid_df.to_dict(orient='dict')
categorydict = {1: 'Film & Animation',
  2: 'Autos & Vehicles',
  10: 'Music',
  15: 'Pets & Animals',
  17: 'Sports',
  18: 'Short Movies',
  19: 'Travel & Events',
  20: 'Gaming',
  21: 'Videoblogging',
  22: 'People & Blogs',
  23: 'Comedy',
  24: 'Entertainment',
  25: 'News & Politics',
  26: 'Howto & Style',
  27: 'Education',
  28: 'Science & Technology',
  29: 'Nonprofits & Activism',
  30: 'Movies',
  31: 'Anime/Animation',
  32: 'Action/Adventure',
  33: 'Classics',
  34: 'Comedy',
  35: 'Documentary',
  36: 'Drama',
  37: 'Family',
  38: 'Foreign',
  39: 'Horror',
  40: 'Sci-Fi/Fantasy',
  41: 'Thriller',
  42: 'Shorts',
  43: 'Shows',
  44: 'Trailers'}
us_df.replace({"category_id": categorydict},inplace=True)
us_df.rename(columns={'category_id':'category'},inplace=True)
def conv_dates_series(df, col, old_date_format, new_date_format):

    df[col] = pd.to_datetime(df[col], format=old_date_format).dt.strftime(new_date_format)

    return(df)
old_date_format='%y.%d.%m'
new_date_format='%Y-%m-%d'

conv_dates_series(us_df, 'trending_date', old_date_format, new_date_format)
us_df['YYYY'] = us_df['trending_date'].apply(lambda x: x.split('-')[0])
us_df['MM'] = us_df['trending_date'].apply(lambda x: x.split('-')[1])
def view_per_columnname(column_name):
    views_per_column = us_df['views'].groupby(us_df[column_name]).sum()
    views_per_column = pd.DataFrame(views_per_column)
    final = views_per_column.reset_index()
    return final
view_per_columnname('MM')
sns.barplot(x='MM',y='views',data=view_per_columnname('MM'), palette='viridis', ci=None)
view_per_columnname('YYYY')
sns.barplot(x='YYYY',y='views',data=view_per_columnname('YYYY'), palette='viridis', ci=None)
view_per_columnname('category')
plt.figure(figsize=(20,6))
sns.barplot(x='category',y='views',data=view_per_columnname('category'), palette='viridis', ci=None)
plt.tick_params(axis='x', which='major', labelsize=4)
us_df.columns
us_df.corr()
corpus = []
for i in tqdm(range(0,40881)):   
    tags = re.sub('[^a-zA-Z]', ' ', us_df['tags'][i])
    tags = tags.lower()
    tags = tags.split()
    ps = PorterStemmer()
    tags = [ps.stem(word) for word in tags if not word in set(stopwords.words('english'))]
    tags = ' '.join(tags)
    corpus.append(tags)
unique_string=(" ").join(corpus)
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("your_file_name"+".png", bbox_inches='tight')
plt.show()
plt.close()
def getBagofWords(category):
    cat = us_df[us_df['category']==category]['tags'].reset_index()
    countlist = len(cat)
    corpus = []
    for i in tqdm(range(0,countlist)):   
        tags = re.sub('[^a-zA-Z]', ' ', cat['tags'][i])
        tags = tags.lower()
        tags = tags.split()
        ps = PorterStemmer()
        tags = [ps.stem(word) for word in tags if not word in set(stopwords.words('english'))]
        tags = ' '.join(tags)
        corpus.append(tags)
        
    unique_string=(" ").join(corpus)
    wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig("your_file_name"+".png", bbox_inches='tight')
    
    return plt.show()    
getBagofWords('Music')
getBagofWords('Entertainment')
getBagofWords('Film & Animation')
getBagofWords('Comedy')
getBagofWords('People & Blogs')
