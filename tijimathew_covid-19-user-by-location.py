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
# Read the data set into pandas dataframe df

df = pd.read_csv('/kaggle/input/covid19-tweets/covid19_tweets.csv')
# Review the dataset to identify user and geography attributes.

df.head()
df.describe(include='all')
df.info()
df['has_hashtags'] = df.hashtags.notna()

df_task = df.set_index('date')
import ast

from collections import Counter
df['has_covid_tag'] = df.hashtags.str.lower().str.contains('covid') | df.hashtags.str.lower().str.contains('coronavirus')

#hashtags_dict = Counter()

#for tag_list, value in df['hashtags'].str.replace('ー','').str.lower().str.replace('-','').str.replace('_','').value_counts().iteritems():

#    hashtags_dict.update(ast.literal_eval(tag_list))

        

#print(hashtags_dict.)
df['has_covid_text'] = df['text'].str.lower().str.contains('covid') | df.text.str.lower().str.contains('coronavirus')
df_focused = df[['has_hashtags', 'hashtags', 'text', 'has_covid_tag', 'has_covid_text','user_name', 'user_location']]
df_covid_dmg = df_focused[['user_name', 'user_location']].loc[(df_focused.has_covid_tag == True) | (df_focused.has_covid_text == True)]
# Top 5 users tweeting about Covid

df_covid_dmg['user_name'].value_counts()[:5]
# top 5 locations tweeting about Covid

df_covid_dmg['user_location'].value_counts()[:5]