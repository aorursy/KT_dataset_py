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
measure = pd.read_csv('/kaggle/input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv')
measure.head()
measure.isnull().sum()
## First lets drop empty measures

measure = measure[measure['Description of measure implemented'].notna()]



measure = measure[measure['Date Start'].notna()]



measure = measure[measure['Date end intended'].notna()]



measure = measure[measure['Keywords'].notna()]
measure.isnull().sum()
# Getting most Genre type of all times

from wordcloud import WordCloud, STOPWORDS

stopwords = STOPWORDS

wordcloud = WordCloud(width=800,height=400,stopwords=stopwords, min_font_size=10,max_words=150).generate(' '.join(measure['Keywords']).

                                                                                                         join(measure['Country']))

import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Most Popular measures followed", fontsize=25)

plt.show()
## List of countries

measure['Country'].value_counts()
## First country to take action is

measure_value = measure[['Country','Date Start','Date end intended','Keywords','Description of measure implemented']].sort_values(

    ['Date Start','Date end intended'], ascending=(['True','False']))
measure_value.head()
## First country to take action is Vietnam on 'INTERNATIONAL TRAVEL BAN'
## Converting into Date formates

measure['Date Start'] = pd.to_datetime(measure['Date Start'])

measure['Date end intended'] = pd.to_datetime(measure['Date end intended'])
## Finding number of days, that country followed the measure

measure['no. of days'] = measure['Date end intended'] - measure['Date Start']
## Considering only valid values 

measure_imp = measure[['Country','Date Start','Date end intended','Keywords','no. of days','Description of measure implemented']].sort_values(

    ['Date Start','Date end intended', 'no. of days'], ascending=(['True','False','False']))
measure_imp.head()
### List of countries and their measures till date.

measure_value_grp = measure_imp.groupby(['Country','Date end intended','no. of days'])['Keywords'].apply(','.join).reset_index()
measure_value_grp.head()
from collections import Counter

measure_data_c = measure_value_grp['Keywords']

measure_data_count = pd.Series(dict(Counter(','.join(measure_data_c).replace(' school closure','school closure').split(',')))).sort_values(ascending=False)
## Top 50 measurements

measure_data_top = measure_data_count[:50]
measure_data_top
import seaborn as sns

import squarify

plt.rcParams['figure.figsize'] = (120,120)

squarify.plot(sizes=measure_data_top.values,label=measure_data_top.index, color=sns.color_palette('RdGy'),

             linewidth=10, text_kwargs={'fontsize':50,'wrap':True})

plt.axis('off')

plt.title('Top 50 measurements')

plt.show()
## knowing number of countries implemented the same measure.

measure_count = pd.DataFrame(measure_imp['Keywords'].str.split(',').tolist(),measure_imp['Country']).stack()
measure_count = measure_count.reset_index()

measure_count = measure_count.drop(['level_1'],axis=1)

measure_count.columns = ['Country','Keywords']
measure_count.head(15)
import seaborn as sns

plt.figure(figsize=(100,50))

plt.rcParams['font.size'] = 60

sns.countplot(y='Country',data=measure_count,order=measure_count['Country'].value_counts().index)

# set_xlabel('Number of countries',fontsize=20)

# set_ylabel('Countries',fontsize=20)

plt.show()
measure_count.drop_duplicates(keep=False,inplace=True)
import seaborn as sns

plt.figure(figsize=(100,80))

plt.rcParams['font.size'] = 80

sns.countplot(y='Keywords',data=measure_count,order=measure_count['Keywords'].value_counts().index)

# set_xlabel('Number of countries',fontsize=20)

# set_ylabel('Countries',fontsize=20)

plt.title('Number of countries following the measures')

plt.show()