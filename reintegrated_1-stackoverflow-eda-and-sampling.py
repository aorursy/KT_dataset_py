# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth',None)# set options such that the column width is maxed.

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("../input/facebook-recruiting-iii-keyword-extraction/Train.zip")

data.head()
data.columns
data.drop_duplicates(subset=['Title', 'Body', 'Tags'],keep='first',inplace=True)
data.shape 
data.dropna(inplace=True)
print(data.shape)

data.head()
from sklearn.feature_extraction.text import CountVectorizer

vec=CountVectorizer(tokenizer= lambda x:x.split(),binary=True)

tags_vec=vec.fit_transform(data.Tags)
tags_vec.shape
tags=vec.get_feature_names()

tags_data=pd.DataFrame(tags,columns=['tags'])
tags_data['counts']=tags_vec.sum(axis=0).A1

tags_data.head()
tags_sorted=tags_data.sort_values(['counts'],ascending=False)

counts=tags_sorted['counts'].values

plt.plot(counts)

plt.grid()

plt.xlabel('tag_no')

plt.ylabel('frequency')

plt.show()
plt.plot(counts[:10000])

plt.grid()

plt.xlabel('tag_no')

plt.ylabel('frequency')

plt.show()
plt.plot(counts[:1000])

plt.grid()

plt.xlabel('tag_no')

plt.ylabel('frequency')

plt.show()
plt.plot(counts[:500])

plt.grid()

plt.xlabel('tag_no')

plt.ylabel('frequency')

plt.show()
from matplotlib.pyplot import figure

figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

plt.plot(counts[:100],c='b')

plt.scatter(x=list(range(0,100,5)),y=counts[0:100:5],c='orange',label='quant 0.05')

plt.scatter(x=list(range(0,100,25)),y=counts[0:100:25],c='m',label='quant 0.25')

for index,count,tag_name in zip(list(range(0,100,25)),counts[0:100:25],tags_sorted['tags'].values[0:100:25]):

    plt.annotate(s=str(tag_name)+", is tagged "+str(count)+" times",xy=(index,count),xytext=(index+1,count+1000))

plt.grid()

plt.legend()

plt.xlabel('tag_no')

plt.ylabel('frequency')

plt.show()
from wordcloud import WordCloud, ImageColorGenerator

import matplotlib.pyplot as plt

from PIL import Image

import numpy as np

wc=WordCloud().generate_from_frequencies(tags_dict)

fig=plt.figure(figsize=(30,20))

plt.imshow(wc)
i=np.arange(20)

tags_sorted['counts'].head(20).plot(kind='bar')

plt.xticks(ticks=np.arange(20),labels=list(tags_sorted['tags'].values[:20]))

plt.show()
data=data.sample(n=1000000)

data.shape
data.to_csv('data_1_million.csv')