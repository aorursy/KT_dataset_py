# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sas

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import os

from wordcloud import WordCloud

from collections import Counter

import json

from PIL import Image

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv',index_col=0)

test=pd.read_csv('../input/test.csv',index_col=0)
train.head()
test['revenue']=-99
test.head()
train
train[['belongs_to_collection']].info()
train[['belongs_to_collection','title']].head()
train=train.drop('belongs_to_collection',axis=1)

test=test.drop('belongs_to_collection',axis=1)
train.loc[train.index == 16,'revenue'] = 192864         

train.loc[train.index == 90,'budget'] = 30000000                  

train.loc[train.index== 118,'budget'] = 60000000       

train.loc[train.index== 149,'budget'] = 18000000       

train.loc[train.index== 313,'revenue'] = 12000000       

train.loc[train.index == 451,'revenue'] = 12000000      

train.loc[train.index == 464,'budget'] = 20000000       

train.loc[train.index == 470,'budget'] = 13000000       

train.loc[train.index== 513,'budget'] = 930000         

train.loc[train.index == 797,'budget'] = 8000000        

train.loc[train.index == 819,'budget'] = 90000000       

train.loc[train.index == 850,'budget'] = 90000000       

train.loc[train.index == 1007,'budget'] = 2              

train.loc[train.index== 1112,'budget'] = 7500000       

train.loc[train.index == 1131,'budget'] = 4300000        

train.loc[train.index == 1359,'budget'] = 10000000       

train.loc[train.index == 1542,'budget'] = 1             

train.loc[train.index == 1570,'budget'] = 15800000       

train.loc[train.index== 1571,'budget'] = 4000000        

train.loc[train.index == 1714,'budget'] = 46000000       

train.loc[train.index == 1721,'budget'] = 17500000       

train.loc[train.index== 1865,'revenue'] = 25000000      

train.loc[train.index == 1885,'budget'] = 12             

train.loc[train.index == 2091,'budget'] = 10             

train.loc[train.index == 2268,'budget'] = 17500000       

train.loc[train.index == 2491,'budget'] = 6              

train.loc[train.index == 2602,'budget'] = 31000000       

train.loc[train.index == 2612,'budget'] = 15000000       

train.loc[train.index == 2696,'budget'] = 10000000      

train.loc[train.index == 2801,'budget'] = 10000000       

train.loc[train.index == 335,'budget'] = 2 

train.loc[train.index == 348,'budget'] = 12

train.loc[train.index == 470,'budget'] = 13000000 

train.loc[train.index == 513,'budget'] = 1100000

train.loc[train.index == 640,'budget'] = 6 

train.loc[train.index == 696,'budget'] = 1

train.loc[train.index == 797,'budget'] = 8000000 

train.loc[train.index == 850,'budget'] = 1500000

train.loc[train.index == 1199,'budget'] = 5 

train.loc[train.index == 1282,'budget'] = 9              

train.loc[train.index== 1347,'budget'] = 1

train.loc[train.index== 1755,'budget'] = 2

train.loc[train.index == 1801,'budget'] = 5

train.loc[train.index == 1918,'budget'] = 592 

train.loc[train.index == 2033,'budget'] = 4

train.loc[train.index == 2118,'budget'] = 344 

train.loc[train.index == 2252,'budget'] = 130

train.loc[train.index == 2256,'budget'] = 1 

train.loc[train.index == 2696,'budget'] = 10000000
test.loc[test.index== 3033,'budget'] = 250 

test.loc[test.index== 3051,'budget'] = 50

test.loc[test.index == 3084,'budget'] = 337

test.loc[test.index == 3224,'budget'] = 4  

test.loc[test.index == 3594,'budget'] = 25  

test.loc[test.index == 3619,'budget'] = 500  

test.loc[test.index == 3831,'budget'] = 3  

test.loc[test.index== 3935,'budget'] = 500  

test.loc[test.index == 4049,'budget'] = 995946 

test.loc[test.index== 4424,'budget'] = 3  

test.loc[test.index == 4460,'budget'] = 8  

test.loc[test.index == 4555,'budget'] = 1200000 

test.loc[test.index== 4624,'budget'] = 30 

test.loc[test.index== 4645,'budget'] = 500 

test.loc[test.index == 4709,'budget'] = 450 

test.loc[test.index == 4839,'budget'] = 7

test.loc[test.index== 3125,'budget'] = 25 

test.loc[test.index== 3142,'budget'] = 1

test.loc[test.index == 3201,'budget'] = 450

test.loc[test.index == 3222,'budget'] = 6

test.loc[test.index== 3545,'budget'] = 38

test.loc[test.index == 3670,'budget'] = 18

test.loc[test.index == 3792,'budget'] = 19

test.loc[test.index == 3881,'budget'] = 7

test.loc[test.index == 3969,'budget'] = 400

test.loc[test.index == 4196,'budget'] = 6

test.loc[test.index == 4221,'budget'] = 11

test.loc[test.index == 4222,'budget'] = 500

test.loc[test.index== 4285,'budget'] = 11

test.loc[test.index == 4319,'budget'] = 1

test.loc[test.index == 4639,'budget'] = 10

test.loc[test.index == 4719,'budget'] = 45

test.loc[test.index == 4822,'budget'] = 22

test.loc[test.index == 4829,'budget'] = 20

test.loc[test.index== 4969,'budget'] = 20

test.loc[test.index== 5021,'budget'] = 40 

test.loc[test.index== 5035,'budget'] = 1 

test.loc[test.index== 5063,'budget'] = 14 

test.loc[test.index == 5119,'budget'] = 2 

test.loc[test.index== 5214,'budget'] = 30 

test.loc[test.index== 5221,'budget'] = 50 

test.loc[test.index== 4903,'budget'] = 15

test.loc[test.index == 4983,'budget'] = 3

test.loc[test.index == 5102,'budget'] = 28

test.loc[test.index== 5217,'budget'] = 75

test.loc[test.index == 5224,'budget'] = 3 

test.loc[test.index== 5469,'budget'] = 20 

test.loc[test.index == 5840,'budget'] = 1 

test.loc[test.index == 5960,'budget'] = 30

test.loc[test.index == 6506,'budget'] = 11 

test.loc[test.index== 6553,'budget'] = 280

test.loc[test.index == 6561,'budget'] = 7

test.loc[test.index== 6582,'budget'] = 218

test.loc[test.index == 6638,'budget'] = 5

test.loc[test.index== 6749,'budget'] = 8 

test.loc[test.index==6759,'budget'] = 50 

test.loc[test.index == 6856,'budget'] = 10

test.loc[test.index== 6858,'budget'] =  100

test.loc[test.index == 6876,'budget'] =  250

test.loc[test.index == 6972,'budget'] = 1

test.loc[test.index== 7079,'budget'] = 8000000

test.loc[test.index == 7150,'budget'] = 118

test.loc[test.index == 6506,'budget'] = 118

test.loc[test.index == 7225,'budget'] = 6

test.loc[test.index == 7231,'budget'] = 85

test.loc[test.index == 5222,'budget'] = 5

test.loc[test.index == 5322,'budget'] = 90

test.loc[test.index == 5350,'budget'] = 70

test.loc[test.index == 5378,'budget'] = 10

test.loc[test.index== 5545,'budget'] = 80

test.loc[test.index == 5810,'budget'] = 8

test.loc[test.index== 5926,'budget'] = 300

test.loc[test.index== 5927,'budget'] = 4

test.loc[test.index== 5986,'budget'] = 1

test.loc[test.index == 6053,'budget'] = 20

test.loc[test.index== 6104,'budget'] = 1

test.loc[test.index == 6130,'budget'] = 30

test.loc[test.index == 6301,'budget'] = 150

test.loc[test.index == 6276,'budget'] = 100

test.loc[test.index == 6473,'budget'] = 100

test.loc[test.index== 6842,'budget'] = 30
new_data=pd.concat([train,test],axis=0)

new_data.head()
new_data.isnull().sum()
new_data[new_data['release_date'].isnull()]
new_data['release_date']=new_data['release_date'].fillna('3/20/01')
new_data.head()
new_data['release_month']=new_data['release_date'].str.split('/').apply(lambda x:x[0])

new_data['release_day']=new_data['release_date'].str.split('/').apply(lambda x:x[1])

new_data['release_year']=new_data['release_date'].str.split('/').apply(lambda x:x[2])

new_data[['release_month','release_day','release_year','release_date']].head()
new_data=new_data.drop('release_date',axis=1)
new_data['homepage_fact']=new_data['homepage'].apply(lambda x: 0 if x is np.nan  else 1)
a=new_data.loc[train.index].groupby('homepage_fact').revenue.mean()

sas.barplot([0,1],[a[0],a[1]])
new_data['homepage_end']=new_data[new_data['homepage'].notna()]['homepage'].str.findall(r'\.([a-z]+)(?:\/|$)').apply(lambda x:x[0])

new_data['homepage_end'].head()
new_data['homepage_end']=new_data['homepage_end'].fillna(0)
a=new_data.loc[train.index].groupby('homepage_end').revenue.mean()

sas.barplot(a.index,a)
page=pd.get_dummies(new_data['homepage_end'])

page.head()
new_data=new_data.drop('homepage',axis=1)
new_data.head()
new_data['poster_path'].describe()
new_data=new_data.drop('poster_path',axis=1)
new_data['len_overview']=new_data['overview'].fillna('NAN').apply(lambda x:len(x))
# new_data=new_data.drop('overview',axis=1)
new_data.head()
sns.relplot('budget','revenue',data=train)
new_data['geres_name']=new_data['genres'].str.findall(r'\'name\'\s?:\s?\'(\w+)\'')

new_data['geres_name'].head()
new_data=new_data.drop('genres',axis=1)
new_data.head()
country=new_data['production_countries'].str.findall(r'[A-Z]{2,5}')

new_data['production_countries']=country

new_data.head()
new_data['original_and_new']=new_data['original_title']==new_data['title']
new_data=new_data.drop('original_title',axis=1)

new_data=new_data.drop('title',axis=1)

new_data.head()
new_data=pd.concat([new_data,page],axis=1)

new_data.head()
new_data['production_companies']=new_data['production_companies'].str.findall(r'\'name\'?:\s?\'([A-Za-z]+)')

new_data.fillna('Unknow')
new_data['imdb_id'].describe()
new_data=new_data.drop('imdb_id',axis=1)
new_data['spoken_languages']=new_data['spoken_languages'].str.findall(r'\'[a-z]{2}\'')
new_data.head()
new_data['production_companies']=new_data['production_companies'].fillna('unknow')
new_data['spoken_languages']=new_data['spoken_languages'].fillna('unknow')
new_data.head()
new_data['Keywords']=new_data['Keywords'].str.findall(r'\'?:\s?\'([a-z]+\s?[a-z]+)\'').fillna('unkonw')
new_data.head()
m=list(new_data['cast'].str.findall(r'\'name\'?:\s?\'(\S+\s?\S+)\'').fillna('unknow'))

m
Counter([i for j in m for i in j]).most_common(15)


list_of_keywords=list(new_data['Keywords'])

plt.figure(figsize = (16, 12))

text = ' '.join(['_'.join(i.split(' ')) for j in list_of_keywords for i in j])

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top keywords')

plt.axis("off")

plt.show()