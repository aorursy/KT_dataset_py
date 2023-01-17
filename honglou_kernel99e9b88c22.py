# Libraries



import numpy as np

import pandas as pd

pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sas

%matplotlib inline

plt.style.use('ggplot')

from wordcloud import WordCloud

from collections import Counter

from nltk.corpus import stopwords

from nltk.util import ngrams

stop = set(stopwords.words('english'))

import os

import plotly.offline as py

import plotly.graph_objs as go

import plotly.tools as tls

import socket

from urllib.request import urlopen

from PIL import Image

import time

import warnings

warnings.filterwarnings("ignore")
train=pd.read_csv('../input/train.csv',index_col=0)

test=pd.read_csv('../input/test.csv',index_col=0)
train.head()
test['revenue']=-99
test.head()
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
new_data['release_year']=pd.to_datetime(new_data['release_date']).dt.year

new_data['release_month']=pd.to_datetime(new_data['release_date']).dt.month

new_data['release_day']=pd.to_datetime(new_data['release_date']).dt.day
new_data['release_year'].loc[new_data['release_year']>=2018]-=100
new_data[['release_date','release_year','release_month','release_day']].head()
new_data=new_data.drop('release_date',axis=1)
new_data['homepage_fact']=new_data['homepage'].apply(lambda x: 0 if x is np.nan  else 1)
a=new_data.loc[train.index].groupby('homepage_fact').revenue.mean()

sas.barplot([0,1],[a[0],a[1]])
new_data['homepage'].head()
new_data['homepage_end']=new_data[new_data['homepage'].notna()]['homepage'].str.findall(r'\.([a-z]+)(?:\/|$)').apply(lambda x:x[0])

new_data['homepage_end'].head()
new_data['homepage_end']=new_data['homepage_end'].fillna('unknow')
a=new_data.loc[train.index].groupby('homepage_end').revenue.mean().sort_values(ascending=False)

plt.figure(figsize=(20,10))

sas.barplot(a.index,a)
page=pd.get_dummies(new_data['homepage_end'])

page.head()
new_data=new_data.drop('homepage',axis=1)
new_data['poster_path'].describe()
new_data=new_data.drop('poster_path',axis=1)
new_data['len_overview']=new_data['overview'].fillna('NAN').apply(lambda x:len(x))
new_data.plot(x="len_overview",y="revenue", kind="scatter",figsize=(12,8))
len_rew_sort=new_data['len_overview'].sort_values(ascending=True)

len_rew_sort.head()
length=len(new_data['len_overview'])

m=0.1

n=0.1

arr_len_ove=[]

for i in range(1,11):

    arr_len_ove.append(round(length*m))

    m+=n

arr_len_ove
exam=len_rew_sort.iloc[1480:2959].value_counts()

sas.barplot(exam.index,exam.values)
qu_arr=[]

for i in range(10):

    if i==0:

            x=len_rew_sort.iloc[:arr_len_ove[0]].mean()

    else:

        x=len_rew_sort.iloc[arr_len_ove[i-1]:arr_len_ove[i]].mean()

    qu_arr.append(round(x))
for i in range(10):

    qu=qu_arr[i]

    if i==0:

        new_data['len_overview'].loc[(new_data['len_overview']<len_rew_sort.iloc[arr_len_ove[i]-1])]=qu

    else:

        new_data['len_overview'].loc[(new_data['len_overview']<len_rew_sort.iloc[arr_len_ove[i]-1])&(new_data['len_overview'] >qu_arr[i-1])]=qu

    print(i,qu)
np.sort(new_data['len_overview'].unique())
len_ove_agg=new_data.groupby('len_overview').revenue.aggregate(['min','max','std'])
len_ove_agg.plot()
new_data.head()
sas.relplot('budget','revenue',data=train)
new_data['geres_name']=new_data['genres'].str.findall(r'\'name\'\s?:\s?\'(\w+)\'')

new_data['geres_name'].head()
new_data=new_data.drop('genres',axis=1)
country=new_data['production_countries'].str.findall(r'[A-Z]{2,5}')

new_data['production_countries']=country

new_data['original_and_new']=new_data['original_title']==new_data['title']
new_data=new_data.drop('original_title',axis=1)

# new_data=new_data.drop('title',axis=1)

new_data.head()
new_data=pd.concat([new_data,page],axis=1)
new_data['production_companies']=new_data['production_companies'].str.findall(r'\'name\'?:\s?\'([A-Za-z]+)')

new_data.fillna('Unknow')

print('接下来就是地图可视化了')
new_data['imdb_id'].describe()
new_data=new_data.drop('imdb_id',axis=1)
new_data['spoken_languages']=new_data['spoken_languages'].str.findall(r'\'([a-z]{2})\'')
new_data['production_companies']=new_data['production_companies'].fillna('unknow')
new_data['spoken_languages']=new_data['spoken_languages'].fillna('unknow')
new_data['Keywords']=new_data['Keywords'].str.findall(r'\'?:\s?\'([a-z]+\s?[a-z]+)\'').fillna('unkonw')
new_data.head()
m=list(new_data['cast'].str.findall(r'\'name\'?:\s?\'(\S+\s?\S+)\'').fillna('unknow'))
plt.figure(figsize = (16, 12))

list_of_keywords=list(new_data['Keywords'])

text = ' '.join(['_'.join(i.split(' ')) for j in list_of_keywords for i in j])

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title='Top keywords'

plt.axis("off")

plt.show()
new_data['cast']=new_data['cast'].fillna('unknow')
m=list(new_data['cast'].str.replace(r'\s+\'order\':\s?\d+\S?\s\'profile_path\'?:\s?\'','').str.findall('\'name\'?:\s?\'(\D+)\'\S?\s?\'?(\/\w+.jpg)\'?\}'))

d = Counter([i for j in m for i in j]).most_common(16)

d

fig = plt.figure(figsize=(20, 12))

for i,p in enumerate([j[0] for j in d]):

    p=str(p).split(',')

    m=p[0][2:]

    m=m[:len(m)-1]

    p=p[1]

    p=p[2:len(p)-2]

    print("https://image.tmdb.org/t/p/w600_and_h900_bestv2%s"%p)

    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])

    im = Image.open(urlopen("https://image.tmdb.org/t/p/w600_and_h900_bestv2%s"%p))         

    plt.imshow(im)

    ax.set_title(m)

    
new_data['status']=new_data['status'].fillna('Unknow')
new_data['overview']=new_data['overview'].fillna('')
plt.figure(figsize = (12, 12))

text = ' '.join(train['original_title'].values)

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title='Top words in titles'

plt.axis("off")

plt.show()
plt.figure(figsize = (12, 12))

text = ' '.join(train['overview'].fillna('').values)

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title='Top words in overview'

plt.axis("off")

plt.show()
from sklearn.linear_model import  LinearRegression

import eli5

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

log_re=np.log1p(train['revenue'])

v1=TfidfVectorizer( sublinear_tf=True,analyzer='word',

            token_pattern=r'\w{1,}',

            ngram_range=(1, 2),

            min_df=5)

overview_text = v1.fit_transform(train['overview'].fillna(''))

linre=LinearRegression()

linre.fit(overview_text,log_re)

eli5.show_weights(linre, vec=v1, top=20, feature_filter=lambda x: x !='<BIAS>')
# print('Target value:',log_re[1000])

# eli5.show_prediction(linre, doc=train['overview'].values[1000], vec=v1)

plt.figure(figsize = (12, 12))

text_ta=' '.join(new_data['tagline'].fillna('').values)

wordcloud=WordCloud(max_font_size=None,background_color='white',width=1200,height=1000).generate(text_ta)

plt.imshow(wordcloud)

plt.title='Top words in tagline'

plt.axis("off")

plt.show()


# new_data['crew']=new_data['crew'].str.replace(',','').str.replace('}','').str.replace(r'\S+credit_id\S+:\s?\s?\S?\'(\S+)\S+\'','').str.findall(r'\'job\S?\'\s?:\s?\'(\S+\s?\S+)\S\s?\S?\'name\S?\'\s?:\s?\'(\S+\s?\S+)\'\s?\S?\'profile_path\S?\':\s+\'?(\S+)\S?\s?\s?\S?department\S?\'?:\s?\S?(\S+)\s?\S?\'?') 
new_data['crew']=new_data['crew'].str.replace(',','').str.replace('}','').str.replace(r'\'gender\'\s?\S?:\s?\S?\s+\'id\'\s?:\s?\d+','').str.findall(r'\'department\'\s?\S?:\s+\S?(\S+)\S?\s?\S?\s+\'job\S?\'\s?:\s?\'(\D+)\S\s?\S?\'name\S?\'\s?:\s?\'(\D+)\'\s?\S?\'profile_path\S?\':\s+\'?(\S+)\S?\s?\s?\S?') 
new_data['crew'][1][:5]
train['crew'][1]
# new_data['crew'].str.replace(',','').str.replace('}','').str.replace(r'\S+credit_id\S+:\s?\s?\S?\'(\S+)\S+\'','').str.findall(r'\'job\S?\'\s?:\s?\'(\S+\s?\S+)\S\s?\S?\'name\S?\'\s?:\s?\'(\S+\s?\S+)\'\s?\S?\'profile_path\S?\':\s+\S?(\S+)\S?\s?\s?\S?department\S?\'?:\s?\S?(\S+)\s?\S?')[1] 
# train['crew'].str.replace(',','').str.replace('}','').str.replace(r'\S+credit_id\S+:\s?\s?\S?\'(\S+)\S+\'','').str.findall(r'\'job\S?\'\s?:\s?\'(\S+\s?\S+)\S\s?\S?\'name\S?\'\s?:\s?\'(\S+\s?\S+)\'\s?\S?\'profile_path\S?\':\s+\'?(\S+)\S?\s?\s?\S?department\S?\'?:\s?\S?(\S+)\s?\S?')[1] 
new_train=new_data.loc[np.array(train.index)]
new_train['production_countries']=new_train['production_countries'].fillna('')
new_train['production_countries']=new_train['production_countries'].apply(lambda x:'_'.join(x))
count=new_train['production_countries'].value_counts()

count=count[count>5]
plt.figure(figsize=(20,10))

plt.bar(count.index,count.values)

plt.show()
bud=new_train.groupby('production_countries').budget.mean()
bud=bud.sort_values(ascending=False)[:10]
plt.figure(figsize=(20,12))

plt.bar(bud.index,bud.values)

plt.show()
rev=new_train.groupby('production_countries').revenue.mean()
rev=rev.sort_values(ascending=False)[:10]
plt.figure(figsize=(20,12))

plt.bar(rev.index,rev.values)

plt.show()
new_train['release_month'].unique()
new_train['production_countries'].loc[new_train['production_countries']=='ET']='Ethiopia'

along_co=new_train[new_train['production_countries'].apply(lambda x:1 if len(x)==2 else 0)==1]

along_co.head()
count_rev=along_co[['production_countries','revenue']]
count_rev['production_countries'].unique()
import geopandas as ge

boros = ge.read_file(ge.datasets.get_path("nybb"))

m1=ge.tools.geocode(count_rev['production_countries'].unique(), provider='nominatim', user_agent="my-application")
mn=m1.copy()

mn
world = ge.read_file(ge.datasets.get_path('naturalearth_lowres'))
from mpl_toolkits.axes_grid1 import make_axes_locatable

dd=count_rev['production_countries'].value_counts()

mn['values']=dd.values

mn.head()
mn['address'][0]='United States'

mn['address'][2]='Korea'

mn['address'][19]='Sweden'

mn['address'][3]='Serbia'

mn['address'][4]='United Kingdom'

mn['address'][5]='France'

mn['address'][6]='New Zealand'

mn['address'][7]='Italy'

mn['address'][8]='Belgium'

mn['address'][9]='Czech Rep.'

mn['address'][38]='Poland'

mn['address'][37]='Indonesia'

mn['address'][33]='Romania'

mn['address'][35]='Hungary'

mn['address'][14]='China'

mn['address'][12]='Spain'

mn['address'][11]='Russia'

mn['address'][13]='Turkey'

mn['address'][17]='Iran'

mn['address'][18]='Japan'

mn['address'][20]='Philippines'

mn['address'][24]='Mexico'

mn['address'][28]='Pakistan'

mn['address'][30]='Bulgaria'

mn['address'][31]='Ukraine'

mn['address'][21]='Brazil'

mn['address'][22]='Netherlands'

mn['address'][25]='Germany'

mn['address'][27]='Finland'

mn['address'][29]='Norway'

mn['address'][32]='Denmark'
mn=mn.drop(10)

mn=mn.drop(34)
for i in range(39):

    if i==10 or i==34:

        continue;

    d=world[world['name']==mn['address'][i]]['geometry'].values[0]

    mn['geometry'][i]=d
mn.head()
fig, ax = plt.subplots()



ax.set_aspect('equal')

divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.1)

world.plot(ax=ax, color='white', edgecolor='black')

m1.plot(ax=ax,color='blue')

mn.plot(column='values',ax=ax, legend=True,cmap=plt.get_cmap('rainbow'))

TS = train.loc[:,["original_title","release_date","budget","runtime","revenue"]]

TS.dropna()



TS.release_date = pd.to_datetime(TS.release_date)

TS.loc[:,"Year"] = TS["release_date"].dt.year

TS.loc[:,"Month"] = TS["release_date"].dt.month

TS = TS[TS.Year<2018]
titles = TS.groupby("Year")["original_title"].count()

titles.plot(figsize=(15,8))

plt.xlabel="Year of release"

plt.ylabel="Number of titles released"

plt.xticks(np.arange(1970,2025,5))

plt.show()
rev=TS.groupby('Year')['revenue'].aggregate(['min','mean','max','std'])

rev.plot(figsize=(20,10))

plt.xlabel="Year of release"

plt.ylabel="Revenue"

plt.xticks(np.arange(1970,2025,5))

plt.show()
bud=TS.groupby('Year')['budget'].aggregate(['min','max','mean','std'])

bud.plot(figsize=(20,10))

plt.xlabel="Year of release"

plt.ylabel="Budget"

plt.xticks(np.arange(1970,2025,5))

plt.show()
TS[TS['budget']==0].head()
runtimes = TS.groupby("Year")["runtime"].aggregate(["min","mean","max"])

runtimes.plot(figsize=(15,8))

plt.xlabel="Year of release"

plt.ylabel="Runtime"

plt.xticks(np.arange(1970,2025,5))

plt.show()
r_zeros = TS[TS.runtime==0]

r_zeros.head()
train.plot(x="runtime",y="budget", kind="scatter",figsize=(12,8))

plt.show()
train.plot(x="runtime",y="revenue", kind="scatter",figsize=(12,8))

plt.show()
train.plot(x="popularity",y="budget", kind="scatter",figsize=(12,8))

plt.show()
pop = train[train.popularity<50]

pop=pop[pop['budget']!=0]

pop.plot(x="budget",y="popularity", kind="scatter",figsize=(12,8))

plt.show()
train.head()
top3 = train.sort_values(by='popularity',ascending=False)[:10]

id3=top3[['title','poster_path','revenue']]
fig = plt.figure(figsize=(20, 12))

cont=0

for i in id3.index:

    p=id3.loc[i]

    ax = fig.add_subplot(4, 4, cont+1, xticks=[], yticks=[])

    print("https://image.tmdb.org/t/p/w600_and_h900_bestv2%s"%p['poster_path'])

    im = Image.open(urlopen("https://image.tmdb.org/t/p/w600_and_h900_bestv2%s"%p['poster_path']))

    plt.imshow(im)

    ax.set_title(p['title'])

    cont+=1
old_title=new_data['title'].value_counts()
a=old_title[old_title.values==3].index

b=old_title[old_title==2].index
new_data['fan_pai_2']=new_data['title'].apply(lambda x:1 if x in a else 0)

new_data['pan_pai_3']=new_data['title'].apply(lambda x:1 if x in b else 0)
new_data.loc[new_data['title'].fillna('un').str.contains('Planet of the Apes')]
top10 = train.sort_values(by='revenue',ascending=False)[:10]

id10=top10[['title','poster_path','revenue']]
fig = plt.figure(figsize=(20, 12))

cont=0

for i in id10.index:

    p=id10.loc[i]

    ax = fig.add_subplot(4, 4, cont+1, xticks=[], yticks=[])

    print("https://image.tmdb.org/t/p/w600_and_h900_bestv2%s"%p['poster_path'])

    im = Image.open(urlopen("https://image.tmdb.org/t/p/w600_and_h900_bestv2%s"%p['poster_path']))

    plt.imshow(im)

    ax.set_title(p['title'])

    cont+=1
title_count=train['title'].value_counts()

len(title_count[title_count.values!=1])
train[train['title'].str.contains('Furious')]['title']

test[test['title'].fillna('Unknow').str.contains('Furious')]['title']
list_crew=list(new_data['crew'].fillna('Unknow'))

d = Counter([j for i in list_crew for j in i]).most_common(17)
d.remove(d[7])
fig = plt.figure(figsize=(20, 16))

for i in range(16):

    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])

    p=d[i][0][3]

    p=p[:len(p)-1]

    print("https://image.tmdb.org/t/p/w600_and_h900_bestv2%s"%p)

    if p!='Non':

        im = Image.open(urlopen("https://image.tmdb.org/t/p/w600_and_h900_bestv2%s"%p))

    else:

        im = Image.new('RGB', (5, 5))

    plt.imshow(im)

    ax.set_title(' \n Name: %s\n Job:%s'%(d[i][0][2],d[i][0][1]))
ts=new_train.groupby('release_month').revenue.median()
plt.figure(figsize=(10,5))

sas.pointplot(ts.index,ts.values,ylabel='revenue')
mon_is=[1,6,7,9,10,12]

for i in mon_is:

    new_data['mon_is_'+str(i)]=new_data['release_month'].apply(lambda x:1 if x==i else 0)
plt.figure(figsize=(16,6))

plt.plot(ts.rolling(window=2,center=False).mean(),label='Rolling Mean');

# plt.plot(ts.rolling(window=2,center=False).std(),label='Rolling sd');

plt.plot(ts.rolling(window=3,center=False).mean(),label='Rolling Mean3')

plt.plot(ts.rolling(window=6,center=False).mean(),label='Rolling Mean6')

plt.legend();
plt.plot(ts.diff(2))

plt.show()
tt=TS.groupby('Year').revenue.median()
plt.figure(figsize=(15,8))

plt.plot(tt.index,tt.values)

plt.xlabel='year'

plt.ylabel='revenue'

plt.title='revenue_of_year'

plt.show()
tt1=tt.diff(1)

plt.figure(figsize=(15,8))

plt.plot(tt1.index,tt1.values)

plt.xlabel='year'

plt.ylabel='revenue'

plt.title='revenue_of_year'

plt.show()
new_data.head()
new_data['spoken_languages_count']=new_data['spoken_languages'].apply(lambda x:len(x))
new_data.loc[train.index].groupby('spoken_languages_count').revenue.median()
status_du=pd.get_dummies(new_data['status'])
new_data=pd.concat([new_data,status_du],axis=1)

new_data=new_data.drop(['status'],axis=1)

new_data.head()
new_data=new_data.drop(['original_language'],axis=1)
sum(new_data['geres_name'].isna())
new_data['geres_name']=new_data['geres_name'].fillna('Unknow')
new_data['geres_name_count']=new_data['geres_name'].apply(lambda x:len(x))
ge_name_count_rev=new_data.loc[train.index].groupby(['geres_name_count']).revenue.median().sort_values()
sas.barplot(ge_name_count_rev.index,ge_name_count_rev.values)
new_data=new_data.drop('homepage_end',axis=1)
new_data['original_and_new']=new_data['original_and_new'].apply(lambda x: 1 if x else 0)
plt.figure(figsize = (16, 12))

list_of_keywords=list(new_data['geres_name'])

text = ' '.join(['_'.join(i.split(' ')) for j in list_of_keywords for i in j])

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title='Top geres_name'

plt.axis("off")

plt.show()
list_geres_name=['Comedy','Thriller','Action','Drama','Romance']
for i in list_geres_name:

    new_data['geres_name'+'_'+i]=new_data['geres_name'].apply(lambda x: 1 if i in x else 0 )
new_data.head()
new_data=new_data.drop('geres_name',axis=1)
new_data['production_countries']=new_data['production_countries'].fillna('QQ')
plt.figure(figsize = (16, 12))

list_of_keywords=list(new_data['production_countries'])

text = ' '.join(['_'.join(i.split(' ')) for j in list_of_keywords for i in j])

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title='production_countries'

plt.axis("off")

plt.show()
new_data['production_countries'].loc[new_data['production_countries']=='Unknow']['production_countries']='QQ'
list_pro_coun=list(new_data['production_countries'])

d = Counter([j for i in list_pro_coun for j in i]).most_common(5)

d
for i in d:

     new_data['production_cou_name'+'_'+i[0]]=new_data['production_countries'].apply(lambda x: 1 if i[0] in x else 0 )
new_data.head()
new_data['pro_country_count']=new_data['production_companies'].apply(lambda x:len(x))
pro_count_rev=new_data.loc[train.index].groupby('pro_country_count').revenue.median()
sas.barplot(pro_count_rev.index,pro_count_rev.values)
new_data.head()
new_data=new_data.drop('production_countries',axis=1)
new_data['production_companies'].loc[new_data['production_companies']=='unknow']='u'
plt.figure(figsize = (16, 12))

list_of_pro_com=list(new_data['production_companies'])

text = ' '.join(['_'.join(i.split(' ')) for j in list_of_pro_com for i in j])

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title='production_countries'

plt.axis("off")

plt.show()
d = Counter([j for i in list_of_pro_com for j in i]).most_common(11)

d
d.remove(('u',414))

dd=[]

for i in d:

    dd.append(i[0])
fig = plt.figure(figsize=(20, 16))



for i in range(len(d)):

    m=d[i][0]

    new_data['is_'+m]=new_data['production_companies'].apply(lambda x:1 if m in x else 0)

    com=new_data[new_data['is_'+m]==1]

    val=com.groupby('release_year').budget.median()

    cou=com.release_year.value_counts()

    plt.plot(val.index,val.values)

    plt.legend(dd)  
plt.figure(figsize = (16, 12))

list_of_spk_lag=list(new_data['spoken_languages'].loc[new_data['spoken_languages']!='unknow'])

text = ' '.join(['_'.join(i.split(' ')) for j in list_of_spk_lag for i in j])

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title='spk_lag'

plt.axis("off")

plt.show()
d = Counter([j for i in list_of_spk_lag for j in i]).most_common(5)

d
dd=[]

for i in d:

    dd.append(i[0])
fig = plt.figure(figsize=(20, 16))



for i in range(len(d)):

    m=d[i][0]

    new_data['is_'+m]=new_data['spoken_languages'].apply(lambda x:1 if m in x else 0)

    com=new_data[new_data['is_'+m]==1]

    val=com.groupby('release_year').budget.median()

    cou=com.release_year.value_counts()

    plt.plot(val.index,val.values)

    plt.legend(dd)  

    plt.title='psk_release_median'
new_data=new_data.drop('spoken_languages',axis=1)
new_data.head()
new_data['time_budget']=new_data['budget']/new_data['runtime']
new_data.head()