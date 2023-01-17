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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from ast import literal_eval
from datetime import datetime

train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')
test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')
sub=test.copy()
# train,test 행과 열
print (train.shape)
print (test.shape)
#train dataset에 컬럼당 결측치 개수
train.isnull().sum()
#test dataset에 컬럼당 결측치 개수
test.isnull().sum()
# train data 와 test data를 합함.
data=train.append(test)
data.reset_index(inplace=True)
data.drop('index',axis=1,inplace=True)
data.head(2)
data.shape
data.describe(include='all')
data.info()
#data의 dataset에 결측치 처리를 시작해보자.
data.isnull().sum()
# data의 컬럼당 결측치 비율
# 이 결과 값을 갖고 어떻게 처리할지 고민하자.
print ('The percentage of missing value of each column:')
print ('*'*50)
print (round(data.isnull().sum()/data.shape[0]*100,2))
data['cast'].fillna('0',inplace=True)
data['crew'].fillna('0',inplace=True)
# 아래의 컬럼은 1번째 레코드로 채워넣음.
data['genres'].fillna(data['genres'].mode()[0],inplace=True)
data['production_countries'].fillna(data['production_countries'].mode()[0],inplace=True)
data['production_companies'].fillna(data['production_companies'].mode()[0],inplace=True)
data['spoken_languages'].fillna(data['spoken_languages'].mode()[0],inplace=True)
data['release_date'].fillna('3/20/01',inplace=True)
data['runtime'].fillna(data['runtime'].mean(),inplace=True)
data['title'].fillna(data['original_title'],inplace=True)
data.isnull().sum()
#status 컬럼에서의 released의 값이 대부분이라 유용하지 않은 데이터
data['status'].value_counts()
data['belongs_to_collection'].fillna(0,inplace=True)
data['belongs_to_collection']=data['belongs_to_collection'].apply(lambda x:1 if x!=0 else x)
#분석에 필요 없는데이터는 삭제
notusing=['Keywords',
         'homepage',
         'id',
         'imdb_id',
         'original_language',
         'original_title',
         'overview',
         'poster_path',
         'status',
         'tagline']
data.drop(notusing,axis=1,inplace=True)
data.head(3)
def find_name(string):
    s=eval(string) # list of dict
    l=[]
    for i in s:
        l.append(i['name'])
    return l

def find_language(string):
        t=eval(string)
        l=[]
        for i in t:
            l.append(i['iso_639_1'])
        return l

def find_actors(string):
    if eval(string)==0:
        return 0
    else:
        t=eval(string)
        l=[]
        for i in t:
            l.append(i['name'])
        return l
data['cast']=data['cast'].apply(find_actors)
data['crew']=data['crew'].apply(find_actors)
data['genres']=data['genres'].apply(find_name)
data['production_companies']=data['production_companies'].apply(find_name)
data['production_countries']=data['production_countries'].apply(find_name)
data['spoken_languages']=data['spoken_languages'].apply(find_language)
data['no_of_cast']=data['cast'].apply(lambda x:len(x) if x!=0 else 0)
data['no_of_crew']=data['crew'].apply(lambda x:len(x) if x!=0 else 0)

data.drop(['cast','crew'],axis=1,inplace=True)

data.head()
data.head()
#대부분의 영화는 여러장르를 포함한다.
print ('Movies with each no. of genres')
print ('*'*50)
print (data['genres'].apply(lambda x:len(x)).value_counts())
#장르를 더미변수로  변환.
data=pd.get_dummies(data['genres'].apply(pd.Series).stack()).sum(level=0).merge(data,left_index=True,right_index=True)
data.head()
#각 장르의 평균 예산을 계산하고 예산에서 결측치를 채운다.
list_of_genres=[]
for i in data['genres']:
    for j in i:
        if j not in list_of_genres:
            list_of_genres.append(j)

d={}
for i in list_of_genres:
    genre=i
    mean_budget=data.groupby(i)['budget'].mean()
    d[genre]=mean_budget[1]
    
pd.Series(d).sort_values()
list_of_companies=[]
for i in data['production_companies']:
    for j in i:
        if j not in list_of_companies:
            list_of_companies.append(j)

list_of_countries=[]
for i in data['production_countries']:
    for j in i:
        if j not in list_of_countries:
            list_of_countries.append(j)
len(list_of_countries)

list_of_language=[]
for i in data['spoken_languages']:
    for j in i:
        if j not in list_of_language:
            list_of_language.append(j)
len(list_of_language)

print ('The total number of company occurs is {}'.format(len(list_of_companies)))
print ('The total number of country occurs is {}'.format(len(list_of_countries)))
print ('The total number of language occurs is {}'.format(len(list_of_language)))
# budget 0의 값을 nan으로 바꿈
data['budget'].replace(0,np.nan,inplace=True)
data[data['budget'].isnull()][['budget','genres']].head(10)
# 영화의 평균 예산값을 계산
def fill_budget(l):
    el=[]
    for i in l:
        if d[i] not in el:
            el.append(d[i])
    return (np.mean(el))
data['budget'].fillna(data['genres'].apply(fill_budget),inplace=True)
#대부분의 영화는 여러개의 회사가 같이 제작한다.
print ('Movies with each no. of production company')
print ('*'*50)
data['production_companies'].apply(lambda x:len(x)).value_counts()
# 영화는 대부분 2개국 안에서 촬영했다.
print ('Movies with each no. of production_countries')
print ('*'*50)
data['production_countries'].apply(lambda x:len(x)).value_counts()
#예산과 국가,회사에 대한 표
data['no_of_country']=data['production_countries'].apply(lambda x:len(x))
data['no_of_company']=data['production_companies'].apply(lambda x:len(x))
data[['budget','no_of_country','no_of_company']].corr()
data['release_date'].head()
#datetime함수를 사용해 2000년도대와 1900년도대를 잘 처리해야함.
def fix_year(x):
    year=x.split('/')[2]
    if int(year)>18:
        return x[:-2]+'20'+year
    else:
        return x[:-2]+'19'+year
data['release_date']=data['release_date'].apply(fix_year)
data['release_date']=pd.to_datetime(data['release_date'],infer_datetime_format=True)
data['month']=data['release_date'].dt.month
data['day']=data['release_date'].dt.day
data['weekday']=data['release_date'].dt.weekday

# 연도를 사용하면 1911년인지2011년인지 햇갈려서 year을 뺌
data['weekday']=data['weekday'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thur',4:'Fri',5:'Sat',6:'Sun'})
data['month']=data['month'].map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'July',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
data[['release_date','month','day','weekday']].head(10)
data.drop(['release_date'],axis=1,inplace=True)
data.iloc[:5,20:]
data['month'].apply(lambda x:len(x)).value_counts()
#영화의 90퍼센트 이상이 영어로 되어잇는데 외국어가 수입에 영향을 미치는지 알아보자
l=[]
for i in data['spoken_languages']:
    if 'en' in i:
        l.append(i)

len(l)/data.shape[0]
def en_or_not(l):
    if 'en' in l:
        return 1
    else:
        return 0
data['language_en']=data['spoken_languages'].apply(en_or_not)
data.drop('spoken_languages',axis=1,inplace=True)
#영화 만들어지는 곳도 80퍼센트가 미국에서 만들어졌다.
u=[]
for i in data['production_countries']:
    if 'United States of America' in i:
        u.append(i)
        
len(u)/data.shape[0]
def usa_or_not(l):
    if 'United States of America' in l:
        return 1
    else:
        return 0
data['produce_in_USA']=data['production_countries'].apply(usa_or_not)
data.drop('production_countries',axis=1,inplace=True)
#변수의 scale에 영향을 받지 않도록 변수를 정규화.
data['budget']=data['budget'].apply(lambda x:(x-np.min(data['budget']))/(np.max(data['budget']-np.min(data['budget']))))
data['popularity']=data['popularity'].apply(lambda x:(x-np.min(data['popularity']))/(np.max(data['popularity']-np.min(data['popularity']))))
data['runtime']=data['runtime'].apply(lambda x:(x-np.min(data['runtime']))/(np.max(data['runtime']-np.min(data['runtime']))))
data.set_index('title',inplace=True)
data.head()
data.drop('genres',axis=1,inplace=True)
#월 컬럼 더미화.
data=pd.get_dummies(data,columns=['month'])
data
#훈련데이터셋 사용
Train=data[data['revenue'].notnull()]
Train.head(5)
list_of_genres
g={}
for i in list_of_genres:
    mean_rev=Train.groupby(i)['revenue'].mean()
    g[i]=mean_rev[1]

g
#장르에 따른 예산
plt.figure(figsize=(20,8))
pd.Series(g).sort_values().plot.barh()
plt.title('Mean revenue of each genre',fontsize=20)
plt.xlabel('Revenue',fontsize=20)
#언어가 영어인 영화에 따른 수입
print (pd.DataFrame(Train.groupby('language_en')['revenue'].mean()))

plt.figure(figsize=(10,4))
Train.groupby('language_en')['revenue'].mean().sort_values().plot.barh()
plt.title('Mean revenue of is or is not foreign film.',fontsize=20)
plt.xlabel('Revenue',fontsize=20)
#시간에 따른 수입
plt.figure(figsize=(8,8))
plt.scatter(Train['runtime'],Train['revenue'])
plt.title('Scatter plot of runtime vs revenue',fontsize=20)
plt.xlabel('runtime',fontsize=20)
plt.ylabel('Revenue',fontsize=20)
#예산에 따른 수입
plt.figure(figsize=(8,8))
plt.scatter(Train['budget'],Train['revenue'])
plt.title('Scatter plot of budget vs revenue',fontsize=20)
plt.xlabel('budget',fontsize=20)
plt.ylabel('Revenue',fontsize=20)
Train
month=['Jan','Feb','Mar','Apr','May','Jun','July','Aug','Sep','Oct','Nov','Dec']
m={}
for i in month:
    mean=Train.groupby('month_'+i)['revenue'].mean()
    m[i]=mean[1]
pd.Series(m)
for i in month:
    print (i,Train['month_'+i].value_counts()[1])
#월별에 따른 수입통계.
plt.figure(figsize=(20,8))
pd.Series(m).plot.bar()
plt.title('Mean revenue of each month',fontsize=20)
plt.xlabel('Revenue',fontsize=20)
#날짜에 따른 수입통계.
plt.figure(figsize=(20,8))
Train.groupby('day')['revenue'].mean().sort_values().plot.bar()
plt.title('Mean revenue of each day',fontsize=20)
plt.xlabel('Revenue',fontsize=20)