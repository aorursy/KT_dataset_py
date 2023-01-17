# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Data Visualization

import matplotlib.pyplot as plt

import pydot

import statsmodels.api as sn

from mpl_toolkits.mplot3d import axes3d

import seaborn as sns

from wordcloud import WordCloud

from collections import Counter

from datetime import datetime

from tqdm import tqdm



# Sklearn

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.tree import export_graphviz

from tensorflow import keras



# Modeling

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

import lightgbm as lgb

from sklearn.preprocessing import normalize

from sklearn.model_selection import train_test_split

from keras import layers

from keras import models

from sklearn.preprocessing import LabelEncoder

from keras import regularizers

from keras.metrics import mean_squared_logarithmic_error

from sklearn.model_selection import KFold



#Miscellaneous

from tqdm import tqdm_notebook



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory 

from wordcloud import WordCloud

from collections import Counter

from sklearn import preprocessing

from datetime import datetime

import ast

from sklearn.preprocessing import normalize

from sklearn.model_selection import train_test_split

from keras import layers

from keras import models

from sklearn.preprocessing import LabelEncoder

from keras import regularizers

from keras.metrics import mean_squared_logarithmic_error

from sklearn.model_selection import KFold

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

random_seed = 2019



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%%time



train = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/train.csv')

test = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/test.csv')
train.head()
print(train.shape)

print(test.shape)
train.columns
test.columns
train.describe()
train.info()
sns.heatmap(train.isna(), yticklabels=False, cmap="viridis")
import ast



dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



def text_to_dict(df):

    for column in dict_columns:

        train[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

        

dfx = text_to_dict(train)

for col in dict_columns:

       train[col]=dfx[col]
train['belongs_to_collection'].apply(lambda x:len(x) if x!= {} else 0).value_counts()
collections=train['belongs_to_collection'].apply(lambda x : x[0]['name'] if x!= {} else '?').value_counts()[1:15]

sns.barplot(collections,collections.index)

plt.show()
train['tagline'].apply(lambda x:1 if x is not np.nan else 0).value_counts()
plt.figure(figsize=(10,10))

taglines=' '.join(train['tagline'].apply(lambda x:x if x is not np.nan else ''))



wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,

                      width=1200, height=1000).generate(taglines)

plt.imshow(wordcloud)

plt.title('Taglines')

plt.axis("off")

plt.show()
keywords=train['Keywords'].apply(lambda x: ' '.join(i['name'] for i in x) if x != {} else '')

plt.figure(figsize=(10,10))

data=' '.join(words for words in keywords)

wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,

                      width=1200, height=1000).generate(data)

plt.imshow(wordcloud)

plt.title('Taglines')

plt.axis("off")

plt.show()
x=train['production_companies'].apply(lambda x : [x[i]['name'] for i in range(len(x))] if x != {} else []).values

Counter([i for j in x for i in j]).most_common(30)
countries=train['production_countries'].apply(lambda x: [i['name'] for i in x] if x!={} else []).values

count=Counter([j for i in countries for j in i]).most_common(20)

sns.barplot([val[1] for val in count],[val[0] for val in count])
train['spoken_languages'].apply(lambda x:len(x) if x !={} else 0).value_counts()

lang=train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else [])

count=Counter([i for j in lang for i in j]).most_common(5)

sns.barplot([val[1] for val in count],[val[0] for val in count])
genre=train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else [])

count=Counter([i for j in genre for i in j]).most_common(10)

sns.barplot([val[1] for val in count],[val[0] for val in count])
dfx = text_to_dict(test)

for col in dict_columns:

       test[col]=dfx[col]
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

plt.title('skewed data')

sns.distplot(train['revenue'])

plt.subplot(1,2,2)

plt.title('log transformation')

sns.distplot(np.log(train['revenue']))

plt.show()
train['revenue'].describe()
plt.subplots(figsize=(10,5))

plt.subplot(1,2,1)

plt.hist(train['budget']+1,bins=10,color='g')

plt.title('skewed data')

plt.subplot(1,2,2)

plt.hist(np.log(train['budget']+1),bins=10,color='g')

plt.title('log transformation')

plt.show()
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

sns.scatterplot(train['budget'],train['revenue'])

plt.subplot(1,2,2)

sns.scatterplot(np.log1p(train['budget']),np.log1p(train['revenue']))

plt.show()
sns.set()

x = np.array(train["budget"])

y = np.array(train["revenue"])

fig = plt.figure(1, figsize=(10, 8))

sns.regplot(x, y)

plt.xlabel("Budget", fontsize=10)  

plt.ylabel("Revenue", fontsize=10)

plt.title("Link between revenue and budget", fontsize=10)
%%time



sns.pairplot(train)
plt.hist(train['popularity'],bins=30,color='violet')

plt.show()
sns.scatterplot(train['popularity'],train['revenue'],color='violet')

plt.show()
train.loc[train['id'] == 16,'revenue'] = 192864         

train.loc[train['id'] == 90,'budget'] = 30000000                  

train.loc[train['id'] == 118,'budget'] = 60000000       

train.loc[train['id'] == 149,'budget'] = 18000000       

train.loc[train['id'] == 313,'revenue'] = 12000000       

train.loc[train['id'] == 451,'revenue'] = 12000000      

train.loc[train['id'] == 464,'budget'] = 20000000       

train.loc[train['id'] == 470,'budget'] = 13000000       

train.loc[train['id'] == 513,'budget'] = 930000         

train.loc[train['id'] == 797,'budget'] = 8000000        

train.loc[train['id'] == 819,'budget'] = 90000000       

train.loc[train['id'] == 850,'budget'] = 90000000       

train.loc[train['id'] == 1007,'budget'] = 2              

train.loc[train['id'] == 1112,'budget'] = 7500000       

train.loc[train['id'] == 1131,'budget'] = 4300000        

train.loc[train['id'] == 1359,'budget'] = 10000000       

train.loc[train['id'] == 1542,'budget'] = 1             

train.loc[train['id'] == 1570,'budget'] = 15800000       

train.loc[train['id'] == 1571,'budget'] = 4000000        

train.loc[train['id'] == 1714,'budget'] = 46000000       

train.loc[train['id'] == 1721,'budget'] = 17500000       

train.loc[train['id'] == 1865,'revenue'] = 25000000      

train.loc[train['id'] == 1885,'budget'] = 12             

train.loc[train['id'] == 2091,'budget'] = 10             

train.loc[train['id'] == 2268,'budget'] = 17500000       

train.loc[train['id'] == 2491,'budget'] = 6              

train.loc[train['id'] == 2602,'budget'] = 31000000       

train.loc[train['id'] == 2612,'budget'] = 15000000       

train.loc[train['id'] == 2696,'budget'] = 10000000      

train.loc[train['id'] == 2801,'budget'] = 10000000       

train.loc[train['id'] == 335,'budget'] = 2 

train.loc[train['id'] == 348,'budget'] = 12

train.loc[train['id'] == 470,'budget'] = 13000000 

train.loc[train['id'] == 513,'budget'] = 1100000

train.loc[train['id'] == 640,'budget'] = 6 

train.loc[train['id'] == 696,'budget'] = 1

train.loc[train['id'] == 797,'budget'] = 8000000 

train.loc[train['id'] == 850,'budget'] = 1500000

train.loc[train['id'] == 1199,'budget'] = 5 

train.loc[train['id'] == 1282,'budget'] = 9              

train.loc[train['id'] == 1347,'budget'] = 1

train.loc[train['id'] == 1755,'budget'] = 2

train.loc[train['id'] == 1801,'budget'] = 5

train.loc[train['id'] == 1918,'budget'] = 592 

train.loc[train['id'] == 2033,'budget'] = 4

train.loc[train['id'] == 2118,'budget'] = 344 

train.loc[train['id'] == 2252,'budget'] = 130

train.loc[train['id'] == 2256,'budget'] = 1 

train.loc[train['id'] == 2696,'budget'] = 10000000
test.loc[test['id'] == 3033,'budget'] = 250 

test.loc[test['id'] == 3051,'budget'] = 50

test.loc[test['id'] == 3084,'budget'] = 337

test.loc[test['id'] == 3224,'budget'] = 4  

test.loc[test['id'] == 3594,'budget'] = 25  

test.loc[test['id'] == 3619,'budget'] = 500  

test.loc[test['id'] == 3831,'budget'] = 3  

test.loc[test['id'] == 3935,'budget'] = 500  

test.loc[test['id'] == 4049,'budget'] = 995946 

test.loc[test['id'] == 4424,'budget'] = 3  

test.loc[test['id'] == 4460,'budget'] = 8  

test.loc[test['id'] == 4555,'budget'] = 1200000 

test.loc[test['id'] == 4624,'budget'] = 30 

test.loc[test['id'] == 4645,'budget'] = 500 

test.loc[test['id'] == 4709,'budget'] = 450 

test.loc[test['id'] == 4839,'budget'] = 7

test.loc[test['id'] == 3125,'budget'] = 25 

test.loc[test['id'] == 3142,'budget'] = 1

test.loc[test['id'] == 3201,'budget'] = 450

test.loc[test['id'] == 3222,'budget'] = 6

test.loc[test['id'] == 3545,'budget'] = 38

test.loc[test['id'] == 3670,'budget'] = 18

test.loc[test['id'] == 3792,'budget'] = 19

test.loc[test['id'] == 3881,'budget'] = 7

test.loc[test['id'] == 3969,'budget'] = 400

test.loc[test['id'] == 4196,'budget'] = 6

test.loc[test['id'] == 4221,'budget'] = 11

test.loc[test['id'] == 4222,'budget'] = 500

test.loc[test['id'] == 4285,'budget'] = 11

test.loc[test['id'] == 4319,'budget'] = 1

test.loc[test['id'] == 4639,'budget'] = 10

test.loc[test['id'] == 4719,'budget'] = 45

test.loc[test['id'] == 4822,'budget'] = 22

test.loc[test['id'] == 4829,'budget'] = 20

test.loc[test['id'] == 4969,'budget'] = 20

test.loc[test['id'] == 5021,'budget'] = 40 

test.loc[test['id'] == 5035,'budget'] = 1 

test.loc[test['id'] == 5063,'budget'] = 14 

test.loc[test['id'] == 5119,'budget'] = 2 

test.loc[test['id'] == 5214,'budget'] = 30 

test.loc[test['id'] == 5221,'budget'] = 50 

test.loc[test['id'] == 4903,'budget'] = 15

test.loc[test['id'] == 4983,'budget'] = 3

test.loc[test['id'] == 5102,'budget'] = 28

test.loc[test['id'] == 5217,'budget'] = 75

test.loc[test['id'] == 5224,'budget'] = 3 

test.loc[test['id'] == 5469,'budget'] = 20 

test.loc[test['id'] == 5840,'budget'] = 1 

test.loc[test['id'] == 5960,'budget'] = 30

test.loc[test['id'] == 6506,'budget'] = 11 

test.loc[test['id'] == 6553,'budget'] = 280

test.loc[test['id'] == 6561,'budget'] = 7

test.loc[test['id'] == 6582,'budget'] = 218

test.loc[test['id'] == 6638,'budget'] = 5

test.loc[test['id'] == 6749,'budget'] = 8 

test.loc[test['id'] == 6759,'budget'] = 50 

test.loc[test['id'] == 6856,'budget'] = 10

test.loc[test['id'] == 6858,'budget'] =  100

test.loc[test['id'] == 6876,'budget'] =  250

test.loc[test['id'] == 6972,'budget'] = 1

test.loc[test['id'] == 7079,'budget'] = 8000000

test.loc[test['id'] == 7150,'budget'] = 118

test.loc[test['id'] == 6506,'budget'] = 118

test.loc[test['id'] == 7225,'budget'] = 6

test.loc[test['id'] == 7231,'budget'] = 85

test.loc[test['id'] == 5222,'budget'] = 5

test.loc[test['id'] == 5322,'budget'] = 90

test.loc[test['id'] == 5350,'budget'] = 70

test.loc[test['id'] == 5378,'budget'] = 10

test.loc[test['id'] == 5545,'budget'] = 80

test.loc[test['id'] == 5810,'budget'] = 8

test.loc[test['id'] == 5926,'budget'] = 300

test.loc[test['id'] == 5927,'budget'] = 4

test.loc[test['id'] == 5986,'budget'] = 1

test.loc[test['id'] == 6053,'budget'] = 20

test.loc[test['id'] == 6104,'budget'] = 1

test.loc[test['id'] == 6130,'budget'] = 30

test.loc[test['id'] == 6301,'budget'] = 150

test.loc[test['id'] == 6276,'budget'] = 100

test.loc[test['id'] == 6473,'budget'] = 100

test.loc[test['id'] == 6842,'budget'] = 30
release_dates = pd.read_csv('../input/externaldata/release_dates_per_country2.csv')

release_dates['id'] = range(1,7399)

train = pd.merge(train, release_dates, how='left', on=['id'])

test = pd.merge(test, release_dates, how='left', on=['id'])
train.columns
test.columns
corr_mat = train.corr()

corr_mat.revenue.sort_values(ascending=False)
train['revenue'].describe()
train.head()
sns.distplot(train['revenue'])
max_revenue= train[train['revenue']== max(train['revenue'])]

max_revenue.head()
Train = train.copy()

Train.sort_values('revenue',ascending=False,inplace=True)

Train =Train.head(15)

Train[['title', 'original_language','popularity','budget','genres','revenue','release_date','production_companies']]
min_revenue = train[train['revenue']== min(train['revenue'])]

min_revenue.sample()
cols =['revenue','budget','popularity','theatrical','runtime','release_year']

sns.heatmap(train[cols].corr())

plt.show()
%%time



sns.set()

x = np.array(train["budget"])

y = np.array(train["revenue"])

fig = plt.figure(1, figsize=(10, 8))

sns.regplot(x, y)

plt.xlabel("Budget", fontsize=10)  

plt.ylabel("Revenue", fontsize=10)

plt.title("Link between revenue and budget", fontsize=10)
%%time



sns.set()

x=train['revenue']

y=train['popularity']

plt.figure(figsize=(12,6))

sns.regplot(x,y)

plt.xlabel('popularity')

plt.ylabel('revenue')

plt.title('Relationship between popularity and revenue of a movie')
%%time



plt.figure(figsize=(15,12)) 

a1 = sns.swarmplot(x='original_language', y='revenue', 

                   data=train[(train['original_language'].isin((train['original_language'].value_counts()[:10].index.values)))])

a1.set_title("Plot revenue by original language in the movie", fontsize=20) 

a1.set_xticklabels(a1.get_xticklabels(),rotation=45) 

a1.set_xlabel('Original language', fontsize=18)

a1.set_ylabel('revenue', fontsize=18) 



plt.show()
%%time



plt.figure(figsize=(15,8))

sns.countplot(train.release_year)

plt.xticks(rotation=90)

plt.xlabel('Years')

plt.title('Number of movies launched every year')
train['revenue'] = np.log1p(train['revenue'])
x = train.budget.values.reshape(-1,1)

y = train.revenue

reg = LinearRegression().fit(x, y)
print(f'Regression Score: {reg.score(x, y)}')

print(f'Regression Coefficient: {reg.coef_[0]}')

print(f'Regression Intercept: {reg.intercept_}')
predictions = reg.predict(test['budget'].values.reshape(-1,1))
idm = test['id'] 

arrid = np.array(idm) 
Submission = pd.DataFrame({"id" : arrid, "revenue" : predictions}).to_csv("submission.csv", index=False)
submission = pd.read_csv('submission.csv', index_col=0)

submission.head(15)
df_train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

df_test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')

sam_sub = pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')

print( "train dataset:", df_train.shape,"\n","test dataset: ",df_test.shape,"\n","sample_submission dataset:", sam_sub .shape)
df_train.info()
df_train.head(5)
df_train.describe(include='all')
df_train.isna().sum().sort_values(ascending=False)
missing=df_train.isna().sum().sort_values(ascending=False)

sns.barplot(missing[:8],missing[:8].index)

plt.show()
plt.style.use('seaborn')
import ast

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

        

dfx = text_to_dict(df_train)

for col in dict_columns:

       df_train[col]=dfx[col]
df_train['belongs_to_collection'].apply(lambda x:len(x) if x!= {} else 0).value_counts()
collections=df_train['belongs_to_collection'].apply(lambda x : x[0]['name'] if x!= {} else '?').value_counts()[1:15]

sns.barplot(collections,collections.index)

plt.show()
df_train['tagline'].apply(lambda x:1 if x is not np.nan else 0).value_counts()
plt.figure(figsize=(10,10))

taglines=' '.join(df_train['tagline'].apply(lambda x:x if x is not np.nan else ''))



wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,

                      width=1200, height=1000).generate(taglines)

plt.imshow(wordcloud)

plt.title('Taglines')

plt.axis("off")

plt.show()
keywords=df_train['Keywords'].apply(lambda x: ' '.join(i['name'] for i in x) if x != {} else '')

plt.figure(figsize=(10,10))

data=' '.join(words for words in keywords)

wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,

                      width=1200, height=1000).generate(data)

plt.imshow(wordcloud)

plt.title('Taglines')

plt.axis("off")

plt.show()
x=df_train['production_companies'].apply(lambda x : [x[i]['name'] for i in range(len(x))] if x != {} else []).values

Counter([i for j in x for i in j]).most_common(20)
countries=df_train['production_countries'].apply(lambda x: [i['name'] for i in x] if x!={} else []).values

count=Counter([j for i in countries for j in i]).most_common(10)

sns.barplot([val[1] for val in count],[val[0] for val in count])
df_train['spoken_languages'].apply(lambda x:len(x) if x !={} else 0).value_counts()
lang=df_train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else [])

count=Counter([i for j in lang for i in j]).most_common(5)

sns.barplot([val[1] for val in count],[val[0] for val in count])
genre=df_train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else [])

count=Counter([i for j in genre for i in j]).most_common(10)

sns.barplot([val[1] for val in count],[val[0] for val in count])
dfx = text_to_dict(df_test)

for col in dict_columns:

       df_test[col]=dfx[col]
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

plt.title('skewed data')

sns.distplot(df_train['revenue'])

plt.subplot(1,2,2)

plt.title('log transformation')

sns.distplot(np.log(df_train['revenue']))

plt.show()
df_train['log_revenue']=np.log1p(df_train['revenue'])
plt.subplots(figsize=(10,5))

plt.subplot(1,2,1)

plt.hist(df_train['revenue'],bins=10,color='g')

plt.title('skewed data')

plt.subplot(1,2,2)

plt.hist(np.log(df_train['revenue']),bins=10,color='g')

plt.title('log transformation')

plt.show()
df_train['revenue'].describe()
plt.subplots(figsize=(10,5))

plt.subplot(1,2,1)

plt.hist(df_train['budget']+1,bins=10,color='g')

plt.title('skewed data')

plt.subplot(1,2,2)

plt.hist(np.log(df_train['budget']+1),bins=10,color='g')

plt.title('log transformation')

plt.show()
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

sns.scatterplot(df_train['budget'],df_train['revenue'])

plt.subplot(1,2,2)

sns.scatterplot(np.log1p(df_train['budget']),np.log1p(df_train['revenue']))

plt.show()
df_train['log_budget']=np.log1p(df_train['budget'])

df_test['log_budget']=np.log1p(df_train['budget'])
plt.hist(df_train['popularity'],bins=30,color='violet')

plt.show()
sns.scatterplot(df_train['popularity'],df_train['revenue'],color='violet')

plt.show()
def date(x):

    x=str(x)

    year=x.split('/')[2]

    if int(year)<19:

        return x[:-2]+'20'+year

    else:

        return x[:-2]+'19'+year

df_train['release_date']=df_train['release_date'].fillna('1/1/90').apply(lambda x: date(x))

df_test['release_date']=df_test['release_date'].fillna('1/1/90').apply(lambda x: date(x))
#from datetime import datetime

df_train['release_date']=df_train['release_date'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y'))

df_test['release_date']=df_test['release_date'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y'))
df_train['release_day']=df_train['release_date'].apply(lambda x:x.weekday())

df_train['release_month']=df_train['release_date'].apply(lambda x:x.month)

df_train['release_year']=df_train['release_date'].apply(lambda x:x.year)
df_test['release_day']=df_test['release_date'].apply(lambda x:x.weekday())

df_test['release_month']=df_test['release_date'].apply(lambda x:x.month)

df_test['release_year']=df_test['release_date'].apply(lambda x:x.year)
day=df_train['release_day'].value_counts().sort_index()

sns.barplot(day.index,day)

plt.gca().set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],rotation='45')

plt.ylabel('No of releases')
sns.catplot(x='release_day',y='revenue',data=df_train)

plt.gca().set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],rotation='90')

plt.show()
sns.catplot(x='release_day',y='runtime',data=df_train)

plt.gca().set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],rotation='90')

plt.show()
plt.figure(figsize=(10,15))

sns.catplot(x='release_month',y='revenue',data=df_train)

month_lst = ['January', 'February', 'March', 'April', 'May', 'June', 'July','August', 'September', 'October', 'November', 'December']

plt.gca().set_xticklabels(month_lst,rotation='90')

plt.show()
plt.figure(figsize=(15,8))

yearly=df_train.groupby(df_train['release_year'])['revenue'].agg('mean')

plt.plot(yearly.index,yearly)

plt.xlabel('year')

plt.ylabel("Revenue")

plt.savefig('fig')
plt.figure(figsize=(10,5))



plt.subplot(1,2,1)

sns.distplot(np.log1p(df_train['runtime'].fillna(0)))



plt.subplot(1,2,2)

sns.scatterplot(np.log1p(df_train['runtime'].fillna(0)),np.log1p(df_train['revenue']))
df_train['homepage'].value_counts().sort_values(ascending=False)[:10]
genres=df_train.loc[df_train['genres'].str.len()==1][['genres','revenue','budget','popularity','runtime']].reset_index(drop=True)

genres['genres']=genres.genres.apply(lambda x :x[0]['name'])
genres=genres.groupby(genres.genres).agg('mean')
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.barplot(genres['revenue'],genres.index)



plt.subplot(2,2,2)

sns.barplot(genres['budget'],genres.index)



plt.subplot(2,2,3)

sns.barplot(genres['popularity'],genres.index)



plt.subplot(2,2,4)

sns.barplot(genres['runtime'],genres.index)
crew=df_train['crew'].apply(lambda x:[i['name'] for i in x] if x != {} else [])

Counter([i for j in crew for i in j]).most_common(20)
cast=df_train['cast'].apply(lambda x:[i['name'] for i in x] if x != {} else [])

Counter([i for j in cast for i in j]).most_common(20)
def  prepare_data(df):

    df['_budget_runtime_ratio'] = (df['budget']/df['runtime']).replace([np.inf,-np.inf,np.nan],0)

    df['_budget_popularity_ratio'] = df['budget']/df['popularity']

    df['_budget_year_ratio'] = df['budget'].fillna(0)/(df['release_year']*df['release_year'])

    df['_releaseYear_popularity_ratio'] = df['release_year']/df['popularity']

    df['_releaseYear_popularity_ratio2'] = df['popularity']/df['release_year']

    df['budget']=np.log1p(df['budget'])

    

    df['collection_name']=df['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)

    df['has_homepage']=0

    df.loc[(pd.isnull(df['homepage'])),'has_homepage']=1

    

    le=LabelEncoder()

    le.fit(list(df['collection_name'].fillna('')))

    df['collection_name']=le.transform(df['collection_name'].fillna('').astype(str))

    

    le=LabelEncoder()

    le.fit(list(df['original_language'].fillna('')))

    df['original_language']=le.transform(df['original_language'].fillna('').astype(str))

    

    df['_num_Keywords'] = df['Keywords'].apply(lambda x: len(x) if x != {} else 0)

    df['_num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)

    

    df['isbelongto_coll']=0

    df.loc[pd.isna(df['belongs_to_collection']),'isbelongto_coll']=1

    

    df['isTaglineNA'] = 0

    df.loc[df['tagline'] == 0 ,"isTaglineNA"] = 1 



    df['isOriginalLanguageEng'] = 0 

    df.loc[ df['original_language'].astype(str) == "en" ,"isOriginalLanguageEng"] = 1

    

    df['ismovie_released']=1

    df.loc[(df['status']!='Released'),'ismovie_released']=0

    

    df['no_spoken_languages']=df['spoken_languages'].apply(lambda x: len(x))

    df['original_title_letter_count'] = df['original_title'].str.len() 

    df['original_title_word_count'] = df['original_title'].str.split().str.len() 





    df['title_word_count'] = df['title'].str.split().str.len()

    df['overview_word_count'] = df['overview'].str.split().str.len()

    df['tagline_word_count'] = df['tagline'].str.split().str.len()

    

    

    df['collection_id'] = df['belongs_to_collection'].apply(lambda x : np.nan if len(x)==0 else x[0]['id'])

    df['production_countries_count'] = df['production_countries'].apply(lambda x : len(x))

    df['production_companies_count'] = df['production_companies'].apply(lambda x : len(x))

    df['cast_count'] = df['cast'].apply(lambda x : len(x))

    df['crew_count'] = df['crew'].apply(lambda x : len(x))



    df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

    df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

    df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))



    for col in  ['genres', 'production_countries', 'spoken_languages', 'production_companies'] :

        df[col] = df[col].map(lambda x: sorted(list(set([n if n in train_dict[col] else col+'_etc' for n in [d['name'] for d in x]])))).map(lambda x: ','.join(map(str, x)))

        temp = df[col].str.get_dummies(sep=',')

        df = pd.concat([df, temp], axis=1, sort=False)

    df.drop(['genres_etc'], axis = 1, inplace = True)

    

    cols_to_normalize=['runtime','popularity','budget','_budget_runtime_ratio','_budget_year_ratio','_budget_popularity_ratio','_releaseYear_popularity_ratio',

    '_releaseYear_popularity_ratio2','_num_Keywords','_num_cast','no_spoken_languages','original_title_letter_count','original_title_word_count',

    'title_word_count','overview_word_count','tagline_word_count','production_countries_count','production_companies_count','cast_count','crew_count',

    'genders_0_crew','genders_1_crew','genders_2_crew']

    for col in cols_to_normalize:

        print(col)

        x_array=[]

        x_array=np.array(df[col].fillna(0))

        X_norm=normalize([x_array])[0]

        df[col]=X_norm

    

    df = df.drop(['belongs_to_collection','genres','homepage','imdb_id','overview','id'

    ,'poster_path','production_companies','production_countries','release_date','spoken_languages'

    ,'status','title','Keywords','cast','crew','original_language','original_title','tagline', 'collection_id'

    ],axis=1)

    

    df.fillna(value=0.0, inplace = True) 



    return df
def get_json(df):

    global dict_columns

    result=dict()

    for col in dict_columns:

        d=dict()

        rows=df[col].values

        for row in rows:

            if row is None: continue

            for i in row:

                if i['name'] not in d:

                    d[i['name']]=0

                else:

                    d[i['name']]+=1

            result[col]=d

    return result

    

    



    

train_dict=get_json(df_train)

test_dict=get_json(df_test)
df_train.shape
df_test.shape
for col in dict_columns :

    

    remove = []

    train_id = set(list(train_dict[col].keys()))

    test_id = set(list(test_dict[col].keys()))   

    

    remove += list(train_id - test_id) + list(test_id - train_id)

    for i in train_id.union(test_id) - set(remove) :

        if train_dict[col][i] < 10 or i == '' :

            remove += [i]

    for i in remove :

        if i in train_dict[col] :

            del train_dict[col][i]

        if i in test_dict[col] :

            del test_dict[col][i]
df_test['revenue']=np.nan

all_data=prepare_data((pd.concat([df_train,df_test]))).reset_index(drop=True)

train=all_data.loc[:df_train.shape[0]-1,:]

test=all_data.loc[df_train.shape[0]:,:]

print(train.shape)
all_data.head()
train.drop('revenue',axis=1,inplace=True)
y=train['log_revenue']

X=train.drop(['log_revenue'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

kfold=KFold(n_splits=3,random_state=42,shuffle=True)
X.columns
from keras import optimizers



    

model=models.Sequential()

model.add(layers.Dense(356,activation='relu',kernel_regularizer=regularizers.l1(.001),input_shape=(X.shape[1],)))

model.add(layers.Dense(256,kernel_regularizer=regularizers.l1(.001),activation='relu'))

model.add(layers.Dense(1))

model.compile(optimizer=optimizers.rmsprop(lr=.001),loss='mse',metrics=['mean_squared_logarithmic_error'])
model.summary()
keras.utils.plot_model(model)
epochs=50
hist=model.fit(X_train,y_train,epochs=50,verbose=0,validation_data=(X_test,y_test))
hist.params
pd.DataFrame(hist.history).plot(figsize=(8,5))

plt.grid(True)

plt.gca().set_ylim(0,1)

plt.show()
mae=hist.history['mean_squared_logarithmic_error']

plt.plot(range(1,epochs),mae[1:],label='mae')

plt.xlabel('epochs')

plt.ylabel('mean_abs_error')

mae=hist.history['val_mean_squared_logarithmic_error']

plt.plot(range(1,epochs),mae[1:],label='val_mae')

plt.legend()
mae=hist.history['loss']

plt.plot(range(1,epochs),mae[1:],label='trraining loss')

plt.xlabel('epochs')

plt.ylabel('loss')

mae=hist.history['val_loss']

plt.plot(range(1,epochs),mae[1:],label='val_loss')

plt.legend()
test.drop(['revenue','log_revenue'],axis=1,inplace=True)
y=np.expm1(model.predict(test))

df_test['revenue']=y

df_test[['id','revenue']].to_csv('submission2.csv',index=False)
submission = pd.read_csv('submission2.csv', index_col=0)

submission.head(15)