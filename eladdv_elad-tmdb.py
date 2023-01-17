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
# Libraries

import numpy as np

import pandas as pd

pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')

import datetime

import lightgbm as lgb

from scipy import stats

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, KFold

from wordcloud import WordCloud

from collections import Counter

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.preprocessing import StandardScaler

stop = set(stopwords.words('english'))

import os

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import xgboost as xgb

import lightgbm as lgb

from sklearn import model_selection

from sklearn.metrics import accuracy_score

import json

import ast

import eli5

import shap

from tqdm import tqdm

from catboost import CatBoostRegressor

from urllib.request import urlopen

from PIL import Image

from sklearn.preprocessing import LabelEncoder

import time

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn import linear_model
trainAdditionalFeatures = pd.read_csv('../input/tmdb-competition-additional-features/TrainAdditionalFeatures.csv')

testAdditionalFeatures = pd.read_csv('../input/tmdb-competition-additional-features/TestAdditionalFeatures.csv')



train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')



train = pd.merge(train, trainAdditionalFeatures, how='left', on=['imdb_id'])

test = pd.merge(test, testAdditionalFeatures, how='left', on=['imdb_id'])

test['revenue'] = -np.inf

train.loc[train['id'] == 16,'revenue'] = 192864          # Skinning

train.loc[train['id'] == 90,'budget'] = 30000000         # Sommersby          

train.loc[train['id'] == 118,'budget'] = 60000000        # Wild Hogs

train.loc[train['id'] == 149,'budget'] = 18000000        # Beethoven

train.loc[train['id'] == 313,'revenue'] = 12000000       # The Cookout 

train.loc[train['id'] == 451,'revenue'] = 12000000       # Chasing Liberty

train.loc[train['id'] == 464,'budget'] = 20000000        # Parenthood

train.loc[train['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II

train.loc[train['id'] == 513,'budget'] = 930000          # From Prada to Nada

train.loc[train['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol

train.loc[train['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip

train.loc[train['id'] == 850,'budget'] = 90000000        # Modern Times

train.loc[train['id'] == 1007,'budget'] = 2              # Zyzzyx Road 

train.loc[train['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman

train.loc[train['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   

train.loc[train['id'] == 1359,'budget'] = 10000000       # Stir Crazy 

train.loc[train['id'] == 1542,'budget'] = 1              # All at Once

train.loc[train['id'] == 1570,'budget'] = 15800000       # Crocodile Dundee II

train.loc[train['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp

train.loc[train['id'] == 1714,'budget'] = 46000000       # The Recruit

train.loc[train['id'] == 1721,'budget'] = 17500000       # Cocoon

train.loc[train['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed

train.loc[train['id'] == 1885,'budget'] = 12             # In the Cut

train.loc[train['id'] == 2091,'budget'] = 10             # Deadfall

train.loc[train['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget

train.loc[train['id'] == 2491,'budget'] = 6              # Never Talk to Strangers

train.loc[train['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus

train.loc[train['id'] == 2612,'budget'] = 15000000       # Field of Dreams

train.loc[train['id'] == 2696,'budget'] = 10000000       # Nurse 3-D

train.loc[train['id'] == 2801,'budget'] = 10000000       # Fracture

train.loc[train['id'] == 335,'budget'] = 2 

train.loc[train['id'] == 348,'budget'] = 12

train.loc[train['id'] == 470,'budget'] = 13000000 

train.loc[train['id'] == 513,'budget'] = 1100000

train.loc[train['id'] == 640,'budget'] = 6 

train.loc[train['id'] == 696,'budget'] = 1

train.loc[train['id'] == 797,'budget'] = 8000000 

train.loc[train['id'] == 850,'budget'] = 1500000

train.loc[train['id'] == 1199,'budget'] = 5 

train.loc[train['id'] == 1282,'budget'] = 9               # Death at a Funeral

train.loc[train['id'] == 1347,'budget'] = 1

train.loc[train['id'] == 1755,'budget'] = 2

train.loc[train['id'] == 1801,'budget'] = 5

train.loc[train['id'] == 1918,'budget'] = 592 

train.loc[train['id'] == 2033,'budget'] = 4

train.loc[train['id'] == 2118,'budget'] = 344 

train.loc[train['id'] == 2252,'budget'] = 130

train.loc[train['id'] == 2256,'budget'] = 1 

train.loc[train['id'] == 2696,'budget'] = 10000000



#Clean Data

test.loc[test['id'] == 6733,'budget'] = 5000000

test.loc[test['id'] == 3889,'budget'] = 15000000

test.loc[test['id'] == 6683,'budget'] = 50000000

test.loc[test['id'] == 5704,'budget'] = 4300000

test.loc[test['id'] == 6109,'budget'] = 281756

test.loc[test['id'] == 7242,'budget'] = 10000000

test.loc[test['id'] == 7021,'budget'] = 17540562       #  Two Is a Family

test.loc[test['id'] == 5591,'budget'] = 4000000        # The Orphanage

test.loc[test['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee

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



# from this kernel: https://www.kaggle.com/gravix/gradient-in-a-box

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

        

train = text_to_dict(train)

test = text_to_dict(test)
train.head()
train.shape, test.shape

def fix_date(x):

    """

    Fixes dates which are in 20xx

    """

    if not isinstance(x, str): return x

    year = x.split('/')[2]

    if int(year) <= 19:

        return x[:-2] + '20' + year

    else:

        return x[:-2] + '19' + year



train.loc[train['release_date'].isnull() == True, 'release_date'] = '01/01/19'

test.loc[test['release_date'].isnull() == True, 'release_date'] = '01/01/19'

    

#train["RevByBud"] = train["revenue"] / train["budget"]

    

train['release_date'] = train['release_date'].apply(lambda x: fix_date(x))

test['release_date'] = test['release_date'].apply(lambda x: fix_date(x))

train['release_date'] = pd.to_datetime(train['release_date'])

test['release_date'] = pd.to_datetime(test['release_date'])







train['year'] = train['release_date'].dt.year

train['month'] = train['release_date'].dt.month

train['day'] = train['release_date'].dt.day

train['weekday'] = train['release_date'].dt.weekday





test['year'] = test['release_date'].dt.year

test['month'] = test['release_date'].dt.month

test['day'] = test['release_date'].dt.day

test['weekday'] = test['release_date'].dt.weekday

train_cp = train.copy()



features_to_num = ['production_companies', 'production_countries', 'Keywords', 'cast', 'crew']

for f in features_to_num:

    train_cp['num_'+f] = train_cp[f].apply(lambda x : len(x))



features = ['year','budget', 'revenue', 'popularity', 'num_production_companies', 'num_production_countries', 'runtime', 'num_Keywords', 'num_cast', 'num_crew']



means = train_cp[features].groupby('year').mean()

means['budget'] = np.log1p(means['budget'])

means['revenue'] = np.log1p(means['revenue'])



medians = train_cp[features].groupby('year').median()

medians['budget'] = np.log1p(medians['budget'])

medians['revenue'] = np.log1p(medians['revenue'])



for f in features_to_num:

    train_cp['median_num_'+f] = train_cp[f]

    train_cp['mean_num_'+f] = train_cp[f]

for col in means.columns:

    fig, ax = plt.subplots(figsize = (16, 6))

    z = np.polyfit(range(len(means.index.values)), means[col], 1)

    p = np.poly1d(z)

    plt.plot(means.index.values, means[col]);

    plt.plot(means.index.values,p(range(len(means.index.values))),"b--")

    plt.text(0.1,0.9,"a=%.6f, b=%.6f"%(z[0],z[1]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.title('mean %s as a function of year'%col)

    plt.show()
for col in medians.columns:

    fig, ax = plt.subplots(figsize = (16, 6))

    z = np.polyfit(range(len(medians.index.values)), medians[col], 1)

    p = np.poly1d(z)

    plt.plot(medians.index.values, medians[col]);

    plt.plot(medians.index.values,p(range(len(medians.index.values))),"b--")

    plt.text(0.1,0.9,"a=%.6f, b=%.6f"%(z[0],z[1]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.title('median %s as a function of year'%col)

    plt.show()
fig, ax = plt.subplots(figsize = (16, 6))

plt.scatter(np.log1p(train_cp['budget']), np.log1p(train_cp['revenue']), c=train['year'], cmap=plt.cm.hot_r)

plt.title('Log Revenue vs Log Budget')

plt.xlabel("Log Budget")

label = set(train['year'].values)

plt.xlabel("Log Revenue")

plt.show()
plt.figure(figsize=(20,12))

sns.countplot(train_cp['year'].sort_values())

plt.title("Movie Release count by Year",fontsize=20)

loc, labels = plt.xticks()

plt.xticks(fontsize=12,rotation=90)

plt.show()

plt.figure(figsize=(20,12))

sns.countplot(train['month'].sort_values())

plt.title("Release Month Count",fontsize=20)

loc, labels = plt.xticks()

loc, labels = loc, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.xticks(loc, labels,fontsize=20)

plt.show()
plt.figure(figsize=(20,12))

sns.countplot(train['day'].sort_values())

plt.title("Release Day Count",fontsize=20)

plt.xticks(fontsize=20)

plt.show()
plt.figure(figsize=(20,12))

sns.countplot(train['weekday'].sort_values())

plt.title("Total movies released on Day Of Week",fontsize=20)

loc, labels = plt.xticks()

loc, labels = loc, ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

plt.xticks(loc, labels,fontsize=20)

plt.show()
meanRevenueByMonth = train.groupby("month")["revenue"].aggregate('mean')

meanRevenueByMonth.plot(figsize=(15,10),color="g")

plt.xlabel("Release Month")

plt.ylabel("Revenue")

plt.title("Movie Mean Revenue Release Month",fontsize=20)

plt.show()



medianRevenueByMonth = train.groupby("month")["revenue"].aggregate('median')

medianRevenueByMonth.plot(figsize=(15,10),color="g")

plt.xlabel("Release Month")

plt.ylabel("Revenue")

plt.title("Movie Median Revenue Release Month",fontsize=20)

plt.show()
meanRevenueByDay = train.groupby("day")["revenue"].aggregate('mean')

meanRevenueByDay.plot(figsize=(15,10),color="g")

plt.xlabel("Release Day")

plt.ylabel("Revenue")

plt.title("Movie Mean Revenue Release Day",fontsize=20)

plt.show()



medianRevenueByDay = train.groupby("day")["revenue"].aggregate('median')

medianRevenueByDay.plot(figsize=(15,10),color="g")

plt.xlabel("Release Day")

plt.ylabel("Revenue")

plt.title("Movie Median Revenue Release Day",fontsize=20)

plt.show()
meanRevenueByDay = train.groupby("weekday")["revenue"].aggregate('mean')

meanRevenueByDay.plot(figsize=(15,10),color="g")

plt.xlabel("Release WeekDay")

plt.ylabel("Revenue")

plt.title("Movie Mean Revenue Release WeekDay",fontsize=20)

plt.show()



medianRevenueByDay = train.groupby("weekday")["revenue"].aggregate('median')

medianRevenueByDay.plot(figsize=(15,10),color="g")

plt.xlabel("Release WeekDay")

plt.ylabel("Revenue")

plt.title("Movie Median Revenue Release WeekDay",fontsize=20)

plt.show()
plt.figure(figsize=(20,12))

sns.countplot(train['rating'].sort_values())

plt.title("Train Rating Count",fontsize=20)

plt.show()
train_cp['meanRevenueByRating'] = train.groupby("rating")["revenue"].aggregate('mean')

train_cp['meanRevenueByRating'].plot(figsize=(15,10),color="g")

plt.xlabel("Release Year")

plt.ylabel("Revenue")

plt.title("Movie Mean Revenue By Rating",fontsize=20)

plt.show()
def get_dictionary(s):

    try:

        d = ''

        d_l = [k['name'] for k in s]

        for l in d_l:

            d += l + ','

        d = d[:-1]

    except:

        d = ''

    return d

#train = train

#train['genres'] = train['genres'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

data = pd.concat([train,test], axis=0, sort=False)

genres = data['genres'].apply(get_dictionary)

genres = genres.str.get_dummies(sep=',')

data = pd.concat([data, genres], axis=1, sort=False)



train = data[data['revenue'] != -np.inf]

test = data[data['revenue'] == -np.inf]



print("Action Genres Movie           ", train[train.Action == 1].shape[0])

print("Adventure Genres Movie        ", train[train.Adventure == 1].shape[0])

print("Animation Genres Movie        ", train[train.Animation == 1].shape[0])

print("Comedy Genres Movie           ", train[train.Comedy == 1].shape[0])

print("Crime Genres Movie            ", train[train.Crime == 1].shape[0])

print("Documentary Genres Movie      ", train[train.Documentary == 1].shape[0])

print("Drama Genres Movie            ", train[train.Drama == 1].shape[0])

print("Family Genres Movie           ", train[train.Family == 1].shape[0])

print("Fantasy Genres Movie          ", train[train.Fantasy == 1].shape[0])

print("Foreign Genres Movie          ", train[train.Foreign == 1].shape[0])

print("History Genres Movie          ", train[train.History == 1].shape[0])

print("Music Genres Movie            ", train[train.Music == 1].shape[0])

print("Mystery Genres Movie          ", train[train.Mystery == 1].shape[0])

print("Romance Genres Movie          ", train[train.Romance == 1].shape[0])

print("Science Fiction Genres Movie  ", train[train['Science Fiction'] == 1].shape[0])

print("TV Movie Genres Movie         ", train[train['TV Movie'] == 1].shape[0])

print("Thriller Genres Movie         ", train[train.Thriller == 1].shape[0])

print("War Genres Movie              ", train[train.War == 1].shape[0])

print("Western Genres Movie          ", train[train.Western == 1].shape[0])
print(train.genres[:10])
train['has_homepage'] = 1

train.loc[pd.isnull(train['homepage']) ,"has_homepage"] = 0

plt.figure(figsize=(15,8))

sns.countplot(train['has_homepage'].sort_values())

plt.title("Has Homepage?",fontsize=20)

plt.show()



train['isTaglineNA'] = 0

train.loc[pd.isnull(train['tagline']) ,"isTaglineNA"] = 1

sns.catplot(x="isTaglineNA", y="revenue", data=train)

plt.title('Revenue of movies with and without a tagline')

plt.show()



train['isTitleDifferent'] = 1

train.loc[ train['original_title'] == train['title'] ,"isTitleDifferent"] = 0 

sns.catplot(x="isTitleDifferent", y="revenue", data=train)

plt.title('Revenue of movies with single and multiple titles')

plt.show()



train['isOriginalLanguageEng'] = 0 

train.loc[ train['original_language'] == "en" ,"isOriginalLanguageEng"] = 1

sns.catplot(x="isOriginalLanguageEng", y="revenue", data=train)

plt.title('Revenue of movies when Original Language is English and Not English')

plt.show()





test['has_homepage'] = 1

test.loc[pd.isnull(test['homepage']) ,"has_homepage"] = 0

plt.figure(figsize=(15,8))

sns.countplot(test['has_homepage'].sort_values())

plt.title("Has Homepage?",fontsize=20)

plt.show()



test['isTaglineNA'] = 0

test.loc[pd.isnull(test['tagline']) ,"isTaglineNA"] = 1

#sns.catplot(x="isTaglineNA", y="revenue", data=test)

#plt.title('Revenue of movies with and without a tagline')

#plt.show()



test['isTitleDifferent'] = 1

test.loc[ test['original_title'] == test['title'] ,"isTitleDifferent"] = 0 

#sns.catplot(x="isTitleDifferent", y="revenue", data=test)

#plt.title('Revenue of movies with single and multiple titles')

#plt.show()



test['isOriginalLanguageEng'] = 0 

test.loc[ test['original_language'] == "en" ,"isOriginalLanguageEng"] = 1

#sns.catplot(x="isOriginalLanguageEng", y="revenue", data=test)

#plt.title('Revenue of movies when Original Language is English and Not English')

#plt.show()

import random

def fill(df, col):

    return random.choice(df[df[col] != np.nan][col])



train['rating'] = train['rating'].map(lambda x: fill(train, 'rating') if np.isnan(x) else x)

train['totalVotes'] = train['totalVotes'].map(lambda x: fill(train, 'totalVotes') if np.isnan(x) else x)

#train['totalVotes'].fillna(lambda x: fill(train, 'totalVotes'))



test['rating'] = test['rating'].map(lambda x: fill(test, 'rating') if np.isnan(x) else x)

test['totalVotes'] = test['totalVotes'].map(lambda x: fill(test, 'totalVotes') if np.isnan(x) else x)



#test['rating'].fillna(lambda x: fill(test, 'rating'))

#test['totalVotes'].fillna(lambda x: fill(test, 'totalVotes'))



plt.figure(figsize=(20,12))

sns.countplot(train['rating'].sort_values())

plt.title("Train Rating Count",fontsize=20)

plt.show()



plt.figure(figsize=(20,12))

sns.countplot(train['totalVotes'].sort_values())

plt.title("Train Votes Count",fontsize=20)

plt.show()





train_plot = train[['budget','rating','totalVotes','popularity','runtime','year','month','day', 'weekday','revenue']]

f,ax = plt.subplots(figsize=(10, 8))

sns.heatmap(train_plot.corr(), annot=True)

plt.show()
df_all = pd.concat([train,test], sort=False)

rating_null = df_all.groupby(["year","original_language"])['rating'].mean().reset_index()

df_all[df_all.rating.isna()]['rating'] = df_all.merge(rating_null, how = 'left' ,on = ["year","original_language"])



train = df_all[df_all['revenue'] != -np.inf]  

test = df_all[df_all['revenue'] == -np.inf]  



# Adding Producer and Director column to data

train['Producer'] = train['crew'].apply(lambda x : [str(i['name']) for i in x if i['job']=='Producer'])

test['Producer'] = test['crew'].apply(lambda x : [str(i['name']) for i in x if i['job']=='Producer'])



train['ProducerT'] = train['Producer'].apply(lambda x: str(x)[2:-2])

test['ProducerT'] = test['Producer'].apply(lambda x: str(x)[2:-2])



train['Director'] = train['crew'].apply(lambda x : [i['name'] for i in x if i['job']=='Director'])

test['Director'] = test['crew'].apply(lambda x : [i['name'] for i in x if i['job']=='Director'])



train['DirectorT'] = train['Director'].apply(lambda x: str(x)[2:-2])

test['DirectorT'] = test['Director'].apply(lambda x: str(x)[2:-2])



# Creates list and data frame for producers and directors

directors_list = np.concatenate((train['DirectorT'].unique(),test['DirectorT'].unique()))

unique_directors = np.unique(directors_list).tolist()

df = pd.DataFrame(unique_directors, columns = ['name' ])

df['movies_num'] = 0

df['score'] = 0

df['popularity'] = 0.0

df['director_avg_score'] = 0.0

df['director_avg_popularity'] = 0.0

df.set_index('name', inplace=True)





producers_list = np.concatenate((train['ProducerT'].unique(),test['ProducerT'].unique()))

unique_producers = np.unique(producers_list).tolist()

df2 = pd.DataFrame(unique_producers, columns = ['name' ])

df2['movies_num'] = 0

df2['score'] = 0

df2['popularity'] = 0.0

df2['producer_avg_score'] = 0.0

df2['producer_avg_popularity'] = 0.0

df2.set_index('name', inplace=True)





# Calculates the average rating and popularity of each Director/Producer

for i in tqdm(range(len(train))):

    df.at[train['DirectorT'][i],'score'] += train['rating'][i]

    df.at[train['DirectorT'][i],'popularity'] += train['popularity'][i]

    df.at[train['DirectorT'][i],'movies_num'] += 1

     

    df2.at[train['ProducerT'][i],'score'] += train['rating'][i]

    df2.at[train['ProducerT'][i],'popularity'] += train['popularity'][i]

    df2.at[train['ProducerT'][i],'movies_num'] += 1

        

for i in tqdm(range(len(test))):

    try: 

        df.at[test['DirectorT'][i],'score'] += test['rating'][i]

        df.at[test['DirectorT'][i],'popularity'] += test['popularity'][i]

        df.at[test['DirectorT'][i],'movies_num'] += 1

    except:

        pass

    

    try:

        df2.at[test['ProducerT'][i],'score'] += test['rating'][i]

        df2.at[test['ProducerT'][i],'popularity'] += test['popularity'][i]

        df2.at[test['ProducerT'][i],'movies_num'] += 1

    except:

        pass

for i in tqdm(range(len(unique_directors))):

    df.at[unique_directors[i],'director_avg_score'] = df.at[unique_directors[i],'score']/df.at[unique_directors[i],'movies_num']

    df.at[unique_directors[i],'director_avg_popularity'] = df.at[unique_directors[i],'popularity']/df.at[unique_directors[i],'movies_num']

for i in tqdm(range(len(unique_producers))):   

    df2.at[unique_producers[i],'producer_avg_score'] = df2.at[unique_producers[i],'score']/df2.at[unique_producers[i],'movies_num']

    df2.at[unique_producers[i],'producer_avg_popularity'] = df2.at[unique_producers[i],'popularity']/df2.at[unique_producers[i],'movies_num']

    

# Creates new columns of the average score/popularity of directors/producers in specific film

test['director_avg_score'] = 0.0

test['director_avg_popularity'] = 0.0

train['director_avg_score'] = 0.0

train['director_avg_popularity'] = 0.0



test['producer_avg_score'] = 0.0

test['producer_avg_popularity'] = 0.0

train['producer_avg_score'] = 0.0

train['producer_avg_popularity'] = 0.0



lunch = False

if lunch:

    for i in tqdm(range(len(train))):

        director, ld = df.loc[train['DirectorT'][i]][['director_avg_popularity', 'director_avg_score']],len(train['Director'][i])

        producer, lp = df2.loc[train['ProducerT'][i]][['producer_avg_popularity', 'producer_avg_score']],len(train['Producer'][i])

        if ld==0:

            train['director_avg_popularity'][i] = 0.0

            train['director_avg_score'][i] = 0.0

        else :

            train['director_avg_popularity'][i] = director[0]/ld

            train['director_avg_score'][i] = director[1]/ld 



        if lp==0:

            train['producer_avg_popularity'][i] = 0.0

            train['producer_avg_score'][i] = 0.0

        else :

            train['producer_avg_popularity'][i] = producer[0]/lp

            train['producer_avg_score'][i] = producer[1]/lp 



    for i in tqdm(range(len(test))):

        director, ld = df.loc[test['DirectorT'][i]][['director_avg_popularity', 'director_avg_score']],len(test['Director'][i])

        producer, lp = df2.loc[test['ProducerT'][i]][['producer_avg_popularity', 'producer_avg_score']],len(test['Producer'][i])

        if ld==0:

            test['director_avg_popularity'][i] = 0.0

            test['director_avg_score'][i] = 0.0

        else :

            test['director_avg_popularity'][i] = director[0]/ld

            test['director_avg_score'][i] = director[1]/ld 



        if lp==0:

            test['producer_avg_popularity'][i] = 0.0

            test['producer_avg_score'][i] = 0.0

        else :

            test['producer_avg_popularity'][i] = producer[0]/lp

            test['producer_avg_score'][i] = producer[1]/lp 



    for i in tqdm(range(len(unique_directors))):

        df.loc[unique_directors[i]]['score'] = df.loc[unique_directors[i]]['score']/df.loc[unique_directors[i]]['movies_num']



    # Crew popularity is more relevant than avg rating/popularity 

    train['crew_popularity'] = train['producer_avg_popularity'] + train['director_avg_popularity']

    test['crew_popularity'] = test['producer_avg_popularity'] + test['director_avg_popularity']



    # Deletes producer,director and crew columns

drop_colums = ['Producer', 'Director', 'DirectorT', 'ProducerT']

for i in drop_colums:

    train = train.drop([i], axis=1)

    test = test.drop([i], axis=1)
# get all actors names



df = pd.concat([train,test], sort=False)



def get_all_names(col):

    all_names = []

    for col_list in df[col]:

        for value in col_list:

            if not value['name'] in all_names:

                all_names.append(value['name'])

    return all_names

actors_names = get_all_names('cast')

print('number of actors: ',len(actors_names))



actors = pd.DataFrame(0, index=actors_names, columns=['num_movies', 'mean_revenue'])



for index, row in tqdm(df.iterrows(), total=df.shape[0]):

    movie = row['title']

    cast = row['cast']

    genres = row['genres']

    rev = row['revenue']

    actors_in_movie = [a['name'] for a in cast]

    genres_in_movie = [a['name'] for a in genres]

    # save data on actor

    for a in actors_in_movie:

        if rev != -np.inf:

            actors.loc[a]['num_movies'] += 1

            actors.loc[a]['mean_revenue'] += rev

        

# calculate mean revenue

actors['num_movies'][actors['num_movies'] == 0] = 1

actors['mean_revenue'] = actors['mean_revenue']/actors['num_movies']

actors['mean_revenue'] = actors.apply(lambda row: np.nan if row['num_movies'] < 2 else row['mean_revenue'], axis=1)

# bin actors by profit

q = list(actors.mean_revenue.quantile(q=[0.25, 0.5, 0.75]))

bins = [-np.inf]

bins.extend(q)

bins.append(np.inf)

lables = [0, 1, 2, 3]

actors['level'] = pd.cut(actors['mean_revenue'], bins=bins, labels=lables, include_lowest=False)

actors['level'] = actors['level'].cat.add_categories(4).fillna(4)

#actors['level'] = actors['level'].cat.reorder_categories(new_categories=(0,1,2,3))
def prepare(train_df):

    df = train_df.copy() 

    #print(df.columns)

    

    #rating_null = df.groupby(["year","original_language"])['rating'].mean().reset_index()

    #df[df.rating.isna()]['rating'] = df.merge(rating_null, how = 'left' ,on = ["year","original_language"])



    df['inflationBudget'] = df['budget'] + df['budget']*1.8/100*(2018-df['year']) #Inflation simple formula 

    df['log_budget'] = np.log1p(df['budget'])

    df['log_inflationBudget'] = np.log1p(df['inflationBudget'])

    

    df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

    df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

    df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

    df['is_in_collection'] = df['belongs_to_collection'].apply(lambda x: 1 if x != {} else 0)

    df['movies_in_collection'] = df['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)

    df['num_genres'] = df['genres'].apply(lambda x: len(x) if x != {} else 0)

    df['num_spoken_langs'] = df['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

    df['num_Keywords'] = df['Keywords'].apply(lambda x: len(x) if x != {} else 0)

    df['num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)

    

    df['popularity_mean_year'] = df['popularity'] / df.groupby("year")["popularity"].transform('mean')

    df['runtime_mean_year'] = df['runtime'] / df.groupby("year")["runtime"].transform('mean')

    df['log_budget_mean_year'] = df['log_budget']/ df.groupby("year")["log_budget"].transform('mean')

    df['totalVotes_mean_year'] = df['totalVotes'] / df.groupby("year")["totalVotes"].transform('mean')

    df['rating_mean_year'] = df['rating'] / df.groupby("year")["rating"].transform('mean')

    df['log_budget_runtime_ratio'] = df['log_budget']/df['runtime'] 

    df['log_budget_popularity_ratio'] = df['log_budget']/df['popularity']

    

    df['popularity_totalVotes_ratio'] = df['totalVotes']/df['popularity']

    df['rating_popularity_ratio'] = df['rating']/df['popularity']

    df['rating_totalVotes_ratio'] = df['totalVotes']/df['rating']

    df['budget_rating_ratio'] = df['log_budget']/df['rating']

    df['runtime_rating_ratio'] = df['runtime']/df['rating']

    df['budget_totalVotes_ratio'] = df['log_budget']/df['totalVotes']

    

    df['has_homepage'] = 1

    df.loc[pd.isnull(df['homepage']) ,"has_homepage"] = 0

     

    df['has_tagline'] = 0

    df.loc[df['tagline'] == 0 ,"has_tagline"] = 1 



    df['is_eng'] = 0 

    df.loc[ df['original_language'] == "en" ,"is_eng"] = 1

    

    df['isTitleDifferent'] = 1

    df.loc[ df['original_title'] == df['title'] ,"isTitleDifferent"] = 0 



    # get collection id    

    df['original_title_letter_count'] = df['original_title'].str.len() 

    df['original_title_word_count'] = df['original_title'].str.split().str.len() 





    df['title_word_count'] = df['title'].str.split().str.len()

    df['overview_word_count'] = df['overview'].str.split().str.len()

    df['tagline_word_count'] = df['tagline'].str.split().str.len()

    

    df['production_countries_count'] = df['production_countries'].apply(lambda x : len(x))

    df['production_companies_count'] = df['production_companies'].apply(lambda x : len(x))

    df['cast_count'] = df['cast'].apply(lambda x : len(x))

    df['crew_count'] = df['crew'].apply(lambda x : len(x))

    

    levels = np.zeros((len(df),5))

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):

        cast = row['cast']

        actors_in_movie = [a['name'] for a in cast]

        # save data on actor

        for a in actors_in_movie:

            levels[i, actors.loc[a]['level']] += 1  

    df[["level_0", "level_1", "level_2", "level_3", "level_-1"]] = levels

    df[["level_0", "level_1", "level_2", "level_3", "level_-1"]] /= df['cast_count']



    df = df.drop(['id','belongs_to_collection','genres','homepage','imdb_id','overview','runtime'

    ,'poster_path','production_companies','production_countries','release_date','spoken_languages'

    ,'status','title','Keywords','cast','crew','original_language','original_title','tagline', 'budget'

    ],axis=1)

    #print(df.columns)

    df.fillna(value=0.0, inplace = True) 

    return df



from sklearn.model_selection import KFold





def intersection(lst1, lst2): 

    lst3 = [value for value in lst1 if not value in lst2] 

    return lst3 



print(len(set(test.columns.values)))

print(len(set(train.columns.values)))

print(intersection(list(test.columns.values), list(train.columns.values)))

print(intersection(list(train.columns.values), list(test.columns.values)))

print(test.shape, train.shape)



df = pd.concat([train,test], sort=False)

data_df = prepare(df)

train_df = data_df[data_df['revenue'] != -np.inf]

test_df = data_df[data_df['revenue'] == -np.inf]



X, y = train_df.drop(['revenue'],axis=1), np.log1p(train_df['revenue'])

Xtest = test_df.drop(['revenue'],axis=1)



random_seed = 42

k = 5

fold = list(KFold(k, shuffle = True, random_state = random_seed).split(X,y))

np.random.seed(random_seed)
import xgboost as xgb

import lightgbm as lgb

from catboost import CatBoostRegressor



def cat_model(trn_x, trn_y, val_x, val_y, test, verbose) :

    

    model = CatBoostRegressor(iterations=100000,

                                 learning_rate=0.004,

                                 depth=5,

                                 eval_metric='RMSE',

                                 colsample_bylevel=0.8,

                                 random_seed = random_seed,

                                 bagging_temperature = 0.2,

                                 metric_period = None,

                                 early_stopping_rounds=200

                                )

    model.fit(trn_x, trn_y,

                 eval_set=(val_x, val_y),

                 use_best_model=True,

                 verbose=False)

    

    val_pred = model.predict(val_x)

    test_pred = model.predict(test)

    #print('error:', model.get_best_score())

    return {'val':val_pred, 

            'test':test_pred, 

            'error':model.get_best_score()['validation']['RMSE']}



def xgb_model(trn_x, trn_y, val_x, val_y, test, verbose) :

    params = {'objective': 'reg:linear', 

              'eta': 0.01, 

              'max_depth': 6, 

              'subsample': 0.6, 

              'colsample_bytree': 0.7,  

              'eval_metric': 'rmse', 

              'seed': random_seed, 

              'silent': True,

    }

    

    record = dict()

    model = xgb.train(params

                      , xgb.DMatrix(trn_x, trn_y)

                      , 100000

                      , [(xgb.DMatrix(trn_x, trn_y), 'train'), (xgb.DMatrix(val_x, val_y), 'valid')]

                      , verbose_eval=verbose

                      , early_stopping_rounds=500

                      , callbacks = [xgb.callback.record_evaluation(record)])

    best_idx = np.argmin(np.array(record['valid']['rmse']))



    val_pred = model.predict(xgb.DMatrix(val_x), ntree_limit=model.best_ntree_limit)

    test_pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)



    return {'val':val_pred, 'test':test_pred, 'error':record['valid']['rmse'][best_idx], 'importance':[i for k, i in model.get_score().items()]}



def lgb_model(trn_x, trn_y, val_x, val_y, test, verbose) :



    params = {'objective':'regression',

         'num_leaves' : 30,

         'min_data_in_leaf' : 20,

         'max_depth' : 9,

         'learning_rate': 0.004,

         #'min_child_samples':100,

         'feature_fraction':0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9,

         'lambda_l1': 0.2,

         "bagging_seed": random_seed,

         "metric": 'rmse',

         #'subsample':.8, 

          #'colsample_bytree':.9,

         "random_state" : random_seed,

         "verbosity": -1}



    record = dict()

    model = lgb.train(params

                      , lgb.Dataset(trn_x, trn_y)

                      , num_boost_round = 100000

                      , valid_sets = [lgb.Dataset(val_x, val_y)]

                      , verbose_eval = verbose

                      , early_stopping_rounds = 500

                      , callbacks = [lgb.record_evaluation(record)]

                     )

    best_idx = np.argmin(np.array(record['valid_0']['rmse']))



    val_pred = model.predict(val_x, num_iteration = model.best_iteration)

    test_pred = model.predict(test, num_iteration = model.best_iteration)

    

    return {'val':val_pred, 'test':test_pred, 'error':record['valid_0']['rmse'][best_idx], 'importance':model.feature_importance('gain')}
result_dict = dict()

val_pred = np.zeros(train.shape[0])

test_pred = np.zeros(test.shape[0])

final_err = 0

verbose = False

import time

def now():

    return datetime.datetime.fromtimestamp(time.time())

for i, (trn, val) in enumerate(fold) :

    print(i+1, "fold.    RMSE")

    

    trn_x = X.loc[trn, :]

    trn_y = y[trn]

    val_x = X.loc[val, :]

    val_y = y[val]

    

    fold_val_pred = []

    fold_test_pred = []

    fold_err = []

    

    #""" xgboost

    start = now()

    result = xgb_model(trn_x, trn_y, val_x, val_y, Xtest, verbose)

    fold_val_pred.append(result['val']*0.2)

    fold_test_pred.append(result['test']*0.2)

    fold_err.append(result['error'])

    print("xgb model.", "{0:.5f}".format(result['error']), '(' + str(int((now()-start).seconds/60)) + 'm)')

    #"""

    

    #""" lightgbm

    start = now()

    result = lgb_model(trn_x, trn_y, val_x, val_y, Xtest, verbose)

    fold_val_pred.append(result['val']*0.4)

    fold_test_pred.append(result['test']*0.4)

    fold_err.append(result['error'])

    print("lgb model.", "{0:.5f}".format(result['error']), '(' + str(int((now()-start).seconds/60)) + 'm)')

    #"""

    

    #""" catboost model

    start = now()

    result = cat_model(trn_x, trn_y, val_x, val_y, Xtest, verbose)

    fold_val_pred.append(result['val']*0.4)

    fold_test_pred.append(result['test']*0.4)

    fold_err.append(result['error'])

    print("cat model.", "{0:.5f}".format(result['error']), '(' + str(int((now()-start).seconds/60)) + 'm)')

    #"""

    

    # mix result of multiple models

    val_pred[val] += np.mean(np.array(fold_val_pred), axis = 0)

    #print(fold_test_pred)

    #print(fold_test_pred.shape)

    #print(fold_test_pred.columns)

    test_pred += np.mean(np.array(fold_test_pred), axis = 0) / k

    final_err += (sum(fold_err) / len(fold_err)) / k

    

    print("---------------------------")

    print("avg   err.", "{0:.5f}".format(sum(fold_err) / len(fold_err)))

    print("blend err.", "{0:.5f}".format(np.sqrt(np.mean((np.mean(np.array(fold_val_pred), axis = 0) - val_y)**2))))

    

    print('')

    

print("fianl avg   err.", final_err)

print("fianl blend err.", np.sqrt(np.mean((val_pred - y)**2)))
sub = pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')

df_sub = pd.DataFrame()

df_sub['id'] = sub['id']

df_sub['revenue'] = np.expm1(test_pred*3)

#print(df_sub['revenue'])

df_sub.to_csv("submission.csv", index=False)