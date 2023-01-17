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

#target - revenue

train.head()
train['revenue'].describe()
sns.distplot(train['revenue']);
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);

#scatter plot revenue/budget

var = 'budget'

data = pd.concat([train['revenue'], train[var]], axis=1)

data.plot.scatter(x=var, y='revenue');

#scatter plot revenue/total votes

var = 'totalVotes'

data = pd.concat([train['revenue'], train[var]], axis=1)

data.plot.scatter(x=var, y='revenue');
#scatter plot revenue/total votes

#mabe we need to do log later

var = 'popularity2'

data = pd.concat([train['revenue'], train[var]], axis=1)

data.plot.scatter(x=var, y='revenue');
#scatter plot revenue/total votes

var = 'totalVotes'

data = pd.concat([train['popularity'], train[var]], axis=1)

data.plot.scatter(x=var, y='popularity');
#data frame to date time

def to_datetime(df):

    df.release_date = pd.to_datetime(df.release_date)



to_datetime(train)

to_datetime(test)



#split to columns

def split_date_to_col(df):

    df['day'] = df['release_date'].dt.day

    df['month'] = df['release_date'].dt.month

    df['year'] = df['release_date'].dt.year 

    df['week_day'] = df['release_date'].dt.weekday

    

split_date_to_col(train)

split_date_to_col(test)

#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);

#scatter plot year/bodget 

var = 'year'

data = pd.concat([train['budget'], train[var]], axis=1)

data.plot.scatter(x=var, y='budget');
#box plot month/runtime 

var = 'month'

data = pd.concat([train['rating'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="rating", data=data)
#box plot month/runtime 

var = 'month'

data = pd.concat([train['runtime'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="runtime", data=data)
#box plot month/runtime 

var = 'month'

data = pd.concat([train['revenue'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="revenue", data=data)



#Missing data train

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)

#scatter plot year/bodget 

var = 'day'

data = pd.concat([train['revenue'], train[var]], axis=1)

data.plot.scatter(x=var, y='revenue');

train['belongs_to_collection'][0][0]
def is_collection(df):

    list_of_collection_names = []

    for  i in df['belongs_to_collection']:

        if i !={}:

            list_of_collection_names.append(1)

        else: list_of_collection_names.append(0)

    df['is_collection'] = list_of_collection_names

    

is_collection(train)

is_collection(test)





train = train.drop(columns=['belongs_to_collection'], axis =1)

test = test.drop(columns=['belongs_to_collection'], axis =1)
def genres_lists(df):

    list_of_genres = []

    for i in df['genres']:

            if len(i)>1:

                genre_list_per_one = []

                for genre in i:

                    genre_list_per_one.append(genre['name'])

                list_of_genres.append(genre_list_per_one)

            if len(i)==1:

                list_of_genres.append([i[0]['name']])

            if len(i)==0:

                list_of_genres.append(['None'])

    df['genres_lists'] = list_of_genres



genres_lists(train)

genres_lists(test)



train_dummies =  pd.get_dummies(train['genres_lists'].apply(pd.Series), prefix='', prefix_sep='').sum(level=0, axis=1)

test_dummies =  pd.get_dummies(test['genres_lists'].apply(pd.Series), prefix='', prefix_sep='').sum(level=0, axis=1)



train = pd.concat([train, train_dummies],axis=1)

test = pd.concat([test, test_dummies],axis=1)



train = train.drop(columns=['genres','genres_lists','poster_path','overview'], axis =1)

test = test.drop(columns=['genres','genres_lists','poster_path','overview'], axis =1)

#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);

train = train.drop(columns=['imdb_id','id'], axis =1)

test = test.drop(columns=['imdb_id','id'], axis =1)
def plot_reveneu_over_feature(df,feature):

    sns.set(font_scale = 2)

    f, axes = plt.subplots(1, 3, figsize=(45,15))

    #     targets = [train['lrevenue']]

#     lables = ['log-revenue']

    targets = []

    lables = []



    # Sort the dataframe by target 

    all_values = sorted(df[feature].unique())

    targets = [train.loc[df[feature]==all_values[i]]['lrevenue'] for i in range(len(all_values))]

    lables = all_values



    for lbl, dist in zip(lables, targets):

#         sns.lineplot(dist, hist=False, rug=False, label=lbl)

        sns.distplot(dist, hist=False, rug=True, label=lbl, ax=axes[0])

    # violin

#     sns.violinplot(x=feature, y="revenue", data=train, ax=axes[1])

    sns.barplot(x=feature, y="revenue", data=train, ax=axes[1])

    # pie

    # Plot

    size = train[feature].value_counts().sort_values()

    axes[2].pie(list(size), labels=size.index, autopct='%1.1f%%', shadow=True, startangle=140)

train['has_homepage'] = 1

train.loc[pd.isnull(train['homepage']) ,"has_homepage"] = 0

test['has_homepage'] = 1

test.loc[pd.isnull(test['homepage']) ,"has_homepage"] = 0



plt.figure(figsize=(15,8))

sns.countplot(train['has_homepage'].sort_values())

plt.title("Has Homepage?",fontsize=20)

plt.show()



train['isTaglineNA'] = 0

train.loc[pd.isnull(train['tagline']) ,"isTaglineNA"] = 1



test['isTaglineNA'] = 0

test.loc[pd.isnull(test['tagline']) ,"isTaglineNA"] = 1



sns.catplot(x="isTaglineNA", y="revenue", data=train)

plt.title('Revenue of movies with and without a tagline')

plt.show()



train['isTitleDifferent'] = 1

train.loc[ train['original_title'] == train['title'] ,"isTitleDifferent"] = 0 



test['isTitleDifferent'] = 1

test.loc[ test['original_title'] == test['title'] ,"isTitleDifferent"] = 0 



sns.catplot(x="isTitleDifferent", y="revenue", data=train)

plt.title('Revenue of movies with single and multiple titles')

plt.show()



train['isOriginalLanguageEng'] = 0 

train.loc[ train['original_language'] == "en" ,"isOriginalLanguageEng"] = 1

test['isOriginalLanguageEng'] = 0 

test.loc[ test['original_language'] == "en" ,"isOriginalLanguageEng"] = 1



sns.catplot(x="isOriginalLanguageEng", y="revenue", data=train)

plt.title('Revenue of movies when Original Language is English and Not English')

plt.show()
train = train.drop(columns=['homepage','tagline','release_date','status','original_title','title'], axis =1)

test = test.drop(columns=['homepage','tagline','release_date','status','original_title','title'], axis =1)
train['is_day_2'] = 0

train.loc[ train['week_day'] == 2 ,"is_day_2"] = 1 

test['is_day_2'] = 0

test.loc[ test['week_day'] == 2 ,"is_day_2"] = 1 



sns.catplot(x="is_day_2", y="revenue", data=train)

plt.title('Revenue of movies in the second day of week')

plt.show()

train['is_month_6'] = 0

train.loc[ train['month'] == 6 ,"is_month_6"] = 1 

test['is_month_6'] = 0

test.loc[ test['month'] == 6 ,"is_month_6"] = 1 

sns.catplot(x="is_month_6", y="revenue", data=train)

plt.title('Revenue of movies in the 6 month')

plt.show()

print("test shape: " + str(test.shape))

print("train shape: " + str(train.shape))
#train['budget'] = np.log1p(train['budget'])

#train['revenue'] = np.log1p(train['revenue'])



#test['budget'] = np.log1p(test['budget'])

#test['revenue'] = np.log1p(test['revenue'])
def spoken_lang_lists(df):

    list_of_genres = []

    for i in df['spoken_languages']:

            if len(i)>1:

                genre_list_per_one = []

                for genre in i:

                    genre_list_per_one.append(genre['iso_639_1'])

                list_of_genres.append(genre_list_per_one)

            if len(i)==1:

                list_of_genres.append([i[0]['iso_639_1']])

            if len(i)==0:

                list_of_genres.append(['None'])

    df['spoken_languages_new'] = list_of_genres



spoken_lang_lists(train)

spoken_lang_lists(test)



train_dummies =  pd.get_dummies(train['spoken_languages_new'].apply(pd.Series), prefix='', prefix_sep='').sum(level=0, axis=1)

test_dummies =  pd.get_dummies(test['spoken_languages_new'].apply(pd.Series), prefix='', prefix_sep='').sum(level=0, axis=1)



train = pd.concat([train, train_dummies],axis=1)

test = pd.concat([test, test_dummies],axis=1)
train_dummies.columns

train_lang_to_drop = []

test_lang_to_drop = []



for i in train_dummies.columns:

    if i not in test_dummies.columns:

        train_lang_to_drop.append(i)

        

for i in test_dummies.columns:

    if i not in train_dummies.columns:

        test_lang_to_drop.append(i)

        

train = train.drop(columns = train_lang_to_drop, axis =1)

test = test.drop(columns = test_lang_to_drop, axis =1)

print(test.shape)

print(train.shape)
for i  in zip(train['original_language'],train['spoken_languages_new']):

    if i[0] not in i[1]:

        train['is_spoken_lang_is_origin'] = 0

    else: train['is_spoken_lang_is_origin'] = 1

    



sns.catplot(x="is_spoken_lang_is_origin", y="revenue", data=train)

plt.title('spoken lang_is_origin')

plt.show()



train = train.drop(columns=['is_spoken_lang_is_origin','spoken_languages_new'], axis =1)

test = test.drop(columns=['spoken_languages_new'], axis =1)

#train['production_companies'][0]

def is_production_companies(df):

    list_of_production_copmanies = []

    for  i in df['production_companies']:

        if i !={}:

            list_of_production_copmanies.append(1)

        else: list_of_production_copmanies.append(0)

    df['belong_to_prod_comp'] = list_of_production_copmanies

    

is_production_companies(train)

is_production_companies(test)



sns.catplot(x="belong_to_prod_comp", y="revenue", data=train)

plt.title('belong to prod comp?')

plt.show()

train = train.drop(columns=['production_companies'], axis =1)

test = test.drop(columns=['production_companies'], axis =1)
def prod_cont(df):

    list_of_genres = []

    for i in df['production_countries']:

            if len(i)>1:

                genre_list_per_one = []

                for genre in i:

                    genre_list_per_one.append(genre['iso_3166_1'])

                list_of_genres.append(genre_list_per_one)

            if len(i)==1:

                list_of_genres.append([i[0]['iso_3166_1']])

            if len(i)==0:

                list_of_genres.append(['None'])

    df['production_countries_new'] = list_of_genres

    final_list =[]

    for i in df['production_countries_new']:

        i = i[0]

        final_list.append(i)

    df['production_countries_new'] = final_list



prod_cont(train)

prod_cont(test)



#train['production_countries_new']

#train['production_countries'][5]





#spoken_lang_lists(test)
sns.catplot(x="production_countries_new", y="revenue", data=train, height=10,  aspect=2)

plt.title('prod contries vs revenue?')

plt.xticks(fontsize=12,rotation=90)

plt.show()



plt.figure(figsize=(20,12))

sns.countplot(train['production_countries_new'].sort_values())

plt.title("Movie Release count by Year",fontsize=20)

loc, labels = plt.xticks()

plt.xticks(fontsize=12,rotation=90)

plt.show()
train['is_have_prod_com'] = 1

train.loc[ train['production_countries_new'] == 'None' ,"is_have_prod_com"] = 0 

test['is_have_prod_com'] = 1

test.loc[ test['production_countries_new'] == 'None' ,"is_have_prod_com"] = 0



train = train.drop(columns=['production_countries'], axis =1)

test = test.drop(columns=['production_countries'], axis =1)



target = 'revenue'

col_name = 'original_language'



col = train[col_name]

group_col = train[target].groupby(col).mean().sort_values(ascending  = False)

print (group_col)

selected = list(group_col.index)

g = sns.catplot(x=col_name, y=target, data=pd.concat([col, train[target]], axis=1).reset_index(), height=7, ci = None,   kind = 'bar', order = selected)

train['is_origin_en'] = 0

train.loc[ train['original_language'] == 'en' ,"is_origin_en"] = 1 

test['is_origin_en'] = 0

test.loc[ test['original_language'] == 'en' ,"is_origin_en"] = 1



train['is_origin_zh'] = 0

train.loc[ train['original_language'] == 'zh' ,"is_origin_zh"] = 1 

test['is_origin_zh'] = 0

test.loc[ test['original_language'] == 'zh' ,"is_origin_zh"] = 1



train['is_origin_tr'] = 0

train.loc[ train['original_language'] == 'tr' ,"is_origin_tr"] = 1 

test['is_origin_tr'] = 0

test.loc[ test['original_language'] == 'tr' ,"is_origin_tr"] = 1



train = train.drop(columns=['original_language','production_countries_new','spoken_languages'], axis =1)

test = test.drop(columns=['original_language','production_countries_new','spoken_languages'], axis =1)
for i in train['Keywords']:

    print('start: \n')

    print(i)
def extract_key_words(df):

    list_of_key_words = []

    for i in df['Keywords']:

            if len(i)>1:

                genre_list_per_one = []

                for genre in i:

                    genre_list_per_one.append(genre['name'])

                list_of_key_words.append(genre_list_per_one)

                print(genre_list_per_one)

            if len(i)==1:

                list_of_key_words.append([i[0]['name']])

            if len(i)==0:

                list_of_key_words.append(['None'])

    df['Keywords'] = list_of_key_words

    final_list =[]

    for i in df['Keywords']:

        i = str(i).strip('[]')

        final_list.append(i)

    df['Keywords_lists'] = final_list



extract_key_words(train)

extract_key_words(test)
s2 = train['Keywords_lists'][0]

s2
from nltk.corpus import sentiwordnet as swn

import nltk

from nltk.corpus import stopwords



def nlp_keywords(df):

    pos_or_neg =[]

    for s2 in  df['Keywords_lists']:

        pos = 0.0

        neg = 0.0

        #might be

        obj=0

        total = 0.0

        words = [word for word in s2.split()]# if word.lower() not in stopwords.words('english')]

        for wr in words:

            ls = list(swn.senti_synsets(wr,'r')) # adverb (try with v/n/a/none)

            for s in ls:

                r = swn.senti_synset(s.synset.name())

                pos += r.pos_score()

                neg += r.neg_score()

                obj+= r.obj_score()

        if pos > neg :

             pos_or_neg.append('pos')

        if pos< neg :

            pos_or_neg.append('neg')

        if obj > 0  and pos==0 and neg==0 :

            pos_or_neg.append('obj')

        if pos == 0 and neg ==0 and 0 == obj:

            pos_or_neg.append('unknown')

        if pos == neg and obj>0 and pos>0:

            pos_or_neg.append('obj')

    df['pos_or_neg'] = pos_or_neg

nlp_keywords(train)

train['pos_or_neg']
sns.catplot(x="pos_or_neg", y="revenue", data=train, height=10,  aspect=2)

plt.title('positive &negetive  vs revenue?')

plt.xticks(fontsize=12,rotation=90)

plt.show()
