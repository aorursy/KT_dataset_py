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

train.columns

train['genres']
def DicCol2OneHot (df,ColName):

    ColumnDic = {}

    # Create Combined Dictionary

    for ColDictionaty in df[ColName]:

        for Value in ColDictionaty:

            ColumnDic[Value['id']] = Value['name']

    # Create Empty Columns

    for Value in ColumnDic:

        ValueName = ColName + '_' + str(Value)

        df[ValueName] = 0

    for line in df.index:

        for Value in df.loc[line,ColName]:

            ValueName = ColName + '_' + str(Value['id'])

            df.loc[line,ValueName] = 1

    return ColumnDic, df
genres_dic = {}

genres_dic, train = DicCol2OneHot(train, 'genres')
train


genres_dic = {}

for genres in train.genres:

    for genre in genres:

        genres_dic[genre['id']] = genre['name']

genres_dic

# create columns

for genre  in genres_dic.keys():

    genre_name = 'genre_' + str(genre)

    train[genre_name]  = np.NaN

for line in train.index:

#    print(line)

    for genre in train.loc[line,'genres']:

        genre_name = 'genre_' + str(genre['id'])

        train.loc[line,genre_name] = 1

genre['id']
train[['genres_35', 'genres_18',

       'genres_10751', 'genres_10749', 'genres_53', 'genres_28', 'genres_16',

       'genres_12', 'genres_27', 'genres_99', 'genres_10402', 'genres_80',

       'genres_878', 'genres_9648', 'genres_10769', 'genres_14', 'genres_10752',

       'genres_37', 'genres_36', 'genres_10770']]







def convert_date (df):

    df['Date_release'] = df.apply(lambda x: datetime.datetime.strptime(x['release_date'], "%m/%d/%Y").date() ,axis = 1)

    df['Year_release'] = df.apply(lambda x: x['Date_release'].year,axis=1)

    df['Month_release'] = df.apply(lambda x: x['Date_release'].month,axis=1)

    df['Day_release'] = df.apply(lambda x: x['Date_release'].day,axis=1)

    df['WeekDay_release'] = df.apply(lambda x: x['Date_release'].weekday(),axis=1)

    
convert_date (train)

train.info()
time_Dim = ['Date_release','Year_release','Month_release','WeekDay_release','Day_release']

train [time_Dim]



train['profit_value'] = train.revenue - train.budget 
train['profit_value'].describe()
#fig, saxis = plt.subplots(1, 2,figsize=(16,12))



for timeslice in time_Dim:

    print(train.groupby(by=timeslice).mean()['profit_value'])



#plt.bar()

#sns.barplot(x = 'Year_release', y = 'revenue', data=train, ax = saxis[0,0])

#sns.barplot(x = 'Month_release', y = 'revenue',  data=train, ax = saxis[0,1])

#sns.barplot(x = 'genres', y = 'revenue', order=[1,0], data=train, ax = saxis[0,2])



#sns.pointplot(x = 'FareBin', y = 'revenue',  data=data1, ax = saxis[1,0])

#sns.pointplot(x = 'AgeBin', y = 'revenue',  data=data1, ax = saxis[1,1])



#sns.pointplot(x = 'FamilySize', y = 'revenue', data=data1, ax = saxis[1,2])
type(train.belongs_to_collection)
# lists: 

#lists = ['belongs_to_collection','genres','production_companies', 'production_countries', 'spoken_languages','Keywords','cast', 'crew']

group_by = [ 'budget',  'homepage',

       'original_language', 'popularity', 'runtime', 

       'status', 'tagline', 'title',   

       'popularity2', 'rating', 'totalVotes', 'Date_release', 'Year_release',

       'Month_release']



for col in group_by:

    print(f'{col} :\n')

    print(train.groupby(by=col).mean()['revenue'])
lists = ['title','Collection_Name','belongs_to_collection','genres','production_companies', 'production_countries', 'spoken_languages','Keywords','cast', 'crew']

train['Collection_Name'] = train.apply(lambda x: '' if len(x['belongs_to_collection']) == 0 else x['belongs_to_collection'][0]['name'] ,axis=1)

train [lists]



train['belongs_to_collection'][0][0]['name']
fig, ((axis1,axis2,axis3),(axis4,axis5,axis6)) = plt.subplots(2,3,figsize=(14,12))



sns.scatterplot(x = train['Year_release'], y = train['revenue'],ax = axis1)

sns.boxplot(x = train['WeekDay_release'], y = train['revenue'], ax = axis2)

sns.scatterplot(x = train['Month_release'], y = train['revenue'], ax = axis3)

sns.scatterplot(x = train['Year_release'], y = train['profit_value'],ax = axis4)

sns.boxplot(x = train['WeekDay_release'], y = train['profit_value'], ax = axis5)

sns.scatterplot(x = train['Month_release'], y = train['profit_value'], ax = axis6)



#axis1.set_title('Pclass vs Fare Survival Comparison')



#sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data1, split = True, ax = axis2)

#axis2.set_title('Pclass vs Age Survival Comparison')



#sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = data1, ax = axis3)

#axis3.set_title('Pclass vs Family Size Survival Comparison')
import math

train['budget_log'] = np.log1p(train['budget'])

train['revenue_log'] = np.log1p(train['revenue'])



sns.scatterplot(x = train['budget_log'],y = train['revenue_log'],hue=train['Year_release'])
sns.pairplot(train)
train.genres
#create genres list
