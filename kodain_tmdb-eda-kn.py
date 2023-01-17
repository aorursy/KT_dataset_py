import numpy as np

import pandas as pd  

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import nltk

from nltk.corpus import stopwords

stop = stopwords.words('english')

import string

%matplotlib inline

%precision 3
train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')
train.head()
train.loc[train['id'] == 1336,'runtime'] = 130 #kololyovの上映時間を調べて入力

train.loc[train['id'] == 2303,'runtime'] = 80 #HappyWeekendの上映時間を調べて入力

train.loc[train['id'] == 391,'runtime'] = 96 #The Worst Christmas of My Lifeの上映時間を調べて入力

train.loc[train['id'] == 592,'runtime'] = 90 #А поутру они проснулисьの上映時間を調べて入力

train.loc[train['id'] == 925,'runtime'] = 86 #¿Quién mató a Bambi?の上映時間を調べて入力

train.loc[train['id'] == 978,'runtime'] = 93 #La peggior settimana della mia vitaの上映時間を調べて入力

train.loc[train['id'] == 1256,'runtime'] = 92 #Cry, Onion!の上映時間を調べて入力

train.loc[train['id'] == 1542,'runtime'] = 93 #All at Onceの上映時間を調べて入力

train.loc[train['id'] == 1875,'runtime'] = 93 #Vermistの上映時間を調べて入力

train.loc[train['id'] == 2151,'runtime'] = 108 #Mechenosetsの上映時間を調べて入力

train.loc[train['id'] == 2499,'runtime'] = 86 #Na Igre 2. Novyy Urovenの上映時間を調べて入力

train.loc[train['id'] == 2646,'runtime'] = 98 #My Old Classmateの上映時間を調べて入力

train.loc[train['id'] == 2786,'runtime'] = 111 #Revelationの上映時間を調べて入力

train.loc[train['id'] == 2866,'runtime'] = 96 #Tutto tutto niente nienteの上映時間を調べて入力

test.loc[test['id'] == 3244,'runtime'] = 93 #La caliente niña Julietta	の上映時間を調べて入力

test.loc[test['id'] == 4490,'runtime'] = 90 #Pancho, el perro millonarioの上映時間を調べて入力

test.loc[test['id'] == 4633,'runtime'] = 108 #Nunca en horas de claseの上映時間を調べて入力

test.loc[test['id'] == 6818,'runtime'] = 90 #Miesten välisiä keskustelujaの上映時間を調べて入力

test.loc[test['id'] == 4074,'runtime'] = 103 #Shikshanachya Aaicha Ghoの上映時間を調べて入力

test.loc[test['id'] == 4222,'runtime'] = 91 #Street Knightの上映時間を調べて入力

test.loc[test['id'] == 4431,'runtime'] = 96 #Plus oneの上映時間を調べて入力

test.loc[test['id'] == 5520,'runtime'] = 86 #Glukhar v kinoの上映時間を調べて入力

test.loc[test['id'] == 5845,'runtime'] = 83 #Frau Müller muss weg!の上映時間を調べて入力

test.loc[test['id'] == 5849,'runtime'] = 140 #Shabdの上映時間を調べて入力

test.loc[test['id'] == 6210,'runtime'] = 104 #The Last Breathの上映時間を調べて入力

test.loc[test['id'] == 6804,'runtime'] = 140 #Chaahat Ek Nasha...の上映時間を調べて入力

test.loc[test['id'] == 7321,'runtime'] = 87 #El truco del mancoの上映時間を調べて入力
train_add = pd.read_csv('../input/tmdb-competition-additional-features/TrainAdditionalFeatures.csv')

test_add = pd.read_csv('../input/tmdb-competition-additional-features/TestAdditionalFeatures.csv')



train = pd.merge(train, train_add, how='left', on=['imdb_id'])

test = pd.merge(test, test_add, how='left', on=['imdb_id'])
df = pd.concat([train, test]).set_index("id")
df.loc[df.index == 90,'budget'] = 30000000

df.loc[df.index == 118,'budget'] = 60000000

df.loc[df.index == 149,'budget'] = 18000000

df.loc[df.index == 464,'budget'] = 20000000

df.loc[df.index == 819,'budget'] = 90000000

df.loc[df.index == 1112,'budget'] = 6000000

df.loc[df.index == 1131,'budget'] = 4300000

df.loc[df.index == 1359,'budget'] = 10000000

df.loc[df.index == 1570,'budget'] = 15800000

df.loc[df.index == 1714,'budget'] = 46000000

df.loc[df.index == 1865,'budget'] = 80000000

df.loc[df.index == 2602,'budget'] = 31000000

#idが105と2941のものの予算は不明
# 各ワードの有無を表す 01 のデータフレームを作成

def count_word_list(series):

    len_max = series.apply(len).max() # ジャンル数の最大値

    tmp = series.map(lambda x: x+["nashi"]*(len_max-len(x))) # listの長さをそろえる

    

    word_set = set(sum(list(series.values), [])) # 全ジャンル名のset

    for n in range(len_max):

        word_dfn = pd.get_dummies(tmp.apply(lambda x: x[n]))

        word_dfn = word_dfn.reindex(word_set, axis=1).fillna(0).astype(int)

        if n==0:

            word_df = word_dfn

        else:

            word_df = word_df + word_dfn

    

    return word_df#.drop("nashi", axis=1)
import datetime
df[df["release_date"].isnull()]
# 公開日の欠損1件 id=3829

# May,2000 (https://www.imdb.com/title/tt0210130/) 

# 日は不明。1日を入れておく

df.loc[3829, "release_date"] = "5/1/00"
df["release_year"] = pd.to_datetime(df["release_date"]).dt.year.astype(int)

# 年の20以降を、2020年より後の未来と判定してしまうので、補正。

df.loc[df["release_year"]>2020, "release_year"] = df.loc[df["release_year"]>2020, "release_year"]-100



df["release_month"] = pd.to_datetime(df["release_date"]).dt.month.astype(int)

df["release_day"] = pd.to_datetime(df["release_date"]).dt.day.astype(int)
train["release_year"] = pd.to_datetime(train["release_date"]).dt.year.astype(int)

# 年の20以降を、2020年より後の未来と判定してしまうので、補正。

train.loc[train["release_year"]>2020, "release_year"] = train.loc[train["release_year"]>2020, "release_year"]-100



train["release_month"] = pd.to_datetime(train["release_date"]).dt.month.astype(int)

train["release_day"] = pd.to_datetime(train["release_date"]).dt.day.astype(int)
plt.figure(figsize=(15,8))

sns.lineplot(x="release_year", y="budget", data=train)
plt.figure(figsize=(15,8))

sns.lineplot(x="release_year", y="revenue", data=train)
plt.figure(figsize=(15,8))

sns.countplot(train.release_year)

plt.xticks(rotation=90)

plt.xlabel('Years')
train['budget_releaseyear_ratio'] = train['budget']/train['release_year']
plt.figure(figsize=(15,8))

sns.distplot(train['budget_releaseyear_ratio'])
df['isbelongs_to_collectionNA'] = 1

df.loc[pd.isnull(df['belongs_to_collection']) ,"isbelongs_to_collectionNA"] = 0
train['isbelongs_to_collectionNA'] = 1

train.loc[pd.isnull(train['belongs_to_collection']) ,"isbelongs_to_collectionNA"] = 0
plt.figure(figsize=(15,8))

sns.countplot(x='isbelongs_to_collectionNA', data=train)
plt.figure(figsize=(15,8))

sns.boxplot(x='isbelongs_to_collectionNA', y='revenue', data=train);
# JSON text を辞書型のリストに変換

import ast

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



for col in dict_columns:

    df[col]=df[col].apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x) )

    train[col]=train[col].apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x) )
df["genre_names"] = df["genres"].apply(lambda x : [ i["name"] for i in x])

train["genre_names"] = train["genres"].apply(lambda x : [ i["name"] for i in x])
df['num_genres'] = df['genres'].apply(lambda x: len(x) if x != {} else 0)

train['num_genres'] = train['genres'].apply(lambda x: len(x) if x != {} else 0)
plt.figure(figsize=(15,8))

sns.barplot(x='num_genres', y='revenue', data=train);
df["production_names"] = df["production_companies"].apply(lambda x : [ i["name"] for i in x])
df['production_companies_count'] = df['production_companies'].apply(lambda x : len(x))

train['production_companies_count'] = train['production_companies'].apply(lambda x : len(x))
train['production_companies']
train['production_companies_count'].describe()
plt.figure(figsize=(15,8))

sns.countplot(x='production_companies_count', data=train)
plt.figure(figsize=(15,8))

sns.stripplot(x='production_companies_count', y='revenue', data=train);
tmp = count_word_list(df["production_names"])
df["production_names"]
train['temp_list'] = train['title'].apply(lambda x:str(x).split())

top = Counter([item for sublist in train['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Blues')

#記号の排除

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)
train["title"]=train["title"].apply(lambda x : remove_punct(x))
#全て小文字に変換

def lower_text(text):

    return text.lower()
train["title"]=train["title"].apply(lambda x : lower_text(x))
#短縮形を元に戻す

shortened = {

    '\'m': ' am',

    '\'re': ' are',

    'don\'t': 'do not',

    'doesn\'t': 'does not',

    'didn\'t': 'did not',

    'won\'t': 'will not',

    'wanna': 'want to',

    'gonna': 'going to',

    'gotta': 'got to',

    'hafta': 'have to',

    'needa': 'need to',

    'outta': 'out of',

    'kinda': 'kind of',

    'sorta': 'sort of',

    'lotta': 'lot of',

    'lemme': 'let me',

    'gimme': 'give me',

    'getcha': 'get you',

    'gotcha': 'got you',

    'letcha': 'let you',

    'betcha': 'bet you',

    'shoulda': 'should have',

    'coulda': 'could have',

    'woulda': 'would have',

    'musta': 'must have',

    'mighta': 'might have',

    'dunno': 'do not know',

}

df["title"] = df["title"].replace(shortened)

train["title"] = train["title"].replace(shortened)
def remove_stopword(text):

    return [w for w in text if not w in stop]
train['temp_list'] = train['title'].apply(lambda x:str(x).split())

train['temp_list'] = train['temp_list'].apply(lambda x:remove_stopword(x))
train['temp_list']
top = Counter([item for sublist in train['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Blues')