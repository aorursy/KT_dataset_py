# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import json



from collections import Counter



import itertools



import re

import string

import collections



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error



%matplotlib inline

%precision 3

pd.set_option('precision', 3)



import warnings

warnings.filterwarnings('ignore')
#データを読み取る

#

train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

#

test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')
print(train.shape,test.shape)

train.columns
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
#columnsを確認し、除外する変数をdrop

print(df.columns)

# 使わない列を消す

df = df.drop(["poster_path", "status", "original_title"], axis=1) # "overview",  "imdb_id", 
# logを取っておく

df["log_revenue"] = np.log10(df["revenue"])

# homepage: 有無に

#df["homepage"] =  ~df['homepage'].isnull()

df['has_homepage'] = 1

df.loc[pd.isnull(df['homepage']) ,"has_homepage"] = 0
dfdic_feature = {}
%%time

# JSON text を辞書型のリストに変換

import ast

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



for col in dict_columns:

       df[col]=df[col].apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x) )
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
df["genre_names"] = df["genres"].apply(lambda x : [ i["name"] for i in x])

df["genre_names"]
df.columns
dfdic_feature["genre"] = count_word_list(df["genre_names"])

# TV movie は1件しかないので削除

dfdic_feature["genre"] = dfdic_feature["genre"].drop("TV Movie", axis=1)

dfdic_feature["genre"].head()
# train内の作品数が10件未満の言語は "small" に集約

n_language = df.loc[:train.index[-1], "original_language"].value_counts()

large_language = n_language[n_language>=10].index

df.loc[~df["original_language"].isin(large_language), "original_language"] = "small"
df["original_language"] = df["original_language"].astype("category")
# one_hot_encoding

#dfdic_feature["original_language"] = pd.get_dummies(df["original_language"])

#dfdic_feature["original_language"] = dfdic_feature["original_language"].loc[:, dfdic_feature["original_language"].sum()>0]

#dfdic_feature["original_language"].head()
df["production_names"] = df["production_companies"].apply(lambda x : [ i["name"] for i in x])

#.fillna("[{'name': 'nashi'}]").map(to_name_list)
tmp = count_word_list(df["production_names"])
# train内の件数が多い物のみ選ぶ

def select_top_n(df, topn=9999, nmin=2):  # topn:上位topn件, nmin:作品数nmin以上

#    if "small" in df.columns:

#        df = df.drop("small", axis=1)

    n_word = (df.loc[train["id"]]>0).sum().sort_values(ascending=False)

    # 作品数がnmin件未満

    smallmin = n_word[n_word<nmin].index

    # 上位topn件に入っていない

    smalln = n_word.iloc[topn+1:].index

    small = set(smallmin) | set(smalln)

    # 件数の少ないタグのみの作品

    df["small"] = df[small].sum(axis=1) #>0

    

    return df.drop(small, axis=1)
# trainに50本以上作品のある会社

#dfdic_feature["production_companies"] = select_top_n(tmp, nmin=50)

#dfdic_feature["production_companies"].head()
# 国名のリストに

df["country_names"] = df["production_countries"].apply(lambda x : [ i["name"] for i in x])

df_country = count_word_list(df["country_names"])
# 2か国だったら、0.5ずつに

df_country = (df_country.T/df_country.sum(axis=1)).T.fillna(0)
# 30作品以上の国のみ

#dfdic_feature["production_countries"] = select_top_n(df_country, nmin=30)

#dfdic_feature["production_countries"].head()
df["keyword_list"] = df["Keywords"].apply(lambda x : [ i["name"] for i in x])
df["num_Keywords"] = df["keyword_list"].apply(len)
#crewのname

df_lang = pd.DataFrame(df['spoken_languages'])

list_lang_names = list(df_lang['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

df_lang['num_spoken_languages'] = df_lang['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

df_lang['all_spoken_languages'] = df_lang['spoken_languages'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_lang_names = [m[0] for m in Counter([i for j in list_lang_names for i in j]).most_common(15)]

for g in top_lang_names:

    df_lang[g] = df_lang['all_spoken_languages'].apply(lambda x: 1 if g in x else 0)
df_lang.rename(columns ={'English':'spoken_en',

 'Français':'spoken_fr',

 'Español':'spoken_es',

 'Deutsch':'spoken_gr',

 'Pусский':'spoken_sv',

 'Italiano':'spoken_it',

 '日本語':'spoken_ja',

 '普通话':'spoken_ch1',

 'हिन्दी':'spoken_in',

 '':'spoken_unknown',

 'العربية':'spoken_arb',

 'Português':'spoken_por',

 '广州话 / 廣州話':'spoken_ch2',

 '한국어/조선말':'spoken_kr',

 'Polski':'spoken_pol'}, inplace =True)
df_lang.drop(columns=['spoken_languages', 'all_spoken_languages'], inplace = True)
df["language_names"] = df["spoken_languages"].apply(lambda x : [ i["name"] for i in x])

df["n_language"] = df["language_names"].apply(len)

# 欠損値は１にする(データを見ると無声映画ではない)

df.loc[df["n_language"]==0, "n_language"] = 1
# 英語が含まれるか否か

df["speak_English"] = df["language_names"].apply(lambda x : "English" in x)
df['speak_English'] = pd.get_dummies(df['speak_English'])
df['speak_English']
#Since only last two digits of year are provided, this is the correct way of getting the year.

df[['release_month','release_day','release_year']]=df['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)

# Some rows have 4 digits of year instead of 2, that's why I am applying (df['release_year'] < 100) this condition

df.loc[ (df['release_year'] <= 19) & (df['release_year'] < 100), "release_year"] += 2000

df.loc[ (df['release_year'] > 19)  & (df['release_year'] < 100), "release_year"] += 1900



releaseDate = pd.to_datetime(df['release_date']) 

df['release_dayofweek'] = releaseDate.dt.dayofweek

df['release_quarter'] = releaseDate.dt.quarter
df['mean_revenue_year'] = df.groupby('release_year')['revenue'].transform('mean')

df['mean_revenue_year'].plot(figsize=(15,5))

plt.xticks(np.arange(1920,2018,4))
df['mean_revenue_year']
df['mean_revenue_month'] = df.groupby('release_month')['revenue'].transform('mean')



df['mean_revenue_month'].plot(figsize=(15,5))

plt.xticks(np.arange(1,13))
df['mean_revenue_day'] = df.groupby('release_day')['revenue'].transform('mean')



df['mean_revenue_day'].plot(figsize=(15,5))

plt.xticks(np.arange(1,32))
df['mean_dayofweek'] = df.groupby('release_dayofweek')['revenue'].transform('mean')



df['mean_dayofweek'].plot(figsize=(15,5))

plt.xticks(np.arange(0,7))
df['mean_quarter'] = df.groupby('release_quarter')['revenue'].transform('mean')



df['mean_quarter'].plot(figsize=(15,5))

plt.xticks(np.arange(1,5))
##import datetime

# 公開日の欠損1件 id=3829

# May,2000 (https://www.imdb.com/title/tt0210130/) 

# 日は不明。1日を入れておく

##df.loc[3829, "release_date"] = "5/1/00"



##df["release_year"] = pd.to_datetime(df["release_date"]).dt.year.astype(int)

# 年の20以降を、2020年より後の未来と判定してしまうので、補正。

##df.loc[df["release_year"]>2020, "release_year"] = df.loc[df["release_year"]>2020, "release_year"]-100



##df["release_month"] = pd.to_datetime(df["release_date"]).dt.month.astype(int)

##df["release_day"] = pd.to_datetime(df["release_date"]).dt.day.astype(int)



# datetime型に

##df["release_date"] = df.apply(lambda s: datetime.datetime(

##    year=s["release_year"],month=s["release_month"],day=s["release_day"]), axis=1)



##df["release_dayofyear"] = df["release_date"].dt.dayofyear

##df["release_dayofweek"] = df["release_date"].dt.dayofweek



# 月、曜日は カテゴリ型に

##df["release_month"] = df["release_month"].astype('category')

##df["release_dayofweek"] = df["release_dayofweek"].astype('category')
# collection 名を抽出

df["collection_name"] = df["belongs_to_collection"].apply(lambda x : x[0]["name"] if len(x)>0 else "nashi")

# 無い場合、"nashi"に
# シリーズの作品数

#df = pd.merge( df, df.groupby("collection_name").count()[["budget"]].rename(columns={"budget":"count_collection"}), 

#         on="collection_name", how="left")

# indexがずれるので、戻す

#df.index = df.index+1



df["count_collection"] = df["collection_name"].apply(lambda x : (df["collection_name"]==x).sum())

# シリーズ以外の場合0

df.loc[df["collection_name"]=="nashi", "count_collection"] = 0



# シリーズ何作目か

df["number_in_collection"] = df.sort_values("release_date").groupby("collection_name").cumcount()+1

# シリーズ以外の場合0

df.loc[df["collection_name"]=="nashi", "number_in_collection"] = 0



# 同シリーズの自分より前の作品の平均log(revenue)

df["collection_av_logrevenue"] = [ df.loc[(df["collection_name"]==row["collection_name"]) & 

                                          (df["number_in_collection"]<row["number_in_collection"]),

                                          "log_revenue"].mean() 

     for key,row in df.iterrows() ]

# 欠損(nashi) の場合、nashi での平均

df.loc[df["collection_name"]=="nashi", "collection_av_logrevenue"] = df.loc[df["collection_name"]=="nashi", "log_revenue"].mean()
# train に無くtestだけにあるシリーズの場合、シリーズもの全部の平均

collection_mean = df.loc[df["collection_name"]!="nashi", "log_revenue"].mean()  # シリーズもの全部の平均

df["collection_av_logrevenue"] = df["collection_av_logrevenue"].fillna(collection_mean)  

df_features = pd.concat(dfdic_feature, axis=1)
df.columns
df[["original_language", "collection_name"]] = df[["original_language", "collection_name"]].astype("category")
df_use = df[['budget', 'has_homepage', 'popularity','runtime','n_language', 

             "num_Keywords", "speak_English",

             'release_year', 'release_month','release_day','release_dayofweek', 

             'mean_revenue_year','mean_revenue_day','collection_av_logrevenue' ,"count_collection","number_in_collection"

            ]]

df_use.head()
df_use = pd.get_dummies(df_use)
train_add = pd.read_csv('../input/tmdb-competition-additional-features/TrainAdditionalFeatures.csv')

test_add = pd.read_csv('../input/tmdb-competition-additional-features/TestAdditionalFeatures.csv')

train_add.head()
df = pd.merge(df, pd.concat([train_add, test_add]), on="imdb_id", how="left")
add_cols = ["popularity2", "rating", "totalVotes"]

df[add_cols] = df[add_cols].fillna(df[add_cols].mean())
train2 = pd.read_csv('../input/tmdb-box-office-prediction-more-training-data/additionalTrainData.csv')

train3 = pd.read_csv('../input/tmdb-box-office-prediction-more-training-data/trainV3.csv')

train3.head()
#全て小文字に変換

def lower_text(text):

    return text.lower()



#記号の排除

def remove_punct(text):

    text = text.replace('-', ' ')  # - は単語の区切りとみなす

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



def remove_stopwords(words, stopwords):#不要な単語を削除

    words = [word for word in words if word not in stopwords]

    return words
# 英語でよく使う単語が入っていない文章を確認

#df.loc[df["overview"].apply(lambda x : str(x)).apply(lambda x : lower_text(x)

#                                ).str.contains("nan|the|where|with|from|and|for|his|her|over")==False, "overview"]

#train3.loc[train3["overview"].apply(lambda x : str(x)).apply(lambda x : lower_text(x)).str.contains("nan|the|where|with|from|and|for|his|her|over")==False, "overview"]
no_english_overview_id = [157, 2863, 4616]   # 上のデータを目で確認

no_english_tagline_id = [3255, 3777, 4937]   # Tfidf で非英語の単語があったもの
#単語数

df['overview_word_count'] = df['overview'].apply(lambda x: len(str(x).split()))

#文字数

#df['overview_char_count'] = df['overview'].apply(lambda x: len(str(x)))

# 記号の個数

#df['overview_punctuation_count'] = df['overview'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
# 前処理

df['_overview']=df['overview'].apply(lambda x : str(x)

                            ).apply(lambda x : lower_text(x)).apply(lambda x : remove_punct(x))

#単語数

df['tagline_word_count'] = df['tagline'].apply(lambda x: len(str(x).split()))

#文字数

#df['tagline_char_count'] = df['tagline'].apply(lambda x: len(str(x)))

# 記号の個数

#df['tagline_punctuation_count'] = df['tagline'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
df['_tagline']=df['tagline'].apply(lambda x : str(x)

                                 ).apply(lambda x : lower_text(x)).apply(lambda x : remove_punct(x))

#単語数

df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))

#文字数

#df['title_char_count'] = df['title'].apply(lambda x: len(str(x)))

# 記号の個数

#df['title_punctuation_count'] = df['title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

df_use2 = df[["has_homepage","runtime",'budget']]
#castの中にある俳優の名前をリスト化させる

list_of_cast_names = list(df['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

df['num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)

list_of_cast_genders = list(df['cast'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)



df['genders_0_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

df['genders_1_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

df['genders_2_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))    
#crewのname

list_of_crew_names = list(df['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

df['num_crew'] = df['crew'].apply(lambda x: len(x) if x != {} else 0)

department_count = pd.Series(Counter([job for lst in df["crew"].apply(lambda x : [ i["department"] for i in x]).values for job in lst]))

department_count.sort_values(ascending=False).head(5)
job_count = pd.Series(Counter([job for lst in df["crew"].apply(lambda x : [ i["job"] for i in x]).values for job in lst]))

job_count.sort_values(ascending=False).head(5)
df_crew = { idx : pd.DataFrame([ [crew["department"], crew["job"], crew["name"]] 

                        for crew in x], columns=["department", "job", "name"]) 

    for idx, x in df["crew"].iteritems() }
df_crew = pd.concat(df_crew)
def select_job(list_dict, key, value):

    return [ dic["name"] for dic in list_dict if dic[key]==value]
for department in department_count.index:

    df['dep_{}_num'.format(department)] = df["crew"].apply(select_job, key="department", value=department).apply(len)
df_crewname = pd.DataFrame([], index=df.index)

for job in ["Producer", "Director", "Screenplay", "Casting", "Original Music Composer"]:

    col = 'job_{}_list'.format(job)

    df[col] = df["crew"].apply(select_job, key="job", value=job)



    top_list = [m[0] for m in Counter([i for j in df[col] for i in j]).most_common(15)]

    for i in top_list:

        df_crewname['{}_{}'.format(job,i)] = df[col].apply(lambda x: i in x)
for job in ["Sound", "Art", "Costume & Make-Up", "Camera", "Visual Effects"]:

    col = 'department_{}_list'.format(job)

    df[col] = df["crew"].apply(select_job, key="department", value=job)



    top_list = [m[0] for m in Counter([i for j in df[col] for i in j]).most_common(15)]

    for i in top_list:

        df_crewname['{}_{}'.format(job,i)] = df[col].apply(lambda x: i in x)
list(df)
import pickle

with open("/kaggle/input/private-jhk/df_use_nagano.pkl","rb") as fr:

    df_use_nagano = pickle.load(fr)
df_use_nagano
df_use_nagano = df_use_nagano[['production_countries_count', 'production_companies_count']]
df_use4 = df[add_cols]
df_input = pd.concat([df_use, df_use4, df_features,df_use_nagano], axis=1) # .drop("belongs_to_collection", axis=1) 
# 欠測ナシを確認

df_input.isnull().sum().sum()
df["ln_revenue"] = np.log(df["revenue"]+1)
df_input['log_budget'] = np.log10(df_input['budget'])
df_input['budget/popularity1'] = df_input['budget']/df_input['popularity']

df_input['budget/popularity2'] = df_input['budget']/df_input['popularity2']

df_input['budget/runtime'] = df_input['budget']/df_input['runtime']
sns.distplot(df_input['budget/popularity1'])

plt.show()
sns.distplot(df_input['budget/popularity2'])

plt.show()
sns.distplot(df_input['budget/runtime'])

plt.show()
df_input['_popularity_mean_year']=df['popularity']/df.groupby("release_year")["popularity"].transform('mean')

df_input['_budget_runtime_ratio']=df['budget']/df['runtime']

df_input['_budget_popularity_ratio']=df['budget']/df['popularity']

df_input['_budget_year_ratio']=df['budget']/(df['release_year']*df['release_year'])

df_input['_releaseYear_popularity_ratio']=df['release_year']/df['popularity']

df_input['_releaseYear_popularity_ratio2']=df['popularity']/df['release_year']

df_input['_popularity_totalVotes_ratio']=df['totalVotes']/df['popularity']

df_input['_rating_popularity_ratio']=df['rating']/df['popularity']

df_input['_rating_totalVotes_ratio']=df['totalVotes']/df['rating']

df_input['_totalVotes_releaseYear_ratio']=df['totalVotes']/df['release_year']

df_input['_budget_rating_ratio']=df['budget']/df['rating']

df_input['_runtime_rating_ratio']=df['runtime']/df['rating']

df_input['_budget_totalVotes_ratio']=df['budget']/df['totalVotes']
cols = df_input.loc[:, df_input.isnull().sum()>0].columns

df_input.loc[:, cols] = df_input[cols].fillna(df_input[cols].mean())
# 保存

import pickle

with open('df_input.pkl', 'wb') as f:

      pickle.dump(df_input , f)
# 数値化できい列を確認

no_numeric = df_input.apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull().all()

no_numeric[no_numeric]
df_input.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df_input.columns]
[df_input.isnull().sum()]
X_all = df_input  # .drop(["collection_av_logrevenue"], axis=1)

X_all.drop([0],inplace = True)

y_all = df["ln_revenue"]

y_all.index = X_all.index
X_all.drop(columns = ['budget'],inplace = True)
train.shape
'''

X_all = X_all.drop(columns = ['__genre____Fantasy__',

 '__original_language____cn__',

 '__original_language____it__',

 '__production_companies____Columbia_Pictures_Corporation__',

 '__original_language____ko__',

 '__production_companies____Walt_Disney_Pictures__',

 '__production_companies____Twentieth_Century_Fox_Film_Corporation__',

 '__original_language____ta__',

 '__genre____History__',

 '__production_companies____TriStar_Pictures__',

 '__production_companies____Metro_Goldwyn_Mayer__MGM___',

 '__production_countries____Russia__',

 '__original_language____small__',

 '__production_companies____Warner_Bros___',

 '__genre____War__',

 '__original_language____ru__',

 '__production_countries____Hong_Kong__',

 '__genre____Animation__',

 '__production_companies____Columbia_Pictures__',

 '__original_language____ja__',

 '__production_companies____New_Line_Cinema__',

 '__original_language____de__',

 '__genre____Science_Fiction__',

 '__production_countries____Spain__',

 '__genre____Adventure__',

 '__genre____Mystery__',

 '__original_language____es__',

 '__genre____Music__',

 '__genre____Horror__',

 '__original_language____hi__',

 '__original_language____en__',])

 '''
[ c for c in X_all.columns if "revenue" in str(c)]
#標準化

#X_train_all_mean = X_all[:3000].mean()

#X_train_all_std  = X_all[:3000].std()

#X_all = (X_all-X_train_all_mean)/X_train_all_std
test_X = X_all.iloc[3000:]
test_X.shape
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error 

from sklearn.preprocessing import StandardScaler
train_X, val_X, train_y, val_y = train_test_split(X_all.iloc[:3000], 

                                                  y_all.iloc[:3000], 

                                                  test_size=0.25, random_state=1)
from sklearn.model_selection import KFold



random_seed = 2019

k = 10

fold = list(KFold(k, shuffle = True, random_state = random_seed).split(train))

np.random.seed(random_seed)
from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import GridSearchCV



xgb = XGBRegressor()



'''

params = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'objective':['reg:linear'],

              'learning_rate': [0.03], #so called `eta` value

              'max_depth': [4],

              'min_child_weight': [4],

              'silent': [1],

              'subsample': [0.7],

              'colsample_bytree': [0.3],

              'n_estimators': [500]}



xgb_grid = GridSearchCV(xgb,

                        params,

                        cv = 4,

                        n_jobs = 5,

                        verbose=True)



xgb_grid.fit(train_X, train_y)

print(xgb_grid.best_score_)

print(xgb_grid.best_params_)

'''
xgb_model = XGBRegressor(max_depth=4, 

                            min_child_weight=4,

                            learning_rate=0.03, 

                            n_estimators=500, 

                            objective='reg:linear',

                            nthread = 4,

                            gamma=1.3,  

                            silent=1,

                            subsample=0.7, 

                            colsample_bytree=0.3, 

                            colsample_bylevel=0.5)

xgb_model.fit(train_X,train_y)

xgb_prediction = xgb_model.predict(val_X)

xgb_rmse = mean_squared_error(val_y, xgb_prediction)
plt.figure(figsize=(20,20))

importances = pd.Series(xgb_model.feature_importances_, index = X_all.columns)

importances = importances.sort_values()

importances.plot(kind = "barh")

plt.title("imporance in the xgboost Model")

plt.show()
import math



math.sqrt(xgb_rmse)
xgb_pred = xgb_model.predict(test_X)
pred_xgb = pd.DataFrame(np.exp(xgb_pred)-1,columns=["revenue"])

pred_xgb
test_id = test["id"]
sub=pd.concat([test_id, pred_xgb],axis=1)

sub.to_csv('TMDB_xgb.csv',index=False)
from lightgbm import LGBMRegressor



lgbm = LGBMRegressor()

'''

params = {'n_estimators': [500],

          'objection' :['regressor'],

          'metric':['rmse'],

          'max_depth ': [2],

          'num_leaves':[10],

          'min_child_samples':[500],

          'learning_rate':[0.01],

          'boosting ': ['gbdt'],

          'num_iterations' : [1500],

          'min_data_in_leaf': [10],

          'bagging_freq ': [1],

          'bagging_fraction ': [0.9],

          'feature_fraction' : [0.7],

          'importance_type':['gain'],

          'use_best_model':[True]}



lgbm_grid = GridSearchCV(estimator=lgbm, param_grid=params,cv=4, n_jobs=5, verbose=True)



lgbm_grid.fit(train_X, train_y)

print(lgbm_grid.best_score_)

print(lgbm_grid.best_params_)

'''
lgbm_model = LGBMRegressor(n_estimators= 500,

          objection ='regressor',

          metric='rmse',

          max_depth = 2,

          num_leaves=10,

          min_child_samples=500,

          learning_rate=0.01,

          boosting = 'gbdt',

          num_iterations = 1500,

          min_data_in_leaf= 10,

          bagging_freq = 1,

          bagging_fraction = 0.9,

          feature_fraction = 0.7,

          importance_type='gain',

          use_best_model=True)

lgbm_model.fit(train_X, train_y)

lgbm_prediction = lgbm_model.predict(val_X)

lgbm_rmse = mean_squared_error(val_y, lgbm_prediction)
plt.figure(figsize=(20,20))

importances = pd.Series(lgbm_model.feature_importances_, index = X_all.columns)

importances = importances.sort_values()

importances.plot(kind = "barh")

plt.title("imporance in the LightGBM Model")

plt.show()
math.sqrt(lgbm_rmse)
lgbm_pred = lgbm_model.predict(test_X)
pred_lgbm = pd.DataFrame(np.exp(lgbm_pred)-1,columns=["revenue"])

pred_lgbm
sub=pd.concat([test_id, pred_lgbm],axis=1)

sub.to_csv('TMDB_lgbm.csv',index=False)
from catboost import CatBoostRegressor

'''

cat = CatBoostRegressor()



params = {'iterations' : [2000], 

                                 'learning_rate' : [0.01], 

                                 'depth' : [6], 

                                 'eval_metric' : ['RMSE'],

                                 'colsample_bylevel' : [0.6],

                                 'bagging_temperature' : [0.1],

                                 'early_stopping_rounds' : [200]}



cat_grid = GridSearchCV(estimator=cat, param_grid=params,cv=4, n_jobs=5, verbose=True)



cat_grid.fit(train_X, train_y)

print(cat_grid.best_score_)

print(cat_grid.best_params_)

'''

cat_model = CatBoostRegressor(iterations=2000, 

                                 learning_rate=0.01, 

                                 depth=6, 

                                 eval_metric='RMSE',

                                 colsample_bylevel=0.6,

                                 bagging_temperature = 0.1,

                                 metric_period = None,

                                 early_stopping_rounds=200)

cat_model.fit(train_X, train_y)

cat_prediction = cat_model.predict(val_X)

cat_rmse = mean_squared_error(val_y, cat_prediction)
math.sqrt(cat_rmse)
plt.figure(figsize=(20,20))

importances = pd.Series(cat_model.feature_importances_, index = X_all.columns)

importances = importances.sort_values()

importances.plot(kind = "barh")

plt.title("imporance in the CatBoost Model")

plt.show()
cat_pred = cat_model.predict(test_X)
pred_cat = pd.DataFrame(np.exp(cat_pred)-1,columns=["revenue"])

pred_cat
sub=pd.concat([test_id, pred_cat],axis=1)

sub.to_csv('TMDB_cat.csv',index=False)
ansamble = 0.4 * pred_lgbm["revenue"] + 0.2 * pred_xgb["revenue"] + 0.4 * pred_cat["revenue"]
sub3=pd.concat([test_id, ansamble],axis=1)

sub3
sub3.to_csv('TMDB_ansamble.csv',index=False)
ansamble2 = 0.35 * pred_lgbm["revenue"] + 0.3 * pred_xgb["revenue"] + 0.35 * pred_cat["revenue"]
sub4=pd.concat([test_id, ansamble2],axis=1)

sub4
sub4.to_csv('TMDB_ansamble2.csv',index=False)
ansamble = 0.2 * pred_lgbm["revenue"] + 0.2 * pred_xgb["revenue"] + 0.6 * pred_cat["revenue"]
sub5=pd.concat([test_id, ansamble],axis=1)

sub5
sub3.to_csv('TMDB_ansamble3.csv',index=False)