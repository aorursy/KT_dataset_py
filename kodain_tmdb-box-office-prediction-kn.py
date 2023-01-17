import numpy as np

import pandas as pd  

import matplotlib.pyplot as plt

import seaborn as sns

import collections

from collections import Counter

import xgboost as xgb

import lightgbm as lgb

import catboost as cat

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
df
low_budget = train[train["budget"] <= 100]
low_budget
plt.figure(figsize=(12, 8))

sns.stripplot(x='budget', y='revenue', data = low_budget, jitter=True)

plt.xlabel('Budget [$]', fontsize=15)

plt.ylabel('Revenue [$]', fontsize=15)

plt.title('Revenues of low-budget movies', fontsize=20)
low_budget[low_budget["revenue"] >= 100000000]
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
# 使わない列を消す

df = df.drop(["poster_path", "original_title"], axis=1)
# logを取っておく

df["log_revenue"] = np.log1p(df["revenue"])

df["log_budget"] = np.log1p(df["budget"])
df['isbelongs_to_collectionNA'] = 1

df.loc[pd.isnull(df['belongs_to_collection']) ,"isbelongs_to_collectionNA"] = 0
df['isbelongs_to_collectionNA']
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
dfdic_feature = {}
df["genre_names"] = df["genres"].apply(lambda x : [ i["name"] for i in x])
df['num_genres'] = df['genres'].apply(lambda x: len(x) if x != {} else 0)
dfdic_feature["genre"] = count_word_list(df["genre_names"])

# TV movie は1件しかないので削除

dfdic_feature["genre"] = dfdic_feature["genre"].drop("TV Movie", axis=1)

dfdic_feature["genre"].head()
n_language = df.loc[:train.index[-1], "original_language"].value_counts()

large_language = n_language[n_language>=20].index

df.loc[~df["original_language"].isin(large_language), "original_language"] = "small"
n_language
large_language
df["original_language"] = df["original_language"].astype("category")
df['isOriginalLanguageEng'] = 0 

df.loc[ df['original_language'] == "en" ,"isOriginalLanguageEng"] = 1


# one_hot_encoding

dfdic_feature["original_language"] = pd.get_dummies(df["original_language"])



dfdic_feature["original_language"].head()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['original_language'] = le.fit_transform(df['original_language'])
df["production_names"] = df["production_companies"].apply(lambda x : [ i["name"] for i in x])
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
# trainに2本以上作品のある会社

dfdic_feature["production_companies"] = select_top_n(tmp, topn=9,nmin=2)

dfdic_feature["production_companies"].head()
# 国名のリストに

df["country_names"] = df["production_countries"].apply(lambda x : [ i["name"] for i in x])

df_country = count_word_list(df["country_names"])
# 2か国だったら、0.5ずつに

df_country = (df_country.T/df_country.sum(axis=1)).T.fillna(0)
dfdic_feature["production_countries"] = select_top_n(df_country, topn=10,nmin=2)

dfdic_feature["production_countries"].head()
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
# datetime型に

df["release_date"] = df.apply(lambda s: datetime.datetime(

    year=s["release_year"],month=s["release_month"],day=s["release_day"]), axis=1)
df["release_dayofyear"] = df["release_date"].dt.dayofyear

df["release_dayofweek"] = df["release_date"].dt.dayofweek
df['has_homepage'] = 1

df.loc[ pd.isnull(df['homepage']),'has_homepage'] = 0
df['num_Keywords'] = df['Keywords'].apply(lambda x: len(x) if x != {} else 0)
#単語数

df['overview_word_count'] = df['overview'].apply(lambda x: len(str(x).split()))

#文字数

df['overview_char_count'] = df['overview'].apply(lambda x: len(str(x)))
#単語数

df['tagline_word_count'] = df['tagline'].apply(lambda x: len(str(x).split()))

#文字数

df['tagline_char_count'] = df['tagline'].apply(lambda x: len(str(x)))

#taglineがあるかどうか

df['isTaglineNA'] = 0

df.loc[df['tagline'] == 0 ,"isTaglineNA"] = 1 
#単語数

df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))

#文字数

df['title_char_count'] = df['title'].apply(lambda x: len(str(x)))
df['num_cast'] = df['cast'].apply(len)  # 人数
list_of_cast_genders = list(df['cast'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)



df['genders_0_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

df['genders_1_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

df['genders_2_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

# 欠損は平均で埋める

df[['genders_0_cast', 'genders_1_cast']] = df[['genders_0_cast', 'genders_1_cast']].fillna(df[['genders_0_cast', 'genders_1_cast']].mean())
df['num_crew'] = df['crew'].apply(len)  # 人数
# 部署別　のべ人数

department_count = pd.Series(Counter([job for lst in df["crew"].apply(lambda x : [ i["department"] for i in x]).values for job in lst]))

department_count.sort_values(ascending=False)
# job別　のべ人数(top30)

job_count = pd.Series(Counter([job for lst in df["crew"].apply(lambda x : [ i["job"] for i in x]).values for job in lst]))

job_count.sort_values(ascending=False).head(30)
df_crew = { idx : pd.DataFrame([ [crew["department"], crew["job"], crew["name"]] 

                        for crew in x], columns=["department", "job", "name"]) 

    for idx, x in df["crew"].iteritems() }
df_crew = pd.concat(df_crew)

df_crew.head()
def select_job(list_dict, key, value):

    return [ dic["name"] for dic in list_dict if dic[key]==value]
# 各部署の人数

for department in department_count.index:

    df['dep_{}_num'.format(department)] = df["crew"].apply(select_job, key="department", value=department).apply(len)  
# 重要と思われるjobについて、参加作品数上位15人で one-hot-encoding

# 製作、監督、脚本、キャスティング、作曲

df_crewname = pd.DataFrame([], index=df.index)

for job in ["Producer", "Director", "Screenplay", "Casting", "Original Music Composer","Writer"]:

    col = 'job_{}_list'.format(job)

    df[col] = df["crew"].apply(select_job, key="job", value=job)



    top_list = [m[0] for m in Counter([i for j in df[col] for i in j]).most_common(15)]

    for i in top_list:

        df_crewname['{}_{}'.format(job,i)] = df[col].apply(lambda x: i in x)
# 技術部門はdepartment毎に、参加作品数上位15人で one-hot-encoding

for job in ["Sound", "Art", "Costume & Make-Up", "Camera", "Visual Effects"]:

    col = 'department_{}_list'.format(job)

    df[col] = df["crew"].apply(select_job, key="department", value=job)



    top_list = [m[0] for m in Counter([i for j in df[col] for i in j]).most_common(15)]

    for i in top_list:

        df_crewname['{}_{}'.format(job,i)] = df[col].apply(lambda x: i in x)
df.columns
# Animationの人数（アニメ映画で重要そうなので入れてみる）

df['job_Animation_num'] = df["crew"].apply(select_job, key="job", value="Animation").apply(len)
# crew gender

df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

# 欠損は平均で埋める

df[['genders_0_crew', 'genders_1_crew','genders_2_crew']] = df[['genders_0_crew', 'genders_1_crew','genders_2_crew']].fillna(df[['genders_0_crew', 'genders_1_crew','genders_2_crew']].mean())
df['budget_runtime_ratio'] = df['budget']/df['runtime']
sns.distplot(df['budget_runtime_ratio'])
df['budget_popularity_ratio'] = df['budget']/df['popularity']
sns.distplot(df['budget_popularity_ratio'])
rating_na = df[df["rating"].isnull()]
corrmat = df.corr()

plt.subplots(figsize=(25, 20))

sns.heatmap(corrmat, square=True, cmap='coolwarm', annot=True,vmin=-1)

#plt.savefig("TMDBcorr.png")
rating_na["release_year"]
sns.distplot(rating_na["release_year"])
sns.distplot(df["release_year"])
df['budget_popularity2_ratio'] = df['budget']/df['popularity2']
df['budget_year_ratio'] = df['budget']/df['release_year']
df['production_countries_count'] = df['production_countries'].apply(lambda x : len(x))

df['production_companies_count'] = df['production_companies'].apply(lambda x : len(x))
df["collection_name"] = df["belongs_to_collection"].apply(lambda x : x[0]["name"] if len(x)>0 else 0)

le.fit(list(df['collection_name'].fillna('')))

df['collection_name'] = le.transform(df['collection_name'].fillna('').astype(str))
df['mean_pop1_bud'] = df.groupby('popularity')['budget'].transform('mean')

df['mean_pop2_bud'] = df.groupby('popularity2')['budget'].transform('mean')

df['mean_year_bud'] = df.groupby('release_year')['budget'].transform('mean')

df['mean_pop1_rate'] = df.groupby('popularity')['rating'].transform('mean')

df['mean_pop2_rate'] = df.groupby('popularity2')['rating'].transform('mean')

df['mean_rate_tV'] = df.groupby('rating')['totalVotes'].transform('mean')
df['mean_pop1_bud']
df['runtime_to_mean_year'] = df['runtime'] / df.groupby("release_year")["runtime"].transform('mean')

df['popularity_to_mean_year'] = df['popularity'] / df.groupby("release_year")["popularity"].transform('mean')

df['budget_to_mean_year'] = df['budget'] / df.groupby("release_year")["budget"].transform('mean')
df['runtime_to_mean_year']
df_features = pd.concat(dfdic_feature, axis=1)
df['job_Writer_list']
df_features.index = df.index
df.info()
df_use = df[['num_cast', 'genders_0_cast','runtime_to_mean_year','budget_to_mean_year',"log_budget",

       'genders_1_cast','genders_2_cast', 'num_crew', 'genders_0_crew', 'genders_1_crew','genders_2_crew',

             "tagline_word_count","overview_word_count","title_word_count","has_homepage",

            'popularity','runtime','release_year', 'release_month','release_dayofweek',"num_genres"

            ,"popularity2","rating","totalVotes",'isOriginalLanguageEng',

             'budget_runtime_ratio','budget_popularity_ratio','budget_year_ratio','budget_popularity2_ratio',

            'production_countries_count','production_companies_count','mean_pop1_bud','mean_pop2_bud','mean_year_bud','mean_pop1_rate',

            'mean_pop2_rate','mean_rate_tV',

           'dep_Directing_num', 'dep_Writing_num', 'dep_Production_num',

       'dep_Sound_num', 'dep_Camera_num', 'dep_Editing_num', 'dep_Art_num',

       'dep_Costume & Make-Up_num', 'dep_Crew_num', 'dep_Lighting_num',

       'dep_Visual Effects_num', 'dep_Actors_num', 'job_Animation_num' ]]
df_use = pd.concat([df_use, df_features], axis=1)
df_features.columns
df_use.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df_use.columns]
df_use.columns
df_use.isnull().sum().sum()
df_use
trainX = df_use.iloc[:train.shape[0],:].reset_index(drop=True)

test_X = df_use.iloc[train.shape[0]:,:].reset_index(drop=True)

trainy = np.log1p(train["revenue"])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(trainX,trainy,test_size=0.3,random_state=100)
xgbmodel = xgb.XGBRegressor(max_depth=6, 

                            min_child_weight=3,

                            alpha = 0.5,

                            learning_rate=0.05, 

                            n_estimators=150, 

                            objective='reg:linear', 

                            gamma=0.01,  

                            silent=1,

                            subsample=0.8, 

                            colsample_bytree=0.8)
xgbmodel.fit(X_train, y_train)
pred_train1 = xgbmodel.predict(X_train)

pred_test1 = xgbmodel.predict(X_test)
from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(y_train, pred_train1)))

print(np.sqrt(mean_squared_error(y_test, pred_test1)))
plt.figure(figsize=(20,15))

importances = pd.Series(xgbmodel.feature_importances_, index = df_use.columns)

importances = importances.sort_values()

importances.plot(kind = "barh")

plt.title("imporance in the xgboost Model")

plt.show()
pred_xgb = xgbmodel.predict(test_X)
test_id = test["id"]
pred_xgb = pd.DataFrame(np.exp(pred_xgb)-1,columns=["revenue"])

pred_xgb
sub=pd.concat([test_id, pred_xgb],axis=1)
sub.to_csv('TMDB_xgb.csv',index=False)
lgbmodel = lgb.LGBMRegressor(n_estimators=150, 

                             objective='regression', 

                             metric='rmse',

                             max_depth = 5,

                             num_leaves=30, 

                             min_child_samples=30,

                             learning_rate=0.05,

                             boosting = 'gbdt',

                             min_data_in_leaf= 15,

                             feature_fraction = 0.9,

                             bagging_freq = 1,

                             bagging_fraction = 0.9,

                             importance_type='gain',

                             lambda_l1 = 0.2, 

                             subsample=.8, 

                             colsample_bytree=.8,

                             use_best_model=True)
lgbmodel.fit(X_train, y_train)
pred_train2 = lgbmodel.predict(X_train)

pred_test2 = lgbmodel.predict(X_test)
#rmse

print(np.sqrt(mean_squared_error(y_train, pred_train2)))

print(np.sqrt(mean_squared_error(y_test, pred_test2)))
pred_lgb = lgbmodel.predict(test_X)
pred_lgb = pd.DataFrame(np.exp(pred_lgb)-1,columns=["revenue"])

pred_lgb
sub1=pd.concat([test_id, pred_lgb],axis=1)
sub1.to_csv('TMDB_lgb.csv',index=False)
catmodel = cat.CatBoostRegressor(iterations=2000, 

                                 learning_rate=0.01, 

                                 depth=8, 

                                 eval_metric='RMSE',

                                 colsample_bylevel=0.8,

                                 bagging_temperature = 0.2,

                                 metric_period = None,

                                 early_stopping_rounds=200)
catmodel.fit(X_train, y_train)
pred_train3 = catmodel.predict(X_train)

pred_test3 = catmodel.predict(X_test)
#rmse

print(np.sqrt(mean_squared_error(y_train, pred_train3)))

print(np.sqrt(mean_squared_error(y_test, pred_test3)))
pred_cat = catmodel.predict(test_X)
pred_cat = pd.DataFrame(np.exp(pred_cat)-1,columns=["revenue"])

pred_cat
sub2=pd.concat([test_id, pred_cat],axis=1)
sub2.to_csv('TMDB_cat.csv',index=False)
ansamble = 0.4 * pred_lgb["revenue"] + 0.2 * pred_xgb["revenue"] + 0.4 * pred_cat["revenue"]
sub3=pd.concat([test_id, ansamble],axis=1)
sub3
sub3.to_csv('TMDB_ansamble.csv',index=False)
ansamble2 = 0.35 * pred_lgb["revenue"] + 0.3 * pred_xgb["revenue"] + 0.35 * pred_cat["revenue"]
sub4=pd.concat([test_id, ansamble2],axis=1)
sub4
sub4.to_csv('TMDB_ansamble2.csv',index=False)
ansamble3 = 0.25 * pred_lgb["revenue"] + 0.25 * pred_xgb["revenue"] + 0.5 * pred_cat["revenue"]
sub5=pd.concat([test_id, ansamble3],axis=1)
sub5
sub5.to_csv('TMDB_ansamble3.csv',index=False)