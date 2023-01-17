import numpy as np

import pandas as pd  

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import string

import re

%matplotlib inline

%precision 3
train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')
train.head()
train.info()
train_y = train["revenue"]
df = pd.concat([train, test])
df.drop(columns=['overview','status','imdb_id','poster_path','original_title'], inplace = True)
#辞書型に変換

import ast

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

        

dfx = text_to_dict(train)

for col in dict_columns:

       train[col]=dfx[col]
train[train["runtime"].isnull()]
train.loc[train['id'] == 1336,'runtime'] = 130 #kololyovの上映時間を調べて入力

train.loc[train['id'] == 2303,'runtime'] = 80 #HappyWeekendの上映時間を調べて入力
train["runtime"].isnull().sum()
train[train["runtime"]==0]
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
sns.distplot(train["runtime"], kde=False, rug=False)
test[test["runtime"].isnull()]
test.loc[test['id'] == 3244,'runtime'] = 93 #La caliente niña Julietta	の上映時間を調べて入力

test.loc[test['id'] == 4490,'runtime'] = 90 #Pancho, el perro millonarioの上映時間を調べて入力

test.loc[test['id'] == 4633,'runtime'] = 108 #Nunca en horas de claseの上映時間を調べて入力

test.loc[test['id'] == 6818,'runtime'] = 90 #Miesten välisiä keskustelujaの上映時間を調べて入力
test[test["runtime"]==0]
test.loc[test['id'] == 4074,'runtime'] = 103 #Shikshanachya Aaicha Ghoの上映時間を調べて入力

test.loc[test['id'] == 4222,'runtime'] = 91 #Street Knightの上映時間を調べて入力

test.loc[test['id'] == 4431,'runtime'] = 96 #Plus oneの上映時間を調べて入力

test.loc[test['id'] == 5520,'runtime'] = 86 #Glukhar v kinoの上映時間を調べて入力

test.loc[test['id'] == 5845,'runtime'] = 83 #Frau Müller muss weg!の上映時間を調べて入力

test.loc[test['id'] == 5849,'runtime'] = 140 #Shabdの上映時間を調べて入力

test.loc[test['id'] == 6210,'runtime'] = 104 #The Last Breathの上映時間を調べて入力

test.loc[test['id'] == 6804,'runtime'] = 140 #Chaahat Ek Nasha...の上映時間を調べて入力

test.loc[test['id'] == 7321,'runtime'] = 87 #El truco del mancoの上映時間を調べて入力
sns.distplot(test["runtime"], kde=False, rug=False)
corrmat = train.corr()

plt.subplots(figsize=(12, 8))

sns.heatmap(corrmat, square=True, cmap='coolwarm', annot=True,vmin=-1)

#plt.savefig("TMDBcorr.png")
train[train["budget"].isnull()]
train[train["budget"]==0]
test[test["runtime"].isnull()]
test[test["budget"]==0]
#release_dateを年、月、日に分解

def date_features(df):

    df['release_date'] = pd.to_datetime(df['release_date'])

    df['release_year'] = df['release_date'].dt.year

    df['release_month'] = df['release_date'].dt.month

    df['release_day'] = df['release_date'].dt.day

    df['release_quarter'] = df['release_date'].dt.quarter

    df.drop(columns=['release_date'], inplace=True)

    return df



train=date_features(train)

test=date_features(test)



train['release_year'].head(10)
df = pd.concat([train, test])
df.drop(columns=['status','imdb_id','poster_path','original_title'], inplace = True)
#budgetが0の物を予測（テスト）、0でない物をtrainingデータとする

budget0 = df[df["budget"] == 0]

budget = df[df["budget"] != 0]

train_X = budget[["popularity","runtime"]]

train_y = budget["budget"]

test_X = budget0[["popularity","runtime"]]

test_y = budget0["budget"]
budget0
#budgetが0の物を線形回帰で予測

from sklearn.linear_model import RidgeCV

rcv= RidgeCV(cv=3, alphas = 10**np.arange(-2, 2, 0.1))

rcv.fit(train_X, train_y)

y_pred = rcv.predict(test_X)
budget0["id"].index = range(0,2023)
budget_pred = pd.DataFrame(y_pred,columns=["pred"])

budget_id = pd.DataFrame(budget0["id"],columns=["id"])

budget_pred = pd.concat([budget_id,budget_pred],axis = 1)

budget_pred
budget_pred.describe()
#予算が0を下回っているものはおかしいので0に戻す。

budget_pred.loc[budget_pred["pred"] < 0, "pred"] = 0
df = pd.merge(df, budget_pred, on="id", how="left") 

df.loc[budget_pred["id"]-1, "budget"] = df.loc[budget_pred["id"]-1, "pred"]

df = df.drop("pred", axis=1)
df
corrmat = df.corr()

plt.subplots(figsize=(12, 8))

sns.heatmap(corrmat, square=True, cmap='coolwarm', annot=True,vmin=-1)

#plt.savefig("TMDBcorr.png")
df["overview"]
df["overview"].isnull().sum()
df["overview"]=df["overview"].apply(lambda x : str(x))

train["overview"]=train["overview"].apply(lambda x : str(x))
#全て小文字に変換

def lower_text(text):

    return text.lower()
df["overview"]=df["overview"].apply(lambda x : lower_text(x))

train["overview"]=train["overview"].apply(lambda x : lower_text(x))
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

df["overview"] = df["overview"].replace(shortened)

train["overview"] = train["overview"].replace(shortened)
#記号の排除

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)
df["overview"]=df["overview"].apply(lambda x : remove_punct(x))

train["overview"]=train["overview"].apply(lambda x : remove_punct(x))
# 連続した数字を0で置換

def normalize_number(text):

    replaced_text = re.sub(r'\d+', '0', text)

    return replaced_text
df["overview"]=df["overview"].apply(lambda x : normalize_number(x))

train["overview"]=train["overview"].apply(lambda x : normalize_number(x))
#レンマ化

from nltk.stem.wordnet import WordNetLemmatizer



wnl = WordNetLemmatizer()

df["overview"]=df["overview"].apply(wnl.lemmatize)

train["overview"]=train["overview"].apply(wnl.lemmatize)
#空白ごとの文章の分割

df["overview"]=df["overview"].apply(lambda x : str(x).split())

train["overview"]=train["overview"].apply(lambda x : str(x).split())
df_overview = df["overview"]
def most_common(docs, n=100):#(文章、上位n個の単語)#上位n個の単語を抽出

    fdist = Counter()

    for doc in docs:

        for word in doc:

            fdist[word] += 1

    common_words = {word for word, freq in fdist.most_common(n)}

    print('{}/{}'.format(n, len(fdist)))

    return common_words
most_common(df_overview,100)
def get_stop_words(docs, n=100, min_freq=1):#上位n個の単語、頻度がmin_freq以下の単語を列挙（あまり特徴のない単語等）

    fdist = Counter()

    for doc in docs:

        for word in doc:

            fdist[word] += 1

    common_words = {word for word, freq in fdist.most_common(n)}

    rare_words = {word for word, freq in fdist.items() if freq <= min_freq}

    stopwords = common_words.union(rare_words)

    print('{}/{}'.format(len(stopwords), len(fdist)))

    return stopwords
stopwords = get_stop_words(df_overview)

stopwords
def remove_stopwords(words, stopwords):#不要な単語を削除

    words = [word for word in words if word not in stopwords]

    return words
df["overview"]=df["overview"].apply(lambda x : remove_stopwords(x,stopwords))

train["overview"]=train["overview"].apply(lambda x : remove_stopwords(x,stopwords))
df["overview"]
df["overview"]=[" ".join(review) for review in df["overview"].values]

train["overview"]=[" ".join(review) for review in train["overview"].values]
df["overview"]
from sklearn.feature_extraction.text import TfidfVectorizer#ベクトル化

vec_tfidf = TfidfVectorizer()

X = vec_tfidf.fit_transform(df["overview"])

Tfid_overview = pd.DataFrame(X.toarray(), columns=vec_tfidf.get_feature_names())



X2 = vec_tfidf.fit_transform(df["overview"])

Tfid_train_overview = pd.DataFrame(X2.toarray(), columns=vec_tfidf.get_feature_names())
Tfid_overview
"""

#目的変数とベクトルの線形回帰による単語の重要度比較（途中）

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(X2, np.log1p(train['revenue']))"""
"""

coef = pd.Series(linreg.coef_, index=Tfid_overview.columns)

df_coef = pd.DataFrame(coef[coef!=0], columns=["coef"])

df_coef.sort_values("coef", ascending=False)"""
#単語数

df['tagline_word_count'] = df['tagline'].apply(lambda x: len(str(x).split()))
#文字数

df['tagline_char_count'] = df['tagline'].apply(lambda x: len(str(x)))
# 記号の個数

df['tagline_punctuation_count'] = df['tagline'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
df['tagline']=df['tagline'].apply(lambda x : str(x))
df["tagline"] = df["tagline"].replace(shortened)
df['tagline']=df['tagline'].apply(lambda x : lower_text(x))
df['tagline']=df['tagline'].apply(lambda x : remove_punct(x))
df["tagline"]=df["tagline"].apply(lambda x : normalize_number(x))
df['tagline']=df['tagline'].apply(lambda x : str(x).split())
tagline = df["tagline"]
most_common(tagline)
stopwords = get_stop_words(tagline)
df['tagline']=df['tagline'].apply(lambda x : remove_stopwords(x,stopwords))
nan = {"nan"}

def remove_nan(words):

    words = [word for word in words if word not  in nan]

    return words
df['tagline']=df['tagline'].apply(lambda x : remove_nan(x))
df['tagline']=[" ".join(review) for review in df['tagline'].values]
#ベクトル化

X = vec_tfidf.fit_transform(df['tagline'])

Tfid_tagline = pd.DataFrame(X.toarray(), columns=vec_tfidf.get_feature_names())
Tfid_tagline
df.columns
df_use = df[["runtime",'budget','tagline_char_count']]
df_use.columns
df_use = pd.concat([df_use,Tfid_overview],axis=1)
#使用する変数

df_use = df_use.loc[:,~df_use.columns.duplicated()]
trainX = df_use.iloc[:train.shape[0],:].reset_index(drop=True)

test_X = df_use.iloc[train.shape[0]:,:].reset_index(drop=True)

trainy = train["revenue"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(trainX,trainy,test_size=0.3,random_state=100)
import xgboost as xgb
"""

dtrain = xgb.DMatrix(X_train, label=y_train)  

dvalid = xgb.DMatrix(X_test, label=y_test)"""
#param = {'max_depth': 5, 'eta': 0.5, 'objective': 'reg:squaredlogerror', 'eval_metric': 'rmsle','alpha':0.5} 
"""

evallist = [(dvalid, 'eval'), (dtrain, 'train')]  

num_round = 20

bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=5)  """
reg = xgb.XGBRegressor()
"""

# ハイパーパラメータ探索

from sklearn.model_selection import GridSearchCV

#reg_cv = GridSearchCV(reg, {'max_depth': [2,4,6], 'n_estimators': [100]})

reg.fit(X_train, y_train)

#print (reg_cv.best_params_, reg_cv.best_score_)"""
"""

from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error

best_rmse = 1

for num in range(1,100):

    alpha = num*0.0001

    reg=Lasso(alpha=alpha,max_iter=3000)

    reg.fit(X_train,y_train)

    y_pred=reg.predict(X_test)

    rmse=np.sqrt(mean_squared_error(y_pred,y_test))

    if best_rmse>rmse:

        best_rmse=rmse

        best_alpha=alpha

print("best_alpha",alpha,"rmse",best_rmse)"""
"""

reg1=Lasso(alpha=0.0099,max_iter=3000)

reg1.fit(X_train,y_train)

y_pred1=reg.predict(test_X)"""
#y_pred1