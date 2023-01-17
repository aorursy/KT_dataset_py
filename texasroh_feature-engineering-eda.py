import pandas as pd



train = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/train.csv')



print(train.shape)

train.head()
test = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/test.csv')

print(test.shape)
data = pd.concat([train, test], sort=False).reset_index()

data = data.drop('index', axis=1)

print(data.shape)
import ast



dict_columns = ['belongs_to_collection','genres','production_companies','production_countries','spoken_languages','Keywords','cast','crew']



def get_dict(item):

    try:

        new_item = ast.literal_eval(item)

    except:

        new_item = {}

    return new_item



for col in dict_columns:

#     train[col] = train[col].apply(lambda x: {} if pd.isnull(x) else ast.literal_eval(x))

#     test[col] = test[col].apply(lambda x: {} if pd.isnull(x) else ast.literal_eval(x))

#     data[col] = data[col].apply(lambda x: {} if pd.isnull(x) else ast.literal_eval(x))

#     train[col] = train[col].apply(lambda x: get_dict(x))

#     test[col] = test[col].apply(lambda x: get_dict(x))

    data[col] = data[col].apply(lambda x: get_dict(x))
import matplotlib.pyplot as plt

%matplotlib inline



train_null_pct = train.isnull().sum().sort_values() / len(train)

test_null_pct = test.isnull().sum().sort_values() / len(test)



fig, ax = plt.subplots(1,2, figsize=(10,8), sharey=False)

fig.subplots_adjust(wspace=0.8)

ax[0].barh(train_null_pct.index, train_null_pct)

ax[0].set_title('Train dataset Null')

ax[0].set_xlabel('Null proportion')

ax[1].barh(test_null_pct.index, test_null_pct)

ax[1].set_title('Test dataset Null')

ax[1].set_xlabel('Null proportion')



print(data.isnull().sum())
data['has_homepage'] = data['homepage'].apply(lambda x: 0 if pd.isnull(x) else 1)

data = data.drop('homepage', axis=1)

data.shape
data['tagline'] = data['tagline'].fillna('')
data['overview'] = data['overview'].fillna('')
data = data.drop('poster_path', axis=1)

data.shape
data.loc[data['release_date'].isnull(), 'title']

## Jails, Hospitals & Hip-Hop

# It released on May 2000

data.loc[data['release_date'].isnull(), 'release_date'] = '05/01/2000'
data.loc[data['title'].isnull(), ['id','original_title']]

data.loc[data['id']==5399, 'title'] = 'The Life of Guskou Budori'  #グスコーブドリの伝記

data.loc[data['id']==5426, 'title'] = ''  #La Vérité si je Mens ! 3  # couldn't find english title

data.loc[data['id']==6629, 'title'] = 'Barefoot'  #Barefoot
data.loc[data['runtime'].isnull(), 'runtime'] = data['runtime'].median()

data.loc[data['runtime']==0, 'runtime'] = data['runtime'].median()
print(data['status'].isnull().sum())

print(train['status'].value_counts())

print(test['status'].value_counts())
data = data.drop('status', axis=1)

data.shape
data['revenue'].min()
import numpy as np



fig, ax = plt.subplots(1,2, figsize=(10,5))

ax[0].hist(data['revenue'])

ax[0].set_title('revenue')

ax[1].hist(np.log(data['revenue']+1))

ax[1].set_title('log_revenue')

plt.show()



data['log_revenue'] = np.log(data['revenue']+1)  # add 1 to avoid log(0)
data['belongs_to_collection'] = data['belongs_to_collection'].apply(lambda x: len(x))
tmp_train = data.iloc[:3000]

isin_collection_rev = tmp_train.loc[tmp_train['belongs_to_collection']==1, 'revenue']

notin_collection_rev = tmp_train.loc[tmp_train['belongs_to_collection']==0, 'revenue']



isin_collection_rev_log = tmp_train.loc[tmp_train['belongs_to_collection']==1, 'log_revenue']

notin_collection_rev_log = tmp_train.loc[tmp_train['belongs_to_collection']==0, 'log_revenue']



fig, ax = plt.subplots(1,2, figsize=(12,5))

ax[0].boxplot([isin_collection_rev_log, notin_collection_rev_log])

ax[0].set_title('log_revenue')

ax[0].set_xticklabels(['in_collection', 'not_in_collection'])

ax[1].boxplot([isin_collection_rev, notin_collection_rev])

ax[1].set_title('revenue')

ax[1].set_xticklabels(['in_collection', 'not_in_collection'])

plt.show()
plt.hist(data['budget'])

plt.xlabel('Budget')

plt.ylabel('Counts')
data['log_budget'] = np.log(data['budget']+1)  # add 1 to avoid log(0)

fig, ax = plt.subplots(1,2, figsize=(10,5))

ax[0].scatter(data['budget'], data['revenue'], alpha=0.1)

ax[0].set_title('budget - revenue')

ax[0].set_xlabel('budget')

ax[0].set_ylabel('revenue')

ax[1].scatter(data['log_budget'], data['log_revenue'], alpha=0.1)

ax[1].set_title('log_budget - log_revenue')

ax[1].set_xlabel('log_budget')

ax[1].set_ylabel('log_revenue')
gen_cnt = data['genres'].apply(lambda x: len(x)).value_counts()

plt.bar(gen_cnt.index, gen_cnt)

plt.xticks(range(gen_cnt.index.max()+1))

plt.title('Number of genres that movies in')

plt.xlabel('Number of genres')

plt.ylabel('Movie counts')

plt.show()
import collections

from wordcloud import WordCloud



total_gen_list = []



def gen_list(x):

    for i in x:

        total_gen_list.append(i['name'])



        

fig, ax = plt.subplots(1,2, figsize=(20,7))

data['genres'].apply(lambda x: gen_list(x))

gen_cnt = collections.Counter(total_gen_list).most_common()

for gen, cnt in gen_cnt[::-1]:

    ax[0].barh(gen,cnt)

ax[0].set_title('Genre Frequencies')



wordcloud = WordCloud(background_color='white', width=800, height=500).generate_from_frequencies(dict(gen_cnt))

ax[1].axis('off')

ax[1].imshow(wordcloud, interpolation = 'bilinear')

ax[1].set_title('Genre majorities')

plt.show()
data['genres_list'] = data['genres'].apply(lambda x: [i['name'] for i in x])



for gen in dict(gen_cnt).keys():

    data['genre_'+gen] = data['genres_list'].apply(lambda x: 1 if gen in x else 0)
tmp_train = data[:3000]

plt.figure(figsize=(6,6))

for idx, gen in enumerate(dict(gen_cnt).keys()):

    plt.boxplot(tmp_train.loc[tmp_train['genre_'+gen]==1,'log_revenue'], labels=[gen], positions=range(idx, idx+1), vert=False)

plt.xlabel('log_revenue')
data = data.drop(['genres', 'genres_list'], axis=1)
data = data.drop('imdb_id', axis=1)
data['year'] = data['release_date'].str.split('/').apply(lambda x: 2000+int(x[2]) if int(x[2]) < 19 else 1900+int(x[2]))

data.loc[data['year']==3900,'year'] = 2000   ## there is a typo in dataset

data['month'] = data['release_date'].str.split('/').apply(lambda x: int(x[0]))

data = data.drop('release_date', axis=1)
fig, ax = plt.subplots(2,2, figsize=(13,10))

ax[0,0].scatter(data['year'], data['revenue'], alpha=0.3, s=20)

ax[0,0].set_title('year - revenue')

ax[0,0].set_xlabel('year')

ax[0,0].set_ylabel('revenue')

ax[0,1].scatter(data['year'], data['log_revenue'], alpha=0.3, s=20)

ax[0,1].set_xlabel('year')

ax[0,1].set_ylabel('log_revenue')

ax[0,1].set_title('year - log_revenue')

ax[1,0].scatter(data['month'], data['revenue'], alpha=0.3, s=20)

ax[1,0].set_xlabel('month')

ax[1,0].set_ylabel('revenue')

ax[1,0].set_title('month - revenue')

ax[1,1].scatter(data['month'], data['log_revenue'], alpha=0.3, s=20)

ax[1,1].set_title('month - log_revenue')

ax[1,1].set_xlabel('month')

ax[1,1].set_ylabel('log_revenue')

plt.show()
data = pd.get_dummies(data, columns=['month'], drop_first=True)
def find_director(x):

    director=''

    for i,v in enumerate(x):

        if v['job']=='Director':

            director = v['name']

    return director



def find_writer(x):

    writer=''

    for i,v in enumerate(x):

        if v['job'] == 'Writer':

            writer = v['name']

    return writer



data['director'] = data['crew'].apply(lambda x: find_director(x))

data['writer'] = data['crew'].apply(lambda x: find_writer(x))

collections.Counter(data['director']).most_common(20)
collections.Counter(data['writer']).most_common(20)
data = data.drop(['director', 'writer', 'crew'], axis = 1)
data['cast_list'] = data['cast'].apply(lambda x: [i['name'] for i in x])

data['n_cast'] = data['cast'].apply(lambda x: len(x))
total_cast_list = []

def get_cast_list(x):

    total_cast_list.extend(x)



data['cast_list'].apply(lambda x: get_cast_list(x))

top_cast = list(dict(collections.Counter(total_cast_list).most_common(100)).keys())
collections.Counter(total_cast_list).most_common(20)
for cast in top_cast:

    data['cast_'+cast] = data['cast_list'].apply(lambda x: 1 if cast in x else 0)
tmp_train = data[:3000]

for idx, cast in enumerate(top_cast[:23][::-1]):

    cast_rev = tmp_train.loc[tmp_train['cast_'+cast]==1, 'log_revenue']

    plt.boxplot(cast_rev, positions=range(idx, idx+1), labels=[cast], vert=False)

plt.xlabel('log_revenue')
data = data.drop(['cast', 'cast_list'], axis=1)
# choose language which 

top_langs = dict(data['original_language'].value_counts()[:17]).keys()



for lang in top_langs:

    data['lang_'+lang] = data['original_language'].apply(lambda x: 1 if lang == x else 0)
data['n_spoken_languages'] = data['spoken_languages'].apply(lambda x: len(x))
for lang in list(top_langs)[::-1]:

    plt.barh(lang, data.loc[data['lang_'+lang]==1, 'log_revenue'])

plt.xlabel('log_revenue')

plt.title('original_language')
plt.scatter(data['n_spoken_languages'], data['log_revenue'], alpha=.3)

plt.xlabel('n_spoken_languages')

plt.ylabel('log_revenue')

plt.title('spoken_languages')
data = data.drop(['original_language','spoken_languages','n_spoken_languages'], axis=1)
data['keyword_str'] = data['Keywords'].apply(lambda x: ', '.join([i['name'] for i in x]))
data['text'] = data['title'] + '. '+ data['tagline'] + '. ' + data['overview'] + '. ' + data['keyword_str']
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation



vect = CountVectorizer(ngram_range=(1,3), stop_words='english')

X = vect.fit_transform(data['text'])

lda = LatentDirichletAllocation(n_components=10, random_state = 0)

document_topics = lda.fit_transform(X)
n = 8

# Get features (tokens) from CountVectorizer

feature_names = np.array(vect.get_feature_names())

# Find top n tokens

topics = dict()

for idx, component in enumerate(lda.components_): 

    top_n_indices = component.argsort()[:-(n + 1): -1] 

    topic_tokens = [feature_names[i] for i in top_n_indices] 

    topics[idx] = topic_tokens



topics
plt.figure(figsize=(5,5))

for k, v in collections.Counter(document_topics.argmax(axis=1)).items():

    plt.bar(k,v)

plt.xlabel('Topic clusters')

plt.ylabel('Number of movies')

plt.title('Topic frequencies')
data['topics'] = document_topics.argmax(axis=1)

tmp_train = data[:3000]

plt.figure(figsize=(5,5))

for idx in range(10):

    plt.boxplot(tmp_train.loc[tmp_train['topics']==idx, 'log_revenue'],positions=range(idx, idx+1), labels=[idx])

plt.xlabel('topics')

plt.ylabel('log_revenue')
data = pd.get_dummies(data, columns=['topics'], drop_first=True)
data = data.drop(['original_title','overview','tagline','title','Keywords', 'keyword_str','text'], axis=1)
data['n_production_countries'] = data['production_countries'].apply(lambda x: len(x))

data['n_production_companies'] = data['production_companies'].apply(lambda x: len(x))
company_list = []

def get_company_list(x):

    for i in x:

        company_list.append(i['name'])

data['production_companies'].apply(lambda x: get_company_list(x))

for company in dict(collections.Counter(company_list).most_common(30)).keys():

    data['production_'+company] = data['production_companies'].apply(lambda x: 1 if company in [i['name'] for i in x] else 0)
data = data.drop(['production_companies', 'production_countries'], axis=1)
fig, ax = plt.subplots(1,2, figsize=(12,5))

ax[0].scatter(data['n_production_countries'], data['log_revenue'], alpha=0.1)

ax[0].set_xlabel('n_production_countries')

ax[0].set_ylabel('log_revenue')

ax[1].scatter(data['n_production_companies'], data['log_revenue'], alpha=0.1)

ax[1].set_xlabel('n_production_companies')

ax[1].set_ylabel('log_revenue')
tmp_train = data[:3000]

plt.figure(figsize=(10,10))

for idx, company in enumerate(dict(collections.Counter(company_list).most_common(30)).keys()):

    com_rev = tmp_train.loc[tmp_train['production_'+company]==1, 'log_revenue']

    plt.boxplot(com_rev, positions=range(idx, idx+1), labels = [company], vert=False)

plt.xlabel('log_revenue')
data['runtime_cat'] = pd.qcut(data['runtime'],10, labels=False)
plt.scatter(data['runtime_cat'], data['log_revenue'], alpha=0.1)
tmp_train = data[:3000]

for i in range(10):

    plt.boxplot(tmp_train.loc[data['runtime_cat']==i,'log_revenue'], positions=range(i, i+1))

plt.xlabel('runtime categories')

plt.ylabel('log_revenue')
data = data = pd.get_dummies(data, columns=['runtime_cat'], drop_first=True)

data = data.drop(['runtime'],axis=1)
def rmse_score(y1, y2):

    return np.sqrt(np.power(y1-y2,2).mean())
X = data[:3000].drop(['id','revenue','log_revenue', 'budget'],axis=1)

y = data[:3000]['log_revenue']



sub_X = data[3000:].drop(['id','revenue','log_revenue','budget'], axis=1)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
from sklearn.linear_model import LinearRegression



LR_model = LinearRegression()

LR_model.fit(X_train,y_train)

y_hat = LR_model.predict(X_test)

print(rmse_score(y_hat, y_test))

sub_y = LR_model.predict(sub_X)
sub_csv = pd.DataFrame({'id':data[3000:]['id'], 'revenue': np.exp(sub_y)})

sub_csv.to_csv('LR_predict.csv', index=False)



## 2.40021
from sklearn.svm import SVR



SVR_model = SVR(C = 5)

SVR_model.fit(X_train, y_train)



y_hat = SVR_model.predict(X_test)

print(rmse_score(y_hat, y_test))

sub_y = SVR_model.predict(sub_X)
sub_csv = pd.DataFrame({'id':data[3000:]['id'], 'revenue': np.exp(sub_y)})

sub_csv.to_csv('SVR_predict.csv', index=False)



## 2.27148   # c=1.0

## 2.21529   # c=5

## 2.21807   # c=10
from sklearn.ensemble import RandomForestRegressor



RF_model = RandomForestRegressor(random_state =0, n_estimators=500, max_depth=10)

RF_model.fit(X_train, y_train)



y_hat = RF_model.predict(X_test)

print(rmse_score(y_hat, y_test))

sub_y = RF_model.predict(sub_X)
sub_csv = pd.DataFrame({'id':data[3000:]['id'], 'revenue': np.exp(sub_y)})

sub_csv.to_csv('RF_predict.csv', index=False)



## 2.20717   #n_esimators=200, max_depth=8

## 2.20054   #n_estimators=500, max_depth=10
from sklearn.neural_network import MLPRegressor

MLP_model = MLPRegressor(random_state = 0, hidden_layer_sizes=(50,50))

MLP_model.fit(X_train, y_train)



y_hat = MLP_model.predict(X_test)

print(rmse_score(y_hat, y_test))

sub_y = MLP_model.predict(sub_X)
sub_csv = pd.DataFrame({'id':data[3000:]['id'], 'revenue': np.exp(sub_y)})

sub_csv.to_csv('MLP_predict.csv', index=False)



## 2.29088   hidden=(30,500)

## 2.40497   hidden=(100,)

## 2.26413   hidden=(200,30)

## 2.31559   hidden=(50,50)