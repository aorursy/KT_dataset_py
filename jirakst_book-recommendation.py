# Basic

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt 



# System

import warnings

import os

warnings.filterwarnings("ignore")

%matplotlib inline



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
users = pd.read_csv('/kaggle/input/bx-csv-dump/BX-Users.csv', error_bad_lines=False, delimiter=';', encoding = 'ISO-8859-1') #encoding = "latin-1"
books = pd.read_csv('/kaggle/input/bx-csv-dump/BX-Books.csv', error_bad_lines=False, delimiter=';', encoding = 'ISO-8859-1') #encoding = "latin-1
ratings = pd.read_csv('/kaggle/input/bx-csv-dump/BX-Book-Ratings.csv', error_bad_lines=False, delimiter=';', encoding = 'ISO-8859-1')
users.shape
books.shape
ratings.shape
users.columns
ratings.columns
books.columns
data = pd.merge(ratings, users, on='User-ID', how='inner')
data = pd.merge(data, books, on='ISBN', how='inner')
# Check

data.columns
#data.rename(columns={'Book-Rating':'BookRating', 'User-ID':'UserID'},inplace=True)
# Drop (TODO: image analysis?)

'''to_drop = ['Image-URL-S', 'Image-URL-M', 'Image-URL-L']



data = data.drop(to_drop, axis=1, inplace=False)'''
data.shape
print('Size of the dataset is: ', data.memory_usage().sum() / 1024**2, ' MB')
# TODO: EDA in Power BI
data.shape
data.head(5)
data.info()
print('Number of books: ', data['ISBN'].nunique())
print('Number of users: ',data['User-ID'].nunique())
print('Missing data [%]')

round(data.isnull().sum() / len(data) * 100, 4)
sns.distplot(data['Age'].dropna(), kde=False)
print('Number of outliers: ', sum(data['Age'] > 100))
data['Book-Rating'] = data['Book-Rating'].replace(0, None)
sns.countplot(x='Book-Rating', data=data)
print('Average book rating: ', round(data['Book-Rating'].mean(), 2))
# Publication by Year

year = pd.to_numeric(data['Year-Of-Publication'], 'coerse').fillna(2099, downcast = 'infer')

sns.distplot(year, kde=False, hist_kws={"range": [1945,2020]})
country = data['Location'].apply(lambda row: str(row).split(',')[-1])

data.groupby(country)['Book-Rating'].count().sort_values(ascending=False).head(10)
# Cast to numeric

data['Year-Of-Publication'] = pd.to_numeric(data['Year-Of-Publication'], 'coerse').fillna(2099, downcast = 'infer')
data['Book-Rating'] = data['Book-Rating'].replace(0, None)
data['Age'] = np.where(data['Age']>90, None, data['Age'])
# Categorical feautes

data[['Book-Author', 'Publisher']] = data[['Book-Author', 'Publisher']].fillna('Unknown')
# Check cat features

data[['Book-Author', 'Publisher']].isnull().sum()
# Age

median = data["Age"].median()

std = data["Age"].std()

is_null = data["Age"].isnull().sum()

rand_age = np.random.randint(median - std, median + std, size = is_null)

age_slice = data["Age"].copy()

age_slice[pd.isnull(age_slice)] = rand_age

data["Age"] = age_slice

data["Age"] = data["Age"].astype(int)
# Check Age

data['Age'].isnull().sum()
data['Country'] = data['Location'].apply(lambda row: str(row).split(',')[-1])
# Drop irelevant

data = data.drop('Location', axis=1)
data['Country'].head()
#TODO: country/language analysis (Babel lib?)
#en_list = ['usa', 'canada', 'united kingdom', 'australia']
#data[data['Country'].isin(en_list)]
df = data
# Relevant score

df = df[df['Book-Rating'] >= 6]
# Check

df.groupby('ISBN')['User-ID'].count().describe()
df = df.groupby('ISBN').filter(lambda x: len(x) >= 5)
df.groupby('User-ID')['ISBN'].count().describe()
df = df.groupby('User-ID').filter(lambda x: len(x) >= 5)
df.shape
df_p = df.pivot_table(index='ISBN', columns='User-ID', values='Book-Rating')
# Select users who liked LOTR

lotr = df_p.ix['0345339703'] # Lord of the Rings Part 1

like_lotr = lotr[lotr == 10].to_frame().reset_index()

users = like_lotr['User-ID'].to_frame()
# Trim original dataset

liked = pd.merge(users, df, on='User-ID', how='inner')
rating_count = liked.groupby('ISBN')['Book-Rating'].count().to_frame()
rating_mean = liked.groupby('ISBN')['Book-Rating'].mean().to_frame()
rating_count.rename(columns={'Book-Rating':'Rating-Count'}, inplace=True)
rating_mean.rename(columns={'Book-Rating':'Rating-Mean'}, inplace=True)
liked = pd.merge(liked, rating_count, on='ISBN', how='inner')
liked = pd.merge(liked, rating_mean, on='ISBN', how='inner')
liked['Rating-Mean'] = liked['Rating-Mean'].round(2)
liked['Rating-Count'].hist()
C = liked['Rating-Mean'].mean()

C
m = rating_count.quantile(.995)[0] # .9

m
# IMDB formula; source: http://trailerpark.weebly.com/imdb-rating.html?source=post_page---------------------------

def weighted_rating(x, m=m, C=C):

    v = x['Rating-Count']

    R = x['Rating-Mean']



    return (v/(v+m) * R) + (m/(m+v) * C)
# Create relevant sub-dataset

liked_q = liked.copy().loc[liked['Rating-Count'] >= m]

liked_q.shape
liked_q['Score'] = liked_q.apply(weighted_rating, axis=1)
top_r = liked_q[['Book-Title', 'Rating-Mean']]
top_r = top_r.groupby(['Book-Title'])['Rating-Mean'].mean().to_frame()
top_r.sort_values(by='Rating-Mean', ascending=False).head(10)
top_p = liked[['Book-Title', 'Rating-Count']]
top_p = top_p.groupby(['Book-Title'])['Rating-Count'].mean().to_frame()
top_p.sort_values(by='Rating-Count', ascending=False).head(10)#.plot(kind='barh')
from tqdm import tqdm

from gensim.models import Word2Vec 

import random
users = df["User-ID"].unique().tolist()

len(users)
# shuffle users ID's

random.shuffle(users)



# extract 90% of customer ID's

users_train = [users[i] for i in range(round(0.9*len(users)))]



# split data into train and validation set

train_df = df[df['User-ID'].isin(users_train)]

validation_df = df[~df['User-ID'].isin(users_train)]
# list to capture purchase history of the customers

reads_train = []



# populate the list with the product codes

for i in tqdm(users_train):

    temp = train_df[train_df["User-ID"] == i]["ISBN"].tolist()

    reads_train.append(temp)
# list to capture purchase history of the customers

reads_val = []



# populate the list with the product codes

for i in tqdm(validation_df['User-ID'].unique()):

    temp = validation_df[validation_df["User-ID"] == i]["ISBN"].tolist()

    reads_val.append(temp)
# train word2vec model

model = Word2Vec(window = 10, sg = 1, hs = 0,

                 negative = 10, # for negative sampling

                 alpha=0.03, min_alpha=0.0007,

                 seed = 14)



model.build_vocab(reads_train, progress_per=200)



model.train(reads_train, total_examples = model.corpus_count, 

            epochs=10, report_delay=1)
model.init_sims(replace=True)
print(model)
# extract all vectors

X = model[model.wv.vocab]



X.shape
import umap



cluster_embedding = umap.UMAP(n_neighbors=30, min_dist=0.0,

                              n_components=2, random_state=42).fit_transform(X)



plt.figure(figsize=(10,9))

plt.scatter(cluster_embedding[:, 0], cluster_embedding[:, 1], s=3, cmap='Spectral')
books = train_df[["ISBN", "Book-Title"]]



# remove duplicates

books.drop_duplicates(inplace=True, subset='ISBN', keep="last")



# create product-ID and product-description dictionary

books_dict = books.groupby('ISBN')['Book-Title'].apply(list).to_dict()
# Find LOTR

df[df['Book-Title'].str.contains('Lord of the Rings')].sample()
# Check

books_dict['0345339703']
def similar_books(v, n = 15):

    

    # extract most similar products for the input vector

    ms = model.similar_by_vector(v, topn= n+1)[1:]

    

    # extract name and similarity score of the similar products

    new_ms = []

    for j in ms:

        pair = (books_dict[j[0]][0], j[1])

        new_ms.append(pair)

        

    return new_ms 
# Recommend

similar_books(model['0345339703'])