# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
credits_df=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')
movies_df=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
credits_df.head()
movies_df.head(2)
credits_df=credits_df.rename(columns={'movie_id':'id'})
credits_df.head()
movies_tmdb=movies_df.merge(credits_df,on='id')
movies_tmdb.head(2)
def fun(index,x):
    
    return len(x)
    

import json
movies_tmdb['directors']=movies_tmdb['crew'].apply(lambda x:np.array([i['name'] for i in json.loads(x) if i['job']=='Director']))
movies_tmdb['actors']=movies_tmdb['cast'].apply(lambda x:np.array([i['name'] for i in json.loads(x)]))


movies_tmdb.head(2)
def fun(index,x):
    if len(x)<=index:
        return np.nan
    return x[index]
movies_tmdb['actor_1']=movies_tmdb['actors'].apply(lambda x: fun(0,x))
movies_tmdb['actor_2']=movies_tmdb['actors'].apply(lambda x: fun(1,x))
movies_tmdb['actor_3']=movies_tmdb['actors'].apply(lambda x: fun(2,x))
movies_tmdb.head()
movies_tmdb['keywords']=movies_tmdb['keywords'].apply(lambda x: np.array([i['name'] for i in json.loads(x)]))

import itertools
from collections import Counter
a=Counter(list(itertools.chain(*movies_tmdb['keywords'].values)))

%matplotlib inline 
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS 
wordcloud = WordCloud()
b=wordcloud.generate_from_frequencies(a)
plt.imshow(b)
most_common_keywords=a.most_common(50)
plt.figure(figsize=(25,10))
plt.bar(list(dict(most_common_keywords).keys()),list(dict(most_common_keywords).values()))
plt.xticks(rotation=90,weight='bold',size='large')

movies_tmdb['genres']=movies_tmdb['genres'].apply(lambda x:[i['name'] for i in json.loads(x)])
genres=Counter(list(itertools.chain(*movies_tmdb['genres'])))
plt.figure(figsize=(15,15))
plt.pie(list(genres.values()),labels=list(genres.keys()),autopct='%1.1f%%')

wordcloud=WordCloud().generate_from_frequencies(genres)
plt.imshow(wordcloud)


def stemm_of_key(dataframe, col):
    import nltk
    from nltk.stem import PorterStemmer
    keywords=list(itertools.chain(*movies_tmdb[col].values))
    stemmer=PorterStemmer()
    keywords_roots  = dict()
    keywords_select= dict()
    category_keys=[]
    for word in keywords:
        word=word.lower()
        lem=stemmer.stem(word)
        if lem in keywords_roots:
            keywords_roots[lem].add(word)
        else:
            keywords_roots[lem]={word}
    
    for  word in keywords_roots:
        min_length=10000
        
        if len(keywords_roots[word])>1:
            clef="a"
            for k in keywords_roots[word]:
                if len(k)<min_length:
                    min_length=len(k)
                    c=k
            keywords_select[word]=c
            category_keys.append(clef)
        else:
            keywords_select[word]=list(keywords_roots[word])[0]
            category_keys.append(list(keywords_roots[word])[0])
    print("Nb of keywords in variable '{}': {}".format(col,len(category_keys)))
    return category_keys, keywords_roots, keywords_select
keywords, keywords_roots, keywords_select = stemm_of_key(movies_tmdb,
                                                               'keywords')
cnt = 0
for s in keywords_roots.keys():
    if len(keywords_roots[s]) > 1: 
        cnt += 1
        print(cnt, keywords_roots[s], len(keywords_roots[s]))
for index, row in movies_tmdb.iterrows():
    print(row['keywords'])
    break
movies_tmdb['keywords'][4200]
def replacement_fun(dataframe,word_replacer ,roots):
    llst=[]
    df=dataframe.copy(deep=True)
    from nltk.stem import PorterStemmer
    stemmer=PorterStemmer()
    for index, row in df.iterrows():
            newlist=[]
            for word in row['keywords']:
                nw=stemmer.stem(word) if roots else s
                if nw in word_replacer.keys():
                    newlist.append(word_replacer[nw])
                else :
                    newlist.append(word)
            
            llst.append(np.array(newlist))
                        
    return np.array(llst)
    

movies_tmdb['keywords']=replacement_fun(movies_tmdb,keywords_select, True)


from nltk.corpus import wordnet
def syn(word):
    lst=set()
    for s in wordnet.synsets(word):
        for w in s.lemma_names():
            index=s.name().find('.')+1
            if s.name()[index]=='n':
                lst.add(w.lower().replace('_',' '))
    return lst    
    
wordcount=Counter(list(itertools.chain(*movies_tmdb['keywords'])))
def syn_of_keys(dataframe , col, wd_cnt):
    df=dataframe.copy(deep=True)
    llst=[]
    for index, row in df.iterrows():
        lst=[]
        for w in row[col]:
            word=w
            if wd_cnt[w]<5:
                occr=wd_cnt[w]
                synonyms=syn(w)
                for s in synonyms:
                    if s not in wd_cnt.keys():continue
                    if wd_cnt[s]>occr:
                        occr=wd_cnt[s]
                        word=s
            lst.append(word)
        llst.append(np.array(lst))
             
    return np.array(llst) 

movies_tmdb['keywords']=syn_of_keys(movies_tmdb , 'keywords', wordcount)


new_wordcount=Counter(list(itertools.chain(*movies_tmdb['keywords'])))
len(new_wordcount)
xxx=movies_tmdb.copy(deep=1)
llst=[]
for index, row in xxx.iterrows():
    lst=[]
    for w in row['keywords']:
        if new_wordcount[w]>=3:
            lst.append(w) 
    llst.append(np.array(lst))
movies_tmdb['keywords']=np.array(llst)



movies_tmdb.isna().sum()
movies_tmdb['release_date']=movies_tmdb['release_date'].apply(lambda x:float(str(x)[:4])).fillna(2014.0)
movies_tmdb['original_title']
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
stopwrds=stopwords.words('english')
movies_tmdb['original_title']=movies_tmdb['original_title'].apply(lambda x:[i for i in word_tokenize(x) if i not in stopwrds and len(i)>2])
movies_tmdb['original_title']
newlist=list(Counter(list(itertools.chain(*movies_tmdb['keywords']))).keys())
movies_tmdb['original_title']=movies_tmdb['original_title'].apply(lambda x:list(itertools.chain(*[[i for i in syn(word) if i in newlist] for word in x])))
movies_tmdb['original_title']=movies_tmdb['original_title'].apply(lambda x:' '.join(x))
movies_tmdb['keywords']=movies_tmdb['keywords'].apply(lambda x:' '.join(x))
import seaborn as sns
plt.figure(figsize=(12,10))
a=sns.heatmap(movies_tmdb[['budget','id','popularity','vote_average','vote_count','revenue','runtime']].corr(),annot=True,fmt='.3f')
plt.show()
movies_tmdb['production_companies']=movies_tmdb['production_companies'].apply(lambda x:[i['name'] for i in json.loads(x)])
movies_tmdb['production_countries']=movies_tmdb['production_countries'].apply(lambda x:[i['name'] for i in json.loads(x)])

movies_tmdb[list(genres.keys())]=0
for index,row in movies_tmdb.iterrows():
    for g in row['genres']:
        movies_tmdb.loc[index,g]=1
        


movies_tmdb['keywords']=(movies_tmdb['keywords']+" "+movies_tmdb['original_title'])
movies_tmdb['keywords']=movies_tmdb['keywords'].apply(lambda x:x.split(' '))
Genres=list(genres.keys())
def df_for_recommendation(dataframe,indx,genres):
    df=dataframe.copy(deep=1)
    col=df['keywords'][indx]+list(df['directors'][indx])+list(df['actors'][indx])
    df[col]=0
    for index, row in df.iterrows():
        for word in row['keywords']+list(row['directors'])+list(row['actors']):
            if word in col:
                df.loc[index,word]=1
    
    return df[col]

    
df_for_recommendation(movies_tmdb,1,list(genres.keys()))

def recommand(movies_tmdb,indx,genres):
    df=df_for_recommendation(movies_tmdb,indx,genres)
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix
    matrx=csr_matrix(df.values)
    knn=NearestNeighbors(n_neighbors=30, algorithm='auto', metric='euclidean').fit(matrx)
    return knn.kneighbors(df.iloc[0].values.reshape(1,-1))
    
a,b=recommand(movies_tmdb,500,Genres)

print(movies_tmdb.iloc[0]['title_x'])
movies_tmdb.iloc[b[0]]['title_x']
