# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import ast
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Getting more than one output Line
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
dfmm=pd.read_csv('../input/movies_metadata.csv')
dfc=pd.read_csv('../input/credits.csv')
dfk=pd.read_csv('../input/keywords.csv')
dfr=pd.read_csv('../input/ratings_small.csv')
dfmm.head()
dfc.head()
dfk.head()
dfr.head()
dfmm=dfmm.drop([19730, 29503, 35587])
dfr=dfr.drop([19730, 29503, 35587])
dfmm.shape
dfc.shape
dfk.shape
dfr.shape
dataframes=[dfmm, dfc, dfk]
for dataframe in dataframes:
    dataframe['id']=dataframe['id'].astype('int')
newdf=dfmm.merge(dfc, on='id')
newdf=newdf.merge(dfk, on='id')
newdf.head()
newdf['overview'].fillna('', inplace=True)
newdf.drop(newdf[newdf['vote_average'].isnull()].index, inplace=True)
def get_words(x):
    bagofwords=[]
    for i in x:
        if i[1]=='NN':
            bagofwords.append(i[0])
        elif i[1]=='NNS':
            bagofwords.append(i[0])
        elif i[1]=='NNP':
            bagofwords.append(i[0])
        elif i[1]=='NNPS':
            bagofwords.append(i[0])
        elif i[1]=='JJ':
            bagofwords.append(i[0])
        elif i[1]=='JJR':
            bagofwords.append(i[0])
        elif i[1]=='JJS':
            bagofwords.append(i[0])
        elif i[1]=='RB':
            bagofwords.append(i[0])
        elif i[1]=='RBR':
            bagofwords.append(i[0])
        elif i[1]=='RBS':
            bagofwords.append(i[0])
    return bagofwords
def clean_words(x):
    b=nltk.pos_tag(word_tokenize(x))
    result=get_words(b)
    return result
newdf['bagofwords']=newdf['overview'].apply(clean_words)
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    newdf.loc[:, feature] = newdf.loc[:, feature].apply(ast.literal_eval)
def get_keywords(x):
    names=[i['name'] for i in x]
    if len(names)>6:
        names=names[:6]
    return names
def get_director_producer(x):
    names=[]
    for i in x:
        if i['job']=='Director':
            names.append(i['name'])
        elif i['job']=='Producer':
            names.append(i['name'])
    return names
            
features_new = ['cast', 'keywords', 'genres']
for feature in features_new:
    newdf[feature]=newdf[feature].apply(get_keywords)
newdf['crew']=newdf['crew'].apply(get_director_producer)
newdf['crew']=newdf['crew'].map(lambda x: [i.replace(" ", "") for i in x])
newdf['cast']=newdf['cast'].map(lambda x: [i.replace(" ", "") for i in x])
newdf['document']=newdf['genres']+newdf['cast']+newdf['crew']+newdf['keywords']+newdf['bagofwords']
newdf['document']=newdf['document'].map(lambda x: ' '.join(x))
newdf['document'].head()
pd.qcut(newdf['vote_average'], 10).values
pd.qcut(newdf['vote_count'], 4).values
newdf=newdf[(newdf['vote_average']>3.5) & (newdf['vote_count']>34)]
newdf.reset_index(drop=True, inplace=True)
vectorizer = CountVectorizer(stop_words='english')
matrix = vectorizer.fit_transform(newdf['document'])
similarity = cosine_similarity(matrix, matrix)
similarity
def recommendation(x):
    dataset=newdf.copy()
    ind=dataset[dataset['original_title']==x].index[0]
    sim=sorted(enumerate(similarity[ind]), key=lambda x: x[1], reverse=True)[1:11]
    ind2, scores=zip(*sim)
    recommendation=dataset.loc[ind2, 'original_title']
    return recommendation
recommendation('The Matrix')
recommendation('The Prestige')
dfmm['id']=dfmm['id'].astype('int')
dfr['movieId']=dfr['movieId'].astype('int')
titles=dfmm['original_title'].tolist()
ids=dfmm['id'].tolist()
the_map=dict(zip(ids, titles))
dfr['title']=dfr['movieId'].map(the_map)
dfr.shape
user_item=dfr.pivot_table(index='userId', columns='title', values='rating')
user_item.fillna(0, inplace=True)
user_similarity = cosine_similarity(user_item, user_item)
user_similarity
item_user=user_item.T
item_user.head()
item_similarity = cosine_similarity(item_user, item_user)
item_similarity
def filtering_recommendations(x, a):
    dataset=item_user.copy()
    df=dataset.reset_index()
    ind=df[df['title']==x].index[0]
    sim=sorted(enumerate(item_similarity[ind]), key=lambda x: x[1], reverse=True)[1:51]
    ind2, scores=zip(*sim)
    recommendation=df.loc[ind2, 'title']
    dataset1=user_item.copy()
    df2=dataset1.reset_index()
    ind=df2[df2['userId']==a].index[0]
    sim=sorted(enumerate(user_similarity[ind]), key=lambda x: x[1], reverse=True)[1:51]
    ind2, scores=zip(*sim)
    recommendation2=df2.loc[ind2, 'userId']
    dictionary={}
    for i in recommendation.index:
        lis=[]
        for j in recommendation2.index:
            if (user_item.iloc[j, i]==0):
                continue
            else:
                lis.append(user_item.iloc[j, i])
        dictionary[i]=np.mean(lis)
    keys=[]
    for i in dictionary.keys():
        keys.append(i)
    values=[]
    for i in dictionary.values():
        values.append(i)
    ourdf=pd.Series(values, index=keys)
    sim3=ourdf.fillna(0).sort_values(ascending=False)[0:9].index
    combined_recommendation=df.loc[sim3, 'title']
    return combined_recommendation
filtering_recommendations('1984', 2)
filtering_recommendations('1984', 200)
