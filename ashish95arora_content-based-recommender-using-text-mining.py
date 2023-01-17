import pandas as pd
import numpy as np
#Library used for initial visualization
import matplotlib.pyplot as plt 
#Library used for initial visualization
import csv 
import ast
import re
%matplotlib inline
credits_csv=pd.read_csv('../input/tmdb_5000_credits.csv')#change file path
movies_csv=pd.read_csv('../input/tmdb_5000_movies.csv')#change file path
data=pd.merge(credits_csv, movies_csv, left_on='movie_id', right_on='id')
#To drop columns that have been repeated
data=data.drop(['id','title_x'],axis=1) 
#Renaming particular columns
data.rename(columns={'title_y':'title'}, inplace=True) 
#Descriptive Statistics on the numberical data
data.describe() 
data.boxplot()
data.isnull().sum()
data.dtypes
#This cell should be run only once because the values once converted are no longer string type 
data["genres"]=data["genres"].apply(ast.literal_eval)
data["spoken_languages"]=data["spoken_languages"].apply(ast.literal_eval)
data["cast"]=data["cast"].apply(ast.literal_eval)
data["crew"]=data["crew"].apply(ast.literal_eval)
data["keywords"]=data["keywords"].apply(ast.literal_eval)

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
from collections import Counter
def tf(overview):
    vector_length=0
    overview_words=re.sub("[^\w'-]"," ", str(overview).lower()).split()
    stemmed_words=list()
    for word in overview_words:
        stemmed_words.append(ps.stem(word))
    overview_words=Counter(stemmed_words) 
    
    words_dicts=dict()
    for word,count in overview_words.items():
        vector_length+=((1+np.log10(count))**(2))
    vector_length=vector_length**(0.5)
    for word,count in overview_words.items():
        words_dicts.update({ps.stem(word):((1+np.log10(count))/vector_length)})
    return words_dicts

#pass a list of documents
def idf(idf_data): 
    idf_dict=dict()
    for docs in idf_data:
        doc_words=re.sub("[^\w'-]"," ", str(docs).lower()).split()
        stemmed_words=list()
        for word in doc_words:
            stemmed_words.append(ps.stem(word))
        
        doc_words=list(set(stemmed_words))
        for word in doc_words:
            if ps.stem(word.lower()) not in idf_dict.keys():
                idf_dict.setdefault(ps.stem(word.lower()), 1)
            else:
                idf_dict[ps.stem(word.lower())]+=1
    for key,value in idf_dict.items():
        idf_dict[key]=np.log10(len(idf_data)/value)
    return idf_dict
       

actors=list()

for i in range(0,len(data.index)):
    actors.append([d['name'].strip() for d in data['cast'][i] if d['order'] == 0 or d['order'] == 1 or d['order'] == 2])
    
labels=['actor1','actor2','actor3','actor4','actor5']
actors_df=pd.DataFrame.from_records(actors,columns=labels,exclude=['actor4','actor5'])
tfidf_column=[]
idf_dict=idf(data["overview"])
for overview in data["overview"]:        
    tfidf_dict=tf(overview)
    for key,value in tfidf_dict.items():
        tfidf_dict[key]=value*idf_dict[key]
    tfidf_dict=sorted(tfidf_dict, key=tfidf_dict.get, reverse=True)[:5]
    tfidf_column.append(tfidf_dict)
importantwords_df=pd.DataFrame(tfidf_column,columns=["Key1","Key2","Key3","Key4","Key5"])
importantwords_df
genre_list=[]
for genre in data["genres"]:
    for d in genre:
        if d['name'] not in genre_list:
            genre_list.append(d['name'])

all_movies=[]
for genre in data["genres"]:
    movie_genres=dict()
    for gen in genre_list:
        movie_genres.setdefault(gen, 0)
    for d in genre:
        movie_genres[d['name']]=1
    all_movies.append(movie_genres)
genres_df=pd.DataFrame(all_movies)

directors_list=list()
for crew in data["crew"]:
    director_flag=0
    for d in crew:
        if d['job']=="Director":
            directors_list.append({"Director":d['name']})
            director_flag=1
            break
    if director_flag==0:
        directors_list.append({"Director":''})

directors_df=pd.DataFrame(directors_list)

keywords_column=list()
for movie_keywords in data["keywords"]:
    keywords=''
    for d in movie_keywords:
        keywords=keywords+ps.stem(d['name'])+' '
    keywords_column.append(keywords.strip())
    
keywords_df=pd.DataFrame(keywords_column)

tfidf_column2=[]
idf_dict2=idf(keywords_df[0])
for docs in keywords_df[0]:        
    tfidf_dict=tf(docs)
    for key,value in tfidf_dict.items():
        tfidf_dict[key]=value*idf_dict2[key]
    tfidf_dict=sorted(tfidf_dict, key=tfidf_dict.get, reverse=True)[:5]
    tfidf_column2.append(tfidf_dict)
keywords_5df=pd.DataFrame(tfidf_column2,columns=["Keyword1","Keyword2","Keyword3","Keyword4","Keyword5"])

#Concatenating
result=pd.concat([data,actors_df,directors_df,genres_df,importantwords_df,keywords_5df],axis=1)
#Normalizing vote_average to change the range to 0 to 1
result["vote_average"]=result["vote_average"]*0.1
import ipywidgets as widgets

user_movie=widgets.Dropdown(
    options=sorted(list(result["title"])),
    description='Please choose a movie:',
    disabled=False,
    value='2 Fast 2 Furious'
)
user_movie
import sklearn.metrics.pairwise
print("Movie selected is "+user_movie.value)
user_profile=result.loc[result['title']==str(user_movie.value)]

user_profile=user_profile.drop(['cast','crew','popularity', 'budget', 'genres', 'homepage', 'keywords','original_title','overview', 'production_companies','production_countries','release_date', 'revenue', 'runtime', 'spoken_languages', 'status','tagline','vote_count'],axis=1)
actor1=user_profile['actor1'].tolist()
actor2=user_profile['actor2'].tolist()
actor3=user_profile['actor3'].tolist()
original_language=user_profile['original_language'].tolist()
Director=user_profile['Director'].tolist()
Key1=user_profile['Key1'].tolist()
Key2=user_profile['Key2'].tolist()
Key3=user_profile['Key3'].tolist()
Key4=user_profile['Key4'].tolist()
Key5=user_profile['Key5'].tolist()
Keyword1=user_profile['Keyword1'].tolist()
Keyword2=user_profile['Keyword2'].tolist()
Keyword3=user_profile['Keyword3'].tolist()
Keyword4=user_profile['Keyword4'].tolist()
Keyword5=user_profile['Keyword5'].tolist()


actor_list=[actor1[0],actor2[0],actor3[0]]
key_list=[Key1[0],Key2[0],Key3[0],Key4[0],Key5[0]]
keyword_list=[Keyword1[0],Keyword2[0],Keyword3[0],Keyword4[0],Keyword5[0]]


user_profile['actor1'] = np.where(user_profile.actor1.isin(actor1),1,0)
user_profile['actor2'] = np.where(user_profile.actor2.isin(actor2),1,0)
user_profile['actor3'] = np.where(user_profile.actor3.isin(actor3),1,0)
user_profile['original_language'] = np.where(user_profile.original_language.isin(original_language),1,0)
user_profile['Director'] = np.where(user_profile.Director.isin(Director),1,0)
user_profile['Key1'] = np.where(user_profile.Key1.isin(Key1),1,0)
user_profile['Key2'] = np.where(user_profile.Key2.isin(Key2),1,0)
user_profile['Key3'] = np.where(user_profile.Key3.isin(Key3),1,0)
user_profile['Key4'] = np.where(user_profile.Key4.isin(Key4),1,0)
user_profile['Key5'] = np.where(user_profile.Key5.isin(Key5),1,0)
user_profile['Keyword1'] = np.where(user_profile.Keyword1.isin(Keyword1),1,0)
user_profile['Keyword2'] = np.where(user_profile.Keyword2.isin(Keyword2),1,0)
user_profile['Keyword3'] = np.where(user_profile.Keyword3.isin(Keyword3),1,0)
user_profile['Keyword4'] = np.where(user_profile.Keyword4.isin(Keyword4),1,0)
user_profile['Keyword5'] = np.where(user_profile.Keyword5.isin(Keyword5),1,0)

item_profiles=result.drop(['cast','crew','popularity', 'budget', 'genres', 'homepage', 'keywords','original_title','overview', 'production_companies','production_countries','release_date', 'revenue', 'runtime', 'spoken_languages', 'status','tagline','vote_count'],axis=1)
item_profiles=item_profiles.loc[~(data['original_title']==str(user_movie.value))]
item_profiles['actor1'] = np.where(item_profiles.actor1.isin(actor_list),1,0)
item_profiles['actor2'] = np.where(item_profiles.actor2.isin(actor_list),1,0)
item_profiles['actor3'] = np.where(item_profiles.actor3.isin(actor_list),1,0)
item_profiles['original_language'] = np.where(item_profiles.original_language.isin(original_language),1,0)
item_profiles['Director'] = np.where(item_profiles.Director.isin(Director),1,0)
item_profiles['Key1'] = np.where(item_profiles.Key1.isin(key_list),1,0)
item_profiles['Key2'] = np.where(item_profiles.Key2.isin(key_list),1,0)
item_profiles['Key3'] = np.where(item_profiles.Key3.isin(key_list),1,0)
item_profiles['Key4'] = np.where(item_profiles.Key4.isin(key_list),1,0)
item_profiles['Key5'] = np.where(item_profiles.Key5.isin(key_list),1,0)

item_profiles['Keyword1'] = np.where(item_profiles.Keyword1.isin(keyword_list),1,0)
item_profiles['Keyword2'] = np.where(item_profiles.Keyword2.isin(keyword_list),1,0)
item_profiles['Keyword3'] = np.where(item_profiles.Keyword3.isin(keyword_list),1,0)
item_profiles['Keyword4'] = np.where(item_profiles.Keyword4.isin(keyword_list),1,0)
item_profiles['Keyword5'] = np.where(item_profiles.Keyword5.isin(keyword_list),1,0)


item_profiles
x=user_profile.drop(['title','movie_id'],axis=1)
y=item_profiles.drop(['title','movie_id'],axis=1)


arr = sklearn.metrics.pairwise.cosine_similarity(x,y, dense_output=True)
arr_index=arr.argsort()
arr_index=arr_index.ravel()
final_df=pd.DataFrame(arr_index).tail(5)
final_df
item_profiles=item_profiles.iloc[final_df[0]]
item_profiles=item_profiles.iloc[::-1]
item_profiles
print("Recommended Movies for "+ user_movie.value + ": \n\n"+'\n'.join(list(item_profiles["title"])))