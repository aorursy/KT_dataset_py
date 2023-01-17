import re

import numpy as np

import pandas as pd

from wordcloud import WordCloud

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import nltk

from nltk.corpus import wordnet as wn

from sklearn.neighbors import NearestNeighbors

from fuzzywuzzy import fuzz
import json

import pandas as pd

#___________________________

def load_tmdb_movies(path):

    df = pd.read_csv(path)

    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())

    json_columns = ['genres', 'keywords', 'production_countries',

                    'production_companies', 'spoken_languages']

    for column in json_columns:

        df[column] = df[column].apply(json.loads)

    return df

#___________________________

def load_tmdb_credits(path):

    df = pd.read_csv(path)

    json_columns = ['cast', 'crew']

    for column in json_columns:

        df[column] = df[column].apply(json.loads)

    return df

#___________________

LOST_COLUMNS = [

    'actor_1_facebook_likes',

    'actor_2_facebook_likes',

    'actor_3_facebook_likes',

    'aspect_ratio',

    'cast_total_facebook_likes',

    'color',

    'content_rating',

    'director_facebook_likes',

    'facenumber_in_poster',

    'movie_facebook_likes',

    'movie_imdb_link',

    'num_critic_for_reviews',

    'num_user_for_reviews']

#____________________________________

TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES = {

    'budget': 'budget',

    'genres': 'genres',

    'revenue': 'gross',

    'title': 'movie_title',

    'runtime': 'duration',

    'original_language': 'language',

    'keywords': 'plot_keywords',

    'vote_count': 'num_voted_users'}

#_____________________________________________________

IMDB_COLUMNS_TO_REMAP = {'imdb_score': 'vote_average'}

#_____________________________________________________

def safe_access(container, index_values):

    # return missing value rather than an error upon indexing/key failure

    result = container

    try:

        for idx in index_values:

            result = result[idx]

        return result

    except IndexError or KeyError:

        return pd.np.nan

#_____________________________________________________

def get_director(crew_data):

    directors = [x['name'] for x in crew_data if x['job'] == 'Director']

    return safe_access(directors, [0])

#_____________________________________________________

def pipe_flatten_names(keywords):

    return '|'.join([x['name'] for x in keywords])

#_____________________________________________________

def convert_to_original_format(movies, credits):

    tmdb_movies = movies.copy()

    tmdb_movies.rename(columns=TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES, inplace=True)

    tmdb_movies['title_year'] = pd.to_datetime(tmdb_movies['release_date']).apply(lambda x: x.year)

    # I'm assuming that the first production country is equivalent, but have not been able to validate this

    tmdb_movies['country'] = tmdb_movies['production_countries'].apply(lambda x: safe_access(x, [0, 'name']))

    tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(lambda x: safe_access(x, [0, 'name']))

    tmdb_movies['director_name'] = credits['crew'].apply(get_director)

    tmdb_movies['actor_1_name'] = credits['cast'].apply(lambda x: safe_access(x, [1, 'name']))

    tmdb_movies['actor_2_name'] = credits['cast'].apply(lambda x: safe_access(x, [2, 'name']))

    tmdb_movies['actor_3_name'] = credits['cast'].apply(lambda x: safe_access(x, [3, 'name']))

    tmdb_movies['genres'] = tmdb_movies['genres'].apply(pipe_flatten_names)

    tmdb_movies['plot_keywords'] = tmdb_movies['plot_keywords'].apply(pipe_flatten_names)

    return tmdb_movies

movies=load_tmdb_movies('../input/tmdb_5000_movies.csv')

credits=load_tmdb_credits('../input/tmdb_5000_credits.csv')

df_initial = convert_to_original_format(movies, credits)
tab_info=pd.DataFrame(df_initial.dtypes).T.rename(index={0:'column_type'})

tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().apply(lambda x:len([w for w in x if w==True]))).T.rename({0:'null_values'}))

tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().apply(lambda x:len([w for w in x if w==True])/len(x)*100)).T.rename({0:'%null_values'}))
tab_info

temp=""

for i in df_initial['plot_keywords']:

    temp=temp+'|'+i

    

keywords=set(temp.split('|'))

keywords.remove('')

list_keywords=list(keywords)
## function for counting the frequency of unique words in a column

def word_count(df,col):

    temp=""

    for i in df[col]:

        temp=temp+'|'+i

    

    keywords=set(temp.split('|'))

    if '' in list(keywords):

        keywords.remove('')

    liste=list(keywords)

    dict_key={}

    keywords=[]

    for i in liste:

        dict_key[i]=[]

    lt=temp.split("|")

    for i in liste:

        j=0

        for w in lt:

            if w==i:

                j=j+1

        dict_key[i].append(j)

    for i in liste:

        keywords.append([i,dict_key[i][0]])

    keywords=sorted(keywords,key=lambda x:x[1],reverse=True)

    

    

 

    return keywords

            

        

        

        
keyword_occurences=word_count(df_initial,'plot_keywords')
len(keyword_occurences)
keyword_occurences[:10]
top=keyword_occurences[:50]

dict_key={}

for i in keyword_occurences[:50]:

    dict_key[i[0]]=i[1]

plt.figure(figsize=(12,13))

plt.subplot(211)

wordcloud = WordCloud()

wordcloud.generate_from_frequencies(frequencies=dict_key)

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Word Cloud',size=40,bbox={'facecolor':'k','pad':5},color='w')

plt.axis('off')



plt.subplot(212)

x=[i[0] for i in top]

y=[i[1] for i in top]

sns.barplot(x=x,y=y,color='green')

plt.xticks(fontsize='x-large',rotation='vertical')

plt.title('Histogram',size=40,bbox={'facecolor':'k','pad':5},color='w')







plt.show()
temp=""

for i in df_initial['genres']:

    temp=temp+'|'+i

    

genres=set(temp.split('|'))

genres.remove('')

list_genres=list(genres)
genre_freq=word_count(df_initial,'genres')
len(genre_freq)
genre_freq[:10]


porter= nltk.PorterStemmer()

def word_converter(df,col):

    temp=""

    for i in df[col]:

        temp=temp+'|'+i

    

    genres=set(temp.split('|'))

    if '' in list(genres):

        genres.remove('')

    liste=list(genres)

    keyword_roots={}

    keyword_select={}

    for s in liste:

        root=porter.stem(s)

        if root not in keyword_roots.keys():

            keyword_roots[root]=[s]

        else:

            keyword_roots[root].append(s)

    for s in keyword_roots.keys():

                   

        if len(keyword_roots[s])>1:

            min_length=10000

            for k in keyword_roots[s]:

                if len(k)<min_length:

                    lth=k

                    min_length=len(k)

            keyword_select[s]=lth

        else:

            keyword_select[s]=keyword_roots[s][0]

     

       

    return keyword_roots,keyword_select

        
keyword_roots,keyword_select=word_converter(df_initial,'plot_keywords')
keys=keyword_roots.keys()

l=0

for key in keys:

    print([key,keyword_roots[key]])

    l+=1

    if l>10:

        break



keys=keyword_select.keys()

l=0

for key in keys:

    print([key,keyword_select[key]])

    l+=1

    if l>10:

        break

    
## Now that we have cleaned keywords let's replace them by the selected (as discussed above) keywords in the original Data Frame

def cleaned_dataframe(df,keyword_select,roots=True):

    df_cleaned=df.copy()

    for index,row in df_cleaned.iterrows():

        liste=[]

        t=row['plot_keywords']

        if pd.isnull(t):continue

        keys=t.split('|')

        for s in keys:

            root=porter.stem(s) if roots else s

            if root in keyword_select.keys():

                liste.append(keyword_select[root])

            else:

                liste.append(s)



        df_cleaned.set_value(index,'plot_keywords',"|".join(liste))

                   

    return df_cleaned

        
df_cleaned=cleaned_dataframe(df_initial,keyword_select)
keyword_occurences_filtered=word_count(df_cleaned,'plot_keywords')
key_filter=keyword_occurences_filtered

len(key_filter)
keyword_occurences_filtered[:10]
## functions for getting a synonym of a word and herewe are considering only Nouns.

def get_synonyms(word):

    sys=wn.synsets(word)

    synonyms=set()

    if not len(sys)==0:

        for w in sys:

            if w.name().split('.')[1]=='n':

                for i in w.lemmas():

                    synonyms.add(i.name())

    return list(synonyms)

    

    


def synonym_replace_lessthan(liste,alpha):

    dict_liste={}

    for w in liste:

        dict_liste[w[0]]=w[1]

    replace={}

    replacement=[]

    equal_synonyms=[]

    freq=[w[0] for w in liste if w[1]<alpha]

    for w in freq:

        col=[]

        con=[]

        syn=get_synonyms(w)

      

        for s in syn:

            if s in dict_liste.keys():

                    col.append([s,dict_liste[s]])

        d=sorted(col,key=lambda x:x[1],reverse=True)

   

        if len(col)>1:

            if dict_liste[w]<dict_liste[d[0][0]]:

                replace[w]=d[0][0]  

                replacement.append([w,dict_liste[w],d[0][0],dict_liste[d[0][0]]])

            for s in d:

                if dict_liste[s[0]]==dict_liste[w]:

                    con.append(True)

                else:

                    con.append(False)

            if all(con):

                    equal_synonyms.append(d)

        

    return replace,freq,replacement,equal_synonyms

    

    

    
replace_keys,freq,replacement,equal_synonyms=synonym_replace_lessthan(key_filter,5)
equal_synonyms[:10]



def connect(equal_synonyms,replace_keys,dict_liste):  

    f=replace_keys.copy()

    

    for s in equal_synonyms:

        col=[]

        for w in [i[0] for i in s]:

            if w in replace_keys.keys():

                col.append([replace_keys[w],dict_liste[replace_keys[w]]])

    

        d=sorted(col,key=lambda x:x[1],reverse=True)

        if not len(d)==0: 

               for w in [i[0] for i in s]:

                    f[w]=d[0][0]

        if len(d)==0:

            

            for i in range(1,len(s)):

                if s[i][0] not  in f.keys() and s[i][0] not in f.values():

                    f[s[i][0]]=s[0][0]

    return f

        

            

        

    
dict_keywords={}

for w in key_filter:

    dict_keywords[w[0]]=w[1]

    

second_replace=connect(equal_synonyms,replace_keys,dict_keywords)

keys=second_replace

l=0

for key in keys:

    print([key,second_replace[key]])

    l+=1

    if l>10:

        break
key_value=list(set(list(second_replace.keys())).intersection(list(second_replace.values())))

key_value[:10]
Intersections=[[replace_keys[w],second_replace[w]] for w in list(set(second_replace.keys()).intersection(replace_keys.keys()))]

Intersections[:10]
key_value=list(set(list(second_replace.keys())).intersection(list(second_replace.values())))

while(len(key_value)>0):

    key_value=list(set(list(second_replace.keys())).intersection(list(second_replace.values())))

    i=list(second_replace.values())

    for s in list(second_replace.keys()):

        if s in key_value:

            second_replace[list(second_replace.keys())[i.index(s)]]=second_replace[s]

        

        

    

    
def replace_synonyms(replace_dict,liste):

    dict_liste={}

    for w in liste:

        dict_liste[w[0]]=w[1]

    filtered={}

    filter_list=[]

    for w in dict_liste.keys():

        if w not in replace_dict.keys():

            filtered[w]=dict_liste[w]

    for w in replace_dict.keys():

        filtered[replace_dict[w]]= filtered[replace_dict[w]]+dict_liste[w]

    for w in filtered.keys():

        filter_list.append([w,filtered[w]])

    filter_list=sorted(filter_list,key=lambda x:x[1],reverse=True)

        

        

    return filtered,filter_list

            
filter_dict,filter_list=replace_synonyms(second_replace,key_filter)
df_cleaned2=cleaned_dataframe(df_cleaned,second_replace,roots=False)
processed_keywords=word_count(df_cleaned2,'plot_keywords')
len(processed_keywords)
processed_keywords[:10]
## function for removing words with frequency lower than certain threshold

def frequency_lessthan(liste,alpha):

    new_list=[]

    for w in liste:

        if w[1]>alpha:

            new_list.append(w)

    return new_list

            

        
new_keyword_occurences=frequency_lessthan(processed_keywords,3)

new_keyword_occurences[:10]
## Creating A new dataframe with supressed Keywords

def supress_keywords(df,supress_keywords):

    new_df=df.copy()

    dict_supress={}

    for i in supress_keywords:

        dict_supress[i[0]]=i[1]

    for index,row in new_df.iterrows():

        col=[i for i in row['plot_keywords'].split("|") if i in dict_supress.keys()]

        new_df.set_value(index,'plot_keywords',"|".join(col))

        

    return new_df

        

    
df_cleaned3=supress_keywords(df_cleaned2,new_keyword_occurences)

keyword_list=word_count(df_cleaned3,'plot_keywords')
len(keyword_list)
plt.figure(figsize=(25,8))

x=list(range(1,len(keyword_occurences)+1))

y=[i[1] for i in keyword_occurences]

new_xaxis=list(range(1,len(new_keyword_occurences)+1))

new_yaxis=[i[1] for i in new_keyword_occurences]

plt.plot(x,y,'r--',label='before cleaning',linewidth=3.0)

plt.plot(new_xaxis,new_yaxis,'green',label='after cleaning',linewidth=3.0)

plt.xlabel('keywords index',size=30,weight='bold')

plt.ylabel('No. of occurences',size=30,weight='bold')

plt.axhline(y=3.5,linewidth=2.0,color='black')

plt.text(3000,3.5,'threshold for keyword deletion',fontsize=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.legend(loc='upper right',fontsize=30)

plt.ylim(0,25)## to zoom in the plot

plt.show()
df_var_cleaned=df_cleaned3.copy()

s=df_var_cleaned.isnull().sum(axis=0).reset_index()

s.columns=['column_name','missing_values']

s.sort_values('missing_values',ascending=False,inplace=True)

s['filling_factor']=(df_var_cleaned.shape[0]-s['missing_values']) / df_var_cleaned.shape[0] * 100

s=s.reset_index(drop=True)

missing_df=s.copy()

missing_df[:10]
## This functuion retrievs the labels of the movie entered by the user 

def add_entry(df,id_entry):

    col_labels=[]

    index=list(df['id']).index(id_entry)

    columns=['director_name','actor_1_name','actor_2_name','actor_3_name','plot_keywords','genres']

    for s in columns:

        if pd.isnull(df[s].iloc[index]):continue

        a=df[s].iloc[index].split("|")

        for i in a:

            col_labels.append(i)

    return col_labels

df_initial.head(5)
## This cretaes a datframe including all the  variables given by the add_entry as columns

def new_dataframe(df,ref_var):

    for s in ref_var:df[s]=pd.Series([0 for i in range(len(df_initial))])

    columns=['director_name','actor_1_name','actor_2_name','actor_3_name','plot_keywords','genres']

    for col in columns:

        for index,row in df.iterrows():

            if pd.isnull(row[col]):continue

            t=row[col].split('|')

            for s in t:

                if s in ref_var:df.set_value(index,s,1)

                

    return df

    

    
def make_recommendation(df,id_entry):

    index=list(df['id']).index(id_entry)

    ref_var=add_entry(df,id_entry)

    new_df=new_dataframe(df,ref_var)

    X=new_df.as_matrix(ref_var)

    nearest=NearestNeighbors(n_neighbors=31,algorithm='auto',metric='euclidean').fit(X)

    xtest=new_df.iloc[index,:].as_matrix(ref_var)

    xtest=xtest.reshape(1,-1)

    distance,indices=nearest.kneighbors(xtest)

    indices=indices.ravel()

    return indices   
## Once we have the list of 30 films we extract the variables as stataed above to calculate the score for each movie

def feature_extractor(df,film_indices,id_entry):

    index=list(df['id']).index(id_entry)

    parametre_list=[]

    scores=[]

    col=['vote_average','title_year','num_voted_users','original_title']

    for s in film_indices:

        parametre_list.append(df.iloc[s,:][col].values)

    maximum=0

    for s in parametre_list:

        if s[2]>maximum:

            maximum=s[2]

    title_year_ref=df.iloc[index,:]['title_year']

    for s in parametre_list:

        sc=criterion(s[0],s[1],s[2],s[3],maximum,title_year_ref)

        scores.append([s[3],sc])

    scores.sort(key=lambda x:x[1],reverse=False)

    return parametre_list

    
gaussian_filter = lambda x,mu,sigma: 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-np.power((x - mu)/sigma, 2.)/2)
def criterion(vote_average,title_year,num_voted_users,original_title,maximum,title_year_ref):

    if pd.notnull(vote_average):

        feature1=vote_average

    else:

        feature1=0

    if pd.notnull(title_year):

        feature2=gaussian_filter(title_year,title_year_ref,20)

    else:

        feature2=0

    

    if pd.notnull(num_voted_users):

        feature3=num_voted_users**2

    else:

        feature3=0

    score=feature1**2*feature2*feature3

    return score

    

    
def get_id_entry(title):

    

    index=list(df_var_cleaned['original_title']).index(title)

    return df_var_cleaned['id'].iloc[index]
## This is the final functions which recommends the movie

def recommend_movies(df,title,sequal=False):

    selected=[]

    id_entry=get_id_entry(title)

    

    col_labels=add_entry(df,id_entry)

    film_indices=make_recommendation(df,id_entry)

    parametre_list=feature_extractor(df,film_indices,id_entry)

    for s in parametre_list[:5]:

        selected.append(s[3])

    if sequal==False:selected=remove_sequal(df,parametre_list,id_entry)

    

    

    return selected
def check_sequal(title,title_ref):

    if (fuzz.ratio(title,title_ref) or fuzz.token_sort_ratio(title,title_ref))>60:

        return True

    else:

        return False


def remove_sequal(df,parametre_list,id_entry):

    index=list(df['id']).index(id_entry)

    title_ref=df['original_title'].iloc[index]

    sequal_list=[]

    film_list=[]

    selected=[]

    a=[]

    for i in parametre_list:

        if check_sequal(i[3],title_ref):

            index=list(df['original_title']).index(i[3])

            sequal_list.append([i[3],df['popularity'].iloc[index]])

    c=0



    names,score=zip(*sequal_list)

    for i in parametre_list:

        if c<5:

            if  not i[3] in names:

                film_list.append(i[3])

                a.append(i[3])

                c+=1

    if len(sequal_list)>0:

        sequal_list.sort(key=lambda x:x[1],reverse=True)

        selected.append(sequal_list[0][0])

        for i in film_list[:4]:

            selected.append(i)

        return selected

    else:

        return film_list         
df_var_cleaned.head(5)
recommend_movies(df_var_cleaned,"Avatar",sequal=True)
recommend_movies(df_var_cleaned,"Pirates of the Caribbean: At World's End")
recommend_movies(df_var_cleaned,"Pirates of the Caribbean: At World's End",sequal=True)
recommend_movies(df_var_cleaned,"Spectre")
recommend_movies(df_var_cleaned,"Bound")