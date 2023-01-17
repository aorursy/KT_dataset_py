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

from datetime import datetime

train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')
train_data.describe()
train_data.info()
train_data.head()
def strtoint(x):

    if type(x)==str:

        return 1

    else:

        return 0
data=train_data.copy()









#belongs_to_collection

data['from_collection']=data['belongs_to_collection'].fillna(0)

data['from_collection']=data['from_collection'].apply(strtoint)



#genres

genres_list=[]

unique_genres=[]

for genre in data['genres']:

    try:

        gen=ast.literal_eval(genre)

    except:

        gen=[]

    genres=[]

    for j in gen:

        genres.append(j['name'])

        if j['name'] in unique_genres:

            pass

        else:

            unique_genres.append(j['name'])

    genres_list.append(genres)

data['genres_list']=genres_list

for genre in unique_genres:

    data[genre]=[0]*len(data)

for i,genres in data['genres_list'].iteritems():

    for gen in genres:

        data.set_value(i,gen,1)



#Homepage

data['has_homepage']=data['homepage'].fillna(0)

data['has_homepage']=data['homepage'].apply(strtoint)



#production countries

countries_list=[]

unique_country=[]

for country in data['production_countries']:

    try:

        gen=ast.literal_eval(country)

    except:

        gen=[]

    countrys=[]

    for j in gen:

        countrys.append(j['name'])

        if j['name'] in unique_prod:

            pass

        else:

            unique_prod.append(j['name'])

    countries_list.append(countrys)

data['countries_count']=[len(i) for i in countries_list]



#production companies

companies_list=[]

unique_company=[]

for company in data['production_companies']:

    try:

        gen=ast.literal_eval(company)

    except:

        gen=[]

    companys=[]

    for j in gen:

        companys.append(j['name'])

        if j['name'] in unique_company:

            pass

        else:

            unique_company.append(j['name'])

    companies_list.append(companys)

data['companies_count']=[len(i) for i in companies_list]



#release_date

data['release_date']=data['release_date'].apply(lambda i:datetime.strptime(i,'%m/%d/%y'))



#spoken_languages

languages_list=[]

unique_language=[]

for language in data['spoken_languages']:

    try:

        gen=ast.literal_eval(language)

    except:

        gen=[]

    languages=[]

    for j in gen:

        languages.append(j['name'])

        if j['name'] in unique_language:

            pass

        else:

            unique_language.append(j['name'])

    languages_list.append(languages)

data['languages_count']=[len(i) for i in languages_list]



#tagline

data['has_tagline']=data['tagline'].apply(strtoint)

data['tagline']=data['tagline'].fillna('')

data['tagline_len']=data['tagline'].apply(lambda x:len(x))



#keywords

keywords_list=[]

unique_keyword=[]

for keyword in data['Keywords']:

    try:

        gen=ast.literal_eval(keyword)

    except:

        gen=[]

    keywords=[]

    for j in gen:

        keywords.append(j['name'])

        if j['name'] in unique_keyword:

            pass

        else:

            unique_keyword.append(j['name'])

    keywords_list.append(keywords)

data['keywords_count']=[len(i) for i in keywords_list]



#title

data['title_len']=data['title'].apply(lambda x:len(x))



#cast

casts_count=[]

male_casts_count=[]

for cast in data['cast']:

    try:

        gen=ast.literal_eval(cast)

    except:

        gen=[]

    casts_count.append(len(gen))

    

    male_casts=0

    for j in gen:

        if j['gender']==2:

            male_casts=male_casts+1

        else:

            pass

    male_casts_count.append(male_casts)

data['cast_count']=casts_count

data['male_cast_count']=male_casts_count



#crews

crews_dicts=[]

unique_crews=[]

crew_count_list=[]

for crew in data['crew']:

    try:

        gen=ast.literal_eval(crew)

    except:

        gen=[]

    crews={}

    for j in gen:

        j=j['department']

        try:

            crews[j]=crews[j]+1

        except KeyError:

            crews[j]=1

        if j in unique_crews:

            pass

        else:

            unique_crews.append(j)

    crews_dicts.append(crews)

data['crews_dicts']=crews_dicts

for crew in unique_crews:

    data[crew]=[0]*len(data)

for i,crew in data['crews_dicts'].iteritems():

    for dep,count in crew.items():

        data.set_value(i,dep,count)



redundant_features=['belongs_to_collection','genres','homepage','imdb_id','original_title','overview','poster_path','production_companies','production_countries','spoken_languages','status','tagline','title','Keywords','cast','crew','crews_dicts','genres_list']

data=data.drop(redundant_features,axis=1)
data.head()
data['budget'].hist()