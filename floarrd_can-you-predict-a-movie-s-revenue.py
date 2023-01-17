import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor,plot_importance

import ast

from collections import Counter

from sklearn.cluster import KMeans

import seaborn as sns

import matplotlib.pyplot as plt

import bokeh

from bokeh.plotting import figure

from bokeh.io import output_notebook, show

from bokeh.models import LabelSet, ColumnDataSource, HoverTool

from bokeh.palettes import Category20c, Spectral6

from bokeh.transform import cumsum, factor_cmap, jitter



output_notebook()

# Path of the files to read.

train_path = '../input/tmdb-box-office-prediction/train.csv'

test_path = '../input/tmdb-box-office-prediction/test.csv'

nominations_path='../input/nominations/nominations.csv'



train = pd.read_csv(train_path)

test = pd.read_csv(test_path)

nominations=pd.read_csv(nominations_path)

colons_in_Json = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew','belongs_to_collection']



def get_dictionary(s):

    try:

        d = eval(s)

    except:

        d = {}

    return d



for col in colons_in_Json :

    train[col] = train[col].apply(lambda x : get_dictionary(x))

    

for col in colons_in_Json :

    test[col] = test[col].apply(lambda x : get_dictionary(x))

%%script false



import pandas as pd

from urllib.request import urlopen

from bs4 import BeautifulSoup

import ast

import time

import json

import unidecode

import urllib



path = './train.csv'

data = pd.read_csv(path)



cast_dict = data['cast'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})

cast_ids=[[y['id'] for y in x] for x in cast_dict]

unique_cast = pd.DataFrame(cast_ids).stack().unique()

nominations=pd.DataFrame(columns=['id', 'nominations'])



# output = pd.DataFrame({'id': nominations.id,

#                        'nominations': nominations.nominations})

# output.to_csv('nominations.csv', index=False)



nominations['id'] = [x for x in unique_cast]





i=0 



with open('nominations.csv', 'a') as csv:

    for x in unique_cast[i:]:

        start_time = time.time()



        print(x)

        

        try:

            with urlopen("https://www.themoviedb.org/person/"+str(x)) as page:

                soup = BeautifulSoup(page, 'html.parser')

                name=soup.find('h2').text

                name = unidecode.unidecode(name)

                name=name.replace(" ", "%20")



                print(name)



                url="https://sg.media-imdb.com/suggests/"+name[0].lower()+"/"+name.lower()+".json"

                print(url)

                page = urlopen(url)

                soup = BeautifulSoup(page, 'html.parser')

                id=soup.text.find('"id":"nm')

                id=soup.text[id+6:id+15]



                print(id)

                print(i)

                page = urlopen("https://www.imdb.com/name/"+id+"/awards?ref_=nm_ql_2")



                soup = BeautifulSoup(page, 'html.parser')

                

                a = 0

                academy=0

                golden=0

                if soup.find_all('h3', string="Academy Awards, USA"):

                    soupCarotte = soup.find_all('table', attrs={'class': 'awards'})[a]

                    academy=len(soupCarotte.find_all('td', attrs={'class': 'award_outcome'}))

                    a+=1



                if soup.find_all('h3', string="Golden Globes, USA"):

                    soupSoup = soup.find_all('table', attrs={'class': 'awards'})[a]

                    golden=len(soupSoup.find_all('td', attrs={'class': 'award_outcome'}))



                nominations.loc[i,'nominations']=golden+academy





        except urllib.error.HTTPError:

            nominations.loc[i,'nominations']=0





        print(nominations)



        output = pd.DataFrame({'id': nominations.loc[i, 'id'],

                       'nominations': nominations.loc[i, 'nominations']}, index=[0])

        output.to_csv(csv, header=False, index=False)



        i+=1





print(time.time() - start_time)



# output = pd.DataFrame({'id': nominations.id,

#                        'nominations': nominations.nominations})

# output.to_csv('nominations.csv', index=False)
%%script false



import pandas as pd

import numpy as np

from urllib.request import urlopen

from bs4 import BeautifulSoup

import ast

import time

import json

import urllib

import unidecode

import requests

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

import re

from itertools import islice







path = './train.csv'

data = pd.read_csv(path)





print(data['belongs_to_collection'].iloc[0])





colons_in_Json = ['belongs_to_collection']



def get_dictionary(s):

    try:

        d = eval(s)

    except:

        d = {}

    return d



for col in tqdm(colons_in_Json) :

    data[col] = data[col].apply(lambda x : get_dictionary(x))









google_result=data[['id','title','belongs_to_collection']]

google_result['google_result']=""

movie_title=google_result['title']

list_movie_title=list(movie_title)





google_result['film_belongs_to_collection'] = google_result['belongs_to_collection'].apply(lambda x: 0 if x == {} else 1)



word_movie=' movie'



with open('google_result.csv', 'a') as csv:

    for index, r in google_result.iloc[0:].iterrows():    #choose here to start iterating from the row you want in case of an error during data collection 

        if google_result.iloc[index]['film_belongs_to_collection']==0:

            search=google_result['title'].iloc[index]

            search=search+word_movie

        else:

            words = ['(Theatrical)','(1958 series)','( Series)','- Коллекция','Collection','(Animation)','(Universal Series)','(Heisei)',': The Original Series','- Collezione','(Original)','(Hammer Series)','(Remake)','(Reboot)','( Series)','Trilogy','(1976 series)','(Original Series)','(Universal Series)','Anthology','(Universal)','()','The Klapisch ','(Коллекция)']

            search = [] 

            belongs_to_collection_line=google_result.iloc[index]['belongs_to_collection']

            collection_name = belongs_to_collection_line[0]['name']

            for w in words:

                collection_name = collection_name.replace(w, '')

            search.append(collection_name)

            search=(search[0])  

            search=search+word_movie

      

        try:

            print(search)

            r = requests.get("https://www.google.com/search", params={'q':search})

            soup = BeautifulSoup(r.text, "lxml")

            res = soup.find("div", {"id": "resultStats"})

            nb_result = ''.join(x for x in res.text if x.isdigit())

            print(nb_result)

            google_result.loc[index, 'google_result']=nb_result

        except urllib.error.HTTPError:

            google_result.loc[index,'google_result']=0

  

        output = pd.DataFrame({'id': google_result.loc[index, 'title'],

                    'google_result': google_result.loc[index, 'google_result']}, index=[0])

        output.to_csv(csv, header=False, index=False)

t = train[['id','revenue', 'title']]

          

hover = HoverTool(tooltips = [

            ('Titre','@title'),

            ('Revenue','@revenue'),

            ('id','@id')

           ])





fig = figure(x_axis_label='Films',

             y_axis_label='Revenue',

             title='Revenue for each Films',

            tools=[hover])





fig.square(x='id',

           y='revenue',

          source=t)



show(fig)
t = train[['id','revenue', 'title']]



          

hover = HoverTool(tooltips = [

            ('Titre','@title'),

            ('Revenue in log1p','@revenue'),

            ('id','@id')

           ])





fig = figure(x_axis_label='Films',

             y_axis_label='Revenue Revenue in log1p',

             title='Revenue in log1p for each Films',

            tools=[hover])





fig.square(x='id',

           y='revenue',

          source=t)





show(fig)
t = train[['id','runtime', 'title','revenue']].copy()

t['revenue'] = np.log1p(t.revenue)

          

hover = HoverTool(tooltips = [

            ('Titre','@title'),

            ('Runtime','@runtime'),

            ('id','@id'),

            ('Revenue','@revenue')

           ])





fig = figure(x_axis_label='Films',

             y_axis_label='Runtime',

             title='Runtime for each Films',

            tools=[hover])





fig.square(x='id',

           y='runtime',

          source=t)





show(fig)
t= train[['id','title','runtime','revenue','release_date']].copy()



          

t_150=t.loc[(t['runtime'] >= 150), ['id','title','runtime','revenue','release_date']] 

t_150['revenue'] = np.log1p(t_150.revenue)

          



          

hover = HoverTool(tooltips = [

            ('Titre','@title'),

            ('Runtime','@runtime'),

            ('id','@id'),

            ('Revenue','@revenue'),

            ('Release date','@release_date')

           ])





fig = figure(x_axis_label='Revenue in log1p',

             y_axis_label='Runtime',

             title='Runtime for each Films',

            tools=[hover])





fig.square(x='revenue',

           y='runtime',

          source=t_150)





show(fig)
t= train[['id','title','runtime','revenue']].copy()



t.iloc[1335]=t.iloc[1335].replace(np.nan, int(120))

t.iloc[2302]=t.iloc[2302].replace(np.nan, int(90))





    

t['runtime_cat_min_60'] = t['runtime'].apply(lambda x: 1 if (x <=60) else 0)

t['runtime_cat_61_80'] = t['runtime'].apply(lambda x: 1 if (x >60)&(x<=80) else 0)

t['runtime_cat_81_100'] = t['runtime'].apply(lambda x: 1 if (x >80)&(x<=100) else 0)

t['runtime_cat_101_120'] = t['runtime'].apply(lambda x: 1 if (x >100)&(x<=120) else 0)

t['runtime_cat_121_140'] = t['runtime'].apply(lambda x: 1 if (x >120)&(x<=140) else 0)

t['runtime_cat_141_170'] = t['runtime'].apply(lambda x: 1 if (x >140)&(x<=170) else 0)

t['runtime_cat_171_max'] = t['runtime'].apply(lambda x: 1 if (x >=170) else 0)





t.loc[t.runtime_cat_min_60 == 1,'runtime_category'] = 'cat_min-60'

t.loc[t.runtime_cat_61_80 == 1,'runtime_category'] = 'cat_61-80'

t.loc[t.runtime_cat_81_100 == 1,'runtime_category'] = 'cat_81-100'

t.loc[t.runtime_cat_101_120 == 1,'runtime_category'] = 'cat_101-120'

t.loc[t.runtime_cat_121_140 == 1,'runtime_category'] = 'cat_121-140'

t.loc[t.runtime_cat_141_170 == 1,'runtime_category'] = 'cat_141-170'

t.loc[t.runtime_cat_171_max == 1,'runtime_category'] = 'cat_171-max'





#to count how many samples do we have for a category. We want at at least 15 exemples to categorise a data. 

# print(Counter(t['runtime_cat_171_max']==1))





cat = t['runtime_category']

ctr = Counter(cat)

cat = [x for x in ctr]

unique_names = pd.Series(cat).unique()



dic={}

for a in unique_names:

    mask = t.runtime_category.apply(lambda x: a in x)

    dic[a] = t[mask]['revenue'].mean()

    

t = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'runtime_cat'})



t = t.nlargest(6, 'mean_revenue')



t['color'] = Category20c[6]



hover1 = HoverTool(tooltips = [

            ('Runtime_category','@runtime_cat'),

            ('Revenue','@mean_revenue')

           ])



p = figure(x_range=t.runtime_cat, plot_width=800,plot_height=400, toolbar_location=None, title="Revenue per runtime category", tools=[hover1])

p.vbar(x='runtime_cat', top='mean_revenue', width=0.9, source=t, legend='runtime_cat',

       line_color='white',fill_color='color')



p.xgrid.grid_line_color = None

p.legend.orientation = "horizontal"

p.legend.location = "top_center"



show(p)
# feature engeneering : film by runtime category

train['runtime_cat_min_60'] = train['runtime'].apply(lambda x: 1 if (x <=60) else 0)

train['runtime_cat_61_80'] = train['runtime'].apply(lambda x: 1 if (x >60)&(x<=80) else 0)

train['runtime_cat_81_100'] = train['runtime'].apply(lambda x: 1 if (x >80)&(x<=100) else 0)

train['runtime_cat_101_120'] = train['runtime'].apply(lambda x: 1 if (x >100)&(x<=120) else 0)

train['runtime_cat_121_140'] = train['runtime'].apply(lambda x: 1 if (x >120)&(x<=140) else 0)

train['runtime_cat_141_170'] = train['runtime'].apply(lambda x: 1 if (x >140)&(x<=170) else 0)

train['runtime_cat_171_max'] = train['runtime'].apply(lambda x: 1 if (x >=170) else 0)



test['runtime_cat_min_60'] = test['runtime'].apply(lambda x: 1 if (x <=60) else 0)

test['runtime_cat_61_80'] = test['runtime'].apply(lambda x: 1 if (x >60)&(x<=80) else 0)

test['runtime_cat_81_100'] = test['runtime'].apply(lambda x: 1 if (x >80)&(x<=100) else 0)

test['runtime_cat_101_120'] = test['runtime'].apply(lambda x: 1 if (x >100)&(x<=120) else 0)

test['runtime_cat_121_140'] = test['runtime'].apply(lambda x: 1 if (x >120)&(x<=140) else 0)

test['runtime_cat_141_170'] = test['runtime'].apply(lambda x: 1 if (x >140)&(x<=170) else 0)

test['runtime_cat_171_max'] = test['runtime'].apply(lambda x: 1 if (x >=170) else 0)
t = train[['id','title','runtime','revenue','release_date','budget']].copy()

t['revenue'] = np.log1p(t.revenue)







hover = HoverTool(tooltips = [

            ('Titre','@title'),

            ('Revenue','@revenue'),

            ('Budget','@budget')

           ])





fig = figure(x_axis_label='Budget',

             y_axis_label='Revenue',

             title='log Revenue vs log Budget ',

            tools=[hover])







fig.square('budget', 'revenue',source=t)



show(fig)
# feature engeneering : Films budget  

train['budget'] = np.log1p(train.budget)

test['budget'] = np.log1p(test.budget)
t = train[['id','title','runtime','revenue','release_date','budget','popularity']].copy()

t['revenue'] = np.log1p(t.revenue)



hover = HoverTool(tooltips = [

            ('Titre','@title'),

            ('Revenue','@revenue'),

            ('Popularity','@popularity')

            

           ])





fig = figure(x_axis_label='Popularity',

             y_axis_label='Revenue',

             title='log Revenue vs log Popularity ',

            tools=[hover])







fig.square('popularity', 'revenue',source=t)



show(fig)
# feature engeneering : popularity

train['popularity'] = np.log1p(train.popularity)

test['popularity'] = np.log1p(test.popularity)
#Plot : Revenue for each film that has homepage or not 



t = train[['revenue','homepage','title']].copy()



t['film_that_has_homepage'] = t['homepage'].isnull().apply(lambda x: str(False) if x==True  else str(True))





t = t.groupby('film_that_has_homepage')['revenue'].mean().reset_index()



hover1 = HoverTool(tooltips = [

            ('Mean revenue','@revenue'),

           ])





t['color'] = [Spectral6[1],Spectral6[2]]





p = figure(x_range=['False','True'], plot_width=600,plot_height=400, toolbar_location=None, title="Revenue for a film that has homepage", tools=[hover1])

p.vbar(x='film_that_has_homepage', top='revenue', width=0.9, source=t, legend='film_that_has_homepage',

       line_color='white', fill_color='color')



p.xgrid.grid_line_color = None

p.legend.orientation = "horizontal"

p.legend.location = 'top_left'



show(p)
# feature engeneering : Film that has homepage

train['film_that_has_homepage'] = train['homepage'].isnull().apply(lambda x: 0 if x==True else 1).copy()

test['film_that_has_homepage'] = test['homepage'].isnull().apply(lambda x: 0 if x==True else 1).copy()
t = train[['revenue','original_language','title']].copy()





lang = t['original_language']

ctr = Counter(lang).most_common(17)

lang = [x[0] for x in ctr ]

unique_names = pd.Series(lang).unique()







dic={}

for a in unique_names:

    mask = t.original_language.apply(lambda x: a in x)

    dic[a] = t[mask]['revenue'].mean()



t = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'langue'})



t = t.nlargest(12, 'mean_revenue')



t['color'] = Category20c[12]



hover1 = HoverTool(tooltips = [

            ('Langue','@langue'),

            ('Revenue','@mean_revenue')

           ])



p = figure(x_range=t.langue, plot_width=1400,plot_height=400, toolbar_location=None, title="Revenue per original language", tools=[hover1])

p.vbar(x='langue', top='mean_revenue', width=0.9, source=t, legend='langue',

       line_color='white', fill_color='color')



p.xgrid.grid_line_color = None

p.legend.orientation = "horizontal"

p.legend.location = "top_center"



show(p)
# feature engeneering : one hot encoding for original language that have at least 5 samples

lang = train['original_language']

lang_more_17_samples = [x[0] for x in Counter(pd.DataFrame(lang).stack()).most_common(17)]



for col in lang_more_17_samples :

    train[col] = train['original_language'].apply(lambda x: 1 if x == col else 0)

for col in lang_more_17_samples :

    test[col] = test['original_language'].apply(lambda x: 1 if x == col else 0)

# print(train['Drama'])

google_train_path = '../input/google-result/google_result_train.csv'

google_test_path = '../input/google-result/google_result_test.csv'



google_train = pd.read_csv(google_train_path)

google_test = pd.read_csv(google_test_path)

train['google_result'] = google_train['result']

test['google_result'] = google_test['result']
t = train[['revenue','title','google_result']].copy()

t['revenue']=np.log1p(t.revenue)

t['google_result']=np.log1p(t.google_result)





hover = HoverTool(tooltips = [

            ('Titre','@title'),

            ('Revenue','@revenue'),

            ('Google result number','@google_result')

            

           ])





fig = figure(x_axis_label='Google result number',

             y_axis_label='Revenue',

             title='log Revenue vs log google_result ',

            tools=[hover])







fig.square('google_result', 'revenue',source=t)



show(fig)
# feature engeneering : popularity with google search 

train['google_result']=np.log1p(train.google_result)

test['google_result']=np.log1p(test.google_result)
t = train[['revenue','belongs_to_collection','title']].copy()





t['film_belongs_to_collection'] = t['belongs_to_collection'].apply(lambda x: str(False) if x == {} else str(True))





t = t.groupby('film_belongs_to_collection')['revenue'].mean().reset_index()





hover1 = HoverTool(tooltips = [

            ('Mean revenue','@revenue'),

           ])





t['color'] = [Spectral6[0],Spectral6[1]]





p = figure(x_range=['False','True'], plot_width=600,plot_height=400, toolbar_location=None, title="Mean revenue for a film belonging to a collection", tools=[hover1])

p.vbar(x='film_belongs_to_collection', top='revenue', width=0.9, source=t, legend='film_belongs_to_collection',

       line_color='white', fill_color='color')



p.xgrid.grid_line_color = None

p.legend.orientation = "horizontal"

p.legend.location = 'top_left'



show(p)
# feature engeneering : Film that belongs_to_collection 

train['film_belongs_to_collection'] = train['belongs_to_collection'].apply(lambda x: 0 if x == {} else 1)

test['film_belongs_to_collection'] = test['belongs_to_collection'].apply(lambda x: 0 if x == {} else 1)
t = train[['id','revenue', 'title', 'genres']].copy()

t['genres'] = [[y['name'] for y in x] for x in t['genres']]



genres = t['genres'].sum()

ctr = Counter(genres)

df_genres = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'genre', 0:'count'})       

df_genres=df_genres.sort_values('count', ascending=False)

df_genres = df_genres[df_genres['count'] > 1]

df_genres = df_genres.nlargest(20, 'count')





genres = list(df_genres['genre'])



dic={}

for a in genres:

    mask = t.genres.apply(lambda x: a in x)

    dic[a] = t[mask]['revenue'].mean()



t = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'genre'})



t['color'] = Category20c[len(t)]



hover1 = HoverTool(tooltips = [

            ('Genre','@genre'),

            ('Genre mean revenue','@mean_revenue')

           ])



p = figure(x_range=t.genre, plot_width=1400,plot_height=400, toolbar_location=None, title="Mean revenue per genre", tools=[hover1])

p.vbar(x='genre', top='mean_revenue', width=0.9, source=t, legend='genre',

       line_color='white', fill_color='color')



p.xgrid.grid_line_color = None

p.legend.orientation = "horizontal"

p.legend.location = "top_center"



show(p)

t = train[['id','revenue', 'genres']]

x = [[y['name'] for y in x] for x in t['genres']]

x = Counter(pd.DataFrame(x).stack())

x = pd.Series(x)





data = x.reset_index(name='value').rename(columns={'index':'genre'})

data['angle'] = data['value']/data['value'].sum() * 2*np.pi

data['color'] = Category20c[len(x)]



p = figure(plot_height=350, title="Number of movies per genres", toolbar_location=None,

           tools="hover", tooltips="@genre: @value", x_range=(-0.5, 1.0))



p.wedge(x=0, y=1, radius=0.4,

        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),

        line_color="white", fill_color='color', legend='genre', source=data)



p.axis.axis_label=None

p.axis.visible=False

p.grid.grid_line_color = None



show(p)
train['genres_names'] = [[y['name'] for y in x] for x in train['genres']]



# genres = train['genres_names'].sum()

# ctr = Counter(genres)

# genres=[n for n in ctr if ctr[n] > 249]

# genres_list = pd.Series(genres).unique()



genres_list=['Action', 'Adventure', 'Science Fiction', 'Family', 'Fantasy','Animation']

        

for a in genres_list :

    train['genre_'+a]=train['genres_names'].apply(lambda x: 1 if a in x else 0)

train = train.drop(['genres_names'], axis=1)



test['genres_names'] = [[y['name'] for y in x] for x in test['genres']]

for a in genres_list :

    test['genre_'+a]=test['genres_names'].apply(lambda x: 1 if a in x else 0)

test = test.drop(['genres_names'], axis=1)
# feature engeneering : release date 

def date_features(df):

    df[['release_month','release_day','release_year']]=df['release_date'].str.split('/',expand=True).replace(np.nan, 0).astype(int)

    df['release_year'] = df['release_year']

    df.loc[ (df['release_year'] <= 18) & (df['release_year'] < 100), "release_year"] += 2000

    df.loc[ (df['release_year'] > 18)  & (df['release_year'] < 100), "release_year"] += 1900

    df['release_date'] = pd.to_datetime(df['release_date'])

    df['release_month'] = df['release_date'].dt.month

    # df['release_day'] = df['release_date'].dt.day

    df['release_quarter'] = df['release_date'].dt.quarter

    df.drop(columns=['release_date'], inplace=True)

    

    return df



train=date_features(train)

test=date_features(test)
# mean revenue per year 



t = train[['id','revenue','release_year']].copy()



t = t.groupby('release_year')['revenue'].aggregate('mean')

t=np.log1p(t)



hover = HoverTool(tooltips = [

            ('Year','@x'),

            ('Revenue','@top')

           ])





fig = figure(plot_height=400,

             plot_width=600,

             x_axis_label='Year',

             y_axis_label='Mean revenue',

             title='Log mean revenue for each year',

             tools = [hover])





fig.vbar(x=t.index,

           top=t.values, 

           width=0.9,

           color='royalblue')



show(fig)
# mean revenue per month 



t = train[['id','revenue', 'release_month']]

months_mean_revenues = t.groupby('release_month')['revenue'].aggregate('mean')





hover1 = HoverTool(tooltips = [

            ('Month','@x'),

            ('Revenue','@top')

           ])





fig = figure(plot_height=400,

             plot_width=600,

             x_axis_label='Month',

             y_axis_label='Mean revenue',

             title='Mean revenue for each months',

             tools = [hover1])





fig.vbar(x=months_mean_revenues.index,

           top=months_mean_revenues.values, 

           width=0.9,

           color='royalblue')







show(fig)
# mean revenue per month 



t = train[['id','revenue', 'release_quarter']]

quarters_mean_revenues = t.groupby('release_quarter')['revenue'].aggregate('mean')



hover1 = HoverTool(tooltips = [

            ('Quarter','@x'),

            ('Revenue','@top')

           ])





fig = figure(plot_height=400,

             plot_width=600,

             x_axis_label='Quarter',

             y_axis_label='Mean revenue',

             title='Mean revenue for each quarter',

             tools = [hover1])





fig.vbar(x=quarters_mean_revenues.index,

           top=quarters_mean_revenues.values, 

           width=0.9,

           color='royalblue')







show(fig)
# feature engeneering : Release date per month one hot encoding

for col in range (1,12) :

    train['month'+str(col)] = train['release_month'].apply(lambda x: 1 if x == col else 0)



for col in range (1,12) :

    test['month'+str(col)] = test['release_month'].apply(lambda x: 1 if x == col else 0)

    

# feature engeneering : Release date per quarter one hot encoding

for col in range (1,4) :

    train['quarter'+str(col)] = train['release_quarter'].apply(lambda x: 1 if x == col else 0)



for col in range (1,4) :

    test['quarter'+str(col)] = test['release_quarter'].apply(lambda x: 1 if x == col else 0)



# # feature engeneering : Release date per months mean revenues

# train['months_mean_revenue'] = train['release_month'].apply(lambda x: months_mean_revenues[x])

# train['quarter_mean_revenue'] = train['release_quarter'].apply(lambda x: quarters_mean_revenues[x])



# test['release_quarter'].fillna(0, inplace=True)

# test['release_month'].fillna(0,inplace=True)



# # feature engeneering : Release date per quarter mean revenues

# test['months_mean_revenue'] = test['release_month'].apply(lambda x: months_mean_revenues[x] if x > 0 else 0)

# test['quarter_mean_revenue'] = test['release_quarter'].apply(lambda x: quarters_mean_revenues[x] if x > 0 else 0)
# mean revenue per quarter for animation movies 



t = train[['id','revenue', 'title', 'genres', 'release_quarter']].copy()

t['genres'] = [[y['name'] for y in x] for x in t['genres']]

mask = t.genres.apply(lambda x: 'Animation' in x)

t = t[mask]

t = t.groupby('release_quarter')['revenue'].aggregate('mean')





hover1 = HoverTool(tooltips = [

            ('Quarter','@x'),

            ('Revenue','@top')

           ])





fig = figure(plot_height=400,

             plot_width=600,

             x_axis_label='Quarter',

             y_axis_label='Mean revenue',

             title='Mean revenue for each quarter for animation movies',

             tools = [hover1])





fig.vbar(x=t.index,

           top=t.values, 

           width=0.9,

           color='royalblue')







show(fig)

# mean revenue per quarter for drama movies 



t = train[['id','revenue', 'title', 'genres', 'release_month']].copy()

t['genres'] = t['genres'].apply(lambda x: [y['name'] for y in x])

mask = t.genres.apply(lambda x: 'Drama' in x)

t = t[mask]

t = t.groupby('release_month')['revenue'].aggregate('mean')





hover1 = HoverTool(tooltips = [

            ('Quarter','@x'),

            ('Revenue','@top')

           ])





fig = figure(plot_height=400,

             plot_width=600,

             x_axis_label='Month',

             y_axis_label='Mean revenue',

             title='Mean revenue for each month for drama movies',

             tools = [hover1])





fig.vbar(x=t.index,

           top=t.values, 

           width=0.9,

           color='royalblue')







show(fig)

train = train.drop(['release_month', 'release_quarter'], axis=1)

test = test.drop(['release_month', 'release_quarter'], axis=1)
t = train[['id','revenue', 'title', 'cast']].copy()

t['cast'] = [[y['name'] for y in x] for x in t['cast']]

t['cast'] = t['cast'].apply(lambda x: x[:3])



names = t['cast'].sum()

ctr = Counter(names)

df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'actor', 0:'count'})       

df_names=df_names.sort_values('count', ascending=False)

df_names = df_names[df_names['count'] > 8]

 

p = figure(plot_width=1300, plot_height=500, title="Most common actors",

           x_range=df_names['actor'], toolbar_location=None, tooltips=[("Actor", "@actor"), ("Count", "@count")])



p.vbar(x='actor', top='count', width=1, source=df_names,

       line_color="white" )



p.y_range.start = 0

p.x_range.range_padding = 0.05

p.xgrid.grid_line_color = None

p.xaxis.axis_label = "Actors name"

p.xaxis.major_label_orientation = 1.2

p.outline_line_color = None



show(p)
t = train[['id','revenue', 'title', 'cast']].copy()

t['cast'] = [[y['name'] for y in x] for x in t['cast']]

t['cast'] = t['cast'].apply(lambda x: x[:3])



df_names_revenue = df_names.nlargest(20, 'count')

names = list(df_names_revenue['actor'])



dic={}

for a in names:

    mask = t.cast.apply(lambda x: a in x)

    dic[a] = t[mask]['revenue'].mean()



t = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'actor'})



t = t.nlargest(20, 'mean_revenue')



t['color'] = Category20c[20]



hover1 = HoverTool(tooltips = [

            ('Actor','@actor'),

            ('Movies mean revenue','@mean_revenue')

           ])



p = figure(x_range=t.actor, plot_width=1400,plot_height=400, toolbar_location=None, title="20 most common actors movies mean revenue", tools=[hover1])

p.vbar(x='actor', top='mean_revenue', width=0.9, source=t, legend='actor',

       line_color='white', fill_color='color')



p.xgrid.grid_line_color = None

p.legend.orientation = "horizontal"

p.legend.location = "top_center"



show(p)

df_names = df_names[df_names['count'] > 11]

names_list = list(df_names['actor'])



train['cast_names']=[[y['name'] for y in x] for x in train['cast']]

train['cast_names'] = train['cast_names'].apply(lambda x: x[:3])



dic={}

for a in names_list:

    mask = train['cast_names'].apply(lambda x: a in x)

    dic[a] = train[mask]['revenue'].mean()



actors_mean_revenue = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'actor'})

names_list = list(actors_mean_revenue.nlargest(40, 'mean_revenue')['actor'])



train['actors_mean_revenue'] = train['cast_names'].apply(lambda x: actors_mean_revenue[actors_mean_revenue['actor'].isin(x)].mean()['mean_revenue'])

train['actors_mean_revenue'].fillna(0,inplace=True)





train['total_top_actors_revenue']=train['cast_names'].apply(lambda x: sum([1 for i in x if i in names_list]))

# for a in names_list :

#     train['actor_'+a]=train['cast_names'].apply(lambda x: 1 if a in x else 0)

train = train.drop(['cast_names'], axis=1)



test['cast_names']=[[y['name'] for y in x] for x in test['cast']]

test['cast_names'] = test['cast_names'].apply(lambda x: x[:3])



test['actors_mean_revenue'] = test['cast_names'].apply(lambda x: actors_mean_revenue[actors_mean_revenue['actor'].isin(x)].mean()['mean_revenue'])

test['actors_mean_revenue'].fillna(0,inplace=True)



test['total_top_actors_revenue']=test['cast_names'].apply(lambda x: sum([1 for i in x if i in names_list]))



# for a in names_list :

#     test['actor_'+a]=test['cast_names'].apply(lambda x: 1 if a in x else 0)

test = test.drop(['cast_names'], axis=1)

t = train[['id','revenue', 'title', 'cast']].copy()

t['cast'] = [[y['name'] for y in x] for x in t['cast']]

t['cast'] = t['cast'].apply(lambda x: x[:3])



names = t['cast'].sum()

ctr = Counter(names)

names=[n for n in ctr if ctr[n] > 0]

unique_names = pd.Series(names).unique()



dic={}

for a in unique_names:

    mask = t.cast.apply(lambda x: a in x)

    dic[a] = t[mask]['revenue'].mean()



actors_mean_revenue = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'actor'})



t = actors_mean_revenue.nlargest(20, 'mean_revenue')



t['color'] = Category20c[20]



hover1 = HoverTool(tooltips = [

            ('Actor','@actor'),

            ('Revenue','@mean_revenue')

           ])



p = figure(x_range=t.actor, plot_width=1400,plot_height=400, toolbar_location=None, title="20 actors with highest mean revenue", tools=[hover1])

p.vbar(x='actor', top='mean_revenue', width=0.9, source=t, legend='actor',

       line_color='white', fill_color='color')



p.xgrid.grid_line_color = None

p.legend.orientation = "horizontal"

p.legend.location = "top_center"



show(p)

t = train[['id','popularity', 'title', 'cast']].copy()

t['popularity'] = np.expm1(t['popularity'])

t['cast'] = [[y['name'] for y in x] for x in t['cast']]

t['cast'] = t['cast'].apply(lambda x: x[:3])



names = t['cast'].sum()

ctr = Counter(names)

names=[n for n in ctr if ctr[n] > 0]

unique_names = pd.Series(names).unique()



dic={}

for a in unique_names:

    mask = t.cast.apply(lambda x: a in x)

    dic[a] = t[mask]['popularity'].mean()



t = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'actor'})



t = t.nlargest(20, 'mean_revenue')



t['color'] = Category20c[20]



hover1 = HoverTool(tooltips = [

            ('Actor','@actor'),

            ('Revenue','@mean_revenue')

           ])



p = figure(x_range=t.actor, plot_width=1400,plot_height=400, toolbar_location=None, title="20 actors with highest mean popularity", tools=[hover1])

p.vbar(x='actor', top='mean_revenue', width=0.9, source=t, legend='actor',

       line_color='white', fill_color='color')



p.xgrid.grid_line_color = None

p.legend.orientation = "horizontal"

p.legend.location = "top_center"



show(p)
t = train[['id','revenue', 'title', 'cast']].copy()



t['cast_ids']=[[y['id'] for y in x] for x in t['cast']]

t['cast_ids'] = t['cast_ids'].apply(lambda x: x[:3])

t['nominations'] = t['cast_ids'].apply(lambda x: nominations[nominations['id'].isin(x)]['nominations'].sum())

t=t.drop(['cast'], axis=1)



hover1 = HoverTool(tooltips = [

            ('Titre','@title'),

            ('Revenue','@revenue'),

            ('Nominations','@nominations')

           ])





fig = figure(x_axis_label='Nominations for all main actors',

             y_axis_label='Revenue',

             title='Nominations vs. Revenue',

            tools=[hover1])





fig.square(x='nominations',

           y='revenue',

          source=t)



show(fig)
t = train[['id','revenue', 'title', 'cast']].copy()



t['cast_ids']=[[y['id'] for y in x] for x in t['cast']]

t['cast_ids'] = t['cast_ids'].apply(lambda x: x[:3])

t['nominations'] = t['cast_ids'].apply(lambda x: 'True' if (nominations[nominations['id'].isin(x)]['nominations'] != 0).any() else 'False')

df_has_nominated_actor=t.drop(['cast', 'id'], axis=1)

t = df_has_nominated_actor.groupby('nominations')['revenue'].mean().reset_index()

hover1 = HoverTool(tooltips = [

            ('Mean revenue','@revenue'),

           ])



t['color'] = [Spectral6[1],Spectral6[2]]



p = figure(x_range=['False', 'True'], plot_width=400,plot_height=400, toolbar_location=None, title="Has a nominated actor", tools=[hover1])

p.vbar(x='nominations', top='revenue', width=0.9, source=t, legend='nominations',

       line_color='white', fill_color='color')



p.xgrid.grid_line_color = None

p.legend.orientation = "horizontal"

p.legend.location = "top_center"



show(p)
t = train[['id','revenue', 'title', 'cast']].copy()



t['cast_ids']=[[y['id'] for y in x] for x in t['cast']]

t['cast_ids'] = t['cast_ids'].apply(lambda x: x[:6])

t['nominations'] = t['cast_ids'].apply(lambda x: str((nominations[nominations['id'].isin(x)]['nominations'] != 0).sum()))

t=t.drop(['cast'], axis=1)

t = t.groupby('nominations')['revenue'].mean().reset_index()



hover1 = HoverTool(tooltips = [

            ('Mean revenue','@revenue'),

           ])





t['color'] = Spectral6+[Spectral6[1]]



p = figure(x_range=['0','1','2','3','4','5', '6'], plot_width=500,plot_height=400, toolbar_location=None, title="Revenue vs. number of nominated actors", tools=[hover1])

p.vbar(x='nominations', top='revenue', width=0.9, source=t, legend='nominations',

       line_color='white', fill_color='color')



p.xgrid.grid_line_color = None

p.legend.location='top_left'



show(p)

df_has_nominated_actor['nominations'] = df_has_nominated_actor['nominations'].apply(lambda x: 1 if x == 'True' else 0)

train['has_nominated_actor'] = df_has_nominated_actor['nominations']





test['cast_ids']=[[y['id'] for y in x] for x in test['cast']]

test['cast_ids'] = test['cast_ids'].apply(lambda x: x[:3])

test['has_nominated_actor'] = test['cast_ids'].apply(lambda x: 0 if (nominations[nominations['id'].isin(x)]['nominations'] != 0).any() else 1)

test = test.drop(['cast_ids'], axis=1)





train['cast_ids']=[[y['id'] for y in x] for x in train['cast']]

train['cast_ids'] = train['cast_ids'].apply(lambda x: x[:4])

train['nominated_actors'] = train['cast_ids'].apply(lambda x: (nominations[nominations['id'].isin(x)]['nominations'] != 0).sum())



test['cast_ids']=[[y['id'] for y in x] for x in test['cast']]

test['cast_ids'] = test['cast_ids'].apply(lambda x: x[:4])

test['nominated_actors'] = test['cast_ids'].apply(lambda x: (nominations[nominations['id'].isin(x)]['nominations'] != 0).sum())

t = train[['id','revenue', 'title', 'crew']].copy()

t['crew'] = [[y['name'] for y in x if y['department']=='Directing'] for x in t['crew'] ]

t['crew'] = t['crew'].apply(lambda x: x[:3])



names = t['crew'].sum()

ctr = Counter(names)

df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'actor', 0:'count'})       

df_names=df_names.sort_values('count', ascending=False)

df_names = df_names[df_names['count'] > 4]

 

p = figure(plot_width=1300, plot_height=500, title="Most common directors",

           x_range=df_names['actor'], toolbar_location=None, tooltips=[("Director", "@actor"), ("Count", "@count")])



p.vbar(x='actor', top='count', width=1, source=df_names,

       line_color="white" )



p.y_range.start = 0

p.x_range.range_padding = 0.05

p.xgrid.grid_line_color = None

p.xaxis.axis_label = "Director names"

p.xaxis.major_label_orientation = 1.2

p.outline_line_color = None



show(p)
df_names = df_names[df_names['count'] > 10]

names_list = list(df_names['actor'])



train['crew_names'] = [[y['name'] for y in x if y['department']=='Directing'] for x in train['crew'] ]

train['crew_names'] = train['crew_names'].apply(lambda x: x[:3])



dic={}

for a in names_list:

    mask = train['crew_names'].apply(lambda x: a in x)

    dic[a] = train[mask]['revenue'].mean()



directors_mean_revenue = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'director'})



names_list = list(directors_mean_revenue.nlargest(40, 'mean_revenue')['director'])



# train['total_top_actors_revenue']=train['cast_names'].apply(lambda x: sum([1 for i in x if i in names_list]))



for a in names_list :

    train['director_'+a]=train['crew_names'].apply(lambda x: 1 if a in x else 0)

train = train.drop(['crew_names'], axis=1)



test['crew_names'] = [[y['name'] for y in x if y['department']=='Directing'] for x in test['crew'] ]

test['crew_names'] = test['crew_names'].apply(lambda x: x[:3])

for a in names_list :

    test['director_'+a]=test['crew_names'].apply(lambda x: 1 if a in x else 0)

test = test.drop(['crew_names'], axis=1)

t = train[['id','revenue', 'title', 'production_companies']].copy()

t['production_companies'] = [[y['name'] for y in x] for x in t['production_companies'] ]

t['production_companies'] = t['production_companies'].apply(lambda x: x[:3])



names = t['production_companies'].sum()

ctr = Counter(names)

df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'actor', 0:'count'})       

df_names=df_names.sort_values('count', ascending=False)

df_names = df_names[df_names['count'] > 9]

 

p = figure(plot_width=1300, plot_height=500, title="Number of movies per production company",

           x_range=df_names['actor'], toolbar_location=None, tooltips=[("Company", "@actor"), ("Count", "@count")])



p.vbar(x='actor', top='count', width=1, source=df_names,

       line_color="white" )



p.y_range.start = 0

p.x_range.range_padding = 0.05

p.xgrid.grid_line_color = None

p.xaxis.axis_label = "Production company"

p.xaxis.major_label_orientation = 1.2

p.outline_line_color = None



show(p)
df_names = df_names[df_names['count'] > 9]

names_list = list(df_names['actor'])



train['production_companies'] = [[y['name'] for y in x] for x in train['production_companies'] ]

train['production_companies'] = train['production_companies'].apply(lambda x: x[:3])



dic={}

for a in names_list:

    mask = train['production_companies'].apply(lambda x: a in x)

    dic[a] = train[mask]['revenue'].mean()



companies_mean_revenue = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'company'})



names_list = list(companies_mean_revenue.nlargest(20, 'mean_revenue')['company'])



# train['total_top_companies']=train['production_companies'].apply(lambda x: sum([1 for i in x if i in names_list]))

for a in names_list :

    train['production_'+a]=train['production_companies'].apply(lambda x: 1 if a in x else 0)

train = train.drop(['production_companies'], axis=1)



test['production_companies'] = [[y['name'] for y in x] for x in test['production_companies'] ]

test['production_companies'] = test['production_companies'].apply(lambda x: x[:3])

# test['total_top_companies']=test['production_companies'].apply(lambda x: sum([1 for i in x if i in names_list]))



for a in names_list :

    test['production_'+a]=test['production_companies'].apply(lambda x: 1 if a in x else 0)

test = test.drop(['production_companies'], axis=1)
# Create target object and call it y

y = np.log1p(train.revenue)
# Create X

X = train.drop(['id','runtime', 'release_day'], axis=1)



test_X = test.drop(['id','runtime', 'release_day'], axis=1).select_dtypes(exclude=['object'])



    

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1,test_size=0.33)



train_X=train_X.drop(['revenue'], axis=1).select_dtypes(exclude=['object'])

X=X.drop(['revenue'], axis=1).select_dtypes(exclude=['object'])

val_X_revenue=val_X.pop('revenue')

val_X_title=val_X.pop('title')

val_X=val_X.select_dtypes(exclude=['object'])



xgb_model = XGBRegressor(learning_rate=0.05, 

                            n_estimators=10000,max_depth=4)

xgb_model.fit(train_X, train_y, early_stopping_rounds=100, 

             eval_set=[(val_X, val_y)], eval_metric = 'rmse')

xbg_val_predictions=xgb_model.predict(val_X)

df=val_X.reset_index().join(pd.DataFrame(np.expm1(xbg_val_predictions)).rename(columns={0:'prediction'}))

df=df.join(val_X_revenue.reset_index()['revenue'])

df=df.join(val_X_title.reset_index()['title'])

df_x=df[['revenue','prediction', 'title']]



hover1 = HoverTool(tooltips = [

            ('Titre','@title'),

            ('Revenue','@revenue'),

            ('Prediction','@prediction')

           ])





fig = figure(x_axis_label='Revenue',

             y_axis_label='prediction',

             title='Revenue vs. Prediction',

            tools=[hover1])





fig.square(x='revenue',

           y='prediction',

          source=df_x)



show(fig)



fig, ax = plt.subplots(figsize=(15, 13))

plot_importance(xgb_model, ax=ax)

plt.show()
df_a=df_x[df_x['title']=='Top Gun']

df_a=df_a.append(df_x[df_x['title']=='Tomorrowland'])

df_x=df_a.append(df_x[df_x['title']=='Rambo III'])



fig = figure(x_axis_label='Revenue',

             y_axis_label='prediction',

             title='Revenue vs. Prediction',

            tools=[hover1])



fig.square(x='revenue',

           y='prediction',

          source=df_x)



show(fig)
xgb_model_full = XGBRegressor(n_estimators=145, learning_rate=0.05,max_depth=4)

xgb_model_full.fit(X, y)





test_preds=xgb_model_full.predict(test_X)



output = pd.DataFrame({'id': test.id,

                       'revenue': np.expm1(test_preds)})

output.to_csv('submission.csv', index=False)