from PIL import Image

import pandas as pd

im_frame = Image.open('../input/process-pred/Proceso_pred.png')

im_frame = im_frame.resize((700,650))

display(im_frame)
import numpy as np

import pandas as pd

pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')

import datetime

import lightgbm as lgb

from scipy import stats

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, KFold

from wordcloud import WordCloud

from collections import Counter

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.preprocessing import StandardScaler

stop = set(stopwords.words('english'))

import os

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import xgboost as xgb

import lightgbm as lgb

from sklearn import model_selection

from sklearn.metrics import accuracy_score

import json

import ast

import eli5

import shap

from catboost import CatBoostRegressor

from urllib.request import urlopen

from PIL import Image

from sklearn.preprocessing import LabelEncoder

import time

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn import linear_model

from shutil import copyfile

copyfile(src = "../input/funcdefd/Funciones.py", dst = "../working/Funciones.py")

from Funciones import *
train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')
train.head(2)
## Se hace identifican 7 columnas donde la información es un diccionario de datos. 

## Se transforman estas columnas en el formato apropiado. Para esto nos basamos en el siguiente kernel:

## https://www.kaggle.com/gravix/gradient-in-a-box

## La libreria ast permite identificar el tipo de dato, basados en una iteración sobre cada componente del diccionario

## guardado en cada columna.

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



        

train = text_to_dict(train,dict_columns)

test = text_to_dict(test,dict_columns)
# Inspeccionando el tamaño de nuestros datos

train.shape, test.shape
## Inspección visual de la información. 

for a in dict_columns:

    print(a)

    for i, e in enumerate(train[a][:1]):

        print(i, e)
for a in dict_columns:

    inf=pd.DataFrame(train[a].apply(lambda x: len(x) if x != {} else 0).value_counts())

    print('Porcentaje registros sin infromación: ',(inf[inf.index==0].iloc[0,:]/len(train))*100)

print('Total registros: ',len(train))
lista=['genres','production_companies','production_countries','spoken_languages','cast','crew']

for i in lista:

    list_ = list(train[i].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

    list_ = [i for i in list_ if i]

    

    df=pd.DataFrame(list_)

    df=df[0].str.replace(' ', '_')

    list_=list([df])

    plt.figure(figsize = (12, 8))

    text = ' '.join([i for j in list_ for i in j])

    

    wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,

                          width=1200, height=1000).generate(text)

    plt.imshow(wordcloud)

    plt.title('Top '+i)

    plt.axis("off")

    plt.show()
# Palabras mas comunes

list_ = list(train['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

list_ = [i for i in list_ if i]

plt.figure(figsize = (12, 8))

text = ' '.join([i for j in list_ for i in j])



wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,

                      width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top keywords')

plt.axis("off")

plt.show()

list_of_cast_genders = list(train['cast'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_cast_genders for i in j]).most_common()
list_of_crew_genders = list(train['crew'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_crew_genders for i in j]).most_common()
list_of_cast_characters = list(train['cast'].apply(lambda x: [i['character'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_cast_characters for i in j]).most_common(15)

list_of_crew_jobs = list(train['crew'].apply(lambda x: [i['job'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_crew_jobs for i in j]).most_common(15)
list_of_crew_departments = list(train['crew'].apply(lambda x: [i['department'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_crew_departments for i in j]).most_common(14)
list_of_genres = list(train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_crew_departments for i in j]).most_common(14)
list_of_cast_names = list(train['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_cast_names for i in j]).most_common(15)
list_of_languages = list(train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_languages for i in j]).most_common(15)
list_of_keywords = list(train['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_keywords for i in j]).most_common(15)
list_of_companies = list(train['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_companies for i in j]).most_common(30)
# Diccionario de columnas donde el valor relevante es la columna name:

dict_columns_com = ['belongs_to_collection','genres','production_companies','production_countries','spoken_languages','Keywords','cast','crew']



for i in dict_columns_com:

    train['num_'+i] = train[i].apply(lambda x: len(x) if x != {} else 0)

    train['all_'+i] = train[i].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

    test['num_'+i] = test[i].apply(lambda x: len(x) if x != {} else 0)

    test['all_'+i] = test[i].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

    train['all_'+i]=train['all_'+i].str.replace(' ', '_')

    test['all_'+i]=test['all_'+i].str.replace(' ', '_')
# Columnas con información especial

train['num_cast'] = train['cast'].apply(lambda x: len(x) if x != {} else 0)

train['genders_0_cast'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

train['genders_1_cast'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

train['genders_2_cast'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))



test['num_cast'] = test['cast'].apply(lambda x: len(x) if x != {} else 0)

test['genders_0_cast'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

test['genders_1_cast'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

test['genders_2_cast'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))



train['num_crew'] = train['crew'].apply(lambda x: len(x) if x != {} else 0)

train['genders_0_crew'] = train['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

train['genders_1_crew'] = train['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

train['genders_2_crew'] = train['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

test['num_crew'] = test['crew'].apply(lambda x: len(x) if x != {} else 0)

test['genders_0_crew'] = test['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

test['genders_1_crew'] = test['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

test['genders_2_crew'] = test['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))



train['num_crew_dep'] = train[i].apply(lambda x: len(x) if x != {} else 0)

train['all_crew_dep'] = train[i].apply(lambda x: ' '.join(sorted([i['department'] for i in x])) if x != {} else '')

test['num_crew_dep'] = test[i].apply(lambda x: len(x) if x != {} else 0)

test['all_crew_dep'] = test[i].apply(lambda x: ' '.join(sorted([i['department'] for i in x])) if x != {} else '')

train['num_crew_job'] = train[i].apply(lambda x: len(x) if x != {} else 0)

train['all_crew_job'] = train[i].apply(lambda x: ' '.join(sorted([i['job'] for i in x])) if x != {} else '')

test['num_crew_job'] = test[i].apply(lambda x: len(x) if x != {} else 0)

test['all_crew_job'] = test[i].apply(lambda x: ' '.join(sorted([i['job'] for i in x])) if x != {} else '')



train['collection_name'] = train['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)

train['has_collection'] = train['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)



test['collection_name'] = test['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)

test['has_collection'] = test['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)





for i in dict_columns_com:

    list_ = list(train[i].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

    top_ = [m[0] for m in Counter([i for j in list_ for i in j]).most_common(20)]

    for g in top_:

        train[i+'_' + g] = train['all_'+i].apply(lambda x: 1 if g in x else 0)



    for g in top_:

        test[i+'_' + g] = test['all_'+i].apply(lambda x: 1 if g in x else 0)



top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(15)]

for g in top_genres:

    train['genre_' + g] = train['all_genres'].apply(lambda x: 1 if g in x else 0)

top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(15)]

for g in top_cast_names:

    train['cast_name_' + g] = train['cast'].apply(lambda x: 1 if g in str(x) else 0)

top_cast_characters = [m[0] for m in Counter([i for j in list_of_cast_characters for i in j]).most_common(15)]

for g in top_cast_characters:

    train['cast_character_' + g] = train['cast'].apply(lambda x: 1 if g in str(x) else 0)



for g in top_genres:

    test['genre_' + g] = test['all_genres'].apply(lambda x: 1 if g in x else 0)

for g in top_cast_names:

    test['cast_name_' + g] = test['cast'].apply(lambda x: 1 if g in str(x) else 0)

for g in top_cast_characters:

    test['cast_character_' + g] = test['cast'].apply(lambda x: 1 if g in str(x) else 0)



top_crew_jobs = [m[0] for m in Counter([i for j in list_of_crew_jobs for i in j]).most_common(15)]

for j in top_crew_jobs:

    train['jobs_' + j] = train['crew'].apply(lambda x: sum([1 for i in x if i['job'] == j]))

top_crew_departments = [m[0] for m in Counter([i for j in list_of_crew_departments for i in j]).most_common(15)]

for j in top_crew_departments:

    train['departments_' + j] = train['crew'].apply(lambda x: sum([1 for i in x if i['department'] == j])) 



for j in top_crew_jobs:

    test['jobs_' + j] = test['crew'].apply(lambda x: sum([1 for i in x if i['job'] == j]))



for j in top_crew_departments:

    test['departments_' + j] = test['crew'].apply(lambda x: sum([1 for i in x if i['department'] == j])) 

    

top_companies = [m[0] for m in Counter([i for j in list_of_companies for i in j]).most_common(30)]

for g in top_companies:

    train['production_company_' + g] = train['all_production_companies'].apply(lambda x: 1 if g in x else 0)

for g in top_companies:

    test['production_company_' + g] = train['all_production_companies'].apply(lambda x: 1 if g in x else 0)

    

top_languages = [m[0] for m in Counter([i for j in list_of_languages for i in j]).most_common(30)]

for g in top_languages:

    train['language_' + g] = train['all_spoken_languages'].apply(lambda x: 1 if g in x else 0)

for g in top_languages:

    test['language_' + g] = test['all_spoken_languages'].apply(lambda x: 1 if g in x else 0)



top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(30)]

for g in top_keywords:

    train['keyword_' + g] = train['all_Keywords'].apply(lambda x: 1 if g in x else 0)

for g in top_keywords:

    test['keyword_' + g] = test['all_Keywords'].apply(lambda x: 1 if g in x else 0)



    

for i in dict_columns_com:

    train = train.drop([i], axis=1)

    test = test.drop([i], axis=1)
train.head()
# Función objetivo

fig, ax = plt.subplots(figsize = (16, 6))

plt.subplot(1, 1, 1)

plt.hist(train['revenue']);

plt.title('Distribución de los ingresos');
import scipy.stats as ss

from scipy import stats

dist_continu = [d for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_continuous)]

count, bins, ignored = plt.hist(train["revenue"], 100, density=True, align='mid')

params = stats.lognorm.fit(count)

d, pvalor = stats.kstest(count,"lognorm",params)



if pvalor < 0.05:

    print("No se ajusta a una lognorm")

else:

    print("Se puede ajustar a una lognorm")
evalua_dist(train['revenue'])
import seaborn as sb

numericos = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

datos_num = train.select_dtypes(include=numericos)

datos_num.drop(columns=['id'],inplace=True)

datos_num.dtypes
sb.pairplot(datos_num[[

'revenue',

'budget',

'popularity',

'runtime',

'num_genres',

'num_production_companies',

'num_production_countries',

'num_spoken_languages',

'num_Keywords',

'num_cast',

'num_crew']])

plt.show()
for i in datos_num[[

'revenue',

'budget',

'popularity',



'num_genres',

'num_production_companies',

'num_production_countries',

'num_spoken_languages',

'num_Keywords',

'num_cast',

'num_crew']]:

    print('Distribución '+i)

    evalua_dist(train[i])
import random

color=[]

ingresos=[]

for i in range(1,len(train)):

    prue="rgba("+str(random.randint(0,255))+","+str(random.randint(0,255))+","+str(random.randint(0,255))+",0.5)"

    color.append(prue)

    

for i in train.revenue.iteritems():

    ingresos.append(i)
from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



trace = go.Scatter(

                    x = train['popularity'],

                    y = train['budget'],

                    mode = "markers",

                    marker = dict(color = color,size = (train['revenue']*100)/train['revenue'].max()),

                    text= train.title+' '+train.release_date

)

data = [trace]

layout = dict(title = 'Influencia del Presupuesto y Popularidad en los ingresos',

              xaxis= dict(title= 'Popularidad',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Presupuesto',ticklen= 5,zeroline= False)

             )             

fig = dict(data = data, layout = layout)

iplot(fig)
trace = go.Scatter(

                    x = train['runtime'],

                    y = train['budget'],

                    mode = "markers",

                    marker = dict(color = color,size = (train['revenue']*100)/train['revenue'].max()),

                    text= train.title+' '+train.release_date

)

data = [trace]

layout = dict(title = 'Influencia del Presupuesto y Duracion en los ingresos',

              xaxis= dict(title= 'Duración',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Presupuesto',ticklen= 5,zeroline= False)

             )             

fig = dict(data = data, layout = layout)

iplot(fig)

trace = go.Scatter(

                    x = train['popularity'],

                    y = train['num_cast'],

                    mode = "markers",

                    marker = dict(color = color,size = (train['revenue']*100)/train['revenue'].max()),

                    text= train.title+' '+train.release_date

)

data = [trace]

layout = dict(title = 'Influencia del Presupuesto y Duracion en los ingresos',

              xaxis= dict(title= 'Popularidad',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Numero de actrores',ticklen= 5,zeroline= False)

             )             

fig = dict(data = data, layout = layout)

iplot(fig)

datos=train.sort_values(by=['revenue'], ascending=False)

datos=datos.iloc[:,0:20]

trace1 = go.Bar(

    y=datos.title,

    x=datos.budget,

    name='Presupuesto',

    orientation = 'h',

    marker = dict(

        color = 'rgba(255, 107, 51, 0.5)',

        line = dict(

            color = 'rgba(255, 107, 51, 1.0)',

            width = 3)

    )

)

trace2 = go.Bar(

    y=datos.title,

    x=datos.revenue-datos.budget,

    name='Ganancias',

    orientation = 'h',

    marker = dict(

        color = 'rgba(51, 153, 255, 0.5)',

        line = dict(

            color = 'rgba(51, 153, 255, 1.0)',

            width = 3)

    )

)



data = [trace1, trace2]

layout = go.Layout(title = 'Relación presupuesto ganancias',

    barmode='stack'

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
#Palabras comunes en los titulos

plt.figure(figsize = (12, 12))

text = ' '.join(train['original_title'].values)

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top de palabras en los títulos')

plt.axis("off")

plt.show()
# Palabras que sobre salen en las reseñas

plt.figure(figsize = (12, 12))

text = ' '.join(train['overview'].fillna('').values)

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top de palabas en las reseñas')

plt.axis("off")

plt.show()
test.loc[test['release_date'].isnull() == True, 'release_date'] = '01/01/98'

train['release_date'] = train['release_date'].apply(lambda x: fix_date(x))

test['release_date'] = test['release_date'].apply(lambda x: fix_date(x))

train['release_date'] = pd.to_datetime(train['release_date'])

test['release_date'] = pd.to_datetime(test['release_date'])
# Creando variables basados en fechas

train = process_date(train)

test = process_date(test)
d1 = train['release_date_year'].value_counts().sort_index()

d2 = test['release_date_year'].value_counts().sort_index()

data = [go.Scatter(x=d1.index, y=d1.values, name='train'), go.Scatter(x=d2.index, y=d2.values, name='test')]

layout = go.Layout(dict(title = "Peliculas por año",

                  xaxis = dict(title = 'Year'),

                  yaxis = dict(title = 'Count'),

                  ),legend=dict(

                orientation="v"))

iplot(dict(data=data, layout=layout))
d1 = train['release_date_year'].value_counts().sort_index()

d2 = train.groupby(['release_date_year'])['revenue'].sum()

data = [go.Scatter(x=d1.index, y=d1.values, name='film count'), go.Scatter(x=d2.index, y=d2.values, name='total Ingresos', yaxis='y2')]

layout = go.Layout(dict(title = "Numero de peliculas por año y total de ingresos por año",

                  xaxis = dict(title = 'Año'),

                  yaxis = dict(title = 'Cantidad'),

                  yaxis2=dict(title='Total ingresos', overlaying='y', side='right')

                  ),legend=dict(

                orientation="v"))

iplot(dict(data=data, layout=layout))
d1 = train['release_date_year'].value_counts().sort_index()

d2 = train.groupby(['release_date_year'])['revenue'].mean()

data = [go.Scatter(x=d1.index, y=d1.values, name='film count'), go.Scatter(x=d2.index, y=d2.values, name='Ingresos promedio', yaxis='y2')]

layout = go.Layout(dict(title = "Numero de peliculas e ingreso promedio por año",

                  xaxis = dict(title = 'Año'),

                  yaxis = dict(title = 'Cantiad'),

                  yaxis2=dict(title='Ingreso promedio', overlaying='y', side='right')

                  ),legend=dict(

                orientation="v"))

iplot(dict(data=data, layout=layout))
train['log_revenue'] = np.log1p(train['revenue'])
plt.figure(figsize = (12, 12))

text = ' '.join(train['tagline'].fillna('').values)

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top de palabras tagline')

plt.axis("off")

plt.show()
f, axes = plt.subplots(3, 5, figsize=(24, 12))

plt.suptitle('Ingresos vs Genero')

for i, e in enumerate([col for col in train.columns if 'genre_' in col]):

    sns.violinplot(x=e, y='revenue', data=train, ax=axes[i // 5][i % 5]);
f, axes = plt.subplots(6, 5, figsize=(24, 32))

plt.suptitle('Compañia de producción vs Ingresos')

for i, e in enumerate([col for col in train.columns if 'production_company' in col]):

    sns.violinplot(x=e, y='revenue', data=train, ax=axes[i // 5][i % 5]);
f, axes = plt.subplots(3, 5, figsize=(24, 18))

plt.suptitle('Reparto vs Ingresos')

for i, e in enumerate([col for col in train.columns if 'cast_name' in col]):

    sns.violinplot(x=e, y='revenue', data=train, ax=axes[i // 5][i % 5]);
vectorizer = TfidfVectorizer(

            sublinear_tf=True,

            analyzer='word',

            token_pattern=r'\w{1,}',

            ngram_range=(1, 2),

            min_df=5)



overview_text = vectorizer.fit_transform(train['overview'].fillna(''))

linreg = LinearRegression()

linreg.fit(overview_text, train['revenue'])

eli5.show_weights(linreg, vec=vectorizer, top=20, feature_filter=lambda x: x != '<BIAS>')
print('Target value:', train['revenue'][1000])

eli5.show_prediction(linreg, doc=train['overview'].values[1000], vec=vectorizer)
train = train.drop(['homepage', 'imdb_id', 'poster_path', 'release_date', 'status', 'log_revenue'], axis=1)

test = test.drop(['homepage', 'imdb_id', 'poster_path', 'release_date', 'status'], axis=1)

train = train.drop(['all_belongs_to_collection', 'all_production_companies', 'all_production_countries', 'all_spoken_languages', 'all_Keywords', 'all_cast', 'all_crew', 'all_crew_dep', 'all_crew_job'], axis=1)

test = test.drop(['all_belongs_to_collection', 'all_production_companies', 'all_production_countries', 'all_spoken_languages', 'all_Keywords', 'all_cast', 'all_crew', 'all_crew_dep', 'all_crew_job'], axis=1)
for col in train.columns:

    if train[col].nunique() == 1:

        print(col)

        train = train.drop([col], axis=1)

        test = test.drop([col], axis=1)
train.head(2)
for col in ['original_language', 'collection_name', 'all_genres']:

    le = LabelEncoder()

    le.fit(list(train[col].fillna('')) + list(test[col].fillna('')))

    train[col] = le.transform(train[col].fillna('').astype(str))

    test[col] = le.transform(test[col].fillna('').astype(str))
train_texts = train[['title', 'tagline', 'overview', 'original_title']]

test_texts = test[['title', 'tagline', 'overview', 'original_title']]
for col in ['title', 'tagline', 'overview', 'original_title']:

    train['len_' + col] = train[col].fillna('').apply(lambda x: len(str(x)))

    train['words_' + col] = train[col].fillna('').apply(lambda x: len(str(x.split(' '))))

    train = train.drop(col, axis=1)

    test['len_' + col] = test[col].fillna('').apply(lambda x: len(str(x)))

    test['words_' + col] = test[col].fillna('').apply(lambda x: len(str(x.split(' '))))

    test = test.drop(col, axis=1)
# Datos faltante from https://www.kaggle.com/somang1418/happy-valentines-day-and-keep-kaggling-3

train.loc[train['id'] == 16,'revenue'] = 192864          # Skinning

train.loc[train['id'] == 90,'budget'] = 30000000         # Sommersby          

train.loc[train['id'] == 118,'budget'] = 60000000        # Wild Hogs

train.loc[train['id'] == 149,'budget'] = 18000000        # Beethoven

train.loc[train['id'] == 313,'revenue'] = 12000000       # The Cookout 

train.loc[train['id'] == 451,'revenue'] = 12000000       # Chasing Liberty

train.loc[train['id'] == 464,'budget'] = 20000000        # Parenthood

train.loc[train['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II

train.loc[train['id'] == 513,'budget'] = 930000          # From Prada to Nada

train.loc[train['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol

train.loc[train['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip

train.loc[train['id'] == 850,'budget'] = 90000000        # Modern Times

train.loc[train['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman

train.loc[train['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   

train.loc[train['id'] == 1359,'budget'] = 10000000       # Stir Crazy 

train.loc[train['id'] == 1542,'budget'] = 1              # All at Once

train.loc[train['id'] == 1570,'budget'] = 15800000       # Crocodile Dundee II

train.loc[train['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp

train.loc[train['id'] == 1714,'budget'] = 46000000       # The Recruit

train.loc[train['id'] == 1721,'budget'] = 17500000       # Cocoon

train.loc[train['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed

train.loc[train['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget

train.loc[train['id'] == 2491,'revenue'] = 6800000       # Never Talk to Strangers

train.loc[train['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus

train.loc[train['id'] == 2612,'budget'] = 15000000       # Field of Dreams

train.loc[train['id'] == 2696,'budget'] = 10000000       # Nurse 3-D

train.loc[train['id'] == 2801,'budget'] = 10000000       # Fracture

test.loc[test['id'] == 3889,'budget'] = 15000000       # Colossal

test.loc[test['id'] == 6733,'budget'] = 5000000        # The Big Sick

test.loc[test['id'] == 3197,'budget'] = 8000000        # High-Rise

test.loc[test['id'] == 6683,'budget'] = 50000000       # The Pink Panther 2

test.loc[test['id'] == 5704,'budget'] = 4300000        # French Connection II

test.loc[test['id'] == 6109,'budget'] = 281756         # Dogtooth

test.loc[test['id'] == 7242,'budget'] = 10000000       # Addams Family Values

test.loc[test['id'] == 7021,'budget'] = 17540562       #  Two Is a Family

test.loc[test['id'] == 5591,'budget'] = 4000000        # The Orphanage

test.loc[test['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee



power_six = train.id[train.budget > 1000][train.revenue < 100]



for k in power_six :

    train.loc[train['id'] == k,'revenue'] =  train.loc[train['id'] == k,'revenue'] * 1000000
train
X = train.drop(['id', 'revenue'], axis=1)

y = np.log1p(train['revenue'])

X_test = test.drop(['id'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
params = {'num_leaves': 30,

         'min_data_in_leaf': 20,

         'objective': 'regression',

         'max_depth': 5,

         'learning_rate': 0.01,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.2,

         "verbosity": -1}

model1 = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

model1.fit(X_train, y_train, 

        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=200)
eli5.show_weights(model1, feature_filter=lambda x: x != '<BIAS>')
n_fold = 10

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

print(folds)
def parametros(modelo):

    if modelo=='xgb':

        params = {'eta': 0.01,

                      'objective': 'reg:linear',

                      'max_depth': 15,

                      'subsample': 0.8,

                      'colsample_bytree': 0.8,

                      'eval_metric': 'rmse',

                      'seed': 11,

                      'silent': True}

    if modelo=='lgb':

        params = {'num_leaves': 30,

                 'min_data_in_leaf': 10,

                 'objective': 'regression',

                 'max_depth': 5,

                 'learning_rate': 0.01,

                 "boosting": "gbdt",

                 "feature_fraction": 0.9,

                 "bagging_freq": 1,

                 "bagging_fraction": 0.9,

                 "bagging_seed": 11,

                 "metric": 'rmse',

                 "lambda_l1": 0.2,

                 "verbosity": -1}



    if modelo=='cat':

        params = {'learning_rate': 0.002,

                      'depth': 5,

                      'l2_leaf_reg': 10,

                      # 'bootstrap_type': 'Bernoulli',

                      'colsample_bylevel': 0.8,

                      'bagging_temperature': 0.2,

                      #'metric_period': 500,

                      'od_type': 'Iter',

                      'od_wait': 100,

                      'random_seed': 11,

                      'allow_writing_files': False}

    if modelo=='lgb_1':

        params = {'num_leaves': 30,

                 'min_data_in_leaf': 20,

                 'objective': 'regression',

                 'max_depth': 5,

                 'learning_rate': 0.01,

                 "boosting": "gbdt",

                 "feature_fraction": 0.9,

                 "bagging_freq": 1,

                 "bagging_fraction": 0.9,

                 "bagging_seed": 11,

                 "metric": 'rmse',

                 "lambda_l1": 0.2,

                 "verbosity": -1}



    if modelo=='lgb_2':

        params = {'num_leaves': 30,

                 'min_data_in_leaf': 20,

                 'objective': 'regression',

                 'max_depth': 7,

                 'learning_rate': 0.02,

                 "boosting": "gbdt",

                 "feature_fraction": 0.7,

                 "bagging_freq": 5,

                 "bagging_fraction": 0.7,

                 "bagging_seed": 11,

                 "metric": 'rmse',

                 "lambda_l1": 0.2,

                 "verbosity": -1}

    if modelo=='Random_Forest':

        params={'n_estimators':20000, 

                         'criterion':'rmse', 

                         'max_depth':15, 

                         'bootstrap':True, 

                         'oob_score':False, 

                         'n_jobs':-1, 

                         'random_state':53}

    return params
modelos=['lgb', 'xgb', 'cat', 'lgb_1', 'lgb_2','Random_Forest']

modelos=['cat', 'lgb_1', 'lgb_2']



sub = pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')



for i in modelos:

    params=parametros(i)

    oof_, prediction_= train_model(X, X_test, y, params,folds, i, True,i)

    sub['revenue'] = np.expm1(prediction_)

    sub.to_csv(i+".csv", index=False)
