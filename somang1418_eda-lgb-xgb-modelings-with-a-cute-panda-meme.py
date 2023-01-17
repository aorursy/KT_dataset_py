import numpy as np 

import pandas as pd 

import os

import json

from pandas.io.json import json_normalize

import ast

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')

import seaborn as sns

%matplotlib notebook

from scipy.stats import skew, boxcox

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from mpl_toolkits.mplot3d import Axes3D

import ast

import re

import yaml

import json

from collections import Counter

from nltk.corpus import stopwords

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split, KFold

import xgboost as xgb

import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn import model_selection

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error

import eli5

import time

from datetime import datetime

from sklearn.preprocessing import LabelEncoder

import warnings  

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)

train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')

sam_sub = pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')

print( "train dataset:", train.shape,"\n","test dataset: ",test.shape,"\n","sample_submission dataset:", sam_sub .shape)
# All time winner in the dataset

train.loc[train['revenue'].idxmax(),['title','revenue','release_date']]
# Top 20 revenue movie in the dataset



train.sort_values(by='revenue', ascending=False).head(20)[['title','revenue','release_date']]
#Let's take a look at the dataset

train.info()
test.info()
#Counting NA in dataset



fig = plt.figure(figsize=(15, 10))

train.isna().sum().sort_values(ascending=True).plot(kind='barh',colors='Blue', fontsize=20)

fig = plt.figure(figsize=(15, 10))

test.isna().sum().sort_values(ascending=True).plot(kind='barh',colors='Orange', fontsize=20)
# Revising some wrong information



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

train.loc[train['id'] == 1542,'budget'] = 15800000       # Crocodile Dundee II

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
## From this function, you can convert release_date column from the character data type to the datetime data type



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
train['release_year'].iloc[np.where(train['release_year']> 2019)][:10]
train['release_year']=np.where(train['release_year']> 2019, train['release_year']-100, train['release_year'])

test['release_year']=np.where(test['release_year']> 2019, test['release_year']-100, test['release_year'])
## Filling NA values with mode of each column



fillna_column = {'release_year':'mode','release_month':'mode',

                'release_day':'mode'}



for k,v in fillna_column.items():

    if v == 'mode':

        fill = train[k].mode()[0]

    else:

        fill = v

    print(k, ': ', fill)

    train[k].fillna(value = fill, inplace = True)

    test[k].fillna(value = fill, inplace = True)
# Putting revised year, month, and day together 



def year_month_together(df):

    year = df["release_year"].astype(int).copy().astype(str)

    month=df['release_month'].astype(int).copy().astype(str)

    day=df['release_day'].astype(int).copy().astype(str) 

    df["release_date"]=  month.str.cat(day.str.cat(year,sep="/"), sep ="/") 

    df['release_date']=pd.to_datetime(df['release_date'],format="%m/%d/%Y")

    df['release_dow'] = df['release_date'].dt.dayofweek

    return df 



train=year_month_together(train)

test=year_month_together(test)



train['release_date'].head(10)
# Counting movie number by release date



d1 = train['release_date'].value_counts().sort_index()

d2 = test['release_date'].value_counts().sort_index()

data = [go.Histogram(x=d1.index, y=d1.values, name='train'), go.Histogram(x=d2.index, y=d2.values, name='test')]

layout = go.Layout(dict(title = "Counts of release_date",

                  xaxis = dict(title = 'Month'),

                  yaxis = dict(title = 'Count'),

                  ),legend=dict(

                orientation="v"))

py.iplot(dict(data=data, layout=layout))
# Average revenue by month



fig = plt.figure(figsize=(10,10))



train.groupby('release_month').agg('mean')['revenue'].plot(kind='bar',color='navy',rot=0)

plt.ylabel('Revenue (100 million dollars)')
release_year_mean_data=train.groupby(['release_year'])['budget','popularity','revenue'].mean()

release_year_mean_data.head()



fig = plt.figure(figsize=(10, 10))

release_year_mean_data['popularity'].plot(kind='line')

plt.ylabel('Mean Popularity value')

plt.title('Mean Popularity Over Years')
release_year_mean_data=train.groupby(['release_year'])['budget','popularity','revenue'].mean()

release_year_mean_data.head()



fig = plt.figure(figsize=(13,13))

ax = plt.subplot(111,projection = '3d')



# Data for three-dimensional scattered points

zdata =train.popularity

xdata =train.budget

ydata = train.revenue

ax.scatter3D(xdata, ydata, zdata, c=zdata, s = 200)

ax.set_xlabel('Budget of the Movie',fontsize=17)

ax.set_ylabel('Revenue of the Movie',fontsize=17)

ax.set_zlabel('Popularity of the Movie',fontsize=17)

# Creating correlation matrix 



corr = train.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(11, 9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

        

train = text_to_dict(train)

test = text_to_dict(test)
# Counting NAs as 0

train['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0).value_counts()
train['collection_name'] = train['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)

train['has_collection'] = train['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)



test['collection_name'] = test['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)

test['has_collection'] = test['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)



train = train.drop(['belongs_to_collection'], axis=1)

test = test.drop(['belongs_to_collection'], axis=1)
# Most common collection 

train['collection_name'].value_counts()[1:10]
train['genres'].apply(lambda x: len(x) if x != {} else 0).value_counts()

list_of_genres = list(train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

most_common_genres=Counter([i for j in list_of_genres for i in j]).most_common()

fig = plt.figure(figsize=(10, 6))

data=dict(most_common_genres)

names = list(data.keys())

values = list(data.values())



plt.barh(sorted(range(len(data)),reverse=True),values,tick_label=names,color='teal')

plt.xlabel('Count')

plt.title('Movie Genre Count')

plt.show()

train['num_genres'] = train['genres'].apply(lambda x: len(x) if x != {} else 0)

train['all_genres'] = train['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(15)]

for g in top_genres:

    train['genre_' + g] = train['all_genres'].apply(lambda x: 1 if g in x else 0)

    

test['num_genres'] = test['genres'].apply(lambda x: len(x) if x != {} else 0)

test['all_genres'] = test['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_genres:

    test['genre_' + g] = test['all_genres'].apply(lambda x: 1 if g in x else 0)



train = train.drop(['genres'], axis=1)

test = test.drop(['genres'], axis=1)
# Movie title text analysis 



text = " ".join(review for review in train.title)

print ("There are {} words in the combination of all review.".format(len(text)))



stopwords = set(stopwords.words('english'))

wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)



fig = plt.figure(figsize=(11, 7))

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('movie title in train data')

plt.axis("off")

plt.show()
# Text analysis on top 4 movie genres title 



drama=train.loc[train['genre_Drama']==1,]

comedy=train.loc[train['genre_Comedy']==1,]

action=train.loc[train['genre_Action']==1,]

thriller=train.loc[train['genre_Thriller']==1,]







text_drama = " ".join(review for review in drama.title)

text_comedy = " ".join(review for review in comedy.title)

text_action = " ".join(review for review in action.title)

text_thriller = " ".join(review for review in thriller.title)





wordcloud1 = WordCloud(stopwords=stopwords, background_color="white",colormap="Reds").generate(text_drama)

wordcloud2 = WordCloud(stopwords=stopwords, background_color="white",colormap="Blues").generate(text_comedy)

wordcloud3 = WordCloud(stopwords=stopwords, background_color="white",colormap="Greens").generate(text_action)

wordcloud4 = WordCloud(stopwords=stopwords, background_color="white",colormap="Greys").generate(text_thriller)





fig = plt.figure(figsize=(20, 10))



plt.subplot(221)

plt.imshow(wordcloud1, interpolation='bilinear')

plt.title('drama movie title')

plt.axis("off")



plt.subplot(222)

plt.imshow(wordcloud2, interpolation='bilinear')

plt.title('comedy movie title')

plt.axis("off")

plt.show()



fig = plt.figure(figsize=(20, 10))



plt.subplot(223)

plt.imshow(wordcloud3, interpolation='bilinear')

plt.title('action movie title')

plt.axis("off")



plt.subplot(224)

plt.imshow(wordcloud4, interpolation='bilinear')

plt.title('thriller movie title')

plt.axis("off")

plt.show()

drama_revenue=drama.groupby(['release_year']).mean()['revenue']

comedy_revenue=comedy.groupby(['release_year']).mean()['revenue']

action_revenue=action_revenue=action.groupby(['release_year']).mean()['revenue']

thriller_revenue=thriller.groupby(['release_year']).mean()['revenue']



revenue_concat = pd.concat([drama_revenue,comedy_revenue,action_revenue,thriller_revenue], axis=1)

revenue_concat.columns=['drama','comedy','action','thriller']

revenue_concat.index=train.groupby(['release_year']).mean().index
# Mean revenue over years by top 4 genres 



data = [go.Scatter(x=revenue_concat.index, y=revenue_concat.drama, name='drama'), go.Scatter(x=revenue_concat.index, y=revenue_concat.comedy, name='comedy'),

       go.Scatter(x=revenue_concat.index, y=revenue_concat.action, name='action'),go.Scatter(x=revenue_concat.index, y=revenue_concat.thriller, name='thriller')]

layout = go.Layout(dict(title = 'Mean Revenue by Top 4 Movie Genres Over Years',

                  xaxis = dict(title = 'Year'),

                  yaxis = dict(title = 'Revenue'),

                  ),legend=dict(

                orientation="v"))

py.iplot(dict(data=data, layout=layout))
# Counting the frequency of production company 

list_of_companies = list(train['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)



most_common_companies=Counter([i for j in list_of_companies for i in j]).most_common(20)



fig = plt.figure(figsize=(10, 6))

data=dict(most_common_companies)

names = list(data.keys())

values = list(data.values())



plt.barh(sorted(range(len(data)),reverse=True),values,tick_label=names,color='brown')

plt.xlabel('Count')

plt.title('Top 20 Production Company Count')

plt.show()

train['num_companies'] = train['production_companies'].apply(lambda x: len(x) if x != {} else 0)

train['all_production_companies'] = train['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_companies = [m[0] for m in Counter([i for j in list_of_companies for i in j]).most_common(30)]

for g in top_companies:

    train['production_company_' + g] = train['all_production_companies'].apply(lambda x: 1 if g in x else 0)

    

test['num_companies'] = test['production_companies'].apply(lambda x: len(x) if x != {} else 0)

test['all_production_companies'] = test['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_companies:

    test['production_company_' + g] = test['all_production_companies'].apply(lambda x: 1 if g in x else 0)



train = train.drop(['production_companies', 'all_production_companies'], axis=1)

test = test.drop(['production_companies', 'all_production_companies'], axis=1)
# Getting the mean revenue of top 8 most common production companies 



Warner_Bros=train.loc[train['production_company_Warner Bros.']==1,]

Universal_Pictures=train.loc[train['production_company_Universal Pictures']==1,]

Twentieth_Century_Fox_Film=train.loc[train['production_company_Twentieth Century Fox Film Corporation']==1,]

Columbia_Pictures=train.loc[train['production_company_Columbia Pictures']==1,]

MGM=train.loc[train['production_company_Metro-Goldwyn-Mayer (MGM)']==1,]

New_Line_Cinema=train.loc[train['production_company_New Line Cinema']==1,]

Touchstone_Pictures=train.loc[train['production_company_Touchstone Pictures']==1,]

Walt_Disney=train.loc[train['production_company_Walt Disney Pictures']==1,]



Warner_Bros_revenue=Warner_Bros.groupby(['release_year']).mean()['revenue']

Universal_Pictures_revenue=Universal_Pictures.groupby(['release_year']).mean()['revenue']

Twentieth_Century_Fox_Film_revenue=Twentieth_Century_Fox_Film.groupby(['release_year']).mean()['revenue']

Columbia_Pictures_revenue=Columbia_Pictures.groupby(['release_year']).mean()['revenue']

MGM_revenue=MGM.groupby(['release_year']).mean()['revenue']

New_Line_Cinema_revenue=New_Line_Cinema.groupby(['release_year']).mean()['revenue']

Touchstone_Pictures_revenue=Touchstone_Pictures.groupby(['release_year']).mean()['revenue']

Walt_Disney_revenue=Walt_Disney.groupby(['release_year']).mean()['revenue']





prod_revenue_concat = pd.concat([Warner_Bros_revenue,Universal_Pictures_revenue,Twentieth_Century_Fox_Film_revenue,Columbia_Pictures_revenue,

                                MGM_revenue,New_Line_Cinema_revenue,Touchstone_Pictures_revenue,Walt_Disney_revenue], axis=1)

prod_revenue_concat.columns=['Warner_Bros','Universal_Pictures','Twentieth_Century_Fox_Film','Columbia_Pictures','MGM','New_Line_Cinema','Touchstone_Pictures','Walt_Disney']



fig = plt.figure(figsize=(13, 7))

prod_revenue_concat.agg("mean",axis='rows').sort_values(ascending=True).plot(kind='barh',x='Production Companies',y='Revenue',title='Mean Revenue (100 million dollars) of Top 8 Most Common Production Companies')

plt.xlabel('Revenue (100 million dollars)')
data = [go.Scatter(x=prod_revenue_concat.index, y=prod_revenue_concat.Warner_Bros, name='Warner_Bros'), go.Scatter(x=prod_revenue_concat.index, y=prod_revenue_concat.Universal_Pictures, name='Universal_Pictures'),

       go.Scatter(x=prod_revenue_concat.index, y=prod_revenue_concat.Twentieth_Century_Fox_Film, name='Twentieth_Century_Fox_Film'),go.Scatter(x=prod_revenue_concat.index, y=prod_revenue_concat.Columbia_Pictures, name='Columbia_Pictures'),

       go.Scatter(x=prod_revenue_concat.index, y=prod_revenue_concat.MGM, name='MGM'), go.Scatter(x=prod_revenue_concat.index, y=prod_revenue_concat.New_Line_Cinema, name='New_Line_Cinema'),

       go.Scatter(x=prod_revenue_concat.index, y=prod_revenue_concat.Touchstone_Pictures, name='Touchstone_Pictures'),go.Scatter(x=prod_revenue_concat.index, y=prod_revenue_concat.Walt_Disney, name='Walt_Disney')]





layout = go.Layout(dict(title = 'Mean Revenue of Top 8 Movie Production Companies over Years',

                  xaxis = dict(title = 'Year'),

                  yaxis = dict(title = 'Revenue'),

                  ),legend=dict(

                orientation="v"))

py.iplot(dict(data=data, layout=layout))
list_of_countries = list(train['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

most_common_countries=Counter([i for j in list_of_countries for i in j]).most_common(20)



fig = plt.figure(figsize=(10, 6))

data=dict(most_common_countries)

names = list(data.keys())

values = list(data.values())



plt.barh(sorted(range(len(data)),reverse=True),values,tick_label=names,color='purple')

plt.xlabel('Count')

plt.title('Country Count')

plt.show()

train['num_countries'] = train['production_countries'].apply(lambda x: len(x) if x != {} else 0)

train['all_countries'] = train['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_countries = [m[0] for m in Counter([i for j in list_of_countries for i in j]).most_common(25)]

for g in top_countries:

    train['production_country_' + g] = train['all_countries'].apply(lambda x: 1 if g in x else 0)

    

test['num_countries'] = test['production_countries'].apply(lambda x: len(x) if x != {} else 0)

test['all_countries'] = test['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_countries:

    test['production_country_' + g] = test['all_countries'].apply(lambda x: 1 if g in x else 0)



train = train.drop(['production_countries', 'all_countries'], axis=1)

test = test.drop(['production_countries', 'all_countries'], axis=1)
# English is the majority spoken language 

list_of_languages = list(train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)



most_common_languages=Counter([i for j in list_of_languages for i in j]).most_common(20)



fig = plt.figure(figsize=(10, 6))

data=dict(most_common_languages)

names = list(data.keys())

values = list(data.values())



plt.barh(sorted(range(len(data)),reverse=True),values,tick_label=names,color='gray')

plt.xlabel('Count')

plt.title('Language Count')

plt.show()
train['num_languages'] = train['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

train['all_languages'] = train['spoken_languages'].apply(lambda x: ' '.join(sorted([i['iso_639_1'] for i in x])) if x != {} else '')

top_languages = [m[0] for m in Counter([i for j in list_of_languages for i in j]).most_common(30)]

for g in top_languages:

    train['language_' + g] = train['all_languages'].apply(lambda x: 1 if g in x else 0)

    

test['num_languages'] = test['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

test['all_languages'] = test['spoken_languages'].apply(lambda x: ' '.join(sorted([i['iso_639_1'] for i in x])) if x != {} else '')

for g in top_languages:

    test['language_' + g] = test['all_languages'].apply(lambda x: 1 if g in x else 0)



train = train.drop(['spoken_languages', 'all_languages'], axis=1)

test = test.drop(['spoken_languages', 'all_languages'], axis=1)
list_of_keywords = list(train['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)



most_common_keywords=Counter([i for j in list_of_keywords for i in j]).most_common(20)



fig = plt.figure(figsize=(10, 6))

data=dict(most_common_keywords)

names = list(data.keys())

values = list(data.values())



plt.barh(sorted(range(len(data)),reverse=True),values,tick_label=names,color='purple')

plt.xlabel('Count')

plt.title('Top 20 Most Common Keyword Count')

plt.show()

# Text analysis on keywords by top 4 genres





text_drama = " ".join(review for review in drama['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else ''))

text_comedy = " ".join(review for review in comedy['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else ''))

text_action = " ".join(review for review in action['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else ''))

text_thriller = " ".join(review for review in thriller['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else ''))





wordcloud1 = WordCloud(stopwords=stopwords, background_color="white",colormap="Reds").generate(text_drama)

wordcloud2 = WordCloud(stopwords=stopwords, background_color="white",colormap="Blues").generate(text_comedy)

wordcloud3 = WordCloud(stopwords=stopwords, background_color="white",colormap="Greens").generate(text_action)

wordcloud4 = WordCloud(stopwords=stopwords, background_color="white",colormap="Greys").generate(text_thriller)





fig = plt.figure(figsize=(25, 20))



plt.subplot(221)

plt.imshow(wordcloud1, interpolation='bilinear')

plt.title('Drama Keywords')

plt.axis("off")



plt.subplot(222)

plt.imshow(wordcloud2, interpolation='bilinear')

plt.title('Comedy Keywords')

plt.axis("off")

plt.show()



fig = plt.figure(figsize=(25, 20))



plt.subplot(223)

plt.imshow(wordcloud3, interpolation='bilinear')

plt.title('Action Keywords')

plt.axis("off")



plt.subplot(224)

plt.imshow(wordcloud4, interpolation='bilinear')

plt.title('Thriller Keywords')

plt.axis("off")

plt.show()
train['num_Keywords'] = train['Keywords'].apply(lambda x: len(x) if x != {} else 0)

train['all_Keywords'] = train['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(30)]

for g in top_keywords:

    train['keyword_' + g] = train['all_Keywords'].apply(lambda x: 1 if g in x else 0)

    

test['num_Keywords'] = test['Keywords'].apply(lambda x: len(x) if x != {} else 0)

test['all_Keywords'] = test['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_keywords:

    test['keyword_' + g] = test['all_Keywords'].apply(lambda x: 1 if g in x else 0)



train = train.drop(['Keywords', 'all_Keywords'], axis=1)

test = test.drop(['Keywords', 'all_Keywords'], axis=1)
list_of_cast_names = list(train['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

train['num_cast'] = train['cast'].apply(lambda x: len(x) if x != {} else 0)

train['all_cast'] = train['cast'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(30)]

for g in top_cast_names:

    train['cast_name_' + g] = train['all_cast'].apply(lambda x: 1 if g in x else 0)



    

test['num_cast'] = test['cast'].apply(lambda x: len(x) if x != {} else 0)

test['all_cast'] = test['cast'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_cast_names:

    test['cast_name_' + g] = test['all_cast'].apply(lambda x: 1 if g in x else 0)
#Mean revenue comparison of 10 most common actor/actress 



cast_name_Samuel_L_Jackson=train.loc[train['cast_name_Samuel L. Jackson']==1,]

cast_name_Robert_De_Niro=train.loc[train['cast_name_Robert De Niro']==1,]

cast_name_Morgan_Freeman=train.loc[train['cast_name_Morgan Freeman']==1,]

cast_name_J_K_Simmons=train.loc[train['cast_name_J.K. Simmons']==1,]

cast_name_Bruce_Willis=train.loc[train['cast_name_Bruce Willis']==1,]

cast_name_Liam_Neeson=train.loc[train['cast_name_Liam Neeson']==1,]

cast_name_Susan_Sarandon=train.loc[train['cast_name_Susan Sarandon']==1,]

cast_name_Bruce_McGill=train.loc[train['cast_name_Bruce McGill']==1,]

cast_name_John_Turturro=train.loc[train['cast_name_John Turturro']==1,]

cast_name_Forest_Whitaker=train.loc[train['cast_name_Forest Whitaker']==1,]





cast_name_Samuel_L_Jackson_revenue=cast_name_Samuel_L_Jackson.mean()['revenue']

cast_name_Robert_De_Niro_revenue=cast_name_Robert_De_Niro.mean()['revenue']

cast_name_Morgan_Freeman_revenue=cast_name_Morgan_Freeman.mean()['revenue']

cast_name_J_K_Simmons_revenue=cast_name_J_K_Simmons.mean()['revenue']

cast_name_Bruce_Willis_revenue=cast_name_Bruce_Willis.mean()['revenue']

cast_name_Liam_Neeson_revenue=cast_name_Liam_Neeson.mean()['revenue']

cast_name_Susan_Sarandon_revenue=cast_name_Susan_Sarandon.mean()['revenue']

cast_name_Bruce_McGill_revenue=cast_name_Bruce_McGill.mean()['revenue']

cast_name_John_Turturro_revenue=cast_name_John_Turturro.mean()['revenue']

cast_name_Forest_Whitaker_revenue=cast_name_Forest_Whitaker.mean()['revenue']





cast_revenue_concat = pd.Series([cast_name_Samuel_L_Jackson_revenue,cast_name_Robert_De_Niro_revenue,cast_name_Morgan_Freeman_revenue,cast_name_J_K_Simmons_revenue,

                                cast_name_Bruce_Willis_revenue,cast_name_Liam_Neeson_revenue,cast_name_Susan_Sarandon_revenue,cast_name_Bruce_McGill_revenue,

                                cast_name_John_Turturro_revenue,cast_name_Forest_Whitaker_revenue])

cast_revenue_concat.index=['Samuel L. Jackson','Robert De Niro','Morgan Freeman','J.K. Simmons','Bruce Willis','Liam Neeson','Susan Sarandon','Bruce McGill',

                            'John Turturro','Forest Whitaker']





fig = plt.figure(figsize=(13, 7))

cast_revenue_concat.sort_values(ascending=True).plot(kind='barh',title='Mean Revenue (100 million dollars) by Top 10 Most Common Cast')

plt.xlabel('Revenue (100 million dollars)')
# Consider other factors like gender and characters 



list_of_cast_genders = list(train['cast'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)

list_of_cast_characters = list(train['cast'].apply(lambda x: [i['character'] for i in x] if x != {} else []).values)





train['genders_0'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

train['genders_1'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

train['genders_2'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

top_cast_characters = [m[0] for m in Counter([i for j in list_of_cast_characters for i in j]).most_common(15)]

for g in top_cast_characters:

    train['cast_character_' + g] = train['cast'].apply(lambda x: 1 if g in str(x) else 0)

    



test['genders_0'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

test['genders_1'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

test['genders_2'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

for g in top_cast_characters:

    test['cast_character_' + g] = test['cast'].apply(lambda x: 1 if g in str(x) else 0)



train = train.drop(['cast'], axis=1)

test = test.drop(['cast'], axis=1)
list_of_crew_names = list(train['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

train['num_crew'] = train['crew'].apply(lambda x: len(x) if x != {} else 0)

train['all_crew'] = train['crew'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_crew_names = [m[0] for m in Counter([i for j in list_of_crew_names for i in j]).most_common(30)]

for g in top_crew_names:

    train['crew_name_' + g] = train['all_crew'].apply(lambda x: 1 if g in x else 0)



test['num_crew'] = test['crew'].apply(lambda x: len(x) if x != {} else 0)

test['all_crew'] = test['crew'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_crew_names:

    test['crew_name_' + g] = test['all_crew'].apply(lambda x: 1 if g in x else 0)
crew_name_Avy_Kaufman=train.loc[train['crew_name_Avy Kaufman']==1,]

crew_name_Robert_Rodriguez=train.loc[train['crew_name_Robert Rodriguez']==1,]

crew_name_Deborah_Aquila=train.loc[train['crew_name_Deborah Aquila']==1,]

crew_name_James_Newton_Howard=train.loc[train['crew_name_James Newton Howard']==1,]

crew_name_Mary_Vernieu=train.loc[train['crew_name_Mary Vernieu']==1,]

crew_name_Steven_Spielberg=train.loc[train['crew_name_Steven Spielberg']==1,]

crew_name_Luc_Besson=train.loc[train['crew_name_Luc Besson']==1,]

crew_name_Jerry_Goldsmith=train.loc[train['crew_name_Jerry Goldsmith']==1,]

crew_name_Francine_Maisler=train.loc[train['crew_name_Francine Maisler']==1,]

crew_name_Tricia_Wood=train.loc[train['crew_name_Tricia Wood']==1,]





crew_name_Avy_Kaufman_revenue=crew_name_Avy_Kaufman.mean()['revenue']

crew_name_Robert_Rodriguez_revenue=crew_name_Robert_Rodriguez.mean()['revenue']

crew_name_Deborah_Aquila_revenue=crew_name_Deborah_Aquila.mean()['revenue']

crew_name_James_Newton_Howard_revenue=crew_name_James_Newton_Howard.mean()['revenue']

crew_name_Mary_Vernieu_revenue=crew_name_Mary_Vernieu.mean()['revenue']

crew_name_Steven_Spielberg_revenue=crew_name_Steven_Spielberg.mean()['revenue']

crew_name_Luc_Besson_revenue=crew_name_Luc_Besson.mean()['revenue']

crew_name_Jerry_Goldsmith_revenue=crew_name_Jerry_Goldsmith.mean()['revenue']

crew_name_Francine_Maisler_revenue=crew_name_Francine_Maisler.mean()['revenue']

crew_name_Tricia_Wood_revenue=crew_name_Tricia_Wood.mean()['revenue']





crew_revenue_concat = pd.Series([crew_name_Avy_Kaufman_revenue,crew_name_Robert_Rodriguez_revenue,crew_name_Deborah_Aquila_revenue,crew_name_James_Newton_Howard_revenue,

                                crew_name_Mary_Vernieu_revenue,crew_name_Steven_Spielberg_revenue,crew_name_Luc_Besson_revenue,crew_name_Jerry_Goldsmith_revenue,

                                crew_name_Francine_Maisler_revenue,crew_name_Tricia_Wood_revenue])

crew_revenue_concat.index=['Avy Kaufman','Robert Rodriguez','Deborah Aquila','James Newton Howard','Mary Vernieu','Steven Spielberg','Luc Besson','Jerry Goldsmith',

                            'Francine Maisler','Tricia Wood']





fig = plt.figure(figsize=(13, 7))

crew_revenue_concat.sort_values(ascending=True).plot(kind='barh',title='Mean Revenue (100 million dollars) by Top 10 Most Common Crew')

plt.xlabel('Revenue (100 million dollars)')
# Consider other factors like crew jobs, gender, and department 



list_of_crew_jobs = list(train['crew'].apply(lambda x: [i['job'] for i in x] if x != {} else []).values)

list_of_crew_genders = list(train['crew'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)

list_of_crew_departments = list(train['crew'].apply(lambda x: [i['department'] for i in x] if x != {} else []).values)





train['genders_0'] = train['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

train['genders_1'] = train['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

train['genders_2'] = train['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

top_cast_characters = [m[0] for m in Counter([i for j in list_of_cast_characters for i in j]).most_common(15)]

for g in top_cast_characters:

    train['crew_character_' + g] = train['crew'].apply(lambda x: 1 if g in str(x) else 0)

top_crew_jobs = [m[0] for m in Counter([i for j in list_of_crew_jobs for i in j]).most_common(15)]

for j in top_crew_jobs:

    train['jobs_' + j] = train['crew'].apply(lambda x: sum([1 for i in x if i['job'] == j]))

top_crew_departments = [m[0] for m in Counter([i for j in list_of_crew_departments for i in j]).most_common(15)]

for j in top_crew_departments:

    train['departments_' + j] = train['crew'].apply(lambda x: sum([1 for i in x if i['department'] == j])) 

    



    

test['genders_0'] = test['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

test['genders_1'] = test['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

test['genders_2'] = test['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

for g in top_cast_characters:

    test['crew_character_' + g] = test['crew'].apply(lambda x: 1 if g in str(x) else 0)

for j in top_crew_jobs:

    test['jobs_' + j] = test['crew'].apply(lambda x: sum([1 for i in x if i['job'] == j]))

for j in top_crew_departments:

    test['departments_' + j] = test['crew'].apply(lambda x: sum([1 for i in x if i['department'] == j])) 



train = train.drop(['crew'], axis=1)

test = test.drop(['crew'], axis=1)
# Plot the distribution of the revenue



fig = plt.figure(figsize=(30, 25))



plt.subplot(221)

train['revenue'].plot(kind='hist',bins=100)

plt.title('Distribution of Revenue')

plt.xlabel('Revenue')



plt.subplot(222)

np.log1p(train['revenue']).plot(kind='hist',bins=100)

plt.title('Train Log Revenue Distribution')

plt.xlabel('Log Revenue')





print('Skew of revenue attribute: %0.1f' % skew(train['revenue']))
# Adjusting other skewed variables such as popularity and budget



print('Skew of train budget attribute: %0.1f' % skew(train['budget']))

print('Skew of test budget attribute: %0.1f' % skew(test['budget']))

print('Skew of train popularity attribute: %0.1f' % skew(train['popularity']))

print('Skew of test popularity attribute: %0.1f' % skew(test['popularity']))
# Before log transformation and after log transformation for train budget and train popularity 

fig = plt.figure(figsize=(30, 25))



plt.subplot(221)

train['budget'].plot(kind='hist',bins=100)

plt.title('Train Budget Distribution')

plt.xlabel('Budget')



plt.subplot(222)

np.log1p(train['budget']).plot(kind='hist',bins=100)

plt.title('Train Log Budget Distribution')

plt.xlabel('Log Budget')



plt.show()



fig = plt.figure(figsize=(30, 25))



plt.subplot(223)

test['budget'].plot(kind='hist',bins=100)

plt.title('Train Popularity Distribution')

plt.xlabel('Popularity')



plt.subplot(224)

np.log1p(test['budget']).plot(kind='hist',bins=100)

plt.title('Train Log Popularity Distribution')

plt.xlabel('Log Popularity')

plt.show()

# Revising budget variable 



power_six = train.id[train.budget > 1000][train.revenue < 100]



for k in power_six :

    train.loc[train['id'] == k,'revenue'] =  train.loc[train['id'] == k,'revenue'] * 1000000
# Putting log variables for skewed data 

train['log_budget']=np.log1p(train['budget'])

test['log_budget']=np.log1p(test['budget'])

train['log_popularity']=np.log1p(train['popularity'])

test['log_popularity']=np.log1p(test['popularity'])
def prepare(df):

    df['_budget_runtime_ratio'] = df['budget']/df['runtime'] 

    df['_budget_popularity_ratio'] = df['budget']/df['popularity']

    df['_budget_year_ratio'] = df['budget']/(df['release_year']*df['release_year'])

    df['_releaseYear_popularity_ratio'] = df['release_year']/df['popularity']

    df['_releaseYear_popularity_ratio2'] = df['popularity']/df['release_year']

    df['_year_to_log_budget'] = df['release_year'] / df['log_budget']

    df['_year_to_log_popularity'] = df['release_year'] / df['log_popularity']



    df['has_homepage'] = 0

    df.loc[pd.isnull(df['homepage']) ,"has_homepage"] = 1

    

    df['isTaglineNA'] = 0

    df.loc[df['tagline'] == 0 ,"isTaglineNA"] = 1 

    

    df['isTitleDifferent'] = 1

    df.loc[ df['original_title'] == df['title'] ,"isTitleDifferent"] = 0 



    df['isMovieReleased'] = 1

    df.loc[ df['status'] != "Released" ,"isMovieReleased"] = 0 



    df['original_title_letter_count'] = df['original_title'].str.len() 

    df['original_title_word_count'] = df['original_title'].str.split().str.len() 

    df['title_word_count'] = df['title'].str.split().str.len()

    df['overview_word_count'] = df['overview'].str.split().str.len()

    df['tagline_word_count'] = df['tagline'].str.split().str.len()

    df['meanruntimeByYear'] = df.groupby("release_year")["runtime"].aggregate('mean')

    df['meanPopularityByYear'] = df.groupby("release_year")["popularity"].aggregate('mean')

    df['meanBudgetByYear'] = df.groupby("release_year")["budget"].aggregate('mean')



    return df



train_new=prepare(train)

test_new=prepare(test)

train_new.to_csv("train_new.csv", index=False)

test_new.to_csv("test_new.csv", index=False)
drop_columns=['homepage','imdb_id','poster_path','status','title', 'release_date','tagline', 'overview', 'original_title','all_genres','all_cast',

             'original_language','collection_name','all_crew']

train_new=train_new.drop(drop_columns,axis=1)

test_new=test_new.drop(drop_columns,axis=1)
print( "updated train dataset:", train_new.shape,"\n","updated test dataset: ",test_new.shape)



# Just double checking the difference of variables between train and test 

print(train_new.columns.difference(test_new.columns)) # good to go! 
# Formating for modeling



X = train_new.drop(['id', 'revenue'], axis=1)

y = np.log1p(train_new['revenue'])

X_test = test_new.drop(['id'], axis=1)



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
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



lgb_model = lgb.LGBMRegressor(**params, n_estimators = 10000, nthread = 4, n_jobs = -1)

lgb_model.fit(X_train, y_train, 

        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=200)



eli5.show_weights(lgb_model, feature_filter=lambda x: x != '<BIAS>')
"""

Taking too much time! (I also put too many combinations)



# Print the best parameters found

gridParams = {

    "max_depth": [5,6,7,8],

    "min_data_in_leaf": [15,20,25,30],

    'learning_rate': [0.01,0.005],

    'num_leaves': [15,20,25,30,35,40],

    'boosting_type' : ['gbdt'],

    'objective' : ['regression'],

    'random_state' : [501], # Updated from 'seed'

    'reg_alpha' : [1,1.2],

    'reg_lambda' : [1,1.2,1.4]

    }



grid_search = GridSearchCV(lgb_model, n_jobs=-1, param_grid=gridParams, cv = 3, verbose=5)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

print(grid_search.best_score_)





Random_Search_Params ={

    "max_depth": [4,5,6],

    "min_data_in_leaf": [15,20,25],

    'learning_rate': [0.01,0.005],

    'num_leaves': [25,30,35,40],

    'boosting_type' : ['gbdt'],

    'objective' : ['regression'],

    'random_state' : [501] # Updated from 'seed'

    }



n_HP_points_to_test = 50



random_search = RandomizedSearchCV(

    estimator=lgb_model, param_distributions= Random_Search_Params, 

    n_iter=n_HP_points_to_test,

    cv=3,

    refit=True,

    random_state=314,

    verbose=True)



random_search.fit(X_train, y_train)

print('Best score reached: {} with params: {} '.format(random_search.best_score_, random_search.best_params_))



# Using parameters already set above, replace in the best from the random search



params['learning_rate'] = random_search.best_params_['learning_rate']

params['max_depth'] = random_search.best_params_['max_depth']

params['num_leaves'] = random_search.best_params_['num_leaves']

params['reg_alpha'] = random_search.best_params_['reg_alpha']

params['reg_lambda'] = random_search.best_params_['reg_lambda']





"""
# Obtain from Random Search 



opt_parameters = {'random_state': 501, 'objective': 'regression', 'num_leaves': 40, 'min_data_in_leaf': 15, 'max_depth': 4, 'learning_rate': 0.01, 'boosting_type': 'gbdt'} 



params['learning_rate'] = opt_parameters['learning_rate']

params['max_depth'] = opt_parameters['max_depth']

params['num_leaves'] = opt_parameters['num_leaves']

params['min_data_in_leaf'] = opt_parameters['min_data_in_leaf']
n_fold = 5

random_seed=2222

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)



def train_model(X, X_test, y, params=None, folds=folds, model_type='lgb', plot_feature_importance=True, model=None):



    oof = np.zeros(X.shape[0])

    prediction = np.zeros(X_test.shape[0])

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print('Fold', fold_n, 'started at', time.ctime())

        if model_type == 'sklearn':

            X_train, X_valid = X[train_index], X[valid_index]

        else:

            X_train, X_valid = X.values[train_index], X.values[valid_index]

        y_train, y_valid = y[train_index], y[valid_index]

        

        if model_type == 'lgb':

            model = lgb.LGBMRegressor(**params, n_estimators = 10000, nthread = 4, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',

                    verbose=500, early_stopping_rounds=200)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=10000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test.values), ntree_limit=model.best_ntree_limit)



        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = mean_squared_error(y_valid, y_pred_valid)

            

            y_pred = model.predict(X_test)

            

        if model_type == 'cat':

            model = CatBoostRegressor(iterations=10000,  eval_metric='RMSE', **params)

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        oof[valid_index] = y_pred_valid.reshape(-1,)

        scores.append(mean_squared_error(y_valid, y_pred_valid) ** 0.5)

        

        prediction += y_pred    

        

        if model_type == 'lgb':

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = X.columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= n_fold

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    if model_type == 'lgb':

        feature_importance["importance"] /= n_fold

        if plot_feature_importance:

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

        

            return oof, prediction, feature_importance

        return oof, prediction

    

    else:

        return oof, prediction
start = time.time()

oof_lgb, prediction_lgb, _ = train_model(X, X_test, y, params=params, model_type='lgb')

end = time.time()



print("time elapsed:",end - start, "second")
xgb_params = {'eta': 0.01,

              'objective': 'reg:linear',

              'max_depth': 6,

              'min_child_weight': 3,

              'subsample': 0.8,

              'colsample_bytree': 0.8,

              'eval_metric': 'rmse',

              'seed': 11,

              'silent': True}



start = time.time()

oof_xgb, prediction_xgb = train_model(X, X_test, y, params=xgb_params, model_type='xgb')

end = time.time()

print("time elapsed:",end - start, "second")
sam_sub['revenue'] = np.expm1(prediction_lgb)

sam_sub.to_csv("lgb.csv", index=False)

sam_sub['revenue'] = np.expm1(prediction_xgb)

sam_sub.to_csv("xgb.csv", index=False)

sam_sub['revenue'] = np.expm1((prediction_lgb + prediction_xgb) / 2)

sam_sub.to_csv("blend_lgb_xgb.csv", index=False)