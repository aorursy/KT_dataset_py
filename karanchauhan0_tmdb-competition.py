import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

from wordcloud import WordCloud, STOPWORDS

from collections import OrderedDict

import json

from collections import defaultdict



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
data.head()
data.info()
#we have a lot of null values for homepage

#Checking effect on homepage on revenue

data['has_homepage'] = 0

data.loc[data['homepage'].isnull() == False, 'has_homepage'] = 1

#Checking how homepage reflects on revenue



print('Number of null homepages in data = ',data[data.has_homepage==0]['id'].count(),'/',(data.id).count())

plt.figure(figsize=(6,12))

plt.scatter(data.has_homepage, data.revenue, alpha=0.2,

            s=50, cmap='viridis')

plt.xlabel('Does homepage exist?')

plt.ylabel('Total Revenue');

plt.xticks(np.arange(2), ('No','Yes'))

data=data.drop(['has_homepage'],axis =1)

#Checking effect on tagline on revenue

data['has_tg'] = 0

data.loc[data['tagline'].isnull() == False, 'has_tg'] = 1



print('Number of null taglines in test = ',data[data.has_tg==0]['id'].count(),'/',(data.id).count())



plt.figure(figsize=(6,12))

plt.scatter(data.has_tg, data.revenue, alpha=0.2,

            s=50, cmap='viridis')

plt.xlabel('Does tagline exist?')

plt.ylabel('Total Revenue');



plt.xticks(np.arange(2), ('No','Yes'))

data=data.drop(['has_tg'],axis =1)
# Repeating same process for collections

data['has_collection'] = 0

data.loc[data['belongs_to_collection'].isnull() == False, 'has_collection'] = 1

print('Number of null collections in test = ',data[data.has_collection==0]['id'].count(),'/',(data.id).count())



plt.figure(figsize=(6,12))

plt.scatter(data.has_collection, data.revenue, alpha=0.2,

            s=50, cmap='viridis')

plt.xlabel('Does collection exist?')

plt.ylabel('Total Revenue');



plt.xticks(np.arange(2), ('No','Yes'))

data=data.drop(['has_collection'],axis =1)
#Correlation between revenue, budget, popularity and runtime

col = ['revenue','budget','popularity','runtime']

plt.subplots(figsize=(10, 8))

corr = data[col].corr()

sns.heatmap(corr, xticklabels=col,yticklabels=col, linewidths=.5, cmap="Reds")
sns.regplot(x="budget", y="revenue", data=data)
#Relation of revenue on day of week when movie was released

def get_day_of_week(row):

    return pd.to_datetime(row.release_date).dayofweek

data['dow']= data.apply (lambda row: get_day_of_week(row), axis=1)

revenue_by_dow = data.groupby('dow')['revenue'].sum()

# number_of_movies = data.groupby('dow')['revenue'].count()

days = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')

y_pos = np.arange(len(revenue_by_dow))

plt.figure(figsize=(15,10))

plt.bar(y_pos, revenue_by_dow, align='center', alpha=0.5)

plt.xticks(y_pos, days)

plt.xlabel('Day of Week')

plt.ylabel('Revenue')

plt.title('Total Revenue by Day of Week')

plt.show()
plt.figure(figsize=(15,10))

number_of_movies = data.groupby('dow')['revenue'].count()

normalized_revenue = revenue_by_dow/number_of_movies

plt.bar(y_pos, normalized_revenue, align='center', alpha=0.5)

plt.xticks(y_pos, days)

plt.xlabel('Day of Week')

plt.ylabel('Revenue')

plt.title('Normalized Revenue by Day of Week')

plt.show()
#Relation of revenue on day of week when movie was released

def get_year(row):

    year = pd.to_datetime(row.release_date).year

    if(year>2019):

        return year-100

    return year

data['year']= data.apply (lambda row: get_year(row), axis=1)

revenue_by_year = data.groupby('year')['revenue'].sum().reset_index()

years = list(revenue_by_year.year)

y_pos = np.arange(revenue_by_year.revenue.count())

plt.figure(figsize=(15,10))

plt.bar(y_pos, revenue_by_year.revenue, align='center', alpha=0.5)

plt.xticks(y_pos, np.array(years)%100, rotation='vertical')

plt.xlabel('Year')

plt.ylabel('Revenue')

plt.title('Total Revenue by Year')

plt.show()

# revenue_by_year
mydict = defaultdict(int)

def populate_revenue_by_genre(row):

    if not pd.isnull(row.genres):

        my_genres = json.loads(row.genres.replace("\'", "\""))

        for genre in my_genres:

            mydict[genre["name"]]+=row.revenue

data.apply (lambda row: populate_revenue_by_genre(row), axis=1)

plt.figure(figsize=(25,10))



genres = list(mydict.keys())

revenue_by_genre = list(mydict.values())

y_pos = np.arange(len(genres))

plt.bar(y_pos, revenue_by_genre, align='center', alpha=0.5)

plt.xticks(y_pos, genres)

plt.xlabel('Genres')

plt.ylabel('Revenue')

plt.title('Total Revenue by Genre')

plt.show()

revenue_by_lang = data.groupby('original_language')['revenue'].sum().reset_index()

number_of_languages = revenue_by_lang.revenue.count()

y_pos = np.arange(number_of_languages)

plt.figure(figsize=(15,10))

plt.bar(y_pos, revenue_by_lang['revenue'], align='center', alpha=0.5)

plt.xticks(y_pos, revenue_by_lang['original_language'])

plt.xlabel('Languages')

plt.ylabel('Revenue')

plt.title('Total Revenue by Language')

plt.show()
revenue_by_lang = revenue_by_lang[revenue_by_lang.original_language!='en']

number_of_languages = revenue_by_lang.revenue.count()

y_pos = np.arange(number_of_languages)

plt.figure(figsize=(15,10))

plt.bar(y_pos, revenue_by_lang['revenue'], align='center', alpha=0.5)

plt.xticks(y_pos, revenue_by_lang['original_language'])

plt.xlabel('Languages')

plt.ylabel('Revenue')

plt.title('Total Revenue by Language')

plt.show()
mydict = defaultdict(int)

def populate_revenue_by_company(row):

    if not pd.isnull(row.production_companies):

        try:

            companies = json.loads(row.production_companies.replace("\'", "\""))

        except ValueError:

            companies = []

        for company in companies:

            mydict[company["name"]]+=row.revenue

data.apply (lambda row: populate_revenue_by_company(row), axis=1)



plt.figure(figsize=(20,12))

# rev_by_prod = pd.DataFrame.from_dict(mydict)

ax = sns.barplot(list(mydict.keys()), list(mydict.values()))

plt.title("Movie revenue by production company",fontsize=20)

# loc, labels = plt.xticks()

plt.xticks(fontsize=12,rotation=90)

plt.show()

inflation_percents = np.array([0.94, 1.79, 0, 2.34, 1.14, -1.69, -1.72, 0.6, -6.4, -9.3, -10.3, 0.8, 1.5, 3.0, 1.4, 2.9, -2.8, 0.0, 0.7, 9.9, 9.0, 3.0, 2.3, 2.2, 18.1, 8.8, 3.0, -2.1, 5.9, 6.0, 0.8, 0.7, -0.7, 0.4, 3.0, 2.9, 1.8, 1.7, 1.4, 0.7, 1.3, 1.6, 1.0, 1.9, 3.5, 3.0, 4.7, 6.2, 5.6, 3.3, 3.4, 8.7, 12.3, 6.9, 4.9, 6.7, 9.0, 13.3, 12.5, 8.9, 3.8, 3.8, 3.9, 3.8, 1.1, 4.4, 4.4, 4.6, 6.1, 3.1, 2.9, 2.7, 2.7, 2.5, 3.3, 1.7, 1.6, 2.7, 3.4, 1.6, 2.4, 1.9, 3.3, 3.4, 2.5, 4.1, 0.1, 2.7, 1.5, 3.0, 1.7, 1.5, 0.8, 0.7, 2.1, 2.1, 1.9, 1.8])

inflation_year = range(1921, 2019)

inflation = dict(zip(inflation_year, inflation_percents))

def get_inflation(year):

    return inflation.get(year)

mydict = defaultdict(int)

freqdict = defaultdict(int)

def populate_revenue_by_country(row):

    if not pd.isnull(row.production_countries):

        countries = json.loads(row.production_countries.replace("\'", "\""))

        for country in countries:

            mydict[country["name"]]+=row.revenue

            freqdict[country["name"]]+=1

data.apply (lambda row: populate_revenue_by_country(row), axis=1)



# total_rev = list(mydict.values())

# freq_country = list(freqdict.values())

# normalized_rev = [x/y for x, y in zip(total_rev, freq_country)]





plt.figure(figsize=(20,12))

# rev_by_prod = pd.DataFrame.from_dict(mydict)

ax = sns.barplot(list(mydict.keys()), list(mydict.values()))

plt.title("Movie revenue by production country",fontsize=20)

# loc, labels = plt.xticks()

plt.xticks(fontsize=12,rotation=90)

plt.show()

mydict.pop('United States of America', None)

mydict.pop('United Kingdom', None)

plt.figure(figsize=(20,12))

# rev_by_prod = pd.DataFrame.from_dict(mydict)

ax = sns.barplot(list(mydict.keys()), list(mydict.values()))

plt.title("Movie revenue by production country - {US, UK}",fontsize=20)

# loc, labels = plt.xticks()

plt.xticks(fontsize=12,rotation=90)

plt.show()
