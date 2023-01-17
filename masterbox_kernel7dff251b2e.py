import numpy as np

import pandas as pd

import re

from datetime import datetime
answer_ls = []
df = pd.read_csv('../input/data.csv')
df[df['budget'] == df['budget'].max()][['imdb_id','original_title']]
answer_ls.append(4)
df[df['runtime'] == df['runtime'].max()][['imdb_id','original_title']]
answer_ls.append(2)
df[df['runtime'] == df['runtime'].min()][['imdb_id','original_title']]
answer_ls.append(3)
round(df['runtime'].mean())
answer_ls.append(2)
round(df['runtime'].median())
answer_ls.append(1)
df[df['revenue'] == df['revenue'].max()][['imdb_id','original_title']]
answer_ls.append(5)
df['profit'] = df['revenue'] - df['budget']
df[df['profit'] == df['profit'].min()][['imdb_id','original_title']]
answer_ls.append(2)
df[df['profit'] > 0].shape[0]
answer_ls.append(1)
df[df['profit'] == df[df['release_year'] == 2008]['profit'].max()][['imdb_id','original_title']]
answer_ls.append(4)
df[df['profit'] == df[(df['release_year'] > 2011) & (df['release_year'] < 2015)]['profit'].min()][['imdb_id','original_title']]
answer_ls.append(5)
import collections

ganres = collections.Counter()



for i in df['genres']:

    for j in i.split('|'):

        ganres[j] += 1



ganres.most_common()[0]
answer_ls.append(3)
ganres_prof = collections.Counter()



for i in df[df['profit'] > 0]['genres']:

    for j in i.split('|'):

        ganres_prof[j] += 1



ganres_prof.most_common()[0]
answer_ls.append(1)
answer_ls.append(0)
answer_ls.append(0)
answer_ls.append(0)
act_prof = pd.DataFrame(columns=('acter','profit', 'year', 'budget', 'imdb_id'))

for i, j in df.iterrows():

    for acter in j['cast'].split('|'):

        act_prof = act_prof.append({'acter': acter, 'profit': j['profit'], 'year': j['release_year'], 'budget': j['budget'], 'imdb_id': j['imdb_id']}, ignore_index=True)
act_prof.groupby(['acter'])['profit'].sum().sort_values(ascending=False).reset_index().head(1)
answer_ls.append(1)
act_prof[act_prof['year'] == 2012].groupby(['acter'])['profit'].sum().sort_values(ascending=True).reset_index().head(1)
answer_ls.append(3)
act_prof[act_prof['budget'] > df['budget'].mean()]['acter'].value_counts().reset_index().head(1)
answer_ls.append(3)
nc_genres = collections.Counter()

for i in act_prof[act_prof['acter'] == 'Nicolas Cage']['imdb_id']:

    for j in df[df['imdb_id'] == i]['genres']:

        for k in j.split('|'):

            nc_genres[k] += 1

nc_genres.most_common()[0]
answer_ls.append(2)
companies = collections.Counter()

for i in df['production_companies']:

    for j in i.split('|'):

        companies[j] += 1

companies.most_common()[0]
answer_ls.append(1)
companies_2015 = collections.Counter()

for i in df[df['release_year'] == 2015]['production_companies']:

    for j in i.split('|'):

        companies_2015[j] += 1

companies_2015.most_common()[0]
answer_ls.append(4)
companies_comedy = collections.Counter()

for i, j in df.iterrows():

    for genre in j['genres'].split('|'):

        if genre == 'Comedy':

            for company in j['production_companies'].split('|'):

                companies_comedy[company] += j['profit']

companies_comedy.most_common()[0]
answer_ls.append(2)
companies_2014 = collections.Counter()

for i in df[ df['release_year'] == 2014 ].index:

    for company in df.iloc[i]['production_companies'].split('|'):

        companies_2014[company] += df.iloc[i]['profit']

companies_2014.most_common(5)
companies_2014_v2 = collections.Counter()

for i in df[ df['release_year'] == 2014 ].index:

    for company in df.iloc[i]['production_companies'].split('|'):

        companies_2014_v2[company] += df.iloc[i]['revenue']

companies_2014_v2.most_common(5)
answer_ls.append(2)
pc_films = collections.Counter()

for i in df.index:

    for company in df.iloc[i]['production_companies'].split('|'):

        if company == 'Paramount Pictures':

            pc_films[df.iloc[i]['original_title']] += df.iloc[i]['profit']

pc_films.most_common()[:-2:-1]
answer_ls.append(1)
df.groupby(['release_year'])['profit'].sum().sort_values(ascending=False).reset_index().head(1)
answer_ls.append(5)
wb_profit = collections.Counter()

for i in df.index:

    for j in df.iloc[i]['production_companies'].split('|'):

        result = re.match('^Warner Bros', j)

        if result:

            wb_profit[df.iloc[i]['release_year']] += df.iloc[i]['profit']

wb_profit.most_common()[0]
answer_ls.append(1)
month_count = collections.Counter()

for i in df.index:

    dt = datetime.strptime(df.iloc[i]['release_date'], '%m/%d/%Y')

    month_count[dt.month] += 1

month_count.most_common()[0]
answer_ls.append(4)
summer_made = 0

for i in df.index:

    dt = datetime.strptime(df.iloc[i]['release_date'], '%m/%d/%Y')

    if dt.month > 5 and dt.month < 9:

        summer_made += 1

summer_made
answer_ls.append(2)
answer_ls.append(0)
month_mean = pd.DataFrame(columns=('month','profit', 'year'))

for i in df.index:

    dt = datetime.strptime(df.iloc[i]['release_date'], '%m/%d/%Y')

    month_mean = month_mean.append({'month': dt.month, 'profit': df.iloc[i]['profit'], 'year': df.iloc[i]['release_year']}, ignore_index=True)
mdf = month_mean.groupby(['year','month'])['profit'].agg(['sum'])
answer_ls.append(0)
answer_ls.append(0)
answer_ls.append(0)
answer_ls.append(0)
answer_ls.append(0)
answer_ls.append(0)
len(answer_ls)
submission = pd.DataFrame({'Id':range(1,len(answer_ls)+1), 'Answer':answer_ls}, columns=['Id', 'Answer'])
submission
df.head()