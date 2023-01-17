import sqlite3, datetime

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import binom

from scipy.signal import savgol_filter



pd.set_option('precision', 2)

np.set_printoptions(precision=2)



con = sqlite3.connect('../input/database.sqlite')

all_reviews = pd.read_sql('SELECT * FROM reviews WHERE pub_year < 2017', con)

years = pd.read_sql('SELECT * FROM years', con)

con.close()
# convert pub_date to datetime object, get unix timestamps

reviews = all_reviews.copy(deep = True) # slice to new object

reviews['pub_date'] = pd.to_datetime(reviews.pub_date, format = '%Y-%m-%d')

reviews['unix_time'] = reviews.pub_date.astype(np.int64, copy = True) // 10**9



# find the first best new music, get rid of everything before it

first_bnm = reviews.loc[reviews.best_new_music == True, 'unix_time'].min()

reviews = reviews.loc[reviews.unix_time >= first_bnm]



# print out date of first bnm

idx = (reviews.unix_time == first_bnm) & (reviews.best_new_music == True)

first_bnm_str = datetime.datetime.fromtimestamp(first_bnm)

print('First best new music: ' + first_bnm_str.strftime('%B %d, %Y'))



# remove sunday reviews

reviews = reviews.loc[reviews.pub_weekday < 6]



# remove multi-year reviews

year_counts = years.groupby('reviewid').count().reset_index()

keepers = year_counts.loc[year_counts.year == 1, 'reviewid']

reviews = reviews.loc[reviews.reviewid.isin(keepers)]



# find overall proportion of best new music

proportion_bnm = np.mean(reviews.best_new_music)

print('Global p(bnm): ' + str(proportion_bnm))
# compute oberved and expected number of best new music

table = reviews.groupby('pub_year')['best_new_music'].agg(['sum', 'count'])

table['expected'] = table['count'] * proportion_bnm

table.rename(columns = {'sum': 'observed'}, inplace=True)

table.plot(y = ['expected', 'observed'], kind = 'bar')

plt.ylabel('Frequency')

plt.xlabel('')

plt.title('Observed vs. Expected Best New Music Frequency')

plt.show()
p_bmn = reviews.copy(deep = True)

p_bmn['score'] = np.round(p_bmn['score'])

p_bmn = pd.pivot_table(p_bmn, 

                columns = 'score', 

                index = 'pub_year', 

                values = 'best_new_music',

                aggfunc = 'mean',

                fill_value = 0.0)



ax = sns.heatmap(p_bmn, vmin = 0, vmax = 1, annot = True)

plt.ylabel('')

plt.show()
# get counts / observed bnm for each day

g = reviews.groupby('pub_date')

day_counts = g['best_new_music'].agg(['count','sum']).reset_index()

rename_cols = {'count':'N', 'sum':'observed'}

day_counts.rename(columns = rename_cols, inplace = True)

day_counts['pub_year'] = day_counts.pub_date.dt.year



# get probabilities for each year

year_ps = reviews.groupby('pub_year').best_new_music.mean().reset_index()

year_ps.rename(columns = {'best_new_music':'p'}, inplace = True)



# merge into table

data = pd.merge(day_counts, year_ps, on = 'pub_year')



# empty out columns for expected k probabilities

for k in range(5):

    data[k] = pd.Series(0, index = table.index)



# fill in data

for i, row in data.iterrows():

    if row.N < 5: xs = range(row.N + 1)

    else: xs = range(5)

    data.loc[i,xs] = binom.pmf(xs, row.N, row.p)

data.head(n=10)
ks = np.arange(5)



# get observed and expected probs

observed_ps = np.array([sum(data.observed == k) for k in ks])

observed_ps = observed_ps / float(sum(observed_ps))

expected_ps = data[ks].mean().as_matrix()



df = pd.DataFrame(

                index = ks,

                data = dict(Expected = expected_ps,

                            Observed = observed_ps)

            )

df.plot(y = ['Observed','Expected'], kind = 'bar')

plt.ylim([0,1])

plt.ylabel('Probability')

plt.xlabel('# BNM')

plt.show()
# create table where best new music looks ahead to see if there are future bnms

bnm_lookahead = reviews.groupby('unix_time')['best_new_music'].agg('any').to_frame()



lookahead  = 30     # number of days to look ahead

day_length = 86400  # seconds per day



# add lookahead columns

for i in range(lookahead):

   bnm_lookahead[i+1] = pd.Series(index = bnm_lookahead.index) 



# get all pub_dates

all_dates = bnm_lookahead.index.values



# fill out table

for i in all_dates:

    for j in range(1, lookahead+1):

        date = i + j*day_length

        if date not in all_dates: continue

        bnm_lookahead.loc[i, j] = bnm_lookahead.loc[date,'best_new_music']



bnm_lookahead.rename(columns = {'best_new_music':0}, inplace = True)

bnm_lookahead.head()
# plot the table



x = range(lookahead + 1)

y = bnm_lookahead.loc[:,x].mean().as_matrix()

plt.plot(x, y, color='gray', label = 'Global Average')





labels = ['No Best New Music', 'Best New Music']

for tf, rows in bnm_lookahead.groupby(0):



    y = rows.loc[:,x].mean()

    plt.plot(x, y, label = labels[tf])

    



plt.xlabel('Days after $n$')

plt.ylabel('Proportion Best New Music')



plt.legend()

plt.show()
subset = bnm_lookahead.copy(deep = True)

subset = subset[subset.index.values > 1230768000] # after Jan 1 2009





x = range(lookahead + 1)

y = subset.loc[:,x].mean().as_matrix()

plt.plot(x, y, color='gray', label = 'Global Average')





labels = ['No Best New Music', 'Best New Music']

for tf, rows in subset.groupby(0):



    y = rows.loc[:,x].mean()

    plt.plot(x, y, label = labels[tf])

    



plt.xlabel('Days after $n$')

plt.ylabel('Proportion Best New Music')



plt.legend()

plt.show()