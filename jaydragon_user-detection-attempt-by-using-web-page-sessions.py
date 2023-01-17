import numpy as np 

import pandas as pd 

import os

import seaborn as sns

import matplotlib.pyplot as plt

import collections

import pickle

import warnings

warnings.filterwarnings("ignore")

import datetime

from sklearn.linear_model import LogisticRegression
#function that to get a list of the sites by mapping the key to the webpage

def get_site_name(site_key, site_dictionary):

    site_list = []

    for i in range(len(site_key)):

        for key, value in site_dictionary.items():

            if value == site_key[i]:

                site_list.append(key)

    return site_list
print(os.listdir("../input"))
df_train = pd.read_csv("../input/train_sessions.csv")

df_train.head()
nrow, ncol = df_train.shape

print('There are %i rows and %i columns in the train dataset.' % (nrow, ncol))
df_test = pd.read_csv("../input/test_sessions.csv")

df_test.head()
nrow, ncol = df_test.shape

print('There are %i rows and %i columns in the test dataset.' % (nrow, ncol))
with open('../input/site_dic.pkl', 'rb') as f:

    site_dic = pickle.load(f)
Nan_num_train= df_train.isna().sum()

Nan_num_test= df_test.isna().sum()

print('The number of NaNs in the training set per column \n', Nan_num_train)

print('The number of NaNs in the testing set per column \n ', Nan_num_test)
df_train = df_train.fillna(0)

df_test = df_test.fillna(0)



Nan_num_train= df_train.isna().sum()

Nan_num_test= df_test.isna().sum()

print('The number of NaNs in the training set per column \n', Nan_num_train)

print('The number of NaNs in the testing set per column \n ', Nan_num_test)
df_train.dtypes
df_test.dtypes
timelist = ['time%s' % i for i in range(1, 11)]

df_train[timelist] = df_train[timelist].apply(pd.to_datetime)

df_test[timelist] = df_test[timelist].apply(pd.to_datetime)
df_train.dtypes
df_test.dtypes
df_train.describe(include='all')
df_test.describe(include='all')
df_train[list(df_train)].corr()
df_test[list(df_test)].corr()
sns.set(style="whitegrid")

sns.countplot(x="target", data=df_train).set_title('Counts per Target')
sites = [c for c in df_train if c.startswith('site')]

sites_df = pd.melt(df_train, id_vars='target', value_vars=sites, value_name='sites')

sites_df
sns.distplot(sites_df['sites']).set_title('Sites Distribution Plot')
popular_sites = collections.Counter(sites_df['sites']).most_common(11)

print(popular_sites)
popular_sites.remove((0.0, 122730))

#print(popular_sites)
site, count = zip(*popular_sites)

site_labels = get_site_name(site,site_dic)

#print(site_labels)



y_pos = np.arange(len(site_labels))

plt.barh(y_pos, count, align='center', alpha=0.5)

plt.yticks(y_pos, site_labels)

plt.xlabel('Count')

plt.title('Top %i Sites Overall' % len(site_labels))



plt.show()
alice_sites = sites_df[sites_df['target'] == 1]

alice_sites.head(10)

#sns.distplot(alice_sites['sites']).set_title('Alice\'s Sites Distribution Plot')

alice_popular_sites = collections.Counter(alice_sites['sites']).most_common(11)

#print(alice_popular_sites)



site, count = zip(*alice_popular_sites)

site_labels = get_site_name(site,site_dic)

#print(site_labels)



y_pos = np.arange(len(site_labels))

plt.barh(y_pos, count, align='center', alpha=0.5)

plt.yticks(y_pos, site_labels)

plt.xlabel('Count')

plt.title('Alice\'s Top %i Sites Overall' % len(site_labels))



plt.show()
notalice_sites = sites_df[sites_df['target'] == 0]

notalice_popular_sites = collections.Counter(notalice_sites['sites']).most_common(11)

#print(notalice_popular_sites)

notalice_popular_sites.remove((0.0, 122529))



site, count = zip(*notalice_popular_sites)

site_labels = get_site_name(site,site_dic)

#print(site_labels)



y_pos = np.arange(len(site_labels))

plt.barh(y_pos, count, align='center', alpha=0.5)

plt.yticks(y_pos, site_labels)

plt.xlabel('Count')

plt.title('Not Alice Top %i Sites Overall' % len(site_labels))



plt.show()
times = [c for c in df_train if c.startswith('time')]

times_df = pd.melt(df_train, id_vars='target', value_vars=times, value_name='times')

times_df
alice_timesdf = times_df[times_df['target']==1]

notalice_timesdf = times_df[times_df['target']==0]
year_df = times_df['times'].dt.year

year_df.value_counts().plot('bar').set_title('Counts per Year')
year_alice = alice_timesdf['times'].dt.year

year_alice.value_counts().plot('bar').set_title('Counts per Year For Alice')
year_notalice = notalice_timesdf['times'].dt.year

year_notalice.value_counts().plot('bar').set_title('Counts per Year For Not Alice')
month_df = times_df['times'].dt.month

month_df.value_counts().plot('bar').set_title('Counts per Month')
month_alice = alice_timesdf['times'].dt.month

month_alice.value_counts().plot('bar').set_title('Counts per Month For Alice')
month_notalice = notalice_timesdf['times'].dt.month

month_notalice.value_counts().plot('bar').set_title('Counts per Month For Not Alice')
hour_df = times_df['times'].map(lambda x: x.strftime('%H'))

hour_df.value_counts().plot('bar').set_title('Counts per Hour')
hour_alice = alice_timesdf['times'].map(lambda x: x.strftime('%H'))

hour_alice.value_counts().plot('bar').set_title('Counts per Hour For Alice')
hour_notalice = notalice_timesdf['times'].map(lambda x: x.strftime('%H'))

hour_notalice.value_counts().plot('bar').set_title('Counts per Hour For Not Alice')
myr_df = times_df['times'].map(lambda x: x.strftime('%m-%Y'))

myr_df.value_counts().plot('bar').set_title('Counts per Month-Year')
myr_alice = alice_timesdf['times'].map(lambda x: x.strftime('%m-%Y'))

myr_alice.value_counts().plot('bar').set_title('Counts per Month-Year For Alice')
myr_notalice = notalice_timesdf['times'].map(lambda x: x.strftime('%m-%Y'))

myr_notalice.value_counts().plot('bar').set_title('Counts per Month-Year For Not Alice')
wkday_df = times_df['times'].map(lambda x: x.weekday())

wkday_df.value_counts().plot('bar').set_title('Counts per Day of the Week')

#Saturday and Sunday are, respectively. the less active days of the week
wkday_alice = alice_timesdf['times'].map(lambda x: x.weekday())

wkday_alice.value_counts().plot('bar').set_title('Counts per Day of the Week For Alice')

#Saturday, Wednesday, and Sunday are, respectively, the less active days of the week
wkday_notalice = notalice_timesdf['times'].map(lambda x: x.weekday())

wkday_notalice.value_counts().plot('bar').set_title('Counts per Day of the Week For Not Alice')

#Saturday and Sunday are, respectively, the less active days of the week
timedelta = np.zeros((df_train.shape[0], len(timelist)+1))

#len with be 11, columns 0 to 10

timedelta[:,len(timelist)] = df_train['target']

#column 10 is the target

for i in range(len(timelist)-2):

    timedelta[:,i] = (df_train[timelist[i]] - df_train[timelist[i+1]]).abs().dt.seconds
#timedelta.mean(axis=1)

#row mean

alice_timedelta = timedelta[timedelta[:,10] == 1]

alice_avgtimedelta = alice_timedelta.mean(axis=1)

plt.hist(alice_avgtimedelta, 4, facecolor='blue', alpha=0.5)

plt.xlabel('Seconds')

plt.ylabel('Count')

plt.title('Alice\'s Average Time Between Sites')

plt.show()
notalice_timedelta = timedelta[timedelta[:,10] == 0]

notalice_avgtimedelta = notalice_timedelta.mean(axis=1)

plt.hist(notalice_avgtimedelta, 4, facecolor='blue', alpha=0.5)

plt.xlabel('Seconds')

plt.ylabel('Count')

plt.title('Not Alice Average Time Between Sites')

plt.show()
df_new_train = df_train

df_new_train[timelist] = df_new_train[timelist].applymap(lambda x: x.weekday())

df_new_train.head()
df_new_test = df_test

df_new_test[timelist] = df_new_test[timelist].applymap(lambda x: x.weekday())

df_new_test.head()
X_train = df_new_train[df_new_train.columns[1:20]]

y_train = df_new_train[df_new_train.columns[21]]

X_test = df_new_test[df_new_test.columns[1:20]]
logreg = LogisticRegression(random_state=3, solver='lbfgs').fit(X_train, y_train)

predictions = logreg.predict(X_test)

predictions_probabilities = logreg.predict_proba(X_test)
values, counts = np.unique(predictions, return_counts=True)

print(values)

print(counts)