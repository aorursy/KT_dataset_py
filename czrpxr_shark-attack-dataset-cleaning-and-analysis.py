!pip install countryinfo
from countryinfo import CountryInfo

import numpy as np

import pandas as pd

import re

import matplotlib.pyplot as plt

from subprocess import check_output

import seaborn as sns
def clean_colnames(df):

    '''Clean columns names when passing a pandas dataframe: params (df - dataframe)'''

    col_clean = []

    for col in df.columns:

        col = col.strip().lower()

        col = col.replace('.',' ')

        col_clean.append(col)

        

    df.columns = col_clean

    return df.columns



def change_col_pos(col,pos,df):

    '''Change the position of a column in a dataframe: params(col - column name, 

    pos - new index position, df - dataframe)'''

    all_cols = df.columns.tolist()

    temp = all_cols.pop(all_cols.index(col))

    all_cols.insert(pos,col)

    

    df = df[all_cols]

    

    return df



def country_hem(col):

    '''Return a list if countries are part of the north (1) or south (0) hemispheres of planet earth. If country

    is not recognized, NaN is returned'''

    hem_lst = []

    for row in col:

        try:

            country = CountryInfo(row)

        except AttributeError:

            row = 'x'

   

        try:

            pos = country.latlng()

        except KeyError:

            pos = (0,0)

    

        if pos[0] > 0:

            temp_row = 1

        elif pos[0] < 0:

            temp_row = 0

        else:

            temp_row = np.nan

    

        hem_lst.append(temp_row)

    

    return hem_lst     
df_original = pd.read_csv('../input/gsaf5csv/GSAF5.csv', encoding = 'latin-1')

df = df_original.copy()
df.head()
df.shape
df.columns
clean_colnames(df)
#Drop all duplicates considering all columns

df.drop_duplicates(subset=list(df.columns))
null_cols = df.isnull().sum()

null_cols
df = df.drop(axis = 1, columns = ['unnamed: 22', 'unnamed: 23', 'href', 'href formula', 'pdf', 'original order',

                                  'investigator or source', 'time', 'fatal (y/n)', 'injury', 'case number 1', 

                                  'case number 2', 'type', 'name', 'species', 'case number', 'age', 'case number'])
df.head()
df.info()
temp_lst = []

for row in df['date']:

    temp_row = ''.join(re.findall('\-[A-Za-z]{3}\-',row)).lower()

    temp_row = re.sub('\-','',temp_row)

        

        

    if temp_row == '':

        temp_row = np.nan



    temp_lst.append(temp_row)



df['month'] = temp_lst
df = change_col_pos('month',3,df)
df['month'].value_counts(dropna=False)
df.shape
df.dropna(subset=['month'], inplace=True)
for row in df['month']:

    if len(row) > 3:

        df['month'].replace(row,row[:3], inplace=True)

    elif row == 'jut':

        df['month'].replace(row,'jun', inplace=True)
df['month'].value_counts(dropna=False)
df['country'].unique()
for row in df['country']:

    if isinstance(row, str):

        new_row = re.sub('\/.+|\(.+\)|\.|\?', '', row)

        new_row = re.sub('\&', 'and', new_row.strip().lower())

        

        if new_row == 'usa':

            new_row = new_row.replace(new_row, 'united states')

        elif new_row == 'bahamas':

            new_row = new_row.replace(new_row, 'the bahamas')

        elif new_row == 'england' or new_row == 'british isles':

            new_row = new_row.replace(new_row, 'united kingdom')

        elif new_row == 'reunion':

            new_row = new_row.replace(new_row, 'réunion')

        elif new_row == 'okinawa':

            new_row = new_row.replace(new_row, 'japan')

        elif new_row == 'azores':

            new_row = new_row.replace(new_row, 'portugal')

        elif new_row == 'red sea':

            new_row = new_row.replace(new_row, 'egypt')

        elif new_row == 'okinawa':

            new_row = new_row.replace(new_row, 'japan')

        elif new_row == 'columbia':

            new_row = new_row.replace(new_row, 'colombia')

        elif new_row == 'new britain' or new_row == 'new guinea' or new_row == 'british new guinea' or new_row == 'admiralty islands':

            new_row = new_row.replace(new_row, 'papua new guinea')

        

        df['country'].replace(row,new_row, inplace=True)

    else:

        df['country'].replace(row,np.nan, inplace=True)
df['country'].unique()
df.dropna(how='all', subset=['country','area','location'], inplace=True)
df['hemisphere'] = country_hem(df['country'])
df = change_col_pos('hemisphere',6,df)
df['hemisphere'].value_counts(dropna=False)
nh = ['north sea', 'scotland', 'north pacific ocean', 'turks and caicos', 'caribbean sea', 'persian gulf', 'micronesia', 

     'burma', 'north atlantic ocean', 'montenegro', 'the balkans', 'northern arabian sea', 'netherlands antilles', 'mediterranean sea',

     'grand cayman', 'netherlands antilles', 'south china sea', 'st martin', 'andaman', 'palestinian territories', 'johnston island',

     'nevis', 'bay of bengal']

sh = ['south atlantic ocean', 'western samoa', 'southwest pacific ocean', 'tasman sea', 'st helena']
for row in df.index:

    if df.at[row,'country'] in nh:

        df.at[row, 'hemisphere'] = 1

    elif df.at[row, 'country'] in sh:

        df.at[row, 'hemisphere'] = 0
df['hemisphere'].value_counts(dropna=False)
df[df['country'] == 'atlantic ocean']
df.at[1282, 'hemisphere'] = 1

df.at[1293, 'hemisphere'] = 1

df.at[3493, 'hemisphere'] = 1

df.at[3704, 'hemisphere'] = 1

df.at[4475, 'hemisphere'] = 1

df.at[4476, 'hemisphere'] = 0
df[df['country'] == 'pacific ocean']
df.at[3376, 'hemisphere'] = 1

df.at[3736, 'hemisphere'] = 1

df.at[3924, 'hemisphere'] = 1

df.at[3325, 'hemisphere'] = 1

df.at[3961, 'hemisphere'] = 1

df.at[4147, 'hemisphere'] = 1

df.at[4438, 'hemisphere'] = 0

df.at[4456, 'hemisphere'] = 0
df['hemisphere'].value_counts(dropna=False)
df.dropna(subset=['hemisphere'], inplace=True)
df['hemisphere'].value_counts(dropna=False)
n_seasons = {'spring': ['mar', 'apr', 'may'],

             'summer': ['jun', 'jul', 'aug'],

             'autumm': ['sep', 'oct', 'nov'],

             'winter': ['dec', 'jan', 'feb']}



s_seasons = {'spring': ['sep', 'oct', 'nov'],

             'summer': ['dec', 'jan', 'feb'],

             'autumm': ['mar', 'apr', 'may'],

             'winter': ['jun', 'jul', 'aug']}
temp_seasons = []

for row in df.index:

    if df.at[row, 'hemisphere'] == 1:

        if df.at[row, 'month'] in n_seasons['spring']:

            temp_seasons.append('spring')

        elif df.at[row, 'month'] in n_seasons['summer']:

            temp_seasons.append('summer')

        elif df.at[row, 'month'] in n_seasons['autumm']:

            temp_seasons.append('autumm')

        elif df.at[row, 'month'] in n_seasons['winter']:

            temp_seasons.append('winter')

    

    elif df.at[row, 'hemisphere'] == 0:

        if df.at[row, 'month'] in s_seasons['spring']:

            temp_seasons.append('spring')

        elif df.at[row, 'month'] in s_seasons['summer']:

            temp_seasons.append('summer')

        elif df.at[row, 'month'] in s_seasons['autumm']:

            temp_seasons.append('autumm')

        elif df.at[row, 'month'] in s_seasons['winter']:

            temp_seasons.append('winter')

df['seasons'] = temp_seasons
df = change_col_pos('seasons',5,df)
df.head()
df['seasons'] = df['seasons'].astype('category')
df.info()
group_seasons = df['seasons'].value_counts()
%matplotlib inline

plt.figure(figsize=[12,12])

#df['seasons'].value_counts().plot(kind='bar')

#sns.countplot(x='seasons', data=df, palette='Set2', ax=ax)

graph1 = sns.countplot(x='seasons', data=df, palette='Set2')

plt.xlabel('Seasons of the year')

plt.ylabel ('Ocurrencies')

plt.title('Shark attacks by season')