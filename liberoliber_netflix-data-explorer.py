# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import datetime as dt

from datetime import timedelta



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





import pandas as pd 

df = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")



#print(list(df.columns))



#print(df.head())

#print("Number of rows: ", len(df))
df.tail()





# Exploration
def return_counter(data_frame, column_name, limit):

   from collections import Counter    

   print(dict(Counter(data_frame[column_name].values).most_common(limit)))

   

return_counter(df, 'country', 5)

#df['director'].dropna(inplace = True)

return_counter(df, 'director', 5)



df_d1 = df[df['director'] =='Raúl Campos, Jan Suter']

print(set(df_d1['title']))

print(set(df_d1['country']))



df_d2 = df[df['director'] =='Marcus Raboy']

print(set(df_d2['title']))



df = df[df['type'] =='Movie']

df['duration'] = df['duration'].map(lambda x: x.rstrip('min')).astype(int)

print(set(df['duration']))

print(df.head())
def return_statistics(data_frame, categorical_column, numerical_column):

    mean = []

    std = []

    field = []

    for i in set(list(data_frame[categorical_column].values)):

        new_data = data_frame[data_frame[categorical_column] == i]

        field.append(i)

        mean.append(new_data[numerical_column].mean())

        std.append(new_data[numerical_column].std())

    df = pd.DataFrame({'{}'.format(categorical_column): field, 'mean {}'.format(numerical_column): mean, 'std in {}'.format(numerical_column): std})

    df.sort_values('mean {}'.format(numerical_column), inplace = True, ascending = False)

    df.dropna(inplace = True)

    return df



stats = return_statistics(df, 'listed_in', 'duration')

print(stats.head(15))
def get_boxplot_of_categories(data_frame, categorical_column, numerical_column, limit):

    import seaborn as sns

    from collections import Counter

    keys = []

    for i in dict(Counter(df[categorical_column].values).most_common(limit)):

        keys.append(i)

    print(keys)

    df_new = df[df[categorical_column].isin(keys)]

    sns.set()

    sns.boxplot(x = df_new[categorical_column], y =      df_new[numerical_column])

    

#get_boxplot_of_categories(df, 'listed_in', 'duration', 5)
def get_histogram(data_frame, numerical_column):

    df_new = data_frame

    df_new[numerical_column].hist(bins=100)

    

get_histogram(df, 'duration')
df.head()
df['date_added'] = pd.to_datetime(df['date_added'])
df.tail()
start_date = "2019-01-01"

end_date   = "2020-12-31"

released   = 2019

#greater than the start date and smaller than the end date

added_mask    = (df['date_added'] > start_date) & (df['date_added'] <= end_date)

released_mask = (df['release_year'] >= released)

df = df.loc[added_mask]
df
df = df.loc[released_mask]

df