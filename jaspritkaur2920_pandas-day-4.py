# import libraries

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load data

df = pd.read_csv('/kaggle/input/stocknews/upload_DJIA_table.csv')
df.sort_values(by = ['Date'], ascending = False)
df.sort_values(by = ['Open', 'Close'], ascending = False)
df.sort_index()
df.rename(columns= {'Date' : 'new_date'}).head()
# we will make the data as it is by again renaming new_date to Date

df.rename(columns= {'new_date' : 'Date'}).head(1)
df['Difference'] = df.High - df.Low

df.head()
# check for the current name of the index

print(df.index.name)
# giving name to the index 

df.index.name = 'index'

df.head()
df.columns = map(str.lower, df.columns)

df.columns
df.columns = map(str.upper, df.columns)

df.columns
# let's make a dataframe df2

df2 = pd.DataFrame({'Animal': ['Falcon', 'Falcon',

                              'Parrot', 'Parrot'],

                   'Max Speed': [380., 370., 24., 26.]})

df2
# grouping 'Animals' with the mean of their max-speed

df2.groupby(['Animal']).mean()
# creating a hierarchical index dataframe df3

arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],

          ['Captive', 'Wild', 'Captive', 'Wild']]



index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))



df3 = pd.DataFrame({'Max Speed': [390., 350., 30., 20.]},

                  index=index)

df3
# grouping based on 'Animal'== (level=0), because it is the first index

df3.groupby(level=0).mean()
df3.groupby(level="Type").mean()