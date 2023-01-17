import numpy as np

import pandas as pd

import os

ls = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        ls.append(pd.read_csv(os.path.join(dirname, filename)))

df_search = pd.concat(ls)        
df_search.info()
df_search.nunique()
df_search.head(2).T
df_search.duplicated().sum()
df_search['searchTerms'].value_counts()
df_search['displayLink'].value_counts()
cols=['searchTerms', 'rank', 'title', 'displayLink']

df_search_filtered = df_search.loc[df_search['rank'].isin(range(1,5)), cols]

df_search_filtered.head()
df_search_filtered['searchTerms'].value_counts()
top_10_no_filter = df_search['displayLink'].value_counts().nlargest(10).index

top_10_no_filter
top_10 = df_search_filtered['displayLink'].value_counts().nlargest(10).index

top_10
#df_search_filtered[df_search_filtered['displayLink'].isin(top_10)].head()
#df_search_filtered[df_search_filtered['displayLink'].isin(top_10)].groupby(['searchTerms']).size().head()
#https://stackoverflow.com/questions/15222754/groupby-pandas-dataframe-and-select-most-common-value

#df_search_filtered[df_search_filtered['displayLink'].isin(top_10)].groupby(['searchTerms', 'rank'])['displayLink'].agg(pd.Series.mode)
#https://stackoverflow.com/questions/15222754/groupby-pandas-dataframe-and-select-most-common-value

pd.options.display.max_rows = 999

df_search_filtered.groupby(['searchTerms', 'rank'])['displayLink'].agg(pd.Series.mode)
df_output = df_search_filtered.groupby(['searchTerms', 'rank'])['displayLink'].agg(pd.Series.mode)
df_output.reset_index().pivot(index='searchTerms', columns='rank', values='displayLink')
#https://stackoverflow.com/questions/54685894/how-to-extract-numbers-after-string-pattern-in-pandas

df_search['destination'] = df_search['searchTerms'].str.extract(r"(?:flights to |tickets to )(.*)")
cols_desti=['destination', 'rank', 'title', 'displayLink']

df_search_filtered_desti = df_search.loc[df_search['rank'].isin(range(1,5)), cols_desti]

df_output_desti = df_search_filtered_desti.groupby(['destination', 'rank'])['displayLink'].agg(pd.Series.mode)

df_output_desti.reset_index().pivot(index='destination', columns='rank', values='displayLink')
df_output_desti.reset_index().pivot(index='destination', columns='rank', values='displayLink').iloc[:,0].value_counts()