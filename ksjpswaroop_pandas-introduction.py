# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#Check what version of Pandas we are using
pd.__version__
# load data to a variable called df
'''
the data file is tsv which means a tab delimited file
we are going to use read_csv to load the data and will mention the delimiter as '\t' for tabspace.
'''
df = pd.read_csv('../input/gapminder.tsv', delimiter='\t')
# view the first 5 rows
df.head()
# what is this df object?
type(df)
# get num rows and columns
df.shape
# look at the data - summarized information
df.info()
# subset a single column
country_df = df['country']
country_df
# the output above is same as
df.country
#you can check the top 5 values of one column too...
df.country.head()
# subset multiple columns
df[['country', 'continent', 'year']].head()
# delete columns
# this will drop a column in-place
df_new = df.copy()
df_new
del df_new['country']  # del df_new.country won't work
df_new
# this won't unless you use the inplace parameter
df_new.drop('continent', axis=1)
#continent is still in...
df_new.head()
# df is unchanged
df.head()
#display df
df.head()
#pass df to df_new
df_new = df
df_new.head()
len(df),len(df_new)
del df_new['country']
df_new.head()
df.head()
df.columns,df_new.columns
df = pd.read_csv('../input/gapminder.tsv', delimiter='\t')
# first row
df.loc[0]
# 100th row
df.loc[99]
# this will fail
df.loc[-1]
df.iloc[0]
df.iloc[99]
#this will work
df.iloc[-1]




























