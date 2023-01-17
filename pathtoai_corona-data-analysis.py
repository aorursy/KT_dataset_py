# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv")

df_metadata.head()
print(df_metadata.shape)

print(df_metadata.dtypes)
df_metadata.describe(include = 'all').T
df_metadata.isnull().sum()
df_metadata.source_x.value_counts()
print(len(df_metadata.authors.value_counts()))

print(df_metadata.authors.value_counts())
col = ['Microsoft Academic Paper ID','WHO #Covidence']

# Drop Above columns as 90% data is null. 

df_metadata.drop(col, axis =1, inplace = True)
## File NA values in Title and Abstract with BLANK



## Create A New Columns with Titles and Abstract:

df_metadata['title'].fillna("",inplace= True)

df_metadata['abstract'].fillna("",inplace= True)

df_metadata.isna().sum()
df_metadata['title_abstract'] = df_metadata['title'] + df_metadata['abstract']

df_metadata['title_abstract'] .head()
new_col = ['source_x' , 'title']

filtered_df = df_metadata