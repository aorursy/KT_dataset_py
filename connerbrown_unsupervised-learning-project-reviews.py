# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
con = sqlite3.connect("../input/database.sqlite")
df = pd.read_sql_query("SELECT reviews.*, content.* FROM reviews JOIN content ON reviews.reviewid = content.reviewid", con)
display(df.shape)
df.head(3)
author_counts = df['author'].value_counts()
authors = list(author_counts[(author_counts > 199) & (author_counts < 300)].index)
print (authors)
# total word count per author
df['word_count'] = df['content'].apply(lambda x: len(str(x).split(' '))) # add word count for each review
author_groups = df.groupby('author') # sum by author
display(author_groups.mean().loc[authors][['word_count', 'score']])

