# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#importing the data

data_file_path = "../input/books.csv"

data=pd.read_csv(data_file_path, error_bad_lines = False)
# Scanning the columns of the data

print ( data.columns)
# Getting the book list of Ayn Rand

author_Ayn_Rand = data['authors']=='Ayn Rand'

Ayn_Rand_Books = data[author_Ayn_Rand]

ratings_count= Ayn_Rand_Books.groupby('title')['ratings_count'].sum().reset_index().sort_values('ratings_count',

ascending=False)

sns.barplot(y=ratings_count['title'],x=ratings_count['ratings_count'])

print(ratings_count)
ayu = Ayn_Rand_Books.sort_values('average_rating',ascending=False)

sns.barplot(y='title',x= 'average_rating', data= ayu)
maximum_ratings_count = data.groupby('title')['ratings_count'].sum().reset_index().sort_values('ratings_count', 

ascending=False).head()



sns.barplot(x='ratings_count',y='title', data=maximum_ratings_count)
new = pd.concat([maximum_ratings_count,ratings_count] )

sns.barplot(y='title',x= 'ratings_count', data= new)
data_by_authors= data.groupby('authors')['# num_pages'].sum().reset_index().sort_values('# num_pages' ,

ascending=False).head()

sns.barplot(y='authors',x= '# num_pages', data= data_by_authors)