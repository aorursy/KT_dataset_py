# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

data_file=''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        data_file=os.path.join(dirname, filename)

        print(data_file)



# Any results you write to the current directory are saved as output.
#Read the csv into dataframe

df = pd.read_csv(data_file, index_col=0)
df.columns
#See first 5 samples

df.head(5)
#Display general info per column: number of rows, does it include null?, column type

df.info()
#We saw that the data has no null value except the description column.

#Replace the null with empty string

df.Description.fillna('', inplace=True)

df.info()
#See numeric columns statistics

df.describe()
#See all cover types

set(df.Cover.values)

#We can see from the results there are many repeatitves. such as, hard cover, hardcover, Hardcover, Hard Cover, مقوى...all are the same.

#Need some work to combine them togther
#Before cleanup

print('#Of covers before cleanup: ',len(set(df.Cover.values)))



#Do some basic data cleanup:

def covertype(x):

    x = str(x).strip().lower()

    if x=='Hard Cover' or x=='HardCover' or x=='hardcover' or x=='hard cover' or x== 'مقوى':

        return 'hardcover'

    if x=='Paperback' or x=='paperback' or x=='غلاف ورقي' or x=='غلاف ورقي':

        return 'paperback'

    if x=='عادي' or x=='غلاف عادي' or x=='غلاف':

        return 'standard'

    return x

df.Cover=df.Cover.apply(lambda x: covertype(x))





print('#Of covers after cleanup: ',len(set(df.Cover.values)))



print(set(df.Cover.values))
#See all Category types

set(df.Category.values)
a=set(df.Author.values)

print('Unique Authors before cleanup: ', len(a))

#display first 20

list(set(df.Author.values))[:20]



#If you want to display all of the authors, uncomment next line

#set(df.Author.values)
def author_cleanup(x):

    x = str(x).strip().lower()

    return x

df.Author=df.Author.apply(lambda x: covertype(x))



a=set(df.Author.values)

print('Unique Authors before cleanup: ', len(a))
import seaborn as sn

import matplotlib.pyplot as plt
#Author vs Mean Price - plot top 20

df1=df.groupby('Author')['Price'].mean().sort_values(ascending=False)

df1.head(20).plot(kind='bar', grid=True, title='Authos vs. Price(Mean)')
#Category vs Mean Price - plot top 20

df1=df.groupby('Category')['Price'].mean().sort_values(ascending=False)

df1.head(20).plot(kind='bar',grid=True, title='Categiry vs. Price(Mean)')
#Let see the Author vs. num of books graph - filter for the top 20 only

topn=20

df.groupby('Author')['Title'].count().sort_values(ascending=False).head(topn).plot(kind='bar', grid=True, title='Authos vs. Num of books', x='Author', y='Num of books');
#Let see the Category vs. num of books graph - filter for the top 20 only

topn=20

df.groupby('Category')['Title'].count().sort_values(ascending=False).head(topn).plot(kind='bar', grid=True, title='Category vs. Num of books', x='Category', y='Num of books');