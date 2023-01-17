import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime



%matplotlib inline 

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/goodreadsbooks/books.csv',error_bad_lines=False)

df.head()
df.describe(include = 'all')
print(df.dtypes)
df.rename(columns={"  num_pages": "num_pages"}, inplace = True)

df.columns
# df['publish_date'] =

# pd.DatetimeIndex(df['publication_date']).year

# pd.to_datetime(df['publication_date'], format = '%Y')



df['publication_year'] = [i.split('/')[2] for i in df['publication_date']]

df['decade'] = [((int(i)//10)*10) for i in df['publication_year']]



df_lang_year = df.groupby(['decade','language_code']).count().reset_index()

df_lang_year
plt.figure(figsize=(20,10))

plt.xlabel('Year')

plt.ylabel('Number of Books')

    

ax1 = sns.lineplot(x="decade", y="bookID",

             hue="language_code", #style="event",

             data=df_lang_year)



ax1.set_ylabel('Number of Books')

ax1.set_xlabel('Decade')
x = df.groupby('language_code')['bookID'].count().reset_index().sort_values(by = 'bookID',ascending=False)

plt.figure(figsize=(15,10))



ax1 = sns.barplot(x = 'language_code', 

            y = 'bookID',

           data = x)



ax1.set_xlabel('Language Code')

ax1.set_ylabel('Number of Books')

ax1.set_yscale("log")

# ax1.set_ticklabels(x['bookID'], minor=False)
plt.figure(figsize=(15,15))

chart = sns.countplot(

    data=df,

    x='language_code'

)





ax1.set_xlabel('Language Code')

ax1.set_ylabel('Number of Books')

df['updated_language'] = ['en' if i in ('eng','en-US', 'en-GB', 'en-CA') else i for i in df['language_code']]

x = df.groupby('updated_language')['bookID'].count().reset_index().sort_values(by = 'bookID',ascending=False)



plt.figure(figsize=(15,10))



ax1 = sns.barplot(x = 'updated_language', 

            y = 'bookID',

           data = x)



ax1.set_xlabel('Language Code')

ax1.set_ylabel('Number of Books')

ax1.set_yscale("log")

# ax1.set_ticklabels(x['bookID'], minor=False)
authors = df.groupby('authors')['bookID'].count().reset_index().sort_values(by = 'bookID', ascending = False).head(10)



plt.figure(figsize=(15,10))

au = sns.barplot(x = 'authors',

                 y = 'bookID',

                 data = authors)



au.set_xlabel('Authors')

au.set_ylabel('Number of Books')



# Other way to rotate labels

# au.set_xticklabels(au.get_xticklabels(), 

#                    rotation=45,

#                   fontweight='light',

#                   fontsize='x-large')



plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',

    fontsize='x-large'  

)
df['average_rating_rounded'] = df['average_rating'].round(1)

plt.figure(figsize=(20,15))

ax1 = sns.countplot(

    data=df,

    x='average_rating_rounded'

)



ax1.set_xlabel('Average Rating')

ax1.set_ylabel('Number of Books')



fig, ax = plt.subplots(figsize=[14,8])

sns.distplot(df['average_rating'],ax=ax)

ax.set_title('Average rating distribution for all books',fontsize=20)

ax.set_xlabel('Average rating',fontsize=13)

most_rated = df.sort_values(by = 'ratings_count', ascending = False)[['title','ratings_count']]



plt.figure(figsize = (15,10))



ax1 = sns.barplot(x="ratings_count", 

            y="title", 

            data=most_rated.head(10)

           )



ax1.set_yticklabels(ax1.get_yticklabels(), 

                  fontweight='light',

                  fontsize='small')



ax1.set_ylabel('Title')

ax1.get_xaxis().get_major_formatter().set_scientific(False)
pub_data = df.groupby('publisher')['bookID'].count().reset_index().sort_values(by = 'bookID', ascending = False)



ax1 = sns.barplot(x = 'publisher',

                 y = 'bookID',

                 data = pub_data.head(10))



ax1.set_xticklabels(ax1.get_xticklabels(),

                   rotation = 45,

                  fontweight='light',

                  fontsize='small')



ax1.set_ylabel('Number of Books')
plt.figure(figsize=(12, 8))



df_corr = df[['average_rating','num_pages','ratings_count', 'text_reviews_count']].corr()

sns.heatmap(df_corr, 

            xticklabels = df_corr.columns.values,

            yticklabels = df_corr.columns.values,

            annot = True);

plt.figure(figsize=(14, 14))



sns.pairplot(df, diag_kind='kde');
