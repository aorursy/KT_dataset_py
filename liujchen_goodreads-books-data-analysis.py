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
df = pd.read_csv('/kaggle/input/books.csv',error_bad_lines=False)
df.info()
df.isna().sum()
df.describe(include='all')

# this one might not always useful

# count, unique, min, max 
df.corr()
# to have a copy of data frame for comparing before and after change of data cleaning

data = df.copy()

df.head(5)
# select the 1st author

data['authors'] = data['authors'].apply(lambda x: x.split("-")[0])

data.head(5)
freq = data['language_code'].value_counts()[0:10].to_frame()

freq
freq = data['language_code'].value_counts()[0:5].to_frame()

freq
freq = data['authors'].value_counts()[0:5].to_frame()

#print(type(freq))

freq
req = data.groupby(pd.cut(data['average_rating'], [0,1,2,3,4,5]))

req = req[['ratings_count']]  ## to dataframe

#print(type(req))

req.sum().reset_index()
newdf = data[data.average_rating>=4.5]

freq = newdf['authors'].value_counts()[0:5].to_frame()

freq
data[(data.average_rating>=4.5) & (data.authors == 'Bill Watterson')]
data[(data.average_rating>=4.5) & (data.authors == 'J.K. Rowling')]
data[(data.average_rating>=4.5) & (data.authors.str.contains('Rowling') )]
#Create a reference table 

dfLang = pd.DataFrame(columns=['language_code', 'language_name'])

dfLang = dfLang.append({'language_code': 'eng', 'language_name': 'English'}, ignore_index=True)

dfLang = dfLang.append({'language_code': 'cho', 'language_name': 'Chinese'}, ignore_index=True)

dfLang = dfLang.append({'language_code': 'en-US', 'language_name': 'English'}, ignore_index=True)

dfLang = dfLang.append({'language_code': 'en-GB', 'language_name': 'English'}, ignore_index=True)

dfLang = dfLang.append({'language_code': 'ger', 'language_name': 'German'}, ignore_index=True)

dfLang = dfLang.append({'language_code': 'fre', 'language_name': 'French'}, ignore_index=True)



dfLang
# full outer Join

df_outer = pd.merge(data, dfLang, on='language_code', how='outer')

df_outer.count().to_frame()

# linner Join

df_outer = pd.merge(data, dfLang, on='language_code', how='inner')

df_outer.count().to_frame()

# left outer Join

df_outer = pd.merge(data, dfLang, on='language_code', how='left')

df_outer.count().to_frame()

# right Join

df_outer = pd.merge(data, dfLang, on='language_code', how='right')

df_outer.count().to_frame()

# Union all, not remove duplicates

df1= data[(data.average_rating>=4.5) & (data.authors == 'Bill Watterson')]

df2= data[(data.average_rating>=4.5) & (data.authors == 'J.K. Rowling')]

dfUnion = pd.concat([

    df1,df2,df2

],ignore_index=True)

dfUnion.count().to_frame()
# Union , remove duplicates

df1= df[(data.average_rating>=4.5) & (data.authors == 'Bill Watterson')]

df2= df[(data.average_rating>=4.5) & (data.authors == 'J.K. Rowling')]

dfUnion = pd.concat([

    df1,df2,df2

],ignore_index=True).drop_duplicates().reset_index(drop=True)

dfUnion.count().to_frame()
# Import required libaries

import matplotlib.pyplot as plt

import seaborn as sns


bins = (0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5)

arr=plt.hist(data['average_rating'],alpha=.5,bins=bins,color='blue')

plt.xticks((0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5))

plt.title('Average Ratings')

plt.xlabel('Rating')

plt.ylabel('Frequency')

_ = plt.xlim(right=5.5)

for i in range(len(bins)-1):

    plt.text(arr[1][i],arr[0][i],str(arr[0][i]))


sns.violinplot(x=data['average_rating'])

plt.xlim(0,5)

plt.xlabel('Rating (0 to 5 stars)')

plt.title('Distribution of average rating on goodreads')

_ = plt.style.use('seaborn-white')
plt.style.use('seaborn-white')

plt.scatter(data['ratings_count'],data['text_reviews_count'])

plt.title('Rating Count vs. Review Count')

plt.xlabel('Rating Count')

plt.ylabel('Review Count')

plt.xlim(0)

plt.style.use('seaborn-white')

plt.scatter(data['# num_pages'],df['text_reviews_count'])

plt.title('Number of Page vs. Review Count')

plt.xlabel('Number of Page')

plt.ylabel('Review Count')

plt.xlim(0)



f,ax = plt.subplots(figsize=(4, 4))

plt.show()

#f and ax control the subplots size

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)

plt.show()



#annot=True to display the correction value

#fmt= '.2f' how many digits after decimal point

#linewidths=0.5 line width                                         
data.boxplot(column='average_rating')
plt.figure(1, figsize=(15, 7))

plt.title("Which aurthor wrote maximum books")

sns.countplot(x = "authors" ,order=data['authors'].value_counts().index[0:10] ,data=df)

top10Authors = data['authors'].value_counts()[0:10].to_frame()

plt.figure(figsize=(15,10))

sns.barplot(top10Authors['authors'], top10Authors.index, palette='Set3')
sns.distplot(data['average_rating'], 

             kde_kws={"color": "coral", "lw": 1, "label": "KDE"}, 

             hist_kws={"histtype": "stepfilled", "linewidth": 1, "alpha": 1, "color": "skyblue"});
top10NumofPages = data.sort_values('# num_pages', ascending = False).head(10).set_index('title')

plt.figure(figsize=(15,10))

sns.barplot(top10NumofPages['# num_pages'], top10NumofPages.index, palette='Set3')
# This code is copied from https://www.kaggle.com/bellali/select-which-book-to-enjoy

# select data

data_2 = data[data['authors'].isin(['George R.R. Martin', 'J.R.R. Tolkien', 'J.K. Rowling'])]

data_2 = data_2.query('language_code == "eng"')

#data_2.head()

# classify the rating level

bin_edges = [3.5, 4.0, 4.5, 5.0]

bin_names = ['low', 'medium', 'high']

data_2['rating_levels'] = pd.cut(data_2['average_rating'], bin_edges, labels=bin_names)

data_2.head()
#data_3 = pd.DataFrame(data_2.groupby('authors')['rating_levels'].value_counts().unstack().fillna(0))

data_3 = pd.DataFrame(data_2.groupby('authors')['rating_levels'].value_counts())

data_3
#data_3 = pd.DataFrame(data_2.groupby('authors')['rating_levels'].value_counts().unstack().fillna(0))

data_3 = pd.DataFrame(data_2.groupby('authors')['rating_levels'].value_counts().unstack())

data_3
#data_3 = pd.DataFrame(data_2.groupby('authors')['rating_levels'].value_counts().unstack().fillna(0))

data_3 = pd.DataFrame(data_2.groupby('authors')['rating_levels'].value_counts().unstack().fillna(0))

data_3
# This code is copied from https://www.kaggle.com/bellali/select-which-book-to-enjoy

# Pie chart

labels = 'high', 'low', 'medium'

explode = (0.1, 0, 0)

fig1, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize = (12,4))



# George R.R. Martin

ax1.pie(data_3.iloc[0], explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax1.set_title('George R.R. Martin')

ax1.axis('equal')

# J.K. Rowling

ax2.pie(data_3.iloc[1], explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax2.set_title('J.K. Rowling')

ax2.axis('equal')

# J.R.R. Tolkien

ax3.pie(data_3.iloc[2], explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax3.set_title('J.R.R. Tolkien')

ax3.axis('equal')



plt.show()