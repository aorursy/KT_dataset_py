# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines = False)
df.head()
print(df.shape)

df.dtypes
authorsdf=df["authors"].value_counts()
authorsdf
xd=df['authors'].value_counts().iloc[:10]

xd
yd=df['title'].value_counts().iloc[:10]

yd
plt.figure(figsize=(20,10))

plt.bar(xd.keys().tolist(),xd.tolist(),color="orange")
plt.figure(figsize=(25,10))

plt.bar(yd.keys().tolist(),yd.tolist(),color="orange")
#ad=xd.keys().tolist()

#ab=df[(df['authors']==ad[0]) & (df['average_rating']>3)]

#for i in range(1,10):

 #   ab=(df[(df['authors']==ad[i]) & (df['average_rating']>3)])

#ab.sort_values('average_rating',ascending=False).iloc[:20]

topauthor = df[df['average_rating']>=4]

topauthor = topauthor.groupby('authors')['title'].count().reset_index().sort_values('title', ascending = False).head(10).set_index('authors')

topauthor

#zd=df.groupby(['authors'])['title'].value_counts().sort_values(ascending=False).iloc[:10]

#zd

plt.figure(figsize=(10,10))

#plot = sns.countplot(x = topauthor.index, data = topauthor,palette="Set2")

#plt.hist(topauthor['title'],bins=10,color="orange")

ax = sns.barplot(topauthor['title'], topauthor.index, palette='Set2')

plt.scatter(x=topauthor["title"],y=topauthor.index)

plt.ylabel('author')

plt.xlabel('Number of books')
oldestbooks=df.sort_values('publication_date',ascending=True)

oldestbooks=oldestbooks.sort_values('publication_date', ascending = True).head(10).set_index('title')

oldestbooks

plt.figure(figsize=(15,10))

bx = sns.barplot(oldestbooks['average_rating'],oldestbooks.index, palette='Set2')

#plt.hist(oldestbooks['average_rating'],bins=10,color="orange")
ratedbooks=df.sort_values('average_rating',ascending=False)

rb=ratedbooks.groupby('average_rating')

plt.figure(figsize=(15,10))

plt.hist(ratedbooks["average_rating"],bins=35,color="orange")

#plt.scatter(x=ratedbooks["average_rating"])

#rb.head()