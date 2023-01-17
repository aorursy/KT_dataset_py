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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

books = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines = False)
print("Dataset contains {} row and {} colums".format(books.shape[0],books.shape[1]))
books.head()


books.replace(to_replace = 'J.K. Rowling/Mary GrandPré', value= 'J.K Rowling',inplace = True)
books.head()
import seaborn as sns
sns.set_context('poster')
plt.figure(figsize = (20,15))
book = books['title'].value_counts()[:20]
rating = books.average_rating[:20]
sns.barplot(x=book, y = book.index, palette = 'deep')
plt.title("Most occuring books")
plt.xlabel("number of occurances")
plt.ylabel("books")
plt.show()
sns.set_context('paper')
plt.figure(figsize =(15,10))
ax = books.groupby('language_code')['title'].count().plot.bar()
plt.title('Language code')
plt.xticks(fontsize = 15)
most_rated = books.sort_values('ratings_count', ascending = False).head(10).set_index('title')
plt.figure(figsize=(15,10))
sns.barplot(most_rated['ratings_count'], most_rated.index, palette = 'rocket')
sns.set_context('talk')
most_books = books.groupby('authors')['title'].count().reset_index().sort_values('title', ascending=False).head(10).set_index('authors')
plt.figure(figsize=(15,10))
ax = sns.barplot(most_books['title'], most_books.index, palette='icefire_r')
ax.set_title("Top 10 authors with most books")
ax.set_xlabel("Total number of books")
high_rated_author = books[books['average_rating']>=4.3]
high_rated_author = high_rated_author.groupby('authors')['title'].count().reset_index().sort_values('title', ascending = False).head(10).set_index('authors')
plt.figure(figsize=(15,10))
ax = sns.barplot(high_rated_author['title'], high_rated_author.index, palette = 'Set2')
ax.set_xlabel("Number of books")
ax.set_ylabel("Authors")

def segregation(data):
    values=[]
    for val in data.average_rating:
        if val>=0 and val<=1:
            values.append("between 0 and 1")
        elif val>1 and val<=2:
            values.append("between 1 and 2")
        elif val>2 and val<=3:
            values.append("between 2 and 3")
        elif val>3 and val<=4:
            values.append("between 3 and 4")
        elif val>4 and val<=5:
            values.append("between 4 and 5")
        else:
            values.append("NaN")
    print(len(values))
    return values
books.average_rating.isnull().value_counts()
books.dropna(0, inplace=True)
plt.figure(figsize=(10,10))
rating = books.average_rating.astype(float)
sns.distplot(rating, bins =20)
books['Ratings_Dist'] = segregation(books)
ratings_pie = books['Ratings_Dist'].value_counts().reset_index()
labels = ratings_pie['index']
colors = ['lightblue','darkmagenta','coral','bisque', 'black']
percent = 100.*ratings_pie['Ratings_Dist']/ratings_pie['Ratings_Dist'].sum()
fig, ax1 = plt.subplots()
ax1.pie(ratings_pie['Ratings_Dist'],colors = colors, 
        pctdistance=0.85, startangle=90, explode=(0.05, 0.05, 0.05, 0.05, 0.05))
#Draw a circle now:
centre_circle = plt.Circle((0,0), 0.70, fc ='white')
fig1 = plt.gcf()
fig1.gca().add_artist(centre_circle)
#Equal Aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')
plt.tight_layout()
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(labels, percent)]
plt.legend( labels, loc = 'best',bbox_to_anchor=(-0.1, 1.),)
