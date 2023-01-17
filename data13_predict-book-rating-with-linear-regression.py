# import libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn import metrics

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
%matplotlib inline

sns.set(style = 'darkgrid')

sns.set_palette('deep')
# read the data

data = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines = False)
# show the first few rows

data.head(5)
# check basic features and data types

data.info()
# check no. of records

len(data)
# check for doublications

data.duplicated().any()
sns.heatmap(data.isnull(), cmap='viridis')
# ratings distribution

sns.kdeplot(data['average_rating'], shade = True)

plt.title('Rating Distribution\n')

plt.xlabel('Rating')

plt.ylabel('Frequency')
# top 5 languages

data['language_code'].value_counts().head(5).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()
# number of books per rating

sns.barplot(data['average_rating'].value_counts().head(15).index, data['average_rating'].value_counts().head(15))

plt.title('Number of Books Each Rating Received\n')

plt.xlabel('Ratings')

plt.ylabel('Counts')

plt.xticks(rotation=45)
# highest rated books

popular_books = data.nlargest(10, ['ratings_count']).set_index('title')['ratings_count']

sns.barplot(popular_books, popular_books.index)
# highest reviewed books

highest_reviews = data.nlargest(10, ['text_reviews_count'])

sns.barplot(highest_reviews['text_reviews_count'], highest_reviews['title'])
# top 10 books under 200 pages for busy book lovers

under200 = data[data['# num_pages'] <= 200]

top10under200 = under200.nlargest(10, ['ratings_count'])

sns.barplot(top10under200['ratings_count'], top10under200['title'], hue=top10under200['average_rating'])

plt.xticks(rotation=15)
# top 10 longest books

longest_books = data.nlargest(10, ['# num_pages']).set_index('title')

sns.barplot(longest_books['# num_pages'], longest_books.index)
# top languages

data['language_code'].value_counts().plot(kind='bar')

plt.title('Most Popular Language')

plt.ylabel('Counts')

plt.xticks(rotation = 90)
# top published books

sns.barplot(data['title'].value_counts()[:15], data['title'].value_counts().index[:15])

plt.title('Top Published Books')

plt.xlabel('Number of Publications')
# authors with highest rated books

plt.figure(figsize=(10, 5))

authors = data.nlargest(5, ['ratings_count']).set_index('authors')

sns.barplot(authors['ratings_count'], authors.index, ci = None, hue = authors['title'])

plt.xlabel('Total Ratings')
# authors with highest publications

top_authors = data['authors'].value_counts().head(9)

sns.barplot(top_authors, top_authors.index)

plt.title('Authors with Highest Publication Count')

plt.xlabel('No. of Publications')
# visualise a bivariate distribution between ratings & no. of pages

sns.jointplot(x = 'average_rating', y = '# num_pages', data = data)
# visualise a bivariate distribution between ratings & no. of reviews

sns.jointplot(x = 'average_rating', y = 'text_reviews_count', data = data)
# find no. of pages outliers

sns.boxplot(x=data['# num_pages'])
# remove outliers from no. of pages 

data = data.drop(data.index[data['# num_pages'] >= 1000])
# find ratings count outliers

sns.boxplot(x=data['ratings_count'])
# remove outliers from ratings_count

data = data.drop(data.index[data['ratings_count'] >= 1000000])
# remove outliers from text_reviews_count

data = data.drop(data.index[data['text_reviews_count'] >= 20000])
# encode title column

le = preprocessing.LabelEncoder()

data['title'] = le.fit_transform(data['title'])
# encode authors column

data['authors'] = le.fit_transform(data['authors'])
# encode language column

enc_lang = pd.get_dummies(data['language_code'])

data = pd.concat([data, enc_lang], axis = 1)
# divide the data into attributes and labels

X = data.drop(['average_rating', 'language_code', 'isbn'], axis = 1)

y = data['average_rating']
# split 80% of the data to the training set and 20% of the data to test set 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)
lr = LinearRegression()

lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
pred = pd.DataFrame({'Actual': y_test.tolist(), 'Predicted': predictions.tolist()}).head(25)

pred.head(10)
# visualise the above comparison result

pred.plot(kind='bar', figsize=(13, 7))
# evaluate the performance of the algorithm

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))