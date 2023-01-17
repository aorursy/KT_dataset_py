import pandas as pd
books_data = pd.read_csv('../input/goodreadsbooks/books.csv', error_bad_lines = False, index_col = 'bookID')
books_data.head()
books_data.isna().sum().sum()
# Answer to question 1

num_unique_authors = books_data['authors'].nunique()

print('Number of authors = ', num_unique_authors)
# Answer to question 2

# frequency of top 10 frequent authors

books_data['authors'].value_counts()[:10]
import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("dark")
# Answer to question 3

sns.distplot(a = books_data['# num_pages'], kde = False, bins = 2000)

plt.title('Distribution of number of pages')
books_with_many_ratings = books_data.sort_values('ratings_count', 

                                                 ascending = False).head(10).set_index('title')

plt.figure(figsize=(15,10))

sns.barplot(books_with_many_ratings['ratings_count'], books_with_many_ratings.index )
books_data['average_rating'].describe()
top_rated_books = books_data[books_data['average_rating'] > 4.5]

print('Number of top-rated books = ', top_rated_books.shape[0])

print('Top rated books are: ', top_rated_books['title'])
plt.figure(figsize=(6,4))

sns.barplot(y = top_rated_books['title'][:5], 

            x = top_rated_books['average_rating'][:5], palette = 'dark')

plt.show()
plt.figure(figsize=(10, 6))

sns.set_style("dark")

sns.barplot(y = top_rated_books['authors'][:10], 

            x = top_rated_books['average_rating'][:10], palette = 'dark')

plt.show()
sns.catplot(x = 'average_rating', y = 'language_code', 

            kind = 'bar', data = books_data, height = 10)

plt.show()
books_data[books_data['language_code'] == 'wel']
books_data['ratings_count'].describe()
mean_ratings_count = books_data['ratings_count'].describe()[1]
books_data_with_sufficient_ratings = books_data[books_data['ratings_count'] >= mean_ratings_count]
sns.catplot(x = 'average_rating', y = 'language_code'

            , kind = 'bar', data = books_data_with_sufficient_ratings, height = 5)

plt.show()
mean_rating = books_data_with_sufficient_ratings['average_rating'].describe()[1]
books_with_low_rating = books_data_with_sufficient_ratings[

    books_data_with_sufficient_ratings['average_rating'] < mean_rating]
books_with_low_rating.head()
sns.catplot(x = 'average_rating', y = 'language_code',

            kind = 'bar', data = books_with_low_rating, height = 5)

plt.show()
books_data[books_data['authors'] == 'Charles Dickens'].shape[0]
books_data[books_data['authors'] == 'Charles Dickens'][:3]
sns.distplot(a = books_data[books_data['authors'] == 'Charles Dickens']['average_rating'], kde = False)

plt.show()
books_data[books_data['authors'] == 'J.K. Rowling'][:3]
sns.distplot(a = books_data[books_data['authors'] == 'J.K. Rowling']['average_rating'], kde = False)

plt.show()
ax = sns.jointplot(x = "# num_pages", y = "average_rating", data = books_data)

ax.set_axis_labels("Number of Pages", "Average Rating")
books_with_reasonable_num_pages = books_data[books_data['# num_pages'] <= 1000]
ax = sns.jointplot(x = "# num_pages", y = "average_rating", 

                   data = books_with_reasonable_num_pages)

ax.set_axis_labels("Number of Pages", "Average Rating")
books_with_no_reviews = books_data[books_data['text_reviews_count'] == 0]

books_with_reviews = books_data[books_data['text_reviews_count'] > 0]
ax = sns.jointplot(x = "text_reviews_count", y = "average_rating", data = books_with_no_reviews)

ax.set_axis_labels("Number of text reviews", "Average Rating")

plt.title('Books with no reviews')
books_with_no_reviews.shape[0]/books_data.shape[0] * 100
ax = sns.jointplot(x = "text_reviews_count", y = "average_rating", data = books_with_reviews)

ax.set_axis_labels("Number of text reviews", "Average Rating")

plt.title('Books with reviews')
books_with_reviews.shape[0]/books_data.shape[0] * 100
books_with_atmost_200_pages = books_data[books_data['# num_pages'] <= 200]

best_books_with_atmost_200_pages = books_with_atmost_200_pages.nlargest(10, ['ratings_count'])
sns.barplot(best_books_with_atmost_200_pages['ratings_count'],

            best_books_with_atmost_200_pages['title'], 

            hue = best_books_with_atmost_200_pages['average_rating'])

plt.xticks(rotation=25)

plt.title('Top 10 books with <=200 pages')
big_books = books_data[books_data['# num_pages'] >= 1000]

best_big_books = big_books.nlargest(10, ['ratings_count'])
sns.barplot(best_big_books['ratings_count'],

            best_big_books['title'], 

            hue = best_big_books['average_rating'])

plt.xticks(rotation=25)

plt.title('Top 10 books with more >=1000 pages')