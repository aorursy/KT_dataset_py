# Importing the basic libraries for analysis and visualization



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('../input/goodreadsbooks/books.csv', error_bad_lines = False)
# check the erroneous lines in the csv file, used error_bad_lines parameter above to avoid that line.

# file = open('../input/goodreadsbooks/books.csv')

# i = 0

# for line in file:

#     i += 1

#     if i == 4012:

#         print(line)
df.head()

df.describe()
df.info()
# renaming #num_pages column

df.rename(columns = {"# num_pages" : "num_pages"}, inplace = True)

df.replace(to_replace = "J.K. Rowling-Mary GrandPré", value = "J.K. Rowling", inplace = True)

df.head()
# Getting unique authors



authors = df.authors.unique()

print(authors)

len(authors)

# Plotting Top authors in terms of number of books written



books_per_author = df.groupby(['authors']).count()[['title']].nlargest(15, 'title')

print(books_per_author)

# top_authors = books_per_author.nlargest(10, 'title')

# b = books_per_author.sort_values(by = 'title', ascending = False)

# print(top_authors.loc['Agatha Christie'])

# sample_df = books_per_author.sample(n=6)

# print(sample_df.head())

# print(books_per_author.iloc[3455:3460])

plt.figure(figsize = (20,5))

plt.title("Top authors based on number of books written")

plt.xlabel("Author names")

plt.ylabel("Number of books written")

plt.bar(books_per_author.index, books_per_author["title"])

plt.xticks(rotation = 45)
# Top 15 authors based on number of user ratings



num_of_ratings = df.groupby(['authors']).sum()[['ratings_count']].nlargest(15, 'ratings_count')

print(num_of_ratings)

plt.figure(figsize = (20, 5))

plt.title("Top 15 authors with most user ratings")

plt.xlabel("Author names")

plt.ylabel("Ratings_count (In millions)")

plt.bar(num_of_ratings.index, num_of_ratings['ratings_count'])

plt.xticks(rotation = 45)
df['mul_ratings'] = df.average_rating * df.ratings_count # adding column - mul_ratings

# df.head()

ratings_per_author = df.groupby(['authors']).sum().nlargest(15, 'ratings_count') # top 15 authors with most user ratings

ratings_per_author['avg_ratings_per_author'] =  ratings_per_author.mul_ratings / ratings_per_author.ratings_count

top_avg_ratings = ratings_per_author.nlargest(15, 'avg_ratings_per_author')



plt.figure(figsize = (20, 5))

plt.title("Top authors with maximum average ratings")

plt.xlabel("Author names")

plt.ylabel("Average ratings out of 5")

plt.bar(top_avg_ratings.index, top_avg_ratings['avg_ratings_per_author'])

plt.xticks(rotation = 45)