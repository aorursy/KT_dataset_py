import re

import os



try:

    os.remove('/kaggle/working/books_clean.csv')

except OSError:

    pass



pattern = re.compile(r'^(?P<bookID>\d+),(?P<title>.+?(?=,)),(?P<authors>.+(?=,\d+\.))')



wrote_headers = False



with open('../input/goodreadsbooks/books.csv', 'r') as books_file:

  for line in books_file:

    if (not wrote_headers):

      with open('/kaggle/working/books_clean.csv', 'a') as clean_file:

        clean_file.write(line)

      wrote_headers = True



    found = re.match(pattern, line)



    if (found):

      quotified = pattern.sub(r'\g<bookID>,\g<title>,"\g<authors>"', line)

      with open('/kaggle/working/books_clean.csv', 'a') as clean_file:

        clean_file.write(quotified)



print("OK -- Generated /kaggle/working/books_clean.csv")
import pandas as pd



df = pd.read_csv('/kaggle/working/books_clean.csv', encoding = 'utf-8')



df.head()
len(df[df['title'].isna()].index)
len(df[df['text_reviews_count'].isna()].index)
max_reviews_count = df['text_reviews_count'].max()

most_reviewed_book = df[df['text_reviews_count'] == max_reviews_count].iloc[0]

print(f'The book with the highest reviews count is\n\t{most_reviewed_book["title"]}\nwith {max_reviews_count} reviews')
max_rating = df['average_rating'].max()

best_rated_book = df[df['average_rating'] == max_rating].iloc[0]

print(f'The book with the highest rating is\n\t{best_rated_book["title"]}\n with a score of {max_rating}')
rating_sorted_df = df.sort_values('average_rating', ascending = False)

rating_grouping = rating_sorted_df.groupby('average_rating', sort = False)

best_group = list(rating_grouping)[0]

best_rating, group_df = best_group

best_book = group_df.sort_values('text_reviews_count', ascending = False).iloc[0]



print(f'The book with the highest rating and number of text_reviews is\n\t{best_book["title"]}\n with a score of {max_rating} and {best_book["text_reviews_count"]} reviews')
import numpy as np

import matplotlib.pyplot as plt



rating_classes = np.array(range(0, 11)) * 0.5



df['rating_ranges'] = pd.cut(df['average_rating'], bins = rating_classes)



diagram = df['rating_ranges'].value_counts().sort_index().plot(kind = 'bar')

diagram.set_xlabel('Rating Groups')

diagram.set_ylabel('Number of Books')

diagram.set_title('Frequency of books by rating groups')

plt.show()



df['rating_ranges'].value_counts()
counts_df = group_df['text_reviews_count'].value_counts()

diagram = counts_df.sort_index().plot(kind = 'bar', title = 'Number of books with a 5.0 rating, by review count')

diagram.set_xlabel('Number of Reviews')

diagram.set_ylabel('Number of Books')

diagram.set_xticklabels(counts_df.index, rotation = 0)

plt.show()



group_df['text_reviews_count'].value_counts()