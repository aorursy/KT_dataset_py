import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

import altair as alt

print(os.listdir("../input"))
books_data = pd.read_csv('../input/books.csv', error_bad_lines=False)
books_data.head()
books_data.language_code.unique()
books_data['rating_value_count'] = np.log(books_data['average_rating'] * books_data['ratings_count']+1e-3)*books_data['average_rating']
np.min(books_data.rating_value_count), np.max(books_data.rating_value_count)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)



ax1.hist(books_data.rating_value_count, bins=20)



ax2.hist(books_data.average_rating, bins=20)



plt.show()
books_data.sort_values(by=['average_rating', 'ratings_count'], ascending=False).head()
books_data.sort_values(by=['rating_value_count'], ascending=False).head()
books_data.sort_values(by=['ratings_count'], ascending=False).head()
temp_sr = books_data.groupby(['language_code']).count()['bookID'].sort_values(ascending=False)

plt.figure(figsize=(10, 7))

sns.barplot(y=temp_sr.index, x=temp_sr.values)
books_data.columns
plt.figure(figsize=(10, 7))

sns.scatterplot(books_data['# num_pages'], books_data['ratings_count'], hue=books_data['average_rating'])

plt.show()
books_data.head()
authors = books_data.authors.map(lambda x: x.split('-'))

langs = books_data.language_code
auth_lang = list(map(lambda tuple_auths_lang: 

                     list(map(lambda i: (i, tuple_auths_lang[1]), tuple_auths_lang[0])),

                     zip(authors, langs)))
auth_lang[:10]
flat_list = []

for sublist in auth_lang:

    for item in sublist:

        flat_list.append(item)
flat_list = set(flat_list)
len(flat_list)
authors = list(map(lambda i: i[0], flat_list))

langs = list(map(lambda i: i[1], flat_list))

df_auth_lang = pd.DataFrame({'authors': authors, 'langs': langs})
df_auth_lang.head()
df_auth_lang.groupby(['authors']).agg('count').sort_values(by=['langs'], ascending=False).head(20)
books_data.head()