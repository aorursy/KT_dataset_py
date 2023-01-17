import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud



import os
books = pd.read_csv('../input/books.csv', error_bad_lines=False, index_col='bookID')
books.head()
books.info()
books[['average_rating', '# num_pages', 'ratings_count', 'text_reviews_count']].describe()
print('weighted(rating_count) mean ratings: ', np.average(a=books.average_rating, axis=0, weights=books.ratings_count))
fg, ax = plt.subplots(1,2, figsize=(10,10))



sns.boxplot(y=books['average_rating'], data=books, ax=ax[0], color='g')

ax[0].set_title('Average Rating')



sns.boxplot(y=books['# num_pages'], data=books, ax=ax[1], color='r')

ax[1].set_title('Number of Pages')



plt.show()
valid = books[(books['# num_pages'] > 0) & (books['# num_pages'] < 2000)]

print(len(valid)/len(books)*100, '% books')
fg, ax = plt.subplots(1,2, figsize=(10,10))



sns.boxplot(y=valid['average_rating'], data=valid, ax=ax[0], color='g')

ax[0].set_title('Average Rating')



sns.boxplot(y=valid['# num_pages'], data=valid, ax=ax[1], color='r')

ax[1].set_title('Number of Pages')



plt.show()
fg, ax = plt.subplots(1,2, figsize=(20,10))



sns.distplot(valid['average_rating'], ax=ax[0], color='g')

ax[0].set_title('Average Rating')



sns.distplot(valid['# num_pages'], ax=ax[1], color='r')

ax[1].set_title('Number of Pages')



plt.show()
plt.figure(figsize=(10,10))

sns.kdeplot(valid.average_rating, valid['# num_pages'], cmap='Blues', shade=True, shade_lowest=True)

plt.show()
correlation = books[['average_rating', '# num_pages', 'ratings_count', 'text_reviews_count']].corr()

sns.heatmap(correlation, annot=True, vmax=1, vmin=-1, center=0)

plt.show()
books['language_code'].unique()
lang_freq_table = pd.DataFrame(books.language_code.value_counts())

lang_freq_table
lang_freq_table.plot(kind='pie', subplots=True, figsize=(10,10))

plt.show()
plt.figure(figsize=(20,10))

sns.boxplot(y=books.average_rating, x=books.language_code)

plt.show()
plt.figure(figsize=(20,10))

sns.boxplot(y=valid['# num_pages'], x=valid.language_code)

plt.show()
cons_lang = lang_freq_table[lang_freq_table['language_code']>10].index

lang_wise = books[books['language_code'].isin(cons_lang)].groupby('language_code').median()

lang_wise['book_count'] = lang_freq_table.loc[cons_lang]['language_code']
lang_wise[['book_count', 'average_rating', '# num_pages', 'ratings_count', 'text_reviews_count']]
lang_wise.sort_values('average_rating', ascending=False, inplace=True)

plt.figure(figsize=(20,10))

a = sns.barplot(x='average_rating', y=lang_wise.index, data=lang_wise, orient='h')

a.plot([3.96,3.96],[0, len(lang_wise)], linewidth=2)

plt.show()
lang_wise.sort_values('# num_pages', ascending=False, inplace=True)

plt.figure(figsize=(20,10))

a = sns.barplot(x='# num_pages', y=lang_wise.index, data=lang_wise, orient='h')

a.plot([301,301],[0, len(lang_wise)], linewidth=2)

plt.show()
lang_wise.sort_values('ratings_count', ascending=False, inplace=True)

plt.figure(figsize=(20,10))

a = sns.barplot(x='ratings_count', y=lang_wise.index, data=lang_wise, orient='h')

a.plot([630.5,630.5],[0, len(lang_wise)], linewidth=2)

plt.show()
lang_wise.sort_values('text_reviews_count', ascending=False, inplace=True)

plt.figure(figsize=(20,10))

a = sns.barplot(x='text_reviews_count', y=lang_wise.index, data=lang_wise, orient='h')

a.plot([40,40],[0, len(lang_wise)], linewidth=2)

plt.show()
new_dict = {}



for lang in cons_lang:

    df = books[books['language_code']==lang]

    wgt_mean_rating = np.average(a=df.average_rating, axis=0, weights=df.ratings_count)

    new_dict[lang] = wgt_mean_rating

wgt_mean = pd.DataFrame.from_dict(new_dict, orient='index', columns=['wgt_mean_rating'])
wgt_mean = wgt_mean.sort_values('wgt_mean_rating', ascending=False)

wgt_mean
plt.figure(figsize=(20,10))

a = sns.barplot(x='wgt_mean_rating', y=wgt_mean.index, data=wgt_mean, orient='h')

a.plot([4.024,4.024],[0, len(lang_wise)], linewidth=2)

plt.show()
text = ''.join(title for title in books.title)

wc = WordCloud(max_font_size=70, max_words=100, background_color='white').generate(text)

plt.figure(figsize=(16,10))

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.show()
text = ''.join(title for title in books.authors)

wc = WordCloud(max_font_size=70, max_words=100, background_color='white').generate(text)

plt.figure(figsize=(16,10))

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.show()
auth_freq_table = pd.DataFrame(books.authors.value_counts())
cons_auth = auth_freq_table[auth_freq_table['authors']>0].index

authors_wise = books[books['authors'].isin(cons_auth)].groupby('authors').mean()

authors_wise['book_count'] = auth_freq_table.loc[cons_auth]['authors']
authors_wise = authors_wise[['book_count', 'average_rating', '# num_pages', 'ratings_count', 'text_reviews_count']]
top_10_auth_book_count = authors_wise.sort_values(['book_count', 'average_rating'], ascending=False)[:12]

top_10_auth_book_count
top_10_authors = authors_wise[(authors_wise['book_count']>5) & (authors_wise['ratings_count'] > 1e5) ].sort_values(['average_rating', 'book_count', 'ratings_count', '# num_pages', 'text_reviews_count'], ascending=False)[:10]

top_10_authors
top_10_books = books[books['ratings_count']>1e5].sort_values(['average_rating', 'ratings_count', '# num_pages', 'text_reviews_count'], ascending=False)[:11]

top_10_books