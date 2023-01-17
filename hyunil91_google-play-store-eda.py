# Dependencies



import re

import pandas as pd

import numpy as np

import matplotlib.patches as pat

import matplotlib.pyplot as plt

import matplotlib.animation as animation

import seaborn as sns

from datetime import datetime

from dateutil.parser import parse

from sklearn.feature_selection import SelectKBest, chi2, RFE

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import scale, PolynomialFeatures, Normalizer, MinMaxScaler, power_transform

from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, learning_curve

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer

from sklearn.utils import shuffle

from sklearn.svm import SVR

from xgboost import XGBClassifier

from pandas.plotting import register_matplotlib_converters



plt.style.use('ggplot')

register_matplotlib_converters()

%matplotlib inline
# Load the datasets

table = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

review = pd.read_csv('../input/google-play-store-apps/googleplaystore_user_reviews.csv')
table.head()
table  = table.drop(['App', 'Current Ver'], axis=1)
# Count by 'Category'

plt.figure(figsize=(10,10))

sns.barplot(x=table['Category'].value_counts(), y=table['Category'].value_counts().index, palette='Blues_d')

plt.show()
# Count by Genres

plt.figure(figsize=(10, 20))

sns.barplot(x=table['Genres'].value_counts(), y=table['Genres'].value_counts().index, palette='Blues_d')

plt.show()
table = table.drop('Genres', axis=1)

table.info()
plt.figure(figsize=(15, 5))

sns.distplot(table['Rating'].dropna(), bins=35)

plt.show()
table['Rating'].sort_values(ascending=False).head(10)
table[table['Rating'] == table['Rating'].max()]
# Drop index# 10472

table = table.drop(index=10472)
table = table.reset_index(drop=True)
table
# Check the distribution

plt.figure(figsize=(15, 5))

sns.distplot(table['Rating'].dropna(), bins=35)

plt.xticks(np.arange(0, 5.5, step=0.2))

plt.show()
# Remove unnecessary characters or words

table['Installs'] = table['Installs'].str.replace(',', '')

table['Installs'] = table['Installs'].str.replace('+', '')

table['Price'] = table['Price'].str.replace('$', '')

table['Android Ver'] = table['Android Ver'].str.replace('and up', '')



# Change to datetime data type

table['Last Updated'] = pd.to_datetime(table['Last Updated'],format='%B %d, %Y')



# Change to proper data types

table = table.astype({'Category':'category', 'Reviews': 'int', 'Installs': 'int','Type': 'category','Price': 'float'})

table.info()
# Select missing rating data

missing_rating = table[table['Rating'].isnull()]

# Select the rest of the data excluding missing rating data

nmissing_rating = table.iloc[np.delete(table.index, missing_rating.index)]
# Let's see missing values in each category

nmissing = nmissing_rating['Category'].value_counts()

missing = missing_rating['Category'].value_counts().reindex(nmissing_rating['Category'].value_counts().index)





diff = missing /nmissing



total = table['Category'].value_counts().reset_index()

total = total.rename(columns={'Category': 'Total', 'index': 'Category'})



result = pd.concat([nmissing, missing, diff], axis=1)

result.columns = ['Not Missing', 'Missing', 'Difference in %']

result = result.reset_index()

result = result.rename(columns={'index': 'Category'})

result = result.join(total.set_index('Category'), on='Category')

result = result.set_index('Category')
result.head()
total = result['Total'].to_list()

missing = result['Missing'].to_list()

labels = result.index.to_list()



x = np.arange(len(labels))

width = 0.4



f, ax = plt.subplots(figsize=(10, 13))

bar1 = ax.barh(x + width/2, total[::-1],  width, label='Total')

bar2 = ax.barh(x - width/2, missing[::-1], width, label='Missing')



ax.set_yticks(x)

ax.set_yticklabels(labels[::-1])

f.tight_layout()

plt.legend(loc='upper right', fontsize='large')

plt.show()
# Getting number of review counts in descending order

print('Not missing:','\n', nmissing_rating['Reviews'].sort_values(ascending=False).head(10), '\n',

      '\n', 'Missing:','\n',missing_rating['Reviews'].sort_values(ascending=False).head(10))
# Getting general statistics for 'Reviews'

print('Not missing:','\n', nmissing_rating['Reviews'].describe().apply(lambda x: format(round(x,2), 'f')), '\n',

      '\n', 'Missing:','\n',missing_rating['Reviews'].describe().apply(lambda x: format(round(x,2), 'f')))
# Comparision in Reviews

f, ax = plt.subplots(5,1, figsize=(10, 13))

axes = [0, 1, 2, 3, 4, 5]



for i in range(0,5):

    sns.distplot(np.log(nmissing_rating[nmissing_rating['Category'] == result.sort_values('Difference in %', ascending=False).index[i]]['Reviews']+1),\

                 hist=False, kde_kws={'shade': True}, ax=ax[axes[i]], label='Not Missing')

    sns.distplot(np.log(missing_rating[missing_rating['Category'] == result.sort_values('Difference in %', ascending=False).index[i]]['Reviews']+1),\

                 hist=False, kde_kws={'shade': True},ax=ax[axes[i]], label='Missing').set_title(result.sort_values('Difference in %', ascending=False).index[i] + ' - Missing vs Not Missing')

    

plt.tight_layout()

plt.legend()

plt.show()
# Getting number of installs in descending order

print('Not missing:','\n', nmissing_rating['Installs'].sort_values(ascending=False).head(10), '\n',

      '\n', 'Missing:','\n',missing_rating['Installs'].sort_values(ascending=False).head(10))
# Getting general statistics for 'Installs'

print('Not missing:','\n', nmissing_rating['Installs'].describe().apply(lambda x: format(round(x,2), 'f')), '\n',

      '\n', 'Missing:','\n',missing_rating['Installs'].describe().apply(lambda x: format(round(x,2), 'f')))
# Comparision in Installs

f, ax = plt.subplots(5,1, figsize=(10, 13))

axes = [0, 1, 2, 3, 4, 5]



for i in range(0,5):

    sns.distplot(np.log(nmissing_rating[nmissing_rating['Category'] == result.sort_values('Difference in %', ascending=False).index[i]]['Installs']+1),\

                 hist=False, kde_kws={'shade': True}, ax=ax[axes[i]], label='Not Missing')

    sns.distplot(np.log(missing_rating[missing_rating['Category'] == result.sort_values('Difference in %', ascending=False).index[i]]['Installs']+1),\

                 hist=False, kde_kws={'shade': True},ax=ax[axes[i]], label='Missing').set_title(result.sort_values('Difference in %', ascending=False).index[i] + ' - Missing vs Not Missing')

    

plt.tight_layout()

plt.legend()

plt.show()
plt.figure(figsize=(8,8))



sns.distplot(nmissing_rating[nmissing_rating['Reviews'] <= 3248]['Reviews'], kde=True, hist=False,label='Not Missing')

sns.distplot(missing_rating['Reviews'], kde=True, hist=False, label='Missing')



plt.legend()

plt.show()
plt.figure(figsize=(8,8))



sns.distplot(nmissing_rating[nmissing_rating['Installs'] <= 1000000]['Installs'], kde=True, hist=False,label= 'Not Missing')

sns.distplot(missing_rating['Installs'], kde=True, hist=False,label='Missing')



plt.legend()

plt.show()
# Getting 'Price' in descending order

print('Not missing:','\n', nmissing_rating['Price'].sort_values(ascending=False).head(10), '\n',

      '\n', 'Missing:','\n',missing_rating['Price'].sort_values(ascending=False).head(10))
# Getting general statistics for 'Price'

print('Not missing:','\n', nmissing_rating['Price'].describe().apply(lambda x: format(round(x,2), 'f')), '\n',

      '\n', 'Missing:','\n',missing_rating['Price'].describe().apply(lambda x: format(round(x,2), 'f')))
# Comparision in Installs

f, ax = plt.subplots(5,1, figsize=(10, 13))

axes = [0, 1, 2, 3, 4, 5]



for i in range(0,5):

    sns.distplot(np.log(nmissing_rating[nmissing_rating['Category'] == result.sort_values('Difference in %', ascending=False).index[i]]['Price']+1),\

                 hist=True, kde=False, ax=ax[axes[i]], label='Not Missing')

    sns.distplot(np.log(missing_rating[missing_rating['Category'] == result.sort_values('Difference in %', ascending=False).index[i]]['Price']+1),\

                 hist=True, kde=False, ax=ax[axes[i]], label='Missing').set_title(result.sort_values('Difference in %', ascending=False).index[i] + ' - Missing vs Not Missing')



plt.tight_layout()

plt.legend(loc='upper right')

plt.show()
# Dropping missing values

table = table.dropna()



table.info()
f,ax = plt.subplots(1,2, figsize=(15, 6))



ax1 = sns.barplot(x=nmissing_rating['Type'].value_counts().index, y=nmissing_rating['Type'].value_counts(), ax=ax[0]).set_title('Not Missing')

ax2 = sns.barplot(x=missing_rating['Type'].value_counts().index, y=missing_rating['Type'].value_counts(), ax=ax[1]).set_title('Missing')



plt.show()
missing_rating['Type'].value_counts()[1] / missing_rating['Type'].value_counts()[0]
nmissing_rating['Type'].value_counts()[1] / nmissing_rating['Type'].value_counts()[0]
table.head()
# For better comparision, normalize continuous features.

trans_table = table.copy()



trans_table['Reviews'] = trans_table['Reviews'].apply(np.log)

trans_table['Installs'] = trans_table['Installs'].apply(np.log)
trans_table = trans_table.drop(['Type', 'Price'], axis=1)

trans_table['Size'] = trans_table['Size'].str.replace('M', '')
trans_table['Size']
trans_table['Size'] = trans_table['Size'].str.replace('k', '')
# Thinking about how to deal with 'Varies with device'

sizeinfo = trans_table[trans_table['Size'] != 'Varies with device'].loc[:,'Size'].astype('float')
print(' Mean: ', sizeinfo.mean(), '\n', 'Median: ', sizeinfo.median(), '\n', 'Mode: ', sizeinfo.mode()[0])
plt.figure(figsize=(10,6))



sns.boxplot(sizeinfo)



plt.show()
trans_table['Size'] = trans_table['Size'].str.replace('Varies with device', str(sizeinfo.median()))

trans_table['Size'] = trans_table['Size'].astype('float')

# Big gap

trans_table['Size'] = np.log(trans_table['Size'])
sns.pairplot(trans_table, hue='Category', diag_kind='kde', markers=".", plot_kws={'alpha': 0.6, 's':80, 'edgecolor': 'k'}, height=4)

plt.show()
# Drop 'Installs'

trans_table = trans_table.drop('Installs', axis=1)
"""

# Create an animation that shows the difference relationships between 'Reviews' and 'Rating' based on 'Reviews' values

review_range = list(np.arange(0, 19, 0.1))



writer_init = animation.writers['ffmpeg']

writer = writer_init(fps=30, metadata=dict(artist='Hyunil Yoo'), bitrate=1800)



f = plt.figure(figsize=(10, 10))

 

def animate(i):

    plt.cla()

    above = trans_table[trans_table['Reviews'] > review_range[i]]

    below = trans_table[trans_table['Reviews'] <= review_range[i]]

    sns.scatterplot(x=above['Rating'], y=above['Reviews'],alpha= 0.6)

    sns.regplot(x=below['Rating'], y=below['Reviews'], scatter_kws={'alpha': 0.6, 's':80, 'edgecolor': 'k'})

    plt.xlim(0, 19)

    plt.ylim(0, 5)

    

ani = animation.FuncAnimation(f, animate, frames=len(review_range), repeat=True)

ani.save('review_rating_3.mp4', writer=writer)

"""
# For axes

zero = [0] * 11

one = [1] * 11

two = [2] * 11



flat_colax = zero + one + two



rowax = [0 ,1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

rowax = rowax *3
sns.set(font_scale=1.5)

f, ax = plt.subplots(11, 3, figsize=(15, 40))



category = trans_table['Category'].value_counts().index



for i in range(0,33):

    sns.regplot(trans_table[trans_table['Category'] == category[i]]['Rating'], \

                trans_table[trans_table['Category'] == category[i]]['Reviews'], \

                color='b',ax=ax[rowax[i], flat_colax[i]]).set_title(category[i])



plt.tight_layout()

plt.show()



sns.set(font_scale=1)

plt.style.use('ggplot')
"""

# Create an animation that shows the difference relationships between 'Size' and 'Rating' based on 'Size' values

size_range = list(np.arange(0, 8, 0.1))



writer_init = animation.writers['ffmpeg']

writer = writer_init(fps=30, metadata=dict(artist='Hyunil Yoo'), bitrate=1800)



f = plt.figure(figsize=(10, 10))

 

def animate(i):

    plt.cla()

    above = trans_table[trans_table['Size'] > size_range[i]]

    below = trans_table[trans_table['Size'] <= size_range[i]]

    sns.scatterplot(above['Rating'], above['Size'], alpha= 0.6)

    sns.regplot(below['Rating'], below['Size'], scatter_kws={'alpha': 0.6, 's':80, 'edgecolor': 'k'})

    plt.xlim(0, 5)

    plt.ylim(0, 8)

    

ani = animation.FuncAnimation(f, animate, frames=len(size_range), repeat=True)

ani.save('size_rating.mp4', writer=writer)

"""
sns.set(font_scale=1.5)

f, ax = plt.subplots(11, 3, figsize=(15, 40))



category = trans_table['Category'].value_counts().index



for i in range(0,33):

    sns.regplot(trans_table[trans_table['Category'] == category[i]]['Rating'], y=trans_table[trans_table['Category'] == category[i]]['Size'], color='b',ax=ax[rowax[i], flat_colax[i]]).set_title(category[i])



plt.tight_layout()

plt.show()



sns.set(font_scale=1)

plt.style.use('ggplot')
# Drop 'Size'

trans_table = trans_table.drop('Size', axis=1)
trans_table['Last Updated'].max()
plt.figure(figsize=(10,10))



sns.scatterplot(x=trans_table['Last Updated'], y=trans_table['Rating'], hue=trans_table['Category'])

plt.tight_layout()

plt.show()
# Drop Last Update

trans_table = trans_table.drop('Last Updated', axis=1)
version = trans_table['Android Ver'].to_list()



for i in range(len(version)):

    version[i] = re.sub('(.[^.]).', r'\1', version[i])



version = pd.Series(version)

version_stats = version.where(version != 'Vaie wthdeic').dropna()
version_stats[version_stats == '4.03  71.1']

version[338] = version[338][:4]

version[340] = version[340][:4]

version_stats[338] = version[338][:4]

version_stats[340] = version[340][:4]
version[version == '7.0- .11']
version[1490] = version[1490][:3]

version_stats[1490] = version_stats[1490][:3]
version[version == '5.0- .0']
version[2702] = version[2702][:3]

version[3333] = version[3333][:3]

version[6759] = version[6759][:3]

version_stats[2702] = version_stats[2702][:3]

version_stats[3333] = version_stats[3333][:3]

version_stats[6759] = version_stats[6759][:3]
version[version == '4.1- .11']
version[3993] = version[3993][:3]

version_stats[3993] = version_stats[3993][:3]
version_stats = version_stats.astype('float')



plt.figure(figsize=(10,8))

sns.distplot(version_stats, hist=False, kde_kws={'shade': True})

plt.show()
print(' Mean: ', version_stats.mean(), 

      '\n', 'Median: ', version_stats.median(), 

      '\n', 'Mode: ', version_stats.mode()[0])
version = version.str.replace('Vaie wthdeic', str(version_stats.median()))

version = version.astype('float')
plt.figure(figsize=(10,8))

sns.distplot(version, hist=False, kde_kws={'shade': True})

plt.show()
trans_table['Android Ver'] = version.to_list()
plt.figure(figsize=(10,8))

sns.regplot(x=trans_table['Android Ver'], y=trans_table['Rating'])

plt.show()
sns.set(font_scale=1.5)

f, ax = plt.subplots(11, 3, figsize=(15, 40))



category = trans_table['Category'].value_counts().index



for i in range(0,33):

    sns.regplot(x=trans_table[trans_table['Category'] == category[i]]['Android Ver'], y=trans_table[trans_table['Category'] == category[i]]['Rating'], color='b',ax=ax[rowax[i], flat_colax[i]]).set_title(category[i])



plt.tight_layout()

plt.show()



sns.set(font_scale=1)

plt.style.use('ggplot')
f, ax = plt.subplots(11, 3, figsize=(15, 40))



category = trans_table['Category'].value_counts().index



for i in range(0,33):

    sns.violinplot(x=trans_table[trans_table['Category'] == category[i]]['Rating'], y=trans_table[trans_table['Category'] == category[i]]['Content Rating'] , ax=ax[rowax[i], flat_colax[i]]).set_title(category[i])



plt.tight_layout()

plt.show()
plt.figure(figsize=(8,8))



sns.violinplot(x=table['Rating'], y=table['Content Rating'])



plt.show()
# There is not a good way to fill missing data

review = review.dropna()
table_f_cat = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

table_f_cat = table_f_cat[['App', 'Category','Rating']]

table_f_cat.head()
review.head()
review = review.join(table_f_cat.set_index('App'), on='App')

review.head()
review = review.drop_duplicates()

review.info()
review = review.dropna()
plt.figure(figsize=(10,15))



sns.barplot(x=review['Category'].value_counts(), y=review['Category'].value_counts().index, palette='Blues_d')



plt.xlabel('Number of reviews')

plt.show()
from textblob import TextBlob

TextBlob(review['Translated_Review'][0]).sentiment
review['App'].value_counts()
review['Sentiment'].value_counts()
plt.figure(figsize=(8,8))



sns.barplot(review['Sentiment'].value_counts().index, review['Sentiment'].value_counts())



plt.show()
f, ax = plt.subplots(11, 3, figsize=(15, 40))



category = review['Category'].value_counts().index



for i in range(0,33):

    sns.barplot(review[review['Category'] == category[i]]['Sentiment'].value_counts().index, review[review['Category'] == category[i]]['Sentiment'].value_counts(), ax=ax[rowax[i], flat_colax[i]]).set_title(category[i])



plt.tight_layout()

plt.show()
len(review[review['Rating'] >= 4.0])
len(review[review['Rating'] < 4.0])
word_count = []



for i in range(len(review)):

    word_count.append(len(TextBlob(review['Translated_Review'].iloc[i]).words))

    

review['word_count'] = word_count
plt.figure(figsize=(15,8))



sns.distplot(review[review['Sentiment'] == 'Positive'].loc[:, 'word_count'], hist=False, kde_kws={'shade':True}, label='Postive')

sns.distplot(review[review['Sentiment'] == 'Negative'].loc[:, 'word_count'], hist=False, kde_kws={'shade':True}, label='Negative')

sns.distplot(review[review['Sentiment'] == 'Neutral'].loc[:, 'word_count'], hist=False, kde_kws={'shade':True}, label='Neutral')



plt.legend()

plt.show()
plt.figure(figsize=(15,8))



sns.distplot(np.log(review[review['Sentiment'] == 'Positive'].loc[:, 'word_count']), hist=False, kde_kws={'shade':True}, label='Postive')

sns.distplot(np.log(review[review['Sentiment'] == 'Negative'].loc[:, 'word_count']), hist=False, kde_kws={'shade':True}, label='Negative')

sns.distplot(np.log(review[review['Sentiment'] == 'Neutral'].loc[:, 'word_count']), hist=False, kde_kws={'shade':True}, label='Neutral')



plt.legend()

plt.show()
# Code from https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a

# Removing stop words



def get_top_n_words(corpus, n=None):

    vec = CountVectorizer(stop_words = 'english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
common_words_positive = get_top_n_words(review[review['Sentiment'] == 'Positive']['Translated_Review'], 20)

positive_words = pd.DataFrame(common_words_positive, columns=['words', 'counts'])



common_words_negative = get_top_n_words(review[review['Sentiment'] == 'Negative']['Translated_Review'], 20)

negative_words = pd.DataFrame(common_words_negative, columns=['words', 'counts'])



common_words_neutral = get_top_n_words(review[review['Sentiment'] == 'Neutral']['Translated_Review'], 20)

neutral_words = pd.DataFrame(common_words_neutral, columns=['words', 'counts'])
f, ax = plt.subplots(3,1, figsize=(14,10))



ax1 = sns.barplot(x=positive_words['words'], y=positive_words['counts'], ax=ax[0])

ax2 = sns.barplot(x=negative_words['words'], y=negative_words['counts'], ax=ax[1])

ax3 = sns.barplot(x=neutral_words['words'], y=neutral_words['counts'], ax=ax[2])



ax1.tick_params(axis='x', rotation=30, labelsize=15)

ax2.tick_params(axis='x', rotation=30, labelsize=15)

ax3.tick_params(axis='x', rotation=30, labelsize=15)



ax1.set_title('Positive')

ax2.set_title('Negative')

ax3.set_title('Neutral')



ax1.set_xlabel('')

ax2.set_xlabel('')

ax3.set_xlabel('')



plt.tight_layout()

plt.show()
rowax = list(np.arange(0,33))

rowax = rowax * 3



f, ax = plt.subplots(33, 3, figsize=(30, 80))



category = review['Category'].value_counts().index



for i in range(0, 33):

    common_words_positive = get_top_n_words(review[(review['Sentiment'] == 'Positive') & (review['Category'] == category[i])]['Translated_Review'], 20)

    positive_words = pd.DataFrame(common_words_positive, columns=['words', 'counts'])



    common_words_negative = get_top_n_words(review[(review['Sentiment'] == 'Negative') & (review['Category'] == category[i])]['Translated_Review'], 20)

    negative_words = pd.DataFrame(common_words_negative, columns=['words', 'counts'])



    common_words_neutral = get_top_n_words(review[(review['Sentiment'] == 'Neutral') & (review['Category'] == category[i])]['Translated_Review'], 20)

    neutral_words = pd.DataFrame(common_words_neutral, columns=['words', 'counts'])

    

    ax1 = sns.barplot(x=positive_words['words'], y=positive_words['counts'], ax=ax[rowax[i], 0])

    ax2 = sns.barplot(x=negative_words['words'], y=negative_words['counts'], ax=ax[rowax[i], 1])

    ax3 = sns.barplot(x=neutral_words['words'], y=neutral_words['counts'], ax=ax[rowax[i], 2])

    

    ax1.set_title('Positive ' + str(category[i]))

    ax2.set_title('Negative ' + str(category[i]))

    ax3.set_title('Neutral ' + str(category[i]))

    

    ax1.tick_params(axis='x', rotation=30, labelsize=15)

    ax2.tick_params(axis='x', rotation=30, labelsize=15)

    ax3.tick_params(axis='x', rotation=30, labelsize=15)





    ax1.set_xlabel('')

    ax2.set_xlabel('')

    ax3.set_xlabel('')



plt.tight_layout()

plt.show()
game = trans_table[trans_table['Category'] == 'GAME']

game_above_4 = trans_table[(trans_table['Rating'] >= 4.0) & (trans_table['Category'] == 'GAME')]

game_below_4 = trans_table[(trans_table['Rating'] < 4.0) & (trans_table['Category'] == 'GAME')]
plt.figure(figsize=(15,8))



sns.distplot(game_above_4.loc[:,'Reviews'], hist=False, kde_kws={'shade': True}, label='above 4')

sns.distplot(game_below_4.loc[:,'Reviews'], hist=False, kde_kws={'shade': True}, label='Under 4')



plt.legend()

plt.show()
f, ax = plt.subplots(1,2, figsize=(15, 8))



sns.regplot(game['Rating'], game['Reviews'], ax=ax[0])

sns.violinplot(game['Rating'], game['Content Rating'], ax=ax[1])



plt.tight_layout()

plt.show()
game_review = review[review['Category'] == 'GAME']



plt.figure(figsize=(6,6))

sns.barplot(game_review['Sentiment'].value_counts().index, game_review['Sentiment'].value_counts())

plt.show()
game_word_count = []



for i in range(len(game_review)):

    game_word_count.append(len(TextBlob(game_review['Translated_Review'].iloc[i]).words))
game_review['word_count'] = game_word_count
plt.figure(figsize=(15,8))



sns.distplot(game_review[game_review['Sentiment'] == 'Positive'].loc[:, 'word_count'], hist=False, \

             kde_kws={'shade':True}, label='Postive')

sns.distplot(game_review[game_review['Sentiment'] == 'Negative'].loc[:, 'word_count'], hist=False, \

             kde_kws={'shade':True}, label='Negative')

sns.distplot(game_review[game_review['Sentiment'] == 'Neutral'].loc[:, 'word_count'], hist=False, \

             kde_kws={'shade':True}, label='Neutral')



plt.legend()

plt.show()
common_words_positive = get_top_n_words(game_review[game_review['Sentiment'] == 'Positive']['Translated_Review'], 20)

common_words_negative = get_top_n_words(game_review[game_review['Sentiment'] == 'Negative']['Translated_Review'], 20)

common_words_neutral = get_top_n_words(game_review[game_review['Sentiment'] == 'Neutral']['Translated_Review'], 20)



positive_words = pd.DataFrame(common_words_positive, columns=['words', 'counts'])

negative_words = pd.DataFrame(common_words_negative, columns=['words', 'counts'])

neutral_words = pd.DataFrame(common_words_neutral, columns=['words', 'counts'])



# without 'game'

positive_words = positive_words.iloc[1:]

negative_words = negative_words.iloc[1:]

neutral_words = neutral_words.iloc[1:]
f, ax = plt.subplots(3,1, figsize=(14,10))



ax1 = sns.barplot(x=positive_words['words'], y=positive_words['counts'], ax=ax[0])

ax2 = sns.barplot(x=negative_words['words'], y=negative_words['counts'], ax=ax[1])

ax3 = sns.barplot(x=neutral_words['words'], y=neutral_words['counts'], ax=ax[2])



ax1.tick_params(axis='x', rotation=30, labelsize=15)

ax2.tick_params(axis='x', rotation=30, labelsize=15)

ax3.tick_params(axis='x', rotation=30, labelsize=15)



ax1.set_title('Positive')

ax2.set_title('Negative')

ax3.set_title('Neutral')



ax1.set_xlabel('')

ax2.set_xlabel('')

ax3.set_xlabel('')



plt.tight_layout()

plt.show()