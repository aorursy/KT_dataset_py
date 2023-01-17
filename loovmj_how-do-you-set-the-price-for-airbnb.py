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
import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display
cal = pd.read_csv('/kaggle/input/boston/calendar.csv')

lis = pd.read_csv('/kaggle/input/boston/listings.csv')

rev = pd.read_csv('/kaggle/input/boston/reviews.csv')
len(cal)
len(lis)
len(rev)
rev.isnull().sum()
cal.head()
lis.head()
lis.columns
# extract columns I expected to relate with prices

lis[['room_type', 'bathrooms', 'bedrooms', 'beds','square_feet', 'price', 'weekly_price', 'monthly_price']].head(10)
# the number of None data

lis[['accommodates', 'room_type', 'bathrooms', 'bedrooms', 'beds', 'square_feet', 'price', 'weekly_price', 'monthly_price']].isnull().sum()
max(lis.beds)
rev.head()
# check the type

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # show all contents in a seriese

    print(lis.dtypes)
# change the type for price

lis.price = lis.price.str.replace('$', '').str.replace(',', '').astype(float)
# check correlation between numerical variables

lis.describe()
# check  the distribution for numerical variables

lis.hist(bins=20, xlabelsize=20, ylabelsize=20, figsize=(40,40));
# select columns for conditions of price

selected_lis = lis[['accommodates', 'host_listings_count', 'host_total_listings_count', 'bathrooms', 'bedrooms', 

                   'beds', 'number_of_reviews', 'review_scores_accuracy', 'price', 'square_feet']]
# check correlation between extracted variables

plt.figure(figsize=(10,10))

sns.heatmap(selected_lis.corr(), annot=True, fmt='.2f');
# plot the price with scatter

# all properties

plt.figure(figsize=(15, 10));

sns.scatterplot(x='price', y='accommodates', hue='room_type', data=lis);

plt.xticks(rotation='vertical');

plt.title('Distribution of price')

plt.savefig('dist_price.jpg')
# take a look at when accommodates == 2 and room_type == 'Entire home/apt'

tmp_df = lis[(lis.accommodates == 2) & (lis.room_type == 'Entire home/apt')]
# descriptive statistics for price

tmp_df.price.describe()
# check the counts for each city

tmp_df.city.value_counts()
# show average and sd as bar

plt.figure(figsize=(6,4));

sns.barplot(x='city', y='price', data=tmp_df[['city', 'price']].sort_values(by='price', ascending=False), 

            ci="sd", color='orange');

plt.xticks(rotation='vertical');

plt.title('Price variation in cities');

plt.savefig('price_var_cities.jpg')
for i in lis.accommodates.unique():

    print('accommodates = ' + str(i), '\n')

    if len(lis[lis.accommodates == i]) >= 4:

        figsize = (30, 20)

    else:

        figsize = (15, 15)

    lis[lis.accommodates == i].hist(column='price', by=['city', 'room_type'], grid=True, figsize=figsize, bins=20,

                                   xlabelsize=10, ylabelsize=10);

    plt.show();

    plt.close();
# check the number of `available` for each listing

booked = cal[cal.available == 'f'].groupby('listing_id').available.count()

non_booked = cal[cal.available == 't'].groupby('listing_id').available.count()

booking_ratio = booked / (booked + non_booked)
more_booked = booking_ratio[booking_ratio >= 0.7].sort_values(ascending=False)

more_booked.head()
more_booked.describe()
less_booked = booking_ratio[booking_ratio <= 0.3].sort_values(ascending=False)

less_booked.head()
less_booked.describe()
more_booked_id = more_booked.index.to_list()
less_booked_id = less_booked.index.to_list()
def make_series_df(ids_list, target_column, df_return=True):

    '''

    by listing data, make dataframe for price with selected ids

    

    ids_list : list :  list for ids

    target_column : strings : the name of column

    df : bool : True->make and return a dataframe

                False-> make and return a series

    '''

    target_column_list = []

    df = pd.DataFrame()

    for sel_id in ids_list:

        if len(lis[lis.id == sel_id][target_column]) >= 1:

            target_column_list.append(lis[lis.id == sel_id][target_column].values[0])

        

    # when df_return = True    

    if df_return > 0:

        df[target_column] = target_column_list

        return df



    else:

        return pd.Series(target_column_list)
# make a df with more booked id

more_df = make_series_df(more_booked_id,'price', df_return=True)

more_df.head()
# make a df with less booked id

less_df = make_series_df(less_booked_id, 'price', df_return=True)

less_df.head()
more_df.describe()
less_df.describe()
sns.distplot(a=more_df['price'], bins=40, hist=False, color='orange', label='more 70 %');

sns.distplot(a=less_df['price'], bins=40, hist=False, color='gray', label='less 30 %');

plt.title('Price distribution')

plt.savefig('price_dist.jpg')
sns.distplot(a=more_df['price'], bins=40, hist=False, color='orange', label='more 70 %');

sns.distplot(a=less_df['price'], bins=40, hist=False, color='gray', label='less 30 %');

plt.xlim(200, 800);

plt.axvline(x=305, color='green', linestyle='--');

plt.title('Zoom up price distribution')

plt.savefig('intersection_price.jpg')
review_scores = lis[['id', 'review_scores_accuracy', 'review_scores_rating',

    'review_scores_checkin', 'review_scores_cleanliness',

    'review_scores_communication', 'review_scores_location',

    'review_scores_value']].dropna()
review_scores.head()
review_scores.describe()
# check the rating as sum

sns.distplot(a = review_scores.review_scores_rating.dropna(),kde=False, bins = 50);

plt.title('review scores for all data')

plt.savefig('all_scores.jpg')
# check well booked (more 70 % booked)

more_review_scores = make_series_df(more_booked_id, 'review_scores_rating', df_return=False)

more_review_scores.head()
more_review_scores.describe()
sns.distplot(a = more_review_scores, kde = False, bins = 50);

plt.title('review score for well booked');

plt.savefig('well_booked_scores.jpg');
# check well booked (less 30 % booked)

less_review_scores = make_series_df(less_booked_id, 'review_scores_rating', df_return=False)

less_review_scores.head()
less_review_scores.describe()
sns.distplot(a = less_review_scores, kde = False, bins = 50);

plt.title('review score for less booked');

plt.savefig('less_booked_scores.jpg');
def ratio_rev_scores(review_scores, thresh=90):

    '''

    return the ratio of scores with some threshold (numeral)

    

    review_scores : Series : Series of review scores

    thresh : numeral : default is 90

    

    '''

    return (review_scores >= thresh).sum() / len(review_scores)
# the ratio of over 90 scores in reviews for less booked 

round(ratio_rev_scores(less_review_scores) * 100)
# the ratio of over 90 scores in reviews for well booked 

round(ratio_rev_scores(more_review_scores) * 100)
# rev
# rev.dropna(subset=['comments'], how='any', inplace=True)
# rev.isnull().sum()
# make words lower

# rev['low_comments'] = rev['comments'].str.lower()

# rev['low_comments'].head()
# split each comments with space

# rev['low_nospace_com'] = rev['low_comments'].str.split(' ')

# rev['low_nospace_com'] .head()
# comments from reviews of more booked properties

# more_df_rev = []

# for more_id in more_booked_id:

# #     print(more_id)

#     more_df_rev.append(rev[rev.listing_id == more_id].low_nospace_com.values)
# more_df_rev[0][0]
# import re

# text_lis = []

# for one_id in more_df_rev:

#     for comments in one_id:

# #         print(comments)

#         text = [re.sub(r'[0-9]*', '', word) for word in comments if str(word) != 'nan']

#         text_lis.append(text)

# #         words = [x for x in comments if x]

# #             print(text)
# text_lis[0]
# remove stop words

# from nltk.corpus import stopwords

# stopwords_en = stopwords.words('english')
# nonstop_text = [x for one_com in text_lis for x in one_com if x not in stopwords_en]
# nonstop_text