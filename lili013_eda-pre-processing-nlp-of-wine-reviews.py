# Setting up environment

import os

import pandas as pd

import copy

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from scipy import stats

from scipy.stats import levene

import re

%matplotlib inline

sns.set(color_codes=True)



# Setting up environment for language pre processing

from nltk.tokenize import word_tokenize

import nltk

from nltk.corpus import stopwords

import copy

import string

from textblob import Blobber

from textblob.sentiments import NaiveBayesAnalyzer
# importing our data from .csv

data = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv' )

data.drop(data.filter(regex="Unname"),axis=1, inplace=True)
# inspecting the shape of the dataset

print(f'The dataframe has', data.shape[0], 'rows and', data.shape[1], 'columns.')

data.head()
# inspecting the type of the variables

print(data.dtypes)

data.points.describe()[['min', 'max']]
# correct the type variables

data.country = data['country'].astype('category')

data.province = data['province'].astype('category')

data.variety = data['variety'].astype('category')
# inspecting for duplicates

dup_rows = data[data.duplicated(

    subset=['description','title','taster_name','winery'])].sort_values(by='title')



# We found some duplicates. And so, we'll delete them.

print(f'The dataframe has', dup_rows.shape[0], 'duplicates that we need to remove.')

clean_data = copy.deepcopy(data.drop_duplicates(subset=['description','title','taster_name', 'winery']))

print(f'The dataframe now has', clean_data.shape[0], 'rows instead of', data.shape[0], 'rows.')
print(clean_data.isnull().sum())
# inspecting missing values in designation

clean_data['designation'] = clean_data['designation'].fillna(clean_data.title)

is_designation_in_title = clean_data.apply(lambda x: x['designation'] in x['title'] , axis=1).astype(int)



print(f' When we subtract the sum of binary variable is_designation_in_title to the number of rows'

      f' in our original dataset we obtain the value',is_designation_in_title.sum() - clean_data.shape[0])
#cleanning up environment

del is_designation_in_title

del dup_rows



# removing multiple columns (designation, region_1, region_2, taster_name and

# taster_twitter_handle

clean_data = clean_data.drop(['designation', 'region_1', 'region_2', 'taster_name',

                        'taster_twitter_handle'], axis =1)

print(f'The dataframe now has', clean_data.shape[1],'columns instead of',

      data.shape[1])



# inspecting for missing values

print(clean_data.isnull().sum())
#Dropping rows where nan values are found in the columns country and variety

clean_data = clean_data.dropna(subset=['country', 'variety'])

print(clean_data.isnull().sum())
# creating new binary variable describing whether or not the observation doesn't

# have a price.

clean_data['has_price'] = clean_data['price'].notnull().astype(int)

points_with_price= clean_data['points'][clean_data['has_price'] == 1]

points_without_price= clean_data['points'][clean_data['has_price'] == 0]



# box-plot of the two groups

sns.boxplot(x=clean_data['has_price'], y=clean_data['points'])
# levene test to test for equality of variance

levene(points_with_price,points_without_price)

t_stat, p_value = stats.ttest_ind(points_with_price,points_without_price, equal_var=False)

print(f't_stat  = {t_stat:+4.4f}')

print(f'p_value =  {p_value:+4.4f}')
# cleaning up environment

del points_with_price

del points_without_price



# imputation

clean_data['price'].fillna(clean_data['price'].mean(), inplace = True)

print(clean_data.isnull().sum())
# make a deepcopy of the final dataset

df = copy.deepcopy(clean_data)
# descriptive statistics of the variable points

df.points.describe()
# visualizations of points

plt.figure(figsize=(9, 8))

sns.set(style="ticks")



f, (ax_box, ax_hist) = plt.subplots(2, sharex=True\

    ,gridspec_kw={"height_ratios": (.15, .85)})



sns.boxplot(df['points'], ax=ax_box)

sns.distplot(df['points'], ax=ax_hist, bins=20, kde=False)



ax_box.set(yticks=[])

sns.despine(ax=ax_hist)

sns.despine(ax=ax_box, left=True)



print(f'Skewness : {df.points.skew():+4.2f}\n'

      f'Kurtosis : {df.points.kurt():+4.2f}')
# descriptive statistics of the variable price

df.price.describe()
# visualizations of price

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True\

    ,gridspec_kw={"height_ratios": (.15, .85)})



sns.boxplot(df['price'], ax=ax_box)

sns.distplot(df['price'], ax=ax_hist, kde=False)



ax_box.set(yticks=[])

sns.despine(ax=ax_hist)

sns.despine(ax=ax_box, left=True)



print(f'Skewness : {df.price.skew():+4.2f}\n'

      f'Kurtosis : {df.price.kurt():+4.2f}')
# visualizations of log price

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True\

    ,gridspec_kw={"height_ratios": (.15, .85)})



sns.boxplot(np.log(df['price']), ax=ax_box)

sns.distplot(np.log(df['price']), ax=ax_hist, kde=False)



ax_box.set(yticks=[])

sns.despine(ax=ax_hist)

sns.despine(ax=ax_box, left=True)



print(f'Skewness : {np.log(df.price).skew():+4.2f}\n'

      f'Kurtosis : {np.log(df.price).kurt():+4.2f}')
# create new variable log_price

df['log_price'] =np.log(df.price)
df['num_wine_from_winery'] = df.groupby(['winery'])['country'].transform(np.size)
print(df.title[213])

print(df.title[1530])

print(df.title[2262])

print(df.title[63])
# function to extract the largest 4-digits value

def extractmax(str1):

    nums=re.findall("(?:19|20)[0-9][0-9]",str1)

    return max(nums, default=0)



# https://stackoverflow.com/questions/21544159/python-finding-largest-integer-in-string
# extracting the vintage

years= [extractmax(line) for line in df['title']]

years = np.array(years)

df['vintage'] = years
# changing the type of `vintage` to float 

df.vintage = df.vintage.astype(str).astype(float)
# replacing the zeros with nan

df['vintage'].replace(0, np.nan, inplace=True)
# checking if our tactic worked



print(f'title                                           vintage\n\n'

      f'{df.title[213]:<20}{df.vintage[213]:>22}\n'

      f'{df.title[1530]:<20}{df.vintage[1530]:>13}\n'

      f'{df.title[2262]:<20}{df.vintage[2262]:>8}\n'

      f'{df.title[63]:<20}{df.vintage[63]:>12}\n')
# creating variables to play with

df['has_vintage'] = df['vintage'].notnull().astype(int)

points_with_vintage= df['points'][df['has_vintage'] == 1]

points_without_vintage= df['points'][df['has_vintage'] == 0]



# box-plot of the two groups

sns.boxplot(x=df['has_vintage'], y=df['points'])

# levene test to test for equality of variance

levene(points_with_vintage,points_without_vintage)

# We can test whether the rows with missing data differ

# from the ones without missing data on target



t_stat, p_value = stats.ttest_ind(points_with_vintage,points_without_vintage, equal_var=False)

print(f't_stat  = {t_stat:+4.4f}')

print(f'p_value =  {p_value:+4.4f}')
# imputation of vintage with the mean

df['vintage'].fillna(df['vintage'].mean(), inplace = True)

print(df.isnull().sum())
# descriptive statistics 

df.vintage.describe()
# visualization of vintage

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True\

    ,gridspec_kw={"height_ratios": (.15, .85)})



sns.boxplot(df.vintage, ax=ax_box)

sns.distplot(df.vintage, ax=ax_hist, kde=False)



ax_box.set(yticks=[])

sns.despine(ax=ax_hist)

sns.despine(ax=ax_box, left=True)



print(f'Skewness : {df.vintage.skew():+4.2f}\n'

      f'Kurtosis : {df.vintage.kurt():+4.2f}')
# dropping unrealistic observations

df = df.drop(df[df.vintage > 2050].index)
# checking the mean of wines with old vintage

x = df[df.vintage < 1920]

print(f'The average price of the wines with a vintage under year 1920 is {x.price.mean():.2f}$') 
# checking out the prices of old wines starting from 1970

old_wines = df[df.vintage<1970]

print(old_wines.price)
# dropping unrealistic observations

df = df.drop(df[(df.vintage < 1970) & (df.price<50)].index)                 

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True\

    ,gridspec_kw={"height_ratios": (.15, .85)})



sns.boxplot(df.vintage, ax=ax_box)

sns.distplot(df.vintage, ax=ax_hist, kde=False)



ax_box.set(yticks=[])

sns.despine(ax=ax_hist)

sns.despine(ax=ax_box, left=True)



print(f'Skewness : {df.vintage.skew():+4.2f}\n'

      f'Kurtosis : {df.vintage.kurt():+4.2f}')

# create a copy description to work with

desc = list(copy.deepcopy(df['description']))

desc[0]
# creating a set of punctuation

punc = set(string.punctuation)



#loading stop_words

nltk.download('stopwords')



# creating a set of stop words

stop_words = set(stopwords.words('english'))



# combining the 2 sets with an "or" operator (i.e. "|")

all_stops = stop_words | punc



# loop to pre-process data

clean_desc =[]

for item in desc:

    tok_desc = word_tokenize(item)

    lower_data = [i.lower() for i in tok_desc]

    tok_desc_no_num = [i for i in lower_data if i.isalpha()]

    filtered_desc = [i for i in tok_desc_no_num if i not in all_stops]

    clean_desc.append(filtered_desc)
# Organizing the data in a new dataframe

clean_desc_untok = [' '.join(i) for i in clean_desc]

column_names = ['original_desc', 'cleaned_description', 'untok_description']

data_tuple= list(zip(clean_data['description'], clean_desc, clean_desc_untok))

desc_df = pd.DataFrame(data_tuple, columns=column_names)
# Setting up the blobber with a Naive Bayes Analyzer

tb = Blobber(analyzer=NaiveBayesAnalyzer())

blob = [tb(text) for text in desc_df['untok_description']]

sentiment_values = [text.sentiment for text in blob]
# example of positive sentiment

print(f'For this string: \n {desc[1]} \n\n We got this analysis:\n {sentiment_values[1]}')
# example of negative sentiment

print(f'For this string: \n {desc[111]} \n\n We got this analysis:\n {sentiment_values[111]}')
# creating a dataframe that isolate each sentiment values

stats = pd.DataFrame(zip(*sentiment_values)).T

stats.columns=['classification','pos','neg']
# Add the positive sentiment score to the clean dataframe

df['pos_sentiment'] = 0

df['pos_sentiment'] = list(stats.pos)
# save dataframe

df.to_pickle('processed_data.p')
print(f'The final dataframe has {df.shape[0]} rows and {df.shape[1]} columns.')

 