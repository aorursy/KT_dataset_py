import pandas as pd

import operator

import matplotlib.pyplot as plt

import numpy as np 

import nltk
df = pd.read_csv('../input/train.tsv', sep='\t')

df_sub = pd.read_csv('../input/test.tsv', sep='\t')



submission = pd.DataFrame()

submission['train_id'] = df_sub.test_id.copy()



y_target = list(df.price)
def null_percentage(column):

    df_name = column.name

    nans = np.count_nonzero(column.isnull().values)

    total = column.size

    frac = nans / total

    perc = int(frac * 100)

    print('%d%% or %d missing from %s column.' % 

          (perc, nans, df_name))



def check_null(df, columns):

    for col in columns:

        null_percentage(df[col])

        

check_null(df, df.columns)

def merc_imputer(df_temp):

    df_temp.brand_name = df_temp.brand_name.replace(np.nan, 'no_brand')

    df_temp.category_name = df_temp.category_name.replace(np.nan, 'uncategorized/uncategorized')

    df_temp.item_description = df_temp.item_description.replace(np.nan, 'No description yet')

    df_temp.item_description = df_temp.item_description.replace('No description yet', 'no_description')

    return df_temp



df = merc_imputer(df)

df_sub = merc_imputer(df_sub)
print('Training Data')

check_null(df, df.columns)

print('Submission Data')

check_null(df_sub, df_sub.columns)
df.shipping.value_counts()
print('%.1f%% of items have free shipping.' % ((663100 / len(df))*100))
df.columns
print('$1 items: ' + str(df.price[df.price == 1].count()))

print('$2 items: ' + str(df.price[df.price == 2].count()))

print('$3 items: ' + str(df.price[df.price == 3].count()))
plt.figure('Training Price Dist', figsize=(30,10))

plt.title('Price Distribution for Training - 3 Standard Deviations', fontsize=32)

plt.hist(df.price.values, bins=145, normed=False, 

         range=[0, (np.mean(df.price.values) + 3 * np.std(df.price.values))])

plt.axvline(df.price.values.mean(), color='b', linestyle='dashed', linewidth=2)

plt.xticks(fontsize=24)

plt.yticks(fontsize=26)

plt.show()



print('Line indicates mean price.')
print('Free items: %d, representing %.5f%% of all items.' % 

      (df.price[df.price == 0].count(), 

        (df.price[df.price == 0].count() / df.shape[0])))
print('Free items where seller pays shipping: %d.' % 

      df.price[operator.and_(df.price == 0, df.shipping == 1)].count())
print('No description:', str(df.item_description[df.item_description == 'no_description'].count()))

print('Uncategorized:',str(df.category_name[df.category_name == 'uncategorized/uncategorized'].count()))
cat_counts = np.sort(df.category_name.value_counts())

print(str(len(cat_counts)) + ' categories total.')

print(str(df.shape[0]) + ' records total.')

print('Category frequency percentiles, marked by lines: \n25%%: %d, 50%%: %d, 75%%: %d, 95%%: %d, 97.5%%: %d.' % 

     (cat_counts[int(len(cat_counts)*0.25)], 

      cat_counts[int(len(cat_counts)*0.5)],

      cat_counts[int(len(cat_counts)*0.75)],

      cat_counts[int(len(cat_counts)*0.9)],

      cat_counts[int(len(cat_counts)*0.95)]))



title = 'Category Quantity ECDF Without Top 15 Outliers'

plt.figure(title, figsize=(30,10))

plt.title(title, fontsize=32)

x = np.sort(df.category_name.value_counts())

x = x[0:-15]

y = np.arange(1, len(x) + 1) / len(x)

plt.plot(x, y, marker='.', linestyle='none')

plt.xticks(fontsize=24)

plt.yticks(fontsize=26)

plt.axvline(x=x[int(len(x)*0.25)], linewidth=1, color='b')

plt.axvline(x=x[int(len(x)*0.5)], linewidth=1, color='b')

plt.axvline(x=x[int(len(x)*0.75)], linewidth=1, color='b')

plt.axvline(x=x[int(len(x)*0.95)], linewidth=1, color='b')

plt.axvline(x=x[int(len(x)*0.975)], linewidth=1, color='b')

plt.show()
print('The top 75%% of categories represent %.1f%% of the dataset, and the top 50%% represent %.1f%%.' % 

      ((sum([count for count in cat_counts if count > 10]) / len(df))*100, 

       (sum([count for count in cat_counts if count > 76]) / len(df))*100))
title = 'Top 35 Categories'

plt.figure(title, figsize=(30,10))

df.category_name.value_counts()[0:35].plot(kind='bar')

plt.title(title, fontsize=30)

plt.yticks(fontsize=18)

plt.xticks(fontsize=18, rotation=35, ha='right')

plt.show()
brand_counts = np.sort(df.brand_name.value_counts())

print(str(len(brand_counts)) + ' brands total.')

print(str(df.shape[0]) + ' records total.')

print('Category frequency percentiles, marked by lines: \n25%%: %d, 50%%: %d, 75%%: %d, 95%%: %d, 97.5%%: %d.' % 

     (brand_counts[int(len(brand_counts)*0.25)], 

      brand_counts[int(len(brand_counts)*0.5)],

      brand_counts[int(len(brand_counts)*0.75)],

      brand_counts[int(len(brand_counts)*0.9)],

      brand_counts[int(len(brand_counts)*0.95)]))



title = 'Brand Quantity ECDF Without Top 25 Outliers'

plt.figure(title, figsize=(30,10))

plt.title(title, fontsize=32)

x = np.sort(df.brand_name.value_counts())

x = x[0:-25]

y = np.arange(1, len(x) + 1) / len(x)

plt.plot(x, y, marker='.', linestyle='none')

plt.xticks(fontsize=24)

plt.yticks(fontsize=26)

plt.axvline(x=x[int(len(x)*0.25)], linewidth=1, color='b')

plt.axvline(x=x[int(len(x)*0.5)], linewidth=1, color='b')

plt.axvline(x=x[int(len(x)*0.75)], linewidth=1, color='b')

plt.axvline(x=x[int(len(x)*0.95)], linewidth=1, color='b')

plt.axvline(x=x[int(len(x)*0.975)], linewidth=1, color='b')

plt.show()
print('The top 75%% of categories represent %.1f%% of the dataset, and the top 50%% represent %.1f%%.' % 

      ((sum([count for count in brand_counts if count > 1]) / len(df))*100, 

       (sum([count for count in brand_counts if count > 4]) / len(df))*100))
print('%d items, or %.2f%%, are missing a brand name.' % 

      (len(df[df.brand_name == 'no_brand']), 

       len(df[df.brand_name == 'no_brand']) / len(df)))
title = 'Top 35 Brands'

plt.figure(title, figsize=(30,10))

df.brand_name.value_counts()[1:70].plot(kind='bar')

plt.title(title, fontsize=30)

plt.yticks(fontsize=18)

plt.xticks(fontsize=18, rotation=45, ha='right')

plt.show()
title = 'Top Half of Brands'

plt.figure(title, figsize=(30,10))

df.brand_name.value_counts()[50:2500].plot(kind='bar')

plt.title(title, fontsize=30)

plt.yticks(fontsize=18)

plt.xticks(fontsize=0, rotation=45, ha='right')

plt.show()
import nltk

nltk.data.path.append(r'D:\Python\Data Sets\nltk_data')

from nltk.corpus import stopwords 

import string

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer 

from scipy import sparse 
cat_vec = CountVectorizer(stop_words=[stopwords, string.punctuation], max_features=int(len(cat_counts)*0.5))

cat_matrix = cat_vec.fit_transform(df.category_name)

cat_matrix_sub = cat_vec.transform(df_sub.category_name)
# For exploring the tokens. The array is an array inside of an array of one, ravel pulls it out. 

cat_tokens = list(zip(cat_vec.get_feature_names(), np.array(cat_matrix.sum(axis=0)).ravel()))
brand_vec = CountVectorizer(stop_words=[stopwords, string.punctuation], max_features=int(len(brand_counts)*0.5))

brand_matrix = brand_vec.fit_transform(df.brand_name)

brand_matrix_sub = brand_vec.transform(df_sub.brand_name)
brand_tokens = list(zip(brand_vec.get_feature_names(), np.array(brand_matrix.sum(axis=0)).ravel()))
name_vec = TfidfVectorizer(max_features=30000, stop_words=[stopwords, string.punctuation])

name_matrix = name_vec.fit_transform(df.name)

name_matrix_sub = name_vec.transform(df_sub.name)
print('Kept %d words.' % len(name_vec.get_feature_names()))
desc_vec = TfidfVectorizer(max_features=50000,

                           stop_words=[stopwords, string.punctuation])

desc_matrix = desc_vec.fit_transform(df.item_description)

desc_matrix_sub= desc_vec.transform(df_sub.item_description)
cond_matrix = sparse.csr_matrix(pd.get_dummies(df.item_condition_id, sparse=True, drop_first=True))

cond_matrix_sub = sparse.csr_matrix(pd.get_dummies(df_sub.item_condition_id, sparse=True, drop_first=True))
ship_matrix = sparse.csr_matrix(df.shipping).transpose()

ship_matrix_sub = sparse.csr_matrix(df_sub.shipping).transpose()
sparse_matrix = sparse.hstack([cat_matrix, brand_matrix, name_matrix, desc_matrix, 

                               cond_matrix, ship_matrix])

sparse_matrix_sub = sparse.hstack([cat_matrix_sub, brand_matrix_sub, name_matrix_sub, 

                                   desc_matrix_sub, cond_matrix_sub, ship_matrix_sub])
if sparse_matrix.shape[1] == sparse_matrix_sub.shape[1]:

    print('Features check out.')

else:

    print("The number of features in training and test set don't match.")
import gc

del(cat_matrix, brand_matrix, name_matrix, desc_matrix, cond_matrix, ship_matrix)

del(cat_matrix_sub, brand_matrix_sub, name_matrix_sub, desc_matrix_sub, cond_matrix_sub, ship_matrix_sub)

del(df, df_sub)

gc.collect()