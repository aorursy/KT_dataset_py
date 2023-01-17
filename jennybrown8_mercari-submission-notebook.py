# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import lzma

import os

from scipy import stats, integrate

import matplotlib.pyplot as plt

import seaborn as sns

import csv



from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score





pd.options.display.float_format = '{:.0f}'.format
df_train = pd.read_csv('../input/train.tsv', delimiter='\t')

df_test = pd.read_csv('../input/test.tsv', delimiter='\t')



# These don't help us train. Nor do they make any sense.

df_train.drop(df_train[df_train['price'] == 0].index, inplace=True)
# We might need to filter out price=0 lines as weird outliers.

# Replace the brand name NaN with 'Unknown' or some real string.

# We want to split category names on the slash, into separate columns, and provide min/max/median for each of those.

# We want to add columns for length of item description and length of item name.

# Eventually pick up words like 'guarantee' or 'authentic*' or 'lot' or 'brand new' or 'tags' or something,

# but first train a model on purely the numeric information.  We'll come back to these.

# We want to augment brand name columns and category columns for min/max/median.

def clean_df(df):

    # Clean up missing attributes

    df['brand_name']=df['brand_name'].fillna('Unknown')

    df['category_name']=df['category_name'].fillna('Unknown/Unknown/Unknown')

    df['item_description']=df['item_description'].fillna('Unknown')

    df['item_condition_id']=df['item_condition_id'].fillna(3)



    # Handle category split, and count lengths of name/desc

    df[['category1', 'category2', 'category3']] = df['category_name'].str.split('/', 2, expand=True)

    df['category12'] = df['category1'].astype(str)+'_'+df['category2'].astype(str)

    # category123 just = the original category name column.



    df['name_length'] = df['name'].str.len()

    df['item_description_length'] = df['item_description'].str.len()



    # Throw in a couple word indicators for the fun of it

    df['word_brand_new'] = df['item_description'].str.lower().str.contains('brand new')

    df['word_brand_new'] = df['word_brand_new'].astype(int)

    df['word_tag'] = df['item_description'].str.lower().str.contains('tag').astype(int)

    df['word_tag'] = df['word_tag'].astype(int)

    # TODO: Use unicodedata.normalize("NFKD", text.casefold()) eventually instead of .lower() ?



    return df

    

df_train = clean_df(df_train)

df_test = clean_df(df_test)



# From a grouped name/price aggregation, extracts the kv pairs to a dict for fast lookups.

def create_pricedict(grouped, operation_name, orig_colname):

    pricedict = {}

    for index, row in grouped.iterrows():

        pricedict[row[orig_colname]] = row['price']

    return {orig_colname + "_" + operation_name: pricedict}





# This training knowledge creates reusable lookups, let's hold on to it for reuse later as pricedicts!

pricedicts = {}    

for col in ['brand_name', 'category1', 'category12', 'category_name']:

    pricedicts.update(create_pricedict(df_train.groupby(col, as_index=False).min(), 'min', col))

    pricedicts.update(create_pricedict(df_train.groupby(col, as_index=False).median(), 'median', col))

    pricedicts.update(create_pricedict(df_train.groupby(col, as_index=False).mean(), 'mean', col))

    pricedicts.update(create_pricedict(df_train.groupby(col, as_index=False).max(), 'max', col))
# Now we add the features to the chosen df; this is the real work done on each df.

def price_augment_df(df, pricedicts):

    for col in ['brand_name', 'category1', 'category12', 'category_name']:

        for oper in ['_min', '_median', '_mean', '_max']:

            df[col + oper] = df[col].map(pricedicts[col + oper])

            df[col + oper] = df[col + oper].fillna(pricedicts[col + oper].get('Unknown', 5))



price_augment_df(df_train, pricedicts)

price_augment_df(df_test, pricedicts)

xs = ['item_condition_id', 'shipping', 

      'brand_name_min', 'brand_name_max', 'brand_name_median', 'brand_name_mean', 

      'category_name_min', 'category_name_max', 'category_name_median', 'category_name_mean', 'word_brand_new', 'word_tag']



df_train_xs = df_train[xs]

df_train_y = df_train[['price']]



df_test_xs  = df_test[xs]

# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(df_train_xs, df_train_y)



# Make predictions using the testing set

df_test['price'] = regr.predict(df_test_xs)



# Negative prices throw an error, so fix them for now.

df_test.loc[df_test.price < 0, 'price'] = 0



# The coefficients

print('Coefficients: \n', regr.coef_)



# The mean squared error

#print("Mean squared error: %.2f" % mean_squared_error(df_test_y, y_pred))

# Explained variance score: 1 is perfect prediction

# print('Variance score: %.2f' % r2_score(df_test_y, y_pred))

submissiondf = df_test[['test_id', 'price']]

submissiondf.to_csv('sample_submission.csv', index=False)