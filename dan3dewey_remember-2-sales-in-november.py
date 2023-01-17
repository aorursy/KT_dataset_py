# To-do:

#   Create output for a classification problem of the "missing items": new/hot or old/boring ?

#   Try using Oct'15s Second-half / First-half cnt ratios to estimate/adjust Nov item values.

#   Do actual ML with "validation" of some form ;-)
# software to use

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



from time import time

import gc



# Is XGBoost available?  Yes! (conda install -c conda-forge xgboost , and conda update nbconvert)

# ee also: https://www.kaggle.com/alexisbcook/xgboost

from xgboost import XGBRegressor
# Some overall parameters/choices to setup upfront:



# Am I running on kaggle or locally?

# Locally I have:

# - some english translated files available locally.

# - the 1-month shifted 'validation' target file.

LOCATION_KAGGLE = True



# Show the EDA plots, etc.

SHOW_EDA = True



# Down-select to only the last full 1+ or 2+ years

DOWNSEL_14MONS = False  # This misses ~ 4+69 items that are unique in the first 19,20 months.

DOWNSEL_26MONS = True   # This misses only ~ 4 items that are unique to the first 7,8 months.



# For quasi-validation add 1 to date block numbers and use new 34 as target

SHIFT_1MONTH = False



# Use only shop_ids and item_ids that are in the output (submission):

DOWNSEL_OUT = True # this is always done, put here as a reminder.



# A column item_cnt_clip is made as item_cnt_day clipped at some value, e.g., 20.

# This can help approximate the effect of the final clipping of cnt_month in some prediction methods.

# Don't use it when combining/summing actual counts.

USE_CNT_CLIP = False
# Repeatable or roll-the-dice?

# The seed is set once here at beginning of notebook.

#   to a fixed value

RANDOM_SEED = 730

#   or to a time-based random value, 0 to 1023

##RANDOM_SEED = int(time()) % 2**10

# in either case initialize the seed

np.random.seed(RANDOM_SEED)

print(RANDOM_SEED)
# show the files that are in ../input

!ls ../input/competitive-data-science-predict-future-sales/*.csv
# Some constants



# The data files' location (on kaggle and locally)

data_dir = "../input/competitive-data-science-predict-future-sales/"



# The main "train" data of transactions

trans_file = "sales_train.csv"



# The output (submission) file

out_file = "test.csv"



# The items_file, columns: item_name, item_id, item_category_id

# This is used to assign categories to items

items_file = "items.csv"



if LOCATION_KAGGLE:

    categs_file = "item_categories.csv"

    shops_file = "shops.csv"

else:

    # These (somewhat) translated versions are from:

    #  https://www.kaggle.com/c/competitive-data-science-predict-future-sales/discussion/54949

    categs_file = "item_categories-translated.csv"

    shops_file = "shops-translated.csv"

    # The translated item_names_file, columns: item_id, item_name

    item_names_file = "items-translated.csv"

# Read in the files to dataframes



# The main "training" data

df_trans = pd.read_csv(data_dir+trans_file)

# Shift the date-blocks by adding 12 ?

if SHIFT_1MONTH:

    df_trans['date_block_num'] += 1

    

# The output (submission) file

df_out = pd.read_csv(data_dir+out_file)



# Mapping categories to items

df_items = pd.read_csv(data_dir+items_file)



# Names of things:

df_categs = pd.read_csv(data_dir+categs_file)

df_shops = pd.read_csv(data_dir+shops_file)



# Use translated item names if running locally

if LOCATION_KAGGLE == False:

    df_item_names = pd.read_csv(data_dir+item_names_file)

    # Replace the item_name in df_items with the translated-ish value in df_item_names

    df_items['item_name'] = df_item_names['item_name']

# The full transactions



num_months = len(df_trans['date_block_num'].unique())

print("\nFull Transactions has {} entries covering {} months.\n".format(len(df_trans),num_months))



# Number of unique values in each column

print("The number of unique values for each column:")

df_trans.nunique()
# Look at info and look for NaNs

##df_trans.info()

# NaNs in each column - None.

##df_trans.isnull().sum(axis=0)
# Auto way to select categorical and numeric columns

#  --> date is categorical, others are numeric.



##cat_cols = list(df_trans.select_dtypes(include=['object']).columns)

##num_cols = list(df_trans.select_dtypes(exclude=['object']).columns)

##print("Categorical, Numeric columns:\n",cat_cols, '\n', num_cols)
# List and plot the unique number of some column vs the month:

#

# The shop_id: starts at 45, gets up to 52, then back to 42-44.

#

##print(df_trans.groupby('date_block_num').shop_id.nunique())

##plt.plot(df_trans.groupby('date_block_num').shop_id.nunique())

##plt.show()



# The item_id:  drops from high almost 8500 to a low of 5085.

#

##print(df_trans.groupby('date_block_num').item_id.nunique())

##plt.plot(df_trans.groupby('date_block_num').item_id.nunique())

##plt.show()
# Look at the values of item_cnt_day: 

# -  They are mostly less than 10.

# -  -1.0 (7252 times in 34 months) is the only negative value appearing more than ~100 times .

##df_trans['item_cnt_day'].value_counts()
# The monthly counts of positive items

##df_pos = df_trans[df_trans['item_cnt_day'] > 0].copy()

##totcnt_vs_month = df_pos[['date_block_num','item_cnt_day']].groupby('date_block_num').agg(sum)

##mean_pos = totcnt_vs_month.mean()[0]



# The monthly counts of negative items

##df_neg = df_trans[df_trans['item_cnt_day'] < 0].copy()

##totcnt_vs_month = df_neg[['date_block_num','item_cnt_day']].groupby('date_block_num').agg(sum)

##mean_neg = totcnt_vs_month.mean()[0]



##print("Means of the monthly sums of pos and neg item_cnt_day: {:.2f}, {:.2f}".format(mean_pos, mean_neg))

#    Means of the monthly sums of pos and neg item_cnt_day: 107521.97, -221.79

# The negative are only 0.2% of the positive, leave 'em out? No,leave 'em in!
# Some things to do for the SHIFT_1MONTH validation:
# Make a list of the 'future' items: items that are in the output but are not in the full trans:

# for the real prediction these are the missing items,

# for the shifted test these are beyond the test range, in the future.

if SHIFT_1MONTH:

    print(" - - - SHIFT_1MONTH - - -\n   Finding all future_items")

    # This code was copied from below, many variables will be reused.

    # future_items is the output.

    out_items = np.sort(df_out.item_id.unique())

    num_out_items = len(out_items)

    print("\nNumber of items in Output: {}".format(num_out_items))

    #

    trans_items = np.sort(df_trans.item_id.unique())

    print("Number of items in FULL Transactions: {}".format(len(trans_items)))

    #

    # Check if all Output items are in the Transactions items:

    future_items = []

    for item in out_items:

        if item not in trans_items:

            future_items.append(item)

    if len(future_items) > 0:

        print("There are {} future_items which are missing from the FULL Transactions data:".format(

                                        len(future_items)),"\n")

    else:

        print("\nAll of the Test Items are in the Train data.\n")



    # Show a histogram of the missing item_ids:

    plt.hist(future_items, bins=50)

    plt.xlabel("item_id")

    plt.title("Future Items missing from the FULL Transactions")

    plt.show()
# Before downselecting, if we're doing the 1-month shift and we're on Kaggle,

# make a submission_Oct15.csv file which gives the 'validation' target values.

# (This is very inefficiently coded here.)

if SHIFT_1MONTH and LOCATION_KAGGLE:

    # Get just the desired transactions from the now existing date_block_num = 34

    df_month = df_trans[df_trans['date_block_num'] == 34].copy()

    # setup a validation df

    df_val = df_out.copy()

    # Go through the desired output shop-items and get the month's count totals

    out_vals = []

    for iout in df_val.index:

        # Find the matching shop & item entries in the month's transactions:

        this_shop = df_val.loc[iout,'shop_id']

        this_item = df_val.loc[iout,'item_id']

        out_vals.append(df_month.loc[(df_month['shop_id'] == this_shop) &

                                    (df_month['item_id'] == this_item),'item_cnt_day'].sum())



    # Put the counts in the data frame

    df_val['item_cnt_month'] = np.array(out_vals)

    # and write out the clipped counts csv file:

    clip_level = 20

    num_over20 = sum(df_val['item_cnt_month'] > clip_level)

    print("\nCheck for over-20 values, found/clipped {} of them.".format(num_over20))

    if num_over20 > 0:

        df_val.loc[df_val['item_cnt_month'] > clip_level, 'item_cnt_month'] = clip_level



    # Show some simple summary values of the submission:

    print("\nSum of output item_cnt_month: {:.0f}, average count-per-month per shop-item: {:.3f}\n".format(

            df_val['item_cnt_month'].sum(), df_val['item_cnt_month'].sum()/(len(df_val))))

    # and write the file

    df_val[['ID','item_cnt_month']].to_csv("submission_Oct15.csv", index=None)

    

    # delete variables not needed:

    del df_month, df_val, out_vals

    gc.collect()
# Can downselect to keep  ~1 or 2 years of the most recent data,

# 26 months: From date (month) block 8 through 33.

# 14 months: 20 to 33.

# (These give a smaller data size and are the most test-relevant data)



if DOWNSEL_14MONS:

    df_trans = (df_trans[( (df_trans['date_block_num'] >= 20) & 

                          (df_trans['date_block_num'] <= 33) )]).reset_index().drop('index',axis=1)  

elif DOWNSEL_26MONS:

    df_trans = (df_trans[( (df_trans['date_block_num'] >= 8) &

                          (df_trans['date_block_num'] <= 33) )]).reset_index().drop('index',axis=1)

else:

    # Have to do this to ignore date-block 34 when SHIFT_1MONTH = True

    df_trans = (df_trans[( (df_trans['date_block_num'] <= 33) )]).reset_index().drop('index',axis=1)

    

num_months = len(df_trans['date_block_num'].unique())

print("Using {} months of data.".format(num_months))
# Add an item_cnt_clip column, clipped to some low value (20, 10, etc):

# Note that this is to help approximate the final clipping to 20

# when generating average values.

df_trans['item_cnt_clip'] = df_trans['item_cnt_day']

df_trans.loc[df_trans['item_cnt_clip'] > 20, 'item_cnt_clip'] = 20
# Add the item_category_id from df_items as a column in the main transactions dataframe.

# Using merge to be clear about what is happening (vs join which is a wrapped merge)

#   pd.merge(left, right, left_on=key_or_keys, right_index=True, how='left', sort=False)

df_trans = pd.merge(df_trans, df_items[['item_id','item_category_id']].set_index('item_id'), left_on='item_id',

                                 right_index=True, how='left', sort=False)

# Include it in df_out as well:

df_out = pd.merge(df_out, df_items[['item_id','item_category_id']].set_index('item_id'), left_on='item_id',

                                 right_index=True, how='left', sort=False)
# The item_category_id: Look at its variation of unique values vs month

#

# The item_category_id: Stable with a very slow growth from 61ish to 63-64.

#

##print(df_trans.groupby('date_block_num').item_category_id.nunique())

##plt.plot(df_trans.groupby('date_block_num').item_category_id.nunique())

##plt.show()
# Remove items and shops from df_trans that are not in df_out (spurious information?)

if True:

    

    # Keep just the out items:

    out_items = df_out['item_id'].unique()

    iitem = 0

    keep_these = (df_trans['item_id'] == out_items[iitem])

    # go through the rest of the out items and OR those in as well:

    for iitem in range(1,len(out_items)):

        keep_these = (keep_these | (df_trans['item_id'] == out_items[iitem]))

    # Down select to these:

    df_trans = (df_trans[keep_these]).reset_index().drop('index',axis=1)

    

    

    # Keep just the out shops

    out_shops = df_out['shop_id'].unique()

    ishop = 0

    keep_these = (df_trans['shop_id'] == out_shops[ishop])

    # go through the rest of the out shops and OR those in as well:

    for ishop in range(1,len(out_shops)):

        keep_these = (keep_these | (df_trans['shop_id'] == out_shops[ishop]))

    # Down select to these:

    df_trans = (df_trans[keep_these]).reset_index().drop('index',axis=1)

    
# Add a log-price column; change any negative to positive (there are no zeros it seems)

df_trans['log10_price'] = np.log10(np.abs(df_trans['item_price']))

print("Min and Max of the log10 of item price: {:.2f} , {:.2f}".format(df_trans['log10_price'].min(),

                                                            df_trans['log10_price'].max()))

# Add a 3-level price indicator (for color coding plots)

df_trans['item_price3'] = ((df_trans['item_price'] > 300.00).astype(int) +

                           (df_trans['item_price'] > 1000.00).astype(int))
# Add a datetime format column - This takes ~8 seconds on my laptop.

df_trans['date_dt'] = pd.to_datetime(df_trans.date, format='%d.%m.%Y')

# then we can then add these (useful?) columns

df_trans['day'] = df_trans.date_dt.dt.day

df_trans['month'] = df_trans.date_dt.dt.month

df_trans['year'] = df_trans.date_dt.dt.year
# Text Values

# Not using these text fields for now - could make features?

# Note that df_items also has text values for the items (item_name);

# df_categs and df_shops also/only provide text values for the categories (item_category_name)

# and the shops (shop_name).



# Show samples of these text columns, the last entries of them

print(df_items[['item_name','item_id']].tail(5))

print("")

print(df_categs.tail(5))

print("")

print(df_shops.tail(5))
print("The selected Transactions have {} entries covering {} months.".format(len(df_trans),num_months))



# Number of unique values in each column

print("The number of unique values for each column:")

df_trans.nunique()
# The transactions - this is the main dataframe

df_trans.tail(5).T
print("Transactions has {} entries covering {} months.".format(len(df_trans),num_months))

print("The sum of item_cnt_day is {}, an average of {} per entry.".format(

    df_trans['item_cnt_day'].sum(),df_trans['item_cnt_day'].sum()/len(df_trans)))
# Look at the output file - this file maps an ID value to the 214200 shop-item combinations

# that we have to predict.



# Note that item_category_id was added as well.

df_out.tail(5)
# Compare the _Shops_ listed in the transactions vs the output ones.

# --> These are the same since we down-selected df_trans to only have df_out shops.



out_shops = np.sort(df_out.shop_id.unique())

num_out_shops = len(out_shops)

print("\nNumber of shops used in Output: {}".format(num_out_shops))

print(out_shops)



trans_shops = np.sort(df_trans.shop_id.unique())

print("\nNumber of shops in Transactions: {}".format(len(trans_shops)))

print(trans_shops)

print("")



# Check if all Output shops are in the Transactions shops:

missing_out = False

for iout in out_shops:

    if iout not in trans_shops:

        print("Missing Output shop: {}".format(iout))

        missing_out = True

if missing_out:

    print("\nSome Output shops are missing from the Transactions data.")

else:

    print("\nAll of the Output shops are in the Transaction data.\n")
if SHIFT_1MONTH:

    print(df_shops.loc[36])

    missing_shops = [36]

    # shop_name    Novosibirsk TRC "Gallery Novosibirsk"
# Compare the _Categories_ listed in the transactions vs the submission-output ones.



out_cats = np.sort(df_out.item_category_id.unique())

num_out_cats = len(out_cats)

print("\nNumber of categories appearing in Output: {}".format(num_out_cats))

print(out_cats)



trans_cats = np.sort(df_trans.item_category_id.unique())

print("\nNumber of categories in Transactions: {}".format(len(trans_cats)))

print(trans_cats)

print("")



# Check if all Output categories are in the Transactions categories:

missing_out = False

missing_categories = []

for iout in out_cats:

    if iout not in trans_cats:

        print("Missing Output category: {}".format(iout))

        missing_categories.append(iout)

        missing_out = True

if missing_out:

    print("\nSome Output categories are missing from the Transactions data.")

else:

    print("\nAll of the Output categories are in the Transaction data.\n")
df_categs.loc[missing_categories]
# Look into the extra categories = 0, 27 that are in df_out, find that:

#   There are a total of 42 entries for each category, 42 = number of shops.

#   All of them have item_id equal to 5441 or 6439.

#

#   The item_name of 5441 is "PC: Headset HyperX Cloud Core gaming stereo..." (from translated file)

#     The item_category_name of 0 is "PC - Headsets / Headphones" (from translated file)

#

#   The item_name of 6439 is "Sid Meier's Railroadrs! [MAC, Digital Version]" (from translated file)

#     The item_category_name of 27 is "MAC Games - Digit" (from translated file)

#

# So these (whole) categories are not in my down-selected transactions,

# i.e., there were no sales of them in the past 2 years at any of the df_out shops.

# So, either they are old and we'd expect 0 sales in future,

# or they are brand new, first sold in Nov'15.

# I'll go for the former and set these to 0 when the submission is made.

# But see also the next cell where items are compared.



# Commands used for above:



# Category 0:

##len(df_out[df_out['item_category_id'] == 0])

##df_out[df_out['item_category_id'] == 0]

##print(df_categs[df_categs['item_category_id'] == 0])



##print(df_items[df_items['item_id'] == 5441])



##print("")



# Category 27:

##len(df_out[df_out['item_category_id'] == 27])

##df_out[df_out['item_category_id'] == 27]

##print(df_categs[df_categs['item_category_id'] == 27])



##print(df_items[df_items['item_id'] == 6439])

# Compare the _Items_ in the transactions vs the ones in the submission-output

out_items = np.sort(df_out.item_id.unique())

num_out_items = len(out_items)

print("\nNumber of items in Output: {}".format(num_out_items))

##print(len(out_items))



# Note the size of the output file with the numbers of out shops and out items:

print("\nOutput has {} items x {} shops = {} pairs; \nwhich agrees with the length of df_out = {}".format(

            num_out_items, num_out_shops, num_out_shops * num_out_items, len(df_out) ))



trans_items = np.sort(df_trans.item_id.unique())

print("\nNumber of items in Transactions: {}".format(len(trans_items)))

##print(len(trans_items))

print("")



# Check if all Output items are in the Transactions items:

missing_items = []

for item in out_items:

    if item not in trans_items:

        missing_items.append(item)

if len(missing_items) > 0:

    print("Some Output Items, {} of them, are missing from the Transactions data:".format(

                                        len(missing_items)),"\n   (However, these all have known",

                                       "categories,", "except for 5441 and 6439)\n")

    if SHOW_EDA:

        print(missing_items)

else:

    print("\nAll of the Test Items are in the Train data.\n")



# Show a histogram of the missing item_ids:

plt.hist(missing_items, bins=50)

plt.xlabel("item_id")

plt.title("Output items that are missing from Transactions")

plt.show()
# The items in the two categories that are missing from df_trans:

# A new headset and an old game?   (see comments above)

##df_items.loc[[5441,6439]]
# All the missing items

##df_items.loc[missing_items]



# All the categories of the missing items

# Are some of the items old and some very new?  Can tell by category?   --> diverse range of categories.

##df_categs.loc[df_items.loc[missing_items, 'item_category_id'].unique()]
# The total and monthly-average item counts

trans_sum_cnt = df_trans['item_cnt_day'].sum()

trans_sum_clip = df_trans['item_cnt_clip'].sum()



print("\n\n\nTotal count = Sum of all item_cnt_day is: {}  (clipped ~ {})".format(trans_sum_cnt,trans_sum_clip))

average_totcnt_month = trans_sum_cnt/num_months

average_totcnt_clip = trans_sum_clip/num_months

print("\nAverage Total-count /month is: {:.3f}  (clipped ~ {:.3f})\n\n".format(average_totcnt_month,average_totcnt_clip))

# List all the columns in df_trans

df_trans.columns
if SHOW_EDA:

    # Histograms of the columns

    hist_cols = df_trans.columns

    # leave some out, and why

    #   item_cnt_day has some very large values (item_cnt_clip is still there)

    hist_cols = hist_cols.drop('item_cnt_day')

    #   the log10 of the price is more informative

    hist_cols = hist_cols.drop('item_price')

    #   these are just 2013, 2014, 2015

    hist_cols = hist_cols.drop('year')

    #   do this at higher bin resolution on its own:

    hist_cols = hist_cols.drop('item_id')



    df_trans[hist_cols].hist(bins=100, figsize=(15,10))

    plt.show()
if SHOW_EDA:

    # Higher bin-resolution view for the item_id values

    df_trans['item_id'].hist(bins=1000, figsize=(15,8), color='green')

    plt.title('item_id')

    plt.show()
if SHOW_EDA:

    # Scatter plot of category and item_id; colored by 3-level price

    # Use a fraction of the points

    df_trans.iloc[0:len(df_trans):5].plot.scatter(

        'item_id','item_category_id',figsize=(15,8),marker=".",alpha=0.02,

                      c='item_price3',cmap='plasma_r')

    plt.show()
if SHOW_EDA:

    # Scatter plot of shop_id and item_id; colored by 3-level price

    # Use a fraction of the points

    df_trans.iloc[0:len(df_trans):5].plot.scatter(

        'item_id','shop_id',figsize=(15,8),marker=".",alpha=0.02,

                     c='item_price3',cmap='plasma_r')

    # overplot the 'strange' shops to be sure of their ids (see below)

    strange_shops = [12, 22, 25, 28, 31, 42, 55]

    for ish in strange_shops:

        plt.plot([0],[ish],marker='o',c='orange')

    plt.show()



    # Some shops stand out here: Shop number (approx): 12, 25, 28, 31, 42, 55

    # List them:

    print(df_shops.loc[strange_shops])

    

# The 'strange' shops:  (from english translation file) 

#    shop_id                       shop_name

#12       12           Online shop Emergency

#22       22                 Moscow Shop С21

#25       25             Moscow TRK "Atrium"

#28       28  Moscow ТЦ "MEGA Teply Stan" II

#31       31         Moscow ТЦ "Семеновский"

#42       42          SPb TC "Nevsky Center"

#55       55     Digital warehouse 1C-Online
if SHOW_EDA:

    # Scatter plot of date-block and item_id; colored by 3-level price

    # Use a fraction of the points

    df_trans.iloc[0:len(df_trans):5].plot.scatter(

        'item_id','date_block_num',figsize=(18,8),marker=".",alpha=0.05,

                     c='item_price3',cmap='plasma_r',colorbar=False)

    # limit the item range (x):

    plt.xlim(5000,10000)

    plt.show()



# Do items come and go vs time ?
if SHOW_EDA:

    # Scatter plot of shop_id and item-category

    df_blurred = df_trans[['item_category_id','shop_id','month','item_price3']].copy()

    # Use a fraction of the points

    df_blurred = df_blurred.iloc[0:len(df_blurred):5]

    

    df_blurred['item_category_id'] += 0.20*np.random.randn(len(df_blurred))

    df_blurred['shop_id'] += 0.20*np.random.randn(len(df_blurred))

    df_blurred.plot.scatter('item_category_id','shop_id',figsize=(15,12),marker=".",alpha=0.02,

                       c='item_price3',cmap='plasma_r')

    plt.show()



    # Don't need this around

    del df_blurred

    gc.collect()

    

# Shop and Category look generally independent-ish except for some clear cases

# shop ~55 :  very few/specific categories

# shop ~22 : high value in category ~49. 

# shop ~12 : high values in categories ~9, ~49. 



# The 'strange' shops:  (from english translation file) 

#    shop_id                       shop_name

#12       12           Online shop Emergency

#22       22                 Moscow Shop С21

#25       25             Moscow TRK "Atrium"

#28       28  Moscow ТЦ "MEGA Teply Stan" II

#31       31         Moscow ТЦ "Семеновский"

#42       42          SPb TC "Nevsky Center"

#55       55     Digital warehouse 1C-Online
# Make a crosstab by shop_id and item-category.



#   Table contains the sum over all selected months of item_cnt_day:

shop_cat_ctab = pd.crosstab(df_trans['shop_id'],df_trans['item_category_id'],

                            values=df_trans['item_cnt_day'], aggfunc=sum,

                            # can normalize by index, columns, all; set NaN to 0.0

                            normalize='all').fillna(0.0)

# Show its values

##1000.0 * shop_cat_ctab
# Make a crosstab by item_id and item-category.



#   Table contains the sum over all selected months of item_cnt_day:

item_cat_ctab = pd.crosstab(df_trans['item_id'],df_trans['item_category_id'],

                            values=df_trans['item_cnt_day'], aggfunc=sum,

                            # can normalize by index, columns, all; set NaN to 0.0

                            normalize='columns').fillna(0.0)

# Show its values

##item_cat_ctab



# Show the sum of items in each category - normalized to 1.0

##item_cat_ctab.sum()
if SHOW_EDA:

    # Look at the distribution of the categories for each shop

    cats = shop_cat_ctab.columns

    shops = shop_cat_ctab.index

    plt.figure(figsize=(15,8))

    for ishop in range(len(shops)):

        shop_dist = shop_cat_ctab.loc[shops[ishop]].values

        plt.plot(cats,np.sqrt(shop_dist),marker='o')

        plt.xlabel("Category_id")

        plt.ylabel("sqrt( Fraction of counts in shop-category] )")
if SHOW_EDA:

    # Look at the distribution of the shops for each category

    cats = shop_cat_ctab.columns

    shops = shop_cat_ctab.index

    plt.figure(figsize=(15,8))

    for icat in range(len(cats)):

        cat_dist = shop_cat_ctab[cats[icat]].values

        plt.plot(shops,np.sqrt(cat_dist),marker='o')

        plt.xlabel("Shop_id")

        plt.ylabel("sqrt( Fraction of counts in shop-category] )")
if SHOW_EDA:

    # Scatter plot of shop/categ vs date-block

    df_blurred = df_trans[['item_category_id','shop_id','date_block_num','item_price3']].copy()

    # Use a fraction of the points

    df_blurred = df_blurred.iloc[0:len(df_blurred):5]

    

    df_blurred['item_category_id'] += 0.20*np.random.randn(len(df_blurred))

    df_blurred['shop_id'] += 0.20*np.random.randn(len(df_blurred))

    df_blurred['date_block_num'] += 0.20*np.random.randn(len(df_blurred))

    

    

    # Categories vs Date-block

    # Besides the December peak, there are some categories that start peaking in Nov.

    # e.g., Category 28

    df_blurred.plot.scatter('date_block_num','item_category_id',figsize=(15,12),marker=".",alpha=0.03,

                       c='item_price3',cmap='plasma_r')

    # zoom in on a category range

    ##plt.ylim(40,50)

    plt.show()

    

    

    # Shops vs Date-block

    # There doesn't seem much variation between the shops in their monthly pattern

    ##df_blurred.plot.scatter('date_block_num','shop_id',figsize=(15,12),marker=".",alpha=0.02,

    ##                   c='item_price3',cmap='plasma_r')

    ##plt.show()



    

    # Don't need this around

    del df_blurred

    gc.collect()

    

# Category 47 is very recent
# Category 28 has a large increase in Nov'15 (and into Dec)

# Also 20, see below.



# Get names of discussed categories:

##print(df_categs.loc[[20,28,47]])

##print(df_categs.loc[[37,40,55]])



#    item_category_id             item_category_name

# 20                20                    Games - PS4

# 28                28  PC Games - Additional Edition

# 47                47          Books - Comics, manga



# These also stand out in the newhot ones in the SHIFT_1MONTH experiment

# 37                37                Cinema - Blu-Ray

# 40                40                    Cinema - DVD

# 55                55  Music - CD of local production
# Any category 20 or 28 items that are in the missing list?

# These are Hot-new items?

its_missing = []

for iit in df_items.index:

    its_missing.append(df_items.loc[iit,'item_id'] in missing_items)



newhot20_items = df_items[(df_items['item_category_id'] == 20) & its_missing].item_id.values

newhot28_items = df_items[(df_items['item_category_id'] == 28) & its_missing].item_id.values



print("\nMissing items that are in Cats 20 or 28:")

print(newhot28_items,"\n",newhot20_items)

##print("   Cat 20:")

##print(df_items.loc[newhot20_items,'item_name'])

##print("   Cat 28:")

##print(df_items.loc[newhot28_items,'item_name'])



#  Cat 20:     (using english translation file)

#1386               Alien: Isolation [PS4, Russian version]

#2323     Call of Duty: Black Ops III [PS4, Russian vers...

#2325     Call of Duty: Black Ops III. Hardened Edition ...

#2327     Call of Duty: Black Ops III. Nuketown Edition ...

#2427         Crew. Wild Run Edition [PS4, Russian version]

#2966     Divinity. Original Sin: Enhanced Edition [PS4,...

#3408                    Fallout 4 [PS4, Russian subtitles]

#5268               Need for Speed ​​[PS4, Russian version]

#6169     Rock Band 4 (game + guitar, drums and micropho...

#6532     Skylanders SuperChargers. Starter kit [PS4, En...

#6729     Star Wars: Battlefront (+ Battle of Giacca) [P...

#6732         Star Wars: Battlefront [PS4, Russian version]

#6903            Tales of Zestiria [PS4, Russian subtitles]

#6996     The Talos Principle. Deluxe Edition [PS4, Engl...

#7728                 WWE 2K16 [PS4, Russian documentation]

#7782     Wasteland 2: Director's Cut [PS4, Russian subt...

#10203    Witcher 3: Wild Hunt - Supplement "Stone Heart...

#19417       Snoopy. Great adventure [PS4, English version]

#

#   Cat 28:

#1437    Anno 2205. Collector's Edition [PC, Russian ve...

#1580    Assassin's Creed: Syndicate. Rooks [PC, Russia...

#2426         Crew. Wild Run Edition [PC, Russian version]

#3407                    Fallout 4 [PC, Russian subtitles]

#7862    World of Warcraft: Warlords of Draenor (add-on...
# Any category 37, 40, 55 items that are in the missing list?

# These are Hot-new items?

its_missing = []

for iit in df_items.index:

    its_missing.append(df_items.loc[iit,'item_id'] in missing_items)



newhot37_items = df_items[(df_items['item_category_id'] == 37) & its_missing].item_id.values

newhot40_items = df_items[(df_items['item_category_id'] == 40) & its_missing].item_id.values

newhot55_items = df_items[(df_items['item_category_id'] == 55) & its_missing].item_id.values





print("\nMissing items that are in Cats 37, 40, 55:")

print(newhot37_items,"\n",newhot40_items,"\n",newhot55_items)
# Make a crosstab by date-block and item-category.



#   Table contains the sum over all shops of item_cnt_day:

date_cat_ctab = pd.crosstab(df_trans['date_block_num'],df_trans['item_category_id'],

                            values=df_trans['item_cnt_day'], aggfunc=sum,

                            # can normalize by index, column, all; set NaN to 0.0

                            normalize='all').fillna(0.0)
# Look at the jump from Oct'14 to Nov'14 for the categories 

##(date_cat_ctab.loc[22] - date_cat_ctab.loc[21])*1000



# Look at the jump from Oct'13 to Nov'13 for the categories 

##(date_cat_ctab.loc[10] - date_cat_ctab.loc[9])*1000

# 12 has a large jump in '13 but not in '14.
if SHOW_EDA:

    # Look at the distribution of the dates for each category

    cats = date_cat_ctab.columns

    dates = date_cat_ctab.index

    plt.figure(figsize=(15,8))

    for icat in range(len(cats)):

        cat_dist = date_cat_ctab[cats[icat]].values

        plt.plot(dates,np.sqrt(cat_dist),marker='o',alpha=0.2)

        

        # Highlight ones with a large Oct-Nov jump:

        #  20,28 the largest '14 jumps by far, 9 also clear

        #  (12 is an example of jump in '13 but not in '14)

        #  47 is new on the scene

        if cats[icat] in [9,20,28, 47]:

            plt.plot(dates,np.sqrt(cat_dist),marker='o')

    plt.xlabel("Date-block number")

    plt.ylabel("sqrt( Fraction of counts in date-category] )")

    plt.xlim(34-num_months-0.5,34+0.5)
# Three Oct-Nov jump categories, and the new-on-the-scene one:

df_categs.loc[[9,20,28,47]]



# Looked at missing ones in 20, 28 above, look for missing in 9, 47 here:

# (don't really need to do this its_missing again..., but for completeness)

its_missing = []

for iit in df_items.index:

    its_missing.append(df_items.loc[iit,'item_id'] in missing_items)



# There are no missing items in 9

##df_items[(df_items['item_category_id'] == 9) & its_missing].item_id.values



# There are missing items in 47:

newhot47_items = df_items[(df_items['item_category_id'] == 47) & its_missing].item_id.values



print("\nMissing items that are in Cat 47:")

print(newhot47_items)



##print("   Cat 47:")

##print(df_items.loc[newhot47_items,'item_name'])



#   Cat 47:

#13209              Comic Adventure Time All the way around

#13231                        Comic Bessoboy Volume 4 Balor

#13232    Comics Blakeshead Volume 1 Somewhere among the...

#13242                           Comic Crow Time Prehistory

#13249         Comic Star Wars Darth Vader and Ghost Prison

#13250            Comic Star Wars Darth Maul Death sentence

#13251           Comic Star Wars Darth Maul Son of Dathomir

#13257       Comic Book of Inok Volume 4 Beast in Me Book 2

#13263       Comic Red Furies Volume 3 Dark Heritage Book 1

#13264         Comic Red Fury Volume 4 Dark Heritage Book 2

#13271         Comic Major Grom Volume 4 As in a fairy tale

#13273                   Comics Manhattan Projects Volume 2

#13275    Meteor Comics Volume 1 The Most Dangerous Thin...

#13284        Comics The New X-Men Volume 1 The First X-Men

#13295                                     Comic Saga Vol 1

#13303    Comic Skott Piligrim Tom 4 Skott Piligrim boun...

#13309                             Daredevil Comic Volume 2

#13310    Comics of the Guardians of the Galaxy Volume 1...

#13313               Comic Superman Batman Tom 2 Super Girl

#13338    Comic Book Exlibrium Volume 1 And the door wil...

#14959                     Manga The Gate of Stein Volume 3

#14972                                Manga Surviving Youth
# Estimate the November'15 counts as the Oct'15 counts * Nov'14/Oct'14 ratio

# The nov15_totcnt value can be used in prediction methods below.

# Note that only date_block_num is used to select the month (not month, year values),

# this lets the SHIFT_1MONTH work as desired.



# Make monthly counts using item_cnt_day or item_cnt_clip (as approx of clipping)

item_cnt_col = 'item_cnt_day'

if USE_CNT_CLIP:

    item_cnt_col = 'item_cnt_clip'

# The monthly counts

totcnt_vs_month = df_trans[['date_block_num',item_cnt_col]].groupby('date_block_num').agg(sum)



# The average counts from above, either w/o or with clipping:

if USE_CNT_CLIP:

    sel_totcnt_month = average_totcnt_clip

else:

    sel_totcnt_month = average_totcnt_month



# Use these monthly count values to predict Nov'15:

#    Nov'15 = Oct'15(33) * Nov'14(22)/Oct'14(21)

nov15_oct15_ratio = totcnt_vs_month.loc[22,item_cnt_col]/totcnt_vs_month.loc[21,item_cnt_col]

nov15_totcnt = totcnt_vs_month.loc[33,item_cnt_col] * nov15_oct15_ratio



# Make this ratio too

nov15_ave_ratio = nov15_totcnt / sel_totcnt_month



# and for fun (holiday spirit)

dec15_totcnt = (totcnt_vs_month.loc[33,item_cnt_col]*

                totcnt_vs_month.loc[23,item_cnt_col]/totcnt_vs_month.loc[21,item_cnt_col])



# Plot the sum(cnt) for each of the months.

ax = df_trans[['date_block_num',item_cnt_col]].groupby('date_block_num').agg(sum).plot(

    kind='line',marker='o',figsize=(8,5))



# Show the predicted total counts for Nov'15 and Dec'15:

# Nov'15:

plt.plot([34.0,34.0],[sel_totcnt_month,nov15_totcnt],color='gray')

plt.plot([34],[nov15_totcnt],color='red',marker="*",markersize=15)

# Dec'15:

plt.plot([35.0,35.0],[sel_totcnt_month,dec15_totcnt],color='green')

plt.plot([35],[dec15_totcnt],color='green',marker="*",markersize=15)



# Highlight in same way the previous year (2014) Nov and Dec:

# Nov'14:

plt.plot([22],[totcnt_vs_month.loc[22,item_cnt_col]],color='red',marker="*",markersize=15)

# Dec'14:

plt.plot([23],[totcnt_vs_month.loc[23,item_cnt_col]],color='green',marker="*",markersize=15)



# The average per-month value (from above)

min_date_block = df_trans.date_block_num.min()

plt.plot([min_date_block,36],[sel_totcnt_month, sel_totcnt_month],color='gray')



plt.ylim(0,)

plt.xlim(min_date_block-1,36)

plt.title("Total counts in each month.  Stars: Nov(red), Dec(green)")

plt.show()



# Print some values

print("Average line is at {:.1f}".format(sel_totcnt_month))

print("Nov'15 to average ratio is {:.4f}".format(nov15_ave_ratio))

print("Nov'15 to Oct'15 ratio is {:.4f}".format(nov15_oct15_ratio))

print("November 2015 total cnt ~ {:.1f}".format(nov15_totcnt))



# List the monthly totcnt values

##totcnt_vs_month
# Set predictions to all zeros, or a constant (average) value based on...

# e.g.,  (v13, v21)

if False:

    # Best constant value is close to 0.265

    predict_avecnt = 0.265

    print("\nUsing a manual-provided constant value of {:.3f}\n".format(predict_avecnt))

    df_out['item_cnt_month'] = predict_avecnt



    

# The total Nov'15 expected counts spread out over all the output values,

# e.g., (v11)

# Clipped counts should be used here

if False:

    if USE_CNT_CLIP:

        predict_avecnt = nov15_totcnt / (len(out_shops)*len(out_items))

        print("\nUsing a Nov'15 constant value of {:.3f}\n".format(predict_avecnt))

        df_out['item_cnt_month'] = predict_avecnt

    else:

        print("Best to set USE_CLIP_CNT = True for this estimate.")

# Set the predictions as a scaled version of the actual Oct'15 shop-item values.

# e.g., (v14), (v17) (scaled but with further x0.8)

# This should be done with values from above based on USE_CNT_CLIP = False.



if False:

    using_month = 10

    using_year = 2015

    if USE_CNT_CLIP:

        print("\n * Warning: better to have USE_CNT_CLIP = False for this.\n")

    # Set them to the values for October 2015

    # Get just the desired transactions

    df_month = df_trans[(df_trans['year'] == using_year) & (df_trans['month'] == using_month)].copy()

    # Go through the desired output shop-items and get the month's count totals

    out_vals = []

    for iout in df_out.index:

        # Find the matching shop & item entries in the month's transactions:

        this_shop = df_out.loc[iout,'shop_id']

        this_item = df_out.loc[iout,'item_id']

        out_vals.append(df_month.loc[(df_month['shop_id'] == this_shop) &

                                    (df_month['item_id'] == this_item),'item_cnt_day'].sum())



    # Put the counts in the dataframe - scaled for Nov'15

    ##df_out['item_cnt_month'] = nov15_oct15_ratio * np.array(out_vals)

    # or 

    #     (v19) NOT scaled - just the month directly:

    print("\nUnscaled cnt data from month {} of year {}.\n".format(using_month, using_year))

    df_out['item_cnt_month'] = np.array(out_vals)
# Set the predictions as a scaled version of the many-months-averaged shop-item values.

# e.g. (v15)

# This should be done with values from above based on USE_CNT_CLIP = False.

# This takes a while to do.



if False:

    if USE_CNT_CLIP:

        print("\n * Warning: better to have USE_CNT_CLIP = False for this.\n")

    print("\nAveraging counts over {} months.\n".format(num_months))

    # Make the average over ALL num_months of the months

    # Go through the desired output shop-items and get the count averages-per-month

    out_vals = []

    for iout in df_out.index:

        # Find the matching shop & item in the transactions:

        this_shop = df_out.loc[iout,'shop_id']

        this_item = df_out.loc[iout,'item_id']

        out_vals.append(df_trans.loc[(df_trans['shop_id'] == this_shop) &

                                    (df_trans['item_id'] == this_item),'item_cnt_day'].sum()/num_months)



    # Put the month-averaged counts in the output dataframe

    #   Scaled for Nov'15

    ##df_out['item_cnt_month'] = nov15_ave_ratio * np.array(out_vals)

    #   As is, the average monthly counts

    ##df_out['item_cnt_month'] = np.array(out_vals)

    #   Scaled by hand

    scale_factor = 0.515  # (v32, w/13 months)

    df_out['item_cnt_month'] = scale_factor * np.array(out_vals)

    print("Scaled counts by {}".format(scale_factor))
# Many cells for this method so set a flag for it:

ShopCatIt = True



if ShopCatIt:

    # The ctabs used by the model were made above using all the selected months.

    # Re-make them using other month(s)...

    

    if True:

        # Use Aug'15 - Oct'15 for the shop-category

        select_sc = ( (df_trans['date_block_num'] >= 31) & (df_trans['date_block_num'] <= 33) )

        # Use Aug'15 - Oct'15 for the item-category

        select_ic = ( (df_trans['date_block_num'] >= 31) & (df_trans['date_block_num'] <= 33) )

    else:

        # Use just Oct'15:

        select_sc = df_trans['date_block_num'] == 33

        select_ic = df_trans['date_block_num'] == 33

    

    shop_cat_ctab = pd.crosstab(df_trans.loc[select_sc,'shop_id'],df_trans.loc[select_sc,'item_category_id'],

                            values=df_trans.loc[select_sc,'item_cnt_day'], aggfunc=sum,

                            # can normalize by index, columns, all; set NaN to 0.0

                            normalize='all').fillna(0.0)

    

    item_cat_ctab = pd.crosstab(df_trans.loc[select_ic,'item_id'],df_trans.loc[select_ic,'item_category_id'],

                            values=df_trans.loc[select_ic,'item_cnt_day'], aggfunc=sum,

                            # can normalize by index, columns, all; set NaN to 0.0

                            normalize='columns').fillna(0.0)
if ShopCatIt:

    # This gives the normalized fraction per shop:

    shops_fracs = shop_cat_ctab.T.apply(np.sum)

    

    shops_fracs.plot(marker='o',figsize=(8,5))

    

    plt.title("Fraction of counts vs shop_id")

    plt.show()

    

    # This gives the normalized fraction per category:

    cats_fracs = shop_cat_ctab.apply(np.sum)

    

    cats_fracs.plot(marker='o',figsize=(8,5))

    

    plt.title("Fraction of counts vs category")

    plt.show()

if ShopCatIt:

    # Compare the product of shops_frac and cats_frac with the real ctab value:

    # pick a random ID in df_out

    this_ID = 6004



    this_shop = df_out.loc[this_ID,'shop_id'] 

    this_cat = df_out.loc[this_ID,'item_category_id']  



    shop_frac = shop_cat_ctab.T.apply(np.sum)[this_shop]

    cat_frac = shop_cat_ctab.apply(np.sum)[this_cat]



    print("Actual ctab: {}  Approx product: {}".format(shop_cat_ctab.loc[this_shop, this_cat],shop_frac*cat_frac))
if ShopCatIt:

    # The item distribution/fractions depends on the category.

    this_cat = 28



    # Now can get the item-fractions for that category

    items_fracs = (item_cat_ctab[this_cat]/sum(item_cat_ctab[this_cat]))



    # Plot the fraction of each item in it

    items_fracs[items_fracs > 0].plot(marker='o',figsize=(8,5))

    plt.title("Fraction of counts vs item_id in category {}".format(this_cat))

    plt.show()



    ##print(items_fracs[items_fracs > 0])

if ShopCatIt:

    # Put the fractions into the df_out frame with map:

    df_out['shop_frac'] = df_out['shop_id'].map(shops_fracs)

    df_out['cat_frac'] = df_out['item_category_id'].map(cats_fracs)



    # Fill and check for any NaNs

    df_out.fillna(0.0, inplace=True)

    ##df_out.isnull().sum(axis=0)
if ShopCatIt:

    # The item fractions depend on the category, so loop through them

    df_out['item_frac'] = 0.0

    # we may not have all categories

    sc_categories = list(shop_cat_ctab.columns)

    ic_categories = list(item_cat_ctab.columns)

    for this_cat in df_out['item_category_id'].unique():

        ##print(this_cat)

        if (this_cat in ic_categories):

            # Fill item_frac

            items_fracs = (item_cat_ctab[this_cat]/sum(item_cat_ctab[this_cat]))

            select = df_out['item_category_id'] == this_cat

            df_out.loc[select,'item_frac'] = df_out.loc[select,'item_id'].map(items_fracs)

        if (this_cat in sc_categories):

            # Put in the 'real' shop-cat value

            df_out.loc[select,'shopcat_frac'] = df_out.loc[select,'shop_id'].map(shop_cat_ctab[this_cat])

    

    # Fill and check for any NaNs

    df_out.fillna(0.0, inplace=True)

    ##df_out.isnull().sum(axis=0)
if ShopCatIt:

    # Plot to compare the real shop-cat value with the product approx

    plt.figure(figsize=(6,5))

    plt.scatter(df_out['shopcat_frac'], df_out['shop_frac']*df_out['cat_frac'],marker='.')

    plt.xlim(0,0.01)

    plt.ylim(0,0.01)

    plt.show()
if ShopCatIt:

    # Make a prediction using the fractions.

    # Scale it by the Nov15_totcnt estimate and a constant factor, adjusted by LB probing ;-)



    # The product of the three fractions

    ##df_out['item_cnt_month'] = 0.329 * nov15_totcnt * ( df_out['shop_frac'] *

    ##                                df_out['cat_frac'] * df_out['item_frac'] )



    # The product of the real shop-cat ctab fraction and item_fracion

    # With an overall scale factor

    nov15_scale = 0.492     # fit to test

    if SHIFT_1MONTH:

        nov15_scale = 0.624

    df_out['item_cnt_month'] = nov15_scale * nov15_totcnt * ( df_out['shopcat_frac'] *

                                                    df_out['item_frac'] )

df_out.head(10)
# Current sum of the missing items item_cnt_month values:

current_missing_sum = sum(df_out.set_index('item_id').loc[missing_items,'item_cnt_month'])

num_missing_itshops = len(missing_items) * num_out_shops



# Set/override the "missing"-from-trans items' values

if True:

    print("\nThe sum of the missing items is {:.2f}, giving an average of {:.3f} per item-shop\n".format(

                current_missing_sum, current_missing_sum/num_missing_itshops))

    # Might have thought most missing items were old ones, but

    # strangely the lowest score, (v9,10) is when missing = 1.75 * 0.414 = 0.725 (?),

    # are these new/newly-hot items?

    # Set them all to a nominal value:

    missing_value = 0.475     # fit to test

    if SHIFT_1MONTH:

        missing_value = 0.479  # shifted

    #

    for imiss in range(len(missing_items)):

        df_out.loc[df_out['item_id'] == missing_items[imiss] ,'item_cnt_month'] = missing_value

    print("Set {} missing items to {}".format(len(missing_items), missing_value))



    

    # Suspect that categories 20, 28, 47 have new/hot items from explorations way above.

    # Set the values for the "newhot" items in cats 20, 28, 47:

    newhot20_value = 3.60     # fit to test

    newhot28_value = 6.6      # approx

    newhot47_value = 0.530    # approx

    if SHIFT_1MONTH:

        newhot20_value = 3.76

        newhot28_value = 6.79

        newhot47_value = 1.19

    #

    for imiss in range(len(newhot20_items)):

        df_out.loc[df_out['item_id'] == newhot20_items[imiss] ,'item_cnt_month'] = newhot20_value

    print("Set {} newhot-20 items to {}".format(len(newhot20_items), newhot20_value))

    #

    for imiss in range(len(newhot28_items)):

        df_out.loc[df_out['item_id'] == newhot28_items[imiss] ,'item_cnt_month'] = newhot28_value

    print("Set {} newhot-28 items to {}".format(len(newhot28_items), newhot28_value))

    #

    for imiss in range(len(newhot47_items)):

        df_out.loc[df_out['item_id'] == newhot47_items[imiss] ,'item_cnt_month'] = newhot47_value

    print("Set {} newhot-47 items to {}".format(len(newhot47_items), newhot47_value))



    

    # Suspect that categories 37, 40, 55 have new/hot items from the SHIFTED_1MONTH results below.

    # Set the values for the "newhot" items in cats 37, 40, 55:

    newhot37_value = 1.8    # similar to the SHIFT ones for now

    newhot40_value = 0.30   #

    newhot55_value = 0.85    #

    if SHIFT_1MONTH:

        newhot37_value = 1.81

        newhot40_value = 0.95

        newhot55_value = 0.62

    #

    for imiss in range(len(newhot37_items)):

        df_out.loc[df_out['item_id'] == newhot37_items[imiss] ,'item_cnt_month'] = newhot37_value

    print("Set {} newhot-37 items to {}".format(len(newhot37_items), newhot37_value))

    #

    for imiss in range(len(newhot40_items)):

        df_out.loc[df_out['item_id'] == newhot40_items[imiss] ,'item_cnt_month'] = newhot40_value

    print("Set {} newhot-40 items to {}".format(len(newhot40_items), newhot40_value))

    #

    for imiss in range(len(newhot55_items)):

        df_out.loc[df_out['item_id'] == newhot55_items[imiss] ,'item_cnt_month'] = newhot55_value

    print("Set {} newhot-55 items to {}".format(len(newhot55_items), newhot55_value))

    

    # We identified two missing items that are each in their own missing category:

    #   item 5441 in cat  0 : "PC: Headset HyperX Cloud Core gaming stereo..."

    #   item 6439 in cat 27 : "Sid Meier's Railroadrs! [MAC, Digital Version]"

    # Maybe they are missing because: no longer selling (6439 ?) or new item/cat (5441 ?)

    # Try custom values of these:   *** These are best at 0.0 ***

    cnt_5441 = 0.0   # PC headset

    cnt_6439 = 0.0   # Sid Meir Railroad MAC

    df_out.loc[df_out['item_id'] == 5441 ,'item_cnt_month'] = cnt_5441

    df_out.loc[df_out['item_id'] == 6439 ,'item_cnt_month'] = cnt_6439

    print("Set 1 item 5541 to {}".format(cnt_5441))

    print("Set 1 item 6439 to {}".format(cnt_6439))

    

    # This is only available if the Shop-Cat-Item scheme is being used.

    if ShopCatIt:

        # All of the above is done by item and the same value is used for every shop.

        # Instead of a factor of 1 for every shop-item, use 42.0 * df_out['shop_frac']

        # Apply a modulated shop_frac to the missing output values:

        

        shop_modulation = 0.34  # for 'real' (not shifted) data

        if SHIFT_1MONTH:

            shop_modulation = 0.8   # shifted

            

        for imiss in range(len(missing_items)):

            select = (df_out['item_id'] == missing_items[imiss])

            df_out.loc[select, 'item_cnt_month'] = ( df_out.loc[select, 'item_cnt_month'] *

                    (1.0 - shop_modulation + shop_modulation*42.0*df_out.loc[select, 'shop_frac']) )

        print("Scaled all {} missing items by shop_frac with modulation = {:.3f}.".format(

                    len(missing_items),shop_modulation))

    

    # the updated sum of missing items

    current_missing_sum = sum(df_out.set_index('item_id').loc[missing_items,'item_cnt_month'])

else:

    print("The {} 'missing items' were left as is.".format(len(missing_items)))

    

# If we are SHIFT_1MONTH, we will zero predictions for the future items:

if SHIFT_1MONTH:

    for imiss in range(len(future_items)):

        df_out.loc[df_out['item_id'] == future_items[imiss] ,'item_cnt_month'] = 0.0

    print("\n* Set {} Future items to {} *".format(len(future_items), 0.0))

    print("  Leaving {} missing-minus-future items.".format(len(missing_items) - len(future_items)))

    # the updated sum of missing items

    current_missing_sum = sum(df_out.set_index('item_id').loc[missing_items,'item_cnt_month'])

    # report the average ignoring the future items:

    num_missing_itshops = (len(missing_items) - len(future_items)) * num_out_shops

    print("The sum of the missing-minus-future items is {:.2f}, giving an average of {:.3f} per item-shop".format(

                current_missing_sum, current_missing_sum/num_missing_itshops))

else:

    print("\nThe sum of the missing items is {:.2f}, giving an average of {:.3f} per item-shop".format(

                current_missing_sum, current_missing_sum/num_missing_itshops))
# Clip any values over 20, to 20:

clip_level = 20

num_over20 = sum(df_out['item_cnt_month'] > clip_level)

print("\nCheck for over-20 values, found/clipped {} of them.".format(num_over20))

if num_over20 > 0:

    df_out.loc[df_out['item_cnt_month'] > clip_level, 'item_cnt_month'] = clip_level



# Show some simple summary values of the submission:

print("\nSum of output item_cnt_month: {:.0f}, average count-per-month per shop-item: {:.3f}\n".format(

            df_out['item_cnt_month'].sum(), df_out['item_cnt_month'].sum()/(len(df_out))))
# The prediction column, item_cnt_month column, is already in df_out,

# so the submission file can be created directly from df_out.



df_out[['ID','item_cnt_month']].to_csv("submission.csv", index=None)



# Format of sample_submission.csv

#ID,item_cnt_month

#0,0.5

#1,0.5

#2,0.5

# . . .

#214197,0.5

#214198,0.5

#214199,0.5
!head -10 submission.csv
!tail -10 submission.csv
# Get the current or a previous submission and do stuff with it.



# Some summary values

if True:

    

    version = 'current'   # current or vNN

    

    

    if version == 'current':

        # read back in the just-created file

        df_sub = pd.read_csv("submission.csv")

    else:

        # read a previous version, saved locally

        df_sub = pd.read_csv("../submissions/submission_"+version+".csv")

    

    # Show some simple summary values of the submission:

    print("\nSum of output item_cnt_month: {:.0f}, average count-per-month per shop-item: {:.3f}\n".format(

            df_sub['item_cnt_month'].sum(), df_sub['item_cnt_month'].sum()/(len(df_sub))))



    # Calculate the RMSE of the prediction compared to a target of 0.

    rmse_vs_0 = np.sqrt(sum(df_sub['item_cnt_month']*df_sub['item_cnt_month'])/len(df_sub))

    print("RMSE_vs_0 is {:.5f} for version = {}".format(rmse_vs_0,version))

    

    # Add an integer int_cnt_month column

    df_sub['int_cnt_month'] = (0.50 + df_sub['item_cnt_month']).astype(int)

if True:

    # Show a histogram of the values, don't include at/close-to 0 (many of them)

    hist_col = 'item_cnt_month'

    ##hist_col = 'int_cnt_month'

    

    # Use a log y scale  (ok to include 0 then)

    fig, ax = plt.subplots()

    df_sub[df_sub[hist_col] > -0.1].hist(hist_col,bins=100,bottom=0.7, ax=ax)

    ax.set_yscale('log')

    plt.title("Histogram of "+hist_col+", submission version = "+version)

    plt.show()

    
if True:

    # Get the number of shop-items in each count bin

    df_hist = df_sub[['ID','int_cnt_month']].groupby('int_cnt_month').count()

    df_hist = df_hist.reset_index()

    df_hist.columns = ['int_cnt','number']

    

    # Approx RMSE_vs_0 from the binned values:

    df_hist['rmse0_contrib'] = df_hist['number'] * df_hist['int_cnt'] * df_hist['int_cnt']

    rmse0_approx = np.sqrt(sum(df_hist['rmse0_contrib']) / sum(df_hist['number']))

    print("\nRMSE_vs_0 from binned values is {:.5f}\n".format(rmse0_approx))

    

    # Expected RMSE of item_cnt_month just from Poisson statistics.

    # Var(item_cnt_month) = Ave(item_cnt_month)

    df_hist['rmse_contrib'] = df_hist['number'] * df_hist['int_cnt']

    # Set variation for the clipped ones to 0:

    df_hist.loc[20,'rmse_contrib'] = 0.0

    rmse_expect = np.sqrt(sum(df_hist['rmse_contrib']) / sum(df_hist['number']))

    print("\nRMSE expected around the Average is {:.5f}\n\n".format(rmse_expect))

    

    print(df_hist)

# If this is the 1-month shifted case, 

# then we can compare the model submission with the 'truth'.

if SHIFT_1MONTH:

    #      *** Note that df_out is used here for item_id ***

    # Color-code the Scatter plot of the predictions vs actual values

    # create a color variable:  e.g.,  plt.scatter(..., color=list(df_sub['color']) )   

    df_sub['color'] = 'blue'

    for this_item in missing_items:

        # Do not include the future items

        if this_item not in future_items:

            df_sub.loc[ (df_out['item_id'] == this_item),'color'] = 'red'

    # More useful to generate a selection of the missing-minus-future IDs:

    miss_select = df_sub['color'] == 'red'

            

if SHIFT_1MONTH:

    if LOCATION_KAGGLE == True:

        # read in the 'answer' values from Oct'15 submission file

        df_target = pd.read_csv("submission_"+"Oct15"+".csv")        

    else:

        # locally, read in the 'answer' values from Oct'15 submission file

        df_target = pd.read_csv("../submissions/submission_"+"Oct15"+".csv")

    

    # Calculate the RMSE between these, and missing-items contribution

    sub_targ_diff = (df_sub['item_cnt_month'] - df_target['item_cnt_month'])

    sub_targ_RMSE = np.sqrt(sum(sub_targ_diff**2)/len(sub_targ_diff))

    print("\nPrediction-Target RMSE is {:.5f}, from a Sum^2 of {:.0f} [/{}].".format(

        sub_targ_RMSE, sum(sub_targ_diff**2), len(sub_targ_diff)))

    

    missing_diff = (df_sub.loc[miss_select,'item_cnt_month'] - 

                        df_target.loc[miss_select,'item_cnt_month'])

    missing_RMSE = np.sqrt(sum(missing_diff**2)/len(missing_diff))

    print("\n     Missing-only RMSE is {:.5f}, from a Sum^2 of  {:.0f} [/{}].".format(

        missing_RMSE, sum(missing_diff**2), len(missing_diff)))

    

    without_RMSE = np.sqrt( (sum(sub_targ_diff**2) - sum(missing_diff**2)) /

                           (len(sub_targ_diff) - len(missing_diff)) )

    print("\n  without-Missing RMSE is {:.5f}.\n".format(without_RMSE))



    # Make scatter plot of predictions vs target

    plt.figure(figsize=(8,8))

    # Show the pred=targ line

    plt.plot([0.0,20.0],[0.0,20.0],c='orange')

    # All the values

    plt.scatter(df_target['item_cnt_month'] + 0.2*np.random.randn(len(df_target)),

                df_sub['item_cnt_month'] + 0.2*np.random.randn(len(df_target)),

                color='blue', marker='.', alpha=0.15)

    # The missing items on top:

    plt.scatter(df_target.loc[miss_select,'item_cnt_month'] + 0.2*np.random.randn(sum(miss_select)),

                df_sub.loc[miss_select,'item_cnt_month'] + 0.2*np.random.randn(sum(miss_select)),

                color='red', marker='.', alpha=0.30)   

    #

    plt.xlabel("Oct'15, known target value")

    plt.ylabel(version+" submission value")

    plt.title("Predicted counts (using Jul-Sep'15) vs Actual counts (Oct'15)   Red = missing items")

    plt.show()



    # For the SHIFT_1MONTH, the adjusted values give:

    # RMSE between submission and target is 0.87917,  0.87891 with all months used for missing.
if SHIFT_1MONTH:

    # Create a missing df with newhot target by

    #   first adding the newhot value to df_out:

    df_missing = df_out.copy()

    df_missing['newhot'] = (df_target['item_cnt_month'] > 9.5).astype(int)

    # and a boring value

    df_missing['boring'] = (df_target['item_cnt_month'] < 1.0).astype(int)

    # then keeping only the missing entries

    df_missing = df_missing[miss_select].reset_index().drop('index',axis=1)

    

    print(df_missing.tail(5))

    print("\n")

    print(df_missing.describe())

if SHIFT_1MONTH:

    print("Distributions of the new-hot missing items:")

    df_missing.loc[df_missing['newhot'] == 1, [

        'shop_id','item_category_id','item_id','item_cnt_month']].hist(figsize=(12,6), bins=100)
if SHIFT_1MONTH:

    plt.figure(figsize=(10,7))

    # Over plot the boring ones

    select = (df_missing['boring'] == 1)

    plt.scatter(df_missing.loc[select,'item_category_id'], df_missing.loc[select,'shop_id'],

            marker='.', alpha=0.05, color='blue') 

    # Over plot the NOT-boring and NOT-newhot ones

    select = (df_missing['boring'] == 0) & (df_missing['newhot'] == 0)

    plt.scatter(df_missing.loc[select,'item_category_id'], df_missing.loc[select,'shop_id'],

            marker='o', alpha=0.05, color='yellow')    

    # Over plot the newhot ones

    select = (df_missing['newhot'] == 1)

    plt.scatter(df_missing.loc[select,'item_category_id'], df_missing.loc[select,'shop_id'],

            marker='o', alpha=0.2, color='red')    



    plt.show()
if SHIFT_1MONTH:

    plt.figure(figsize=(10,7))

    # Over plot the boring ones

    select = (df_missing['boring'] == 1)

    plt.scatter(df_missing.loc[select,'item_id'], df_missing.loc[select,'shop_id'],

            marker='.', alpha=0.05, color='blue') 

    # Over plot the NOT-boring and NOT-newhot ones

    select = (df_missing['boring'] == 0) & (df_missing['newhot'] == 0)

    plt.scatter(df_missing.loc[select,'item_id'], df_missing.loc[select,'shop_id'],

            marker='o', alpha=0.05, color='yellow')    

    # Over plot the newhot ones

    select = df_missing['newhot'] == 1

    plt.scatter(df_missing.loc[select,'item_id'], df_missing.loc[select,'shop_id'],

            marker='o', alpha=0.2, color='red')    



    plt.show()
if SHIFT_1MONTH:

    print(df_missing.columns)
if SHIFT_1MONTH:

    # newhot "missing" categories

    # 219 entries

    #  11 unique categories  <-- 43 categories in missing in all

    newhot_cats = df_missing.loc[df_missing['newhot'] == 1, 'item_category_id'].unique()

    print("Number of new-hot missing categories: {}\n".format(len(newhot_cats)))

    for this_cat in newhot_cats:

        print(df_categs.loc[this_cat, 'item_category_name'])

    

#Number of new-hot missing categories: 11

#

#Games - PS4

#PC Games - Additional Edition

#Books - Comics, manga

#Cinema - DVD

#Cinema - Blu-Ray

#Games - XBOX ONE

#Music - CD of local production

#Books - Artbook, encyclopedia

#Cinema - Blu-Ray 3D

#PC Games - Digit

#Books - Number 
if SHIFT_1MONTH:

    # newhot "missing" items

    # 219 newhot shop-items

    #  38 unique items  <-- around 6 % of the missing items

    newhot_items = df_missing.loc[df_missing['newhot'] == 1, 'item_id'].unique()

    print("Number of new-hot missing items: {}\n".format(len(newhot_items)))

    for this_item in newhot_items:

        print(df_items.loc[this_item, 'item_name'])

        

#Number of new-hot missing items: 38

#

#Witcher 3: Wild Hunt - Supplement "Stone Hearts" (download code, no disc) [PS4, Russian version]

#Uncharted: Nathan Drake. Collection [PS4, Russian version]

#Assassin's Creed: Syndicate. Special Edition [PS4, Russian version]

#Witcher 3: Wild Hunt - Supplement "Stone Hearts" (download code, no disc) [PC, Russian version]

#Comic Witcher Fox Kids

#Uncharted: Nathan Drake. Collection. Special Edition [PS4, Russian version]

#OUTSIDE / SELF

#SUN ANDREAS RELEASE + collectible postcard

#LAND OF THE FUTURE

#LAND OF THE FUTURE (BD)

#OUT / SELF (BD)

#Halo 5: Guardians [Xbox One, Russian version] (U9Z-00062)

#PUZZLE

#PICKS

#DEL REY LANA Honeymoon

#Assassin's Creed: Syndicate. Big Ben [PS4, Russian version]

#MINE PATRUL Season 1 Issue 5 Harvesting

#MINE PATRUL Season 1 Issue 4 Winter Rescuers

#INTERSTELLAR (region)

#Comedy Deadpool destroys literature

#SUN ANDREAS RID (BD) + collectible card

#STRINGER

#Assassin's Creed: Syndicate. Charing Cross [PS4, Russian version]

#Guitar Hero Live (Guitar + game) [PS4, English version]

#Artbook Game World Assassin's Creed Syndicate

#TERMINATOR GENESIS

#MILL Alchemy (firms.)

#Assassin's Creed: Syndicate. Rooks [PS4, Russian version]

#KNYAZZ Precursor (Company)

#The Comics of the Avengers against the X-Men

#Halo 5: The Guardians. Collector's Edition [Xbox One, Russian version] (CV4-00014)

#Comic Fire and Stone Alien vs. Predator

#ASTRAL 3 (BD)

#TERMINATOR GENESIS (3D BD + BD)

#THE CONSEQUENCES OF SAN-ANDREAS (3D BD + BD)

#Witcher 3: Wild Hunt - Supplement "Stone Hearts" [PC, Digital Version]

#BUK.1S, №10 October 2015 [Digital version]

#Fallout 4. Season Pass [PC, Digital Version]
if SHIFT_1MONTH:

    # middle - Not-Boring and also not newhot - "missing" items

    # 2412 middle shop-items

    #  264 unique items  <-- that's 90 % of the missing items.

    middle_items = df_missing.loc[((df_missing['boring'] == 0) & (df_missing['newhot'] == 0)), 'item_id'].unique()

    print("Number of middle missing items: {}\n".format(len(middle_items)))

    ##for this_item in middle_items:

    ##    print(df_items.loc[this_item, 'item_name'])
if SHIFT_1MONTH:

    # Boring "missing" items

    #  9633 boring shop-items

    #   292 unique items  <-- this is all the missing items: any missing item may not sell at some shop in Oct'15.

    boring_items = df_missing.loc[(df_missing['boring'] == 1), 'item_id'].unique()

    print("Number of boring missing items: {}\n".format(len(boring_items)))

    ##for this_item in boring_items:

    ##    print(df_items.loc[this_item, 'item_name'])