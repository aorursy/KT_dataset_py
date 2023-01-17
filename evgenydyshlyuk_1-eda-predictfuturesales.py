%%writefile libraries.py

# Create a file allowing to import upper level(usefull throughout the whole solution) packages and functions with one line: %run libraries



import os #The functions that the OS module provides allows you to interface with the underlying operating system that Python is running on 



import pickle # Fast saving/loading data



import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)



# Import visualizations

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (30,5) # Set standard output figure size

import seaborn as sns # sns visualization library

from IPython.display import display # Allows to nicely display/output several figures or dataframes in one cell



# Create an output' folder to save data from the notebook

try: os.mkdir('output') # Try to create

except FileExistsError: pass # if already exist pass

        

print('Upper level libraries loaded')
%reset -f

#reset magic function allows one to release all previously used memory. -f (force) parameter allows to run it without confirmation from the user



%run libraries

#jupyter magic function loading standard libraries from the created file.
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Load data from 'input' folder in the current directory

train   = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

items   = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

cats    = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

shops   = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

test    = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

sample  = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
test = test.set_index('ID') #Set index to ID. This way we do not need to drop ID column every time in future calculations
# Show the Loaded Data

# display() allows to output multiple dataframes in one cell

display('train',   train.shape,  train.head(),

        'items',   items.shape,  items.head(),

        'cats',    cats.shape,   cats.head(),

        'shops',   shops.shape,  shops.head(),

        'test',    test.shape,   test.head(),

        'sample',  sample.shape, sample.head()) 
# Define dataframe information function

def df_info(df):

    print('-------------------------------------------shape----------------------------------------------------------------')

    print(df.shape)

    print('-------------------------------------head() and tail(1)---------------------------------------------------------')

    display(df.head(), df.tail(1))

    print('------------------------------------------nunique()-------------------------------------------------------------')

    print(df.nunique())

    print('-------------------------------------describe().round()---------------------------------------------------------')

    print(df.describe().round())

    print('--------------------------------------------info()--------------------------------------------------------------')

    print(df.info())

    print('-------------------------------------------isnull()-------------------------------------------------------------')

    print(df.isnull().sum())

    print('--------------------------------------------isna()--------------------------------------------------------------')

    print(df.isna().sum())

    print('-----------------------------------------duplicated()-----------------------------------------------------------')

    print(len(df[df.duplicated()]))

    print('----------------------------------------------------------------------------------------------------------------')
df_info(train)
# We see 6 duplicates in data, let's drop them

train.drop_duplicates(inplace=True)
# We see a possible typo in item price in train - negative value 

train[train.item_price <= 0 ]
# Only one datapoint - it should be safe to simply remove it

train = train[train.item_price > 0]
#check price distribution

plt.plot(train.item_price)
# There is one clear outlier

print(train[train.item_price > 100000])

print(items[items.item_id == 6066])
# As we see this is a sale of 522 packages in one pack (each one cost 307980/522 = 59 ), let us correct this line

train.item_cnt_day[train.item_id == 6066] = 522

train.item_price[train.item_id == 6066] = 59
# Now let us plot it again

plt.plot(train.item_price)
# Let us plot variation of the mean item price with time

plt.plot(train.groupby(['date_block_num'])['item_price'].mean())
#We do not clearly see much variation of prices within one month of sales 

plt.plot(train[train.date_block_num == 33].item_price)
#Let us see how price is changing for one of the arbitrary taken items

id = 1000 # arbitrary id

plt.figure(figsize=(10,4))

sns.distplot(train[train.item_id == id].item_price, hist_kws={'log':True}, kde = False, bins = 100)



train[train.item_id == id].sort_values(by=['date_block_num'])
# Now let us plot item_cnt_day

plt.plot(train.item_cnt_day)
# Plot the logarithmic histograms for item_cnt_day

sns.distplot(train.item_cnt_day, hist_kws={'log':True}, kde = False, bins = 200)
#Couple outliers above 900

train[train.item_cnt_day > 900]
display(items[items.item_id == 9248])
display(items[items.item_id == 20949],

        items[items.item_id == 11373])
# It's possible that a lot of packets and deliveries were done on some occasion but those have to be some holidays for example.

# I think it's better to remove the points as outliers

train = train[train.item_cnt_day < 900]
# Now let us plot item_cnt_day

plt.plot(train.item_cnt_day)
# Let us see sales distribution per month

sns.countplot(x='date_block_num', data=train);
# Let us see sales distribution over one month

sns.countplot(x='date', data=train[(train.date_block_num == 21)&(train.shop_id == 12)])
# Let's see the sales per shop

sns.countplot(x='shop_id', data=train)
# Let us plot cumulative sales per shop over time. We will use red color for those shops, that are not present in test set.



fig = plt.figure(figsize=(30,36))

for i in range(len(shops)):

    ts=train[train.shop_id == i].groupby(['date_block_num'])['item_cnt_day'].sum()

    plt.subplot(10, 6, i+1)

    plt.bar(ts.index, ts.values)

    plt.xlim((0, 33))

    plt.ylim(0, 12000)

    if i in set(test.shop_id):

        plt.title(str(i) +' '+ shops.shop_name[i], color = 'k')

    else: 

        plt.title(str(i) +' '+ shops.shop_name[i], color = 'r')

plt.show()
# We see that data for some shops was mixed (intentionally I guess), let's fix it

# Якутск Орджоникидзе, 56

train.loc[train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

train.loc[train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

train.loc[train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11

# Now delete those shops from the shops dataframe:

shops.drop([0, 1, 10], inplace = True)



# I think it is also better to remove any data for outbound trade, 

# which is very unusual and misleading (we are not going to predict the outbound trade)

train = train[train.shop_id != 9]

train = train[train.shop_id != 20]

# Now delete those shops from the shops dataframe:

shops.drop([9, 20], inplace = True)



# 12 and 55 are online stores - we cannot remove them because these shops are in test.
# Let us add item_category_id to train and test sets

items_dict = dict(zip(items.item_id, items.item_category_id))

train['item_category_id'] = train['item_id'].map(items_dict)

test['item_category_id'] = test['item_id'].map(items_dict)
# Plot the distribution for sold items relative to the category

fig, ax =plt.subplots(2,1, figsize=(30,10))

sns.countplot(train['item_category_id'], ax=ax[0])

sns.countplot(test['item_category_id'], ax=ax[1])
# We see that some categories are absent in test data but are present in train. Let us remove those categories from train data to make it closer to test.

for i in (set(train.item_category_id) - set(test.item_category_id)):

    train = train[train.item_category_id != i]

    items = items[items.item_category_id != i] # remove them from items

    cats = cats[cats.item_category_id != i]    # remove from cats
# Plot the distribution again

fig, ax =plt.subplots(2,1,  figsize=(30,10))

sns.countplot(train['item_category_id'], ax=ax[0])

sns.countplot(test['item_category_id'], ax=ax[1])
# How many samples in train now?

len(set(train.shop_id))
fig = plt.figure(figsize=(30,60))

i = 1

for shop_id in set(train.shop_id):

    ts=train[train.shop_id == shop_id].groupby(['item_category_id'])['item_cnt_day'].sum()

    plt.subplot(11, 5, i)

    plt.bar(ts.index, ts.values)

    plt.xlim((0, 82))

    if shop_id in set(test.shop_id):

        plt.title(str(shop_id) +' '+ shops.shop_name[shop_id], color = 'k')

    else: 

        plt.title(str(shop_id) +' '+ shops.shop_name[shop_id], color = 'r')

    i+=1

plt.show()
# Shop # 40 showing very different trend from other shops, so let us remove it (it is closed long time ago anyway and we don't need to predict for this shop)

train = train[train.shop_id != 40]

# Now delete the shop from the shops dataframe:

shops.drop([40], inplace = True)



# we do not remove shops # 12 and 55 which are on-line shops and also show different distribution

# shop #55 is an online shop for 1-C Software (business accounting software, #1 in Russia). The sales categories from this shop are only present for this shop and are not present in other shops:

set(train[train.shop_id == 55].item_category_id)
# Now we will save the data, but we do not want to save modifications to data sets at this stage, here we only cleaned the data, so let us drop newly created columns from data. We will modify the data in the next: 2_FeatureEngineering section

train.drop(columns = 'item_category_id', inplace = True)

test.drop(columns = 'item_category_id', inplace = True)



# Save data to the folder to use it in the next part

with open(r'output/1_EDA_data.pkl','wb') as f:

    pickle.dump((train, items, cats, shops, test, sample), f)  

    

'''# Load the saved data in the next section as:

with open(r'output/1_EDA_data.pkl', 'rb') as f:

    (train, items, cats, shops, test, sample) = pickle.load(f)'''