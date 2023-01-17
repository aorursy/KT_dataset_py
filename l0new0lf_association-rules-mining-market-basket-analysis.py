import numpy as np

from itertools import combinations, groupby

from collections import Counter



# Sample data - `(order_id, item_name)`

orders = np.array([

                    (1,'apple'), # can be a list or tuple (to be worked on by `itertools.groupby`) 

                    (1,'egg'),   # eg, `[1,'egg']` or `[1,'milk']`  

                    (1,'milk'), 

                    (2,'egg'), 

                    (2,'milk')

                  ], dtype=object)



# `orders` can be grouped based on `order_id` effficiently using `itertools.groupby`



# Generator that yields item pairs, one-at-a-time

def get_item_pairs(order_item):

    

    # For each order, generate a list of items in that order

    for order_id, order_object in groupby(orders, lambda x: x[0]):

        """ explanation

        

        > `order_id`: "key" based on which differenet `order_object`s will be grouped

                         - it is key because it has index `0` and `lambda x: x[0]` 

                           makes whatever in `0` into key for grouping

        

        > There will be as many iterations as distinct keys.

                         - Here, two iterations – for `1` and for `2`

        

        > `order_object`: list for every iteration for specific "key"

                         - list of `(order_id, item_name)`s or `[order_id, item_name]`s 

                           for that key(here `order_id`) in the iteration

        """

        

        # filter `order_object` which is list of lists/tuples of

        # `(order_id, item_name)` or `[order_id, item_name]` 

        item_list = [item[1] for item in order_object] # at `1` we have specific order (`apple` or `milk` or something else)

                                                       # cz, `orders` is list of `(order_id, item_name)` or `[order_id, item_name]`

                                                            

    

        # For each item list, generate item pairs, one at a time

        for item_pair in combinations(item_list, 2): # returns each possible 2-item set for every iteration

            yield item_pair # see `how to use yeild and why (below)`                                





# Counter iterates through the item pairs returned by our generator and keeps a tally of their occurrence

Counter(get_item_pairs(orders))

def yeild_func(item_list):

    for item_pair in combinations(item_list, 2):

        return item_pair # returns result of first iteration and stops - So, cannot use `return`!!



yeild_func(["x", "y", "z"]) # ('x', 'y') 
def yeild_func(item_list):

    for item_pair in combinations(item_list, 2):

        yield item_pair #return and continues returning with each iteration

                        #must always be inside a function



yeild_func(["x", "y", "z"]) 



# note - as this is a dynamic process, need to use 

# `Counter` (if you are using "keys" as in above `get_item_pairs` function) 

#  or 

# `list` to access it
from collections import Counter

Counter(yeild_func(["x", "y", "z"]))
list(yeild_func(["x", "y", "z"]))
import pandas as pd

import numpy as np

import sys

from itertools import combinations, groupby

from collections import Counter

from IPython.display import display
# Function that returns the size of an object in MB

def size(obj):

    return "{0:.2f} MB".format(sys.getsizeof(obj) / (1000 * 1000))
import os, shutil



base_path = "/kaggle/input/"

files = os.listdir("/kaggle/input/")

extract_dir = "/kaggle/working"

archive_format = "zip"



for file in files:

    print("extracting: ", base_path+file, "/n to: ",extract_dir)

    shutil.unpack_archive(base_path+file, extract_dir, archive_format) 
orders = pd.read_csv('/kaggle/working/order_products__prior.csv')

print('orders -- dimensions: {0};   size: {1}'.format(orders.shape, size(orders)))

display(orders.head())
# Convert from DataFrame to a Series, with `order_id` as index and `product_id` as value

# and name the series-datastructure as `item_id`

orders = orders.set_index('order_id')['product_id'].rename('item_id') # Series will have 'order_id' as it's index values 

                                                                      # and the rest i.e 'product_id' as it's values

                                                                      # note that indices aren't unique

display(orders.head(10))

type(orders)
print('dimensions: {0};   size: {1};   unique_orders: {2};   unique_items: {3}'

      .format(orders.shape, size(orders), len(orders.index.unique()), len(orders.value_counts())))
# Returns frequency counts for items and item pairs (same as term `support-count`)

def freq(iterable):

    if type(iterable) == pd.core.series.Series:

        return iterable.value_counts().rename("freq") # `.value_counts()` on Series returns freq of unique `values` in the series

                                                      # series-datastructure will have name `freq`

    else: 

        return pd.Series(Counter(iterable)).rename("freq") # series-datastructure will have name `freq`



    

# Returns number of unique orders

# `order_item` is Series with `order_id` as index and `product_id` as it's values.

def order_count(order_item):

    return len(set(order_item.index)) # `.index` on Series returns list of all(not unique) index values i.e [1, 2, 2, 2, 1, 1, 4 .... order_ids]

                                      # convert it into set to get unique values and return count





# Returns generator that yields item pairs, one at a time

# `order_item` is Series with `order_id` as index and `product_id` as it's values.

def get_item_pairs(order_item):

    order_item = order_item.reset_index().as_matrix() # `.reset_index()` converts Series into DataFrame with default indices (0,1,2 ...)

                                                      #               changes the index values inside Series into a new col in DataFrame

                                                      # `.as_matrix()` converts DataFrame into 2-D numpy array(forgets col-names) - list of lists

                                                      #                                                                      array([[1, 'apple'],

                                                      #                                                                             [1, 'egg'],

                                                      #                                                                             [1, 'milk'],

                                                      #                                                                             [2, 'apple'],

                                                      #                                                                             [2, 'milk']], dtype=object)

                                                      #                Note: same as `np.array(df)`        

    for order_id, order_object in groupby(order_item, lambda x: x[0]):

        # key - `order_id`

        # `order_object` - list of lists for every key'th iteration

        item_list = [item[1] for item in order_object]

        

        for item_pair in combinations(item_list, 2):

            yield item_pair

            



# Returns frequency and support associated with item

def merge_item_stats(item_pairs, item_stats):

    return (item_pairs

                .merge(item_stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)

                .merge(item_stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True))





# Returns name associated with item

def merge_item_name(rules, item_name):

    columns = ['itemA','itemB','freqAB','supportAB','freqA','supportA','freqB','supportB', 

               'confidenceAtoB','confidenceBtoA','lift']

    rules = (rules

                .merge(item_name.rename(columns={'item_name': 'itemA'}), left_on='item_A', right_on='item_id')

                .merge(item_name.rename(columns={'item_name': 'itemB'}), left_on='item_B', right_on='item_id'))

    return rules[columns]               
def association_rules(order_item, min_support):



    print("Starting order_item: {:22d}".format(len(order_item)))





    # Calculate item frequency and support (`freq` is same as term `support-count`)

    item_stats             = freq(order_item).to_frame("freq") # `.to_frame()` converts the Series into pd.DF with 

                                                               # index - "product_id" ("Milk", "Apple", ...)

                                                               # df with only 1 col - "freq" (freq count of the product)

    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100

                                        # new col `support` added to df with `freq` col





    # Filter from order_item items below min support (Pruning based on Apriori Algorithm)

    qualifying_items       = item_stats[item_stats['support'] >= min_support].index #get indices(i.e "product_id"s) satisfying the condition

    order_item             = order_item[order_item.isin(qualifying_items)] # based on the "indices", filter `order_item`s



    print("Items with support >= {}: {:15d}".format(min_support, len(qualifying_items)))

    print("Remaining order_item: {:21d}".format(len(order_item)))





    # Filter from order_item orders with less than 2 items

    order_size             = freq(order_item.index)

    qualifying_orders      = order_size[order_size >= 2].index

    order_item             = order_item[order_item.index.isin(qualifying_orders)]



    print("Remaining orders with 2+ items: {:11d}".format(len(qualifying_orders)))

    print("Remaining order_item: {:21d}".format(len(order_item)))





    # Recalculate item frequency and support

    item_stats             = freq(order_item).to_frame("freq")

    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100





    # Get item pairs generator

    item_pair_gen          = get_item_pairs(order_item)





    # Calculate item pair frequency and support

    item_pairs              = freq(item_pair_gen).to_frame("freqAB")

    item_pairs['supportAB'] = item_pairs['freqAB'] / len(qualifying_orders) * 100



    print("Item pairs: {:31d}".format(len(item_pairs)))





    # Filter from item_pairs those below min support

    item_pairs              = item_pairs[item_pairs['supportAB'] >= min_support]



    print("Item pairs with support >= {}: {:10d}\n".format(min_support, len(item_pairs)))





    # Create table of association rules and compute relevant metrics

    item_pairs = item_pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})

    item_pairs = merge_item_stats(item_pairs, item_stats)

    

    item_pairs['confidenceAtoB'] = item_pairs['supportAB'] / item_pairs['supportA']

    item_pairs['confidenceBtoA'] = item_pairs['supportAB'] / item_pairs['supportB']

    item_pairs['lift']           = item_pairs['supportAB'] / (item_pairs['supportA'] * item_pairs['supportB'])

    

    

    # Return association rules sorted by lift in descending order

    return item_pairs.sort_values('lift', ascending=False)
%%time

rules = association_rules(orders, 0.01)  
# Replace item ID with item name and display association rules

item_name   = pd.read_csv('/kaggle/working/products.csv')

item_name   = item_name.rename(columns={'product_id':'item_id', 'product_name':'item_name'})

rules_final = merge_item_name(rules, item_name).sort_values('lift', ascending=False)

display(rules_final)