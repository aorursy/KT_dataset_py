import pandas as pd

import numpy as np
cust_df = pd.read_csv("/kaggle/input/parisprocessed/new_cust_df.csv")

prod_df = pd.read_csv("/kaggle/input/parisproductfinal/product_final.csv")
cust_df.head(2)
prod_df.head(2)
ratings = pd.read_excel("/kaggle/input/parisdata/rewiews808K.xlsx")
ratings.shape
ratings.head()
ratings['text'].iloc[0]
ratings['url'].iloc[0]
prod_df['url'].iloc[0]
dummy = [i for i in prod_df['accords']]

total_accords = ','.join(word for word in dummy)

words = total_accords.split(",")

words_set = set(words)
my_list = list(words_set)
my_list.sort()
my_list
prod_df['accords'].iloc[0]
samples = prod_df['accords'].iloc[0]
samples
if "woody" in samples:

    print("hello")
def get_accords(x):

    my_list_vals = []

    for i in my_list:

        if i in x:

            my_list_vals.append(1)

        else:

            my_list_vals.append(0)

    return [i for i in my_list_vals]
sdummy = prod_df['accords'].apply(lambda x: get_accords(x))
prod_df['new_accords'] = sdummy
tags = prod_df['new_accords'].apply(pd.Series)

tags.head()
tags.columns = my_list
prod_df = prod_df.join(tags)
prod_df.shape
del prod_df['accords']

del prod_df['new_accords']
prod_df.to_csv("products_finals_with_accords.csv", index=None)