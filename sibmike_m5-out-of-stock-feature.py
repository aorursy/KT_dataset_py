# General imports

import numpy as np

import pandas as pd

import os, sys, gc, time, warnings, pickle, psutil, random



from math import ceil

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder



warnings.filterwarnings('ignore')
# Now we have 3 sets of features

grid_df = pd.concat([pd.read_pickle('/kaggle/input/m5-simple-fe/grid_part_1.pkl'),

                     pd.read_pickle('/kaggle/input/m5-simple-fe/grid_part_2.pkl').iloc[:,2:],

                     pd.read_pickle('/kaggle/input/m5-simple-fe/grid_part_3.pkl').iloc[:,2:]],

                     axis=1)

grid_df.head()
# Lets pick a product with pronounced gaps: 'HOBBIES_1_288_CA_1_validation'

m = grid_df.id=='HOBBIES_1_288_CA_1_validation'

sales_ts = grid_df.loc[m,'sales'].values

sales_ts
def gap_finder(ts):

    

    # this function finds gaps and calculates their length:

    ts = (~(ts > 0)).astype(int)



    for i, val in enumerate(ts):

        if val == 0: 

            continue

        else: 

            ts[i] += ts[i-1]

            ts[i-1] = -1

    return ts



def gap_counter(ts):

    

    # value_counts for gaps lengths

    

    counts = np.unique(ts, return_counts=True)

    return dict(zip(counts[0], counts[1]))



m = grid_df.id=='HOBBIES_1_288_CA_1_validation'



sales_gaps = gap_counter(gap_finder(grid_df.loc[m,'sales'].values))

sales_gaps
# Lets try to build a synthetic series with gaps.

# We assume products have a constant chance of being sold on any particular day.

# So for every day we will flip an unfair coin with probability of sale equal to 'dates_with sale'/'all_dates'

# >>>This might be an oversimplistic assumption, a moving window might be used instead.



def synth_sales(gaps_dict, min_gap=1000):

    

    sum_sale_days = gaps_dict[0] # key '0' gives number of days with sales

    sum_days = sum(gaps_dict.values()) # sum of all keys - number of all days

    

    # Sum of days in gaps longer than minimum gap length:

    sum_gap_length = sum([k for k in [*sales_gaps.keys()] if k > min_gap])

    

    # Exlude all the gaps longer than min_gap_length from probability calculation:

    p = sum_sale_days/(sum_days-sum_gap_length)

    

    return np.random.binomial(1, p, sum(gaps_dict.values()))



synth_sales_gaps = gap_counter(gap_finder(synth_sales(sales_gaps)))

synth_sales_gaps
# As you can see we dont have results > 20. 

# To make sure this is a consistent result lets run simulation 1,000 times:

# >>>> n equal 1,000 might be an overshot for some ids.



n=1000



sym_df = pd.DataFrame([gap_counter(gap_finder(synth_sales(sales_gaps))) for i in range(n)])

gap_length_prob = (sym_df.sum(axis=0)/n).sort_index()

gap_length_prob
# The shortest gap that has been seen in less than 5% of simulated series:

min_gap_length = min(gap_length_prob[gap_length_prob<0.05].index)

min_gap_length
# Now we exclude the gaps longer than min_gap_length from 'probability of sale' calculation, 

# because we assume them to be non-random.

# Run the simulation recursively until the min_gap_length does not decrease:



n=1000

new_min_gap_length = 0



while new_min_gap_length != min_gap_length:

    

    if new_min_gap_length!=0: min_gap_length=new_min_gap_length



    sym_df = pd.DataFrame([gap_counter(gap_finder(synth_sales(sales_gaps, min_gap_length))) for i in range(n)])

    gap_length_prob = (sym_df.sum(axis=0)/n).sort_index()



    # Lets find the shortes gap that has been seen in les than 5% of simulated series:

    new_min_gap_length = min(gap_length_prob[gap_length_prob<0.05].index)



min_gap_length
# Finally we need to make a feature 'out_of_stock' for the product:

m = grid_df.id=='HOBBIES_1_288_CA_1_validation'

idx = grid_df.loc[m,'sales'].index



gf = pd.Series(gap_finder(grid_df.loc[m,'sales'].values), index = idx)

gf = gf.replace(-1,np.nan).fillna(method='backfill') 

gf = gf > min_gap_length

gf
# Let calculate 'out-of-stock' for the first 30 ids:

grid_df['out_of_stock'] = 0



n=1000

prods = list(grid_df.id.unique())

gap_length_list = []



for prod_id in tqdm(prods[:30]):

    

    m = grid_df.id == prod_id

    idx = grid_df.loc[m,'sales'].index



    sales_gaps = gap_counter(gap_finder(grid_df.loc[m,'sales'].values))



    sym_df = pd.DataFrame([gap_counter(gap_finder(synth_sales(sales_gaps))) for i in range(n)])

    gap_length_prob = (sym_df.sum(axis=0)/n).sort_index()



    min_gap_length = min(gap_length_prob[gap_length_prob<0.05].index)

    new_min_gap_length = 0

    

    while new_min_gap_length < min_gap_length:



        if new_min_gap_length!=0: min_gap_length=new_min_gap_length



        sym_df = pd.DataFrame([gap_counter(gap_finder(synth_sales(sales_gaps, min_gap_length))) for i in range(n)])

        gap_length_prob = (sym_df.sum(axis=0)/n).sort_index()



        # Lets find the shortes gap that has been seen in les than 5% of simulated series:

        new_min_gap_length = min(gap_length_prob[gap_length_prob<0.05].index)



    gf = pd.Series(gap_finder(grid_df.loc[m,'sales'].values), index = idx)

    gf = gf.replace(-1,np.nan).fillna(method='backfill') 

    gf = gf > min_gap_length

    grid_df.loc[m,'out_of_stock'] = gf*1

    gap_length_list += [min_gap_length]
# Let take 'out-of-stock' gap length for different products:

#from collections import Counter

#Counter(gap_length_list)



gap_length_list[:15]
# As you can see the out-of-stock gap length vary extremely from product to product.

# For densily traded products it maybe as short as 8 days, while for rare products it might be over 60 days.



# The current approach to calculating it is SLOW. I will try to refactor it and build the feature in the next notebook.

# Comments are welcome. Please let me know if you spot inefficiencies or mistakes.