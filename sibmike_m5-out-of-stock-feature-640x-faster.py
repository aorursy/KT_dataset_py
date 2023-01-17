# General imports

import numpy as np

import pandas as pd

import os, sys, gc, time, warnings, pickle, psutil, random

from tqdm import tqdm



import seaborn as sns # data visualization library  

import matplotlib.pyplot as plt



warnings.filterwarnings('ignore')
# Load data

grid_df = pd.read_pickle('/kaggle/input/m5-simple-fe/grid_part_1.pkl')

grid_df.head()
def gap_finder(ts):

    

    # this function finds gaps and calculates their length:

    # note ts: 0 = day with sales, 1 = days with 0 sales

    

    for i, gap in enumerate(ts):

        if gap == 0: 

            continue

        elif i!=0: 

            ts[i] += ts[i-1]

            if ts[i-1]!=0: ts[i-1] = -1

    return ts
# Note: in 'gap' column: 1 is a day without sales:

grid_df['gaps'] = (~(grid_df['sales'] > 0)).astype(int)

total_days = 1941



prods = list(grid_df.id.unique())

s_list = [] #list to hold gaps in days

e_list = [] #list to hold expected values of gaps

p_list = [] #list to hold avg probability of no sales



# original 1 hour version

#for prod_id in tqdm(prods):

#    

#    # extract gap_series for a prod_id

#    m = grid_df.id==prod_id

#    sales_gaps = grid_df.loc[m,'gaps']

#    

#    # calculate initial probability

#    zero_days = sum(sales_gaps)

#    p = zero_days/total_days

#    

#    # find and mark gaps

#    sales_gaps[:] = gap_finder(sales_gaps.values.copy())

#    sales_gaps = sales_gaps.astype(int).replace(-1,np.nan).fillna(method='backfill').fillna(method='ffill')

#    s_list += [sales_gaps]



# magic x8 speed booster thanks to @nadare

for prod_id, df in tqdm(grid_df.groupby("id")):   

    # extract gap_series for a prod_id

    sales_gaps = df.loc[:,'gaps']



    # calculate initial probability

    zero_days = sum(sales_gaps)

    p = zero_days/total_days



    # find and mark gaps

    accum_add_prod = np.frompyfunc(lambda x, y: int((x+y)*y), 2, 1)

    sales_gaps[:] = accum_add_prod.accumulate(df["gaps"], dtype=np.object).astype(int)

    sales_gaps[sales_gaps < sales_gaps.shift(-1)] = np.NaN

    sales_gaps = sales_gaps.fillna(method="bfill").fillna(method='ffill')

    s_list += [sales_gaps]

    

    # calculate E/total_days for all possible gap lengths:

    gap_length = sales_gaps.unique()

    

    d = {length: ((1-p**length)/(p**length*(1-p)))/365 for length in gap_length}

    sales_E_years = sales_gaps.map(d)

    

    # cut out supply_gap days and run recursively

    p1 = 0

    while p1 < p:

        

        if p1!=0:

            p=p1

        

        # once in 100 years event; change to your taste here

        gap_days = sum(sales_E_years>100)

            

        p1 = (zero_days-gap_days+0.0001)/(total_days-gap_days)

        

        d = {length: ((1-p1**length)/(p1**length*(1-p1)))/365 for length in gap_length}

        sales_E_years = sales_gaps.map(d)

        

    # add results to list it turns out masked replacemnt is a very expensive operation in pandas, so better do it in one go

    e_list += [sales_E_years]

    p_list += [pd.Series(p,index=sales_gaps.index)]
# add it to grid_df in one go fast!:

grid_df['gap_days'] = pd.concat(s_list)

grid_df['gap_e'] = pd.concat(e_list)

grid_df['sale_prob'] = pd.concat(p_list)

##45664

# Dump to pickle:

grid_df.to_pickle('grid_part_1_gaps.pkl')
# becuase we have some really extreme values lets take a log:

grid_df['gap_e_log10'] = np.log10((grid_df['gap_e'].values+1))
# e over 100 years does not make much sense

m = grid_df['gap_e_log10']>2

grid_df.loc[m,'gap_e_log10']=2



# take a subsample to vizualise:

np.random.seed(19)

depts = list(grid_df.dept_id.unique())



prod_list = []

for d in depts:

    prod_by_dept=grid_df['item_id'][grid_df.dept_id == d].unique()

    prod_list += list(np.random.choice(prod_by_dept,5))

    

m = grid_df.item_id.isin(prod_list)

viz_df = grid_df[m]

viz_df.head()
v_df = viz_df.pivot(index='d', columns='id', values='gap_e_log10')

v_df = v_df.reindex(sorted(v_df.columns), axis=1)
v_df.describe()
f, ax = plt.subplots(figsize=(15, 20))

temp = sns.heatmap(v_df, cmap='Reds')

plt.show()
grid_df.head()
#Finally lets calculate the proportion of non random gaps in original dataset.

# as mentioned by @Amphi2 we should have dropped last 28 days, so lets substract them:



(sum(grid_df['gap_e_log10'] >= 2) - 3049*10*28)/grid_df.shape[0]