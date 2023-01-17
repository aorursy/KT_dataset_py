# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train_cust = pd.read_csv('/kaggle/input/restaurant-recommendation-challenge/train_customers.csv')
df_train_cust.rename(columns={'akeed_customer_id':'customer_id'},inplace=True)
df_train_cust.set_index('customer_id',inplace=True)
df_orders = pd.read_csv('/kaggle/input/restaurant-recommendation-challenge/orders.csv')
df_orders['created_at'] = pd.to_datetime(df_orders.created_at)
df_orders.set_index(['customer_id','akeed_order_id'],inplace=True)

# We can only try to train models based on previous orders
# for users who have more than one order.
df_orders['order_number']=(df_orders
 .groupby(level = 'customer_id')['created_at']
 .rank(method='first',ascending=True)
)

df_orders = df_orders.join(
    df_orders
    .reset_index()
    .groupby('customer_id')
    .agg(
        total_orders = ('akeed_order_id','count')
    )
).sort_index()

# set a flag for the most recent order for each user
df_orders['last_order'] = np.where(df_orders.total_orders == df_orders.order_number,1,np.nan)

# we want to predict the most recent orders.
orders_to_predict = (df_orders
                     .query('last_order == 1')
                     .query('total_orders >= 2')
                    )

to_predict_list = orders_to_predict.reset_index().akeed_order_id.to_list()


# for each customer we want 
# % of times there fav (most frequently visited) restraunt was ordered
# removing the most recent order as this is what we want to predict.
fav_restraunts = (
    df_orders
    .query('akeed_order_id not in @to_predict_list')
    .reset_index()
    .groupby(['customer_id','vendor_id'])
    .agg({'akeed_order_id':'count'})
    .join(
        df_orders
        .query('akeed_order_id not in @to_predict_list')
        .reset_index()
        .groupby('customer_id')
        .agg(total_orders = ('akeed_order_id','count'))
    )
    .assign(
        percentage = lambda x:x['akeed_order_id']/x['total_orders']
    )
    
)

# get the most frequently visited restraunt for each user.
fav_restraunts = fav_restraunts[
    fav_restraunts.groupby(level='customer_id')['percentage'].transform(max) == fav_restraunts.percentage
][['total_orders','percentage']]

# join on to the orders to predict and flag if the n+1 order is
# the same as there most popular
fav_restraunts = fav_restraunts.join(
    orders_to_predict
    .reset_index()
    .set_index('customer_id')['vendor_id'].rename('recent_order'),
    how = 'inner'
)

fav_restraunts['ordered_fav'] = fav_restraunts['recent_order'] == fav_restraunts.index.get_level_values('vendor_id')

fav_restraunts.head()
# bin the percentages 

num_bins = 10
bins = [0 + i/num_bins for i in range(num_bins+1)]
def bin_values(value, bins = bins):
    if bins[-1]<=value:
        return bins[-1]
    for x in bins:
        if value < x:
            return x
    
fav_restraunts['percentage_bin'] = fav_restraunts.percentage.apply(bin_values)
    
    


# compute the probability of choosing most selected restraunt
res = []
for percentage_bin,group in fav_restraunts.groupby('percentage_bin'):
    p_a_b = group.ordered_fav.sum()/group.ordered_fav.count()
    res.append({'bin':percentage_bin, 'probability': p_a_b})
res = pd.DataFrame(res)
display(res)
ax = res.plot(kind='bar',x='bin',y='probability',legend=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
_ = ax.set_ylabel('Probability of revisiting frequently visited vendor')
_ = ax.set_xlabel('% visited most frequently visied vendor')