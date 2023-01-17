%matplotlib inline
import numpy as np
import pandas as pd
import sklearn
import os
import seaborn as sns
import matplotlib.pyplot as plt
base_dir = os.path.join('..', 'input')
start_date = pd.to_datetime('2013-01-01 ')
def read_and_parse(in_path):
    in_df = pd.read_csv(in_path)
    in_df['date'] = pd.to_datetime(in_df['date'])
    in_df['elapsed_days'] = (in_df['date']-start_date).dt.days
    # time tags
    for x_col in 'dayofweek', 'dayofyear', 'day', 'weekday', 'week', 'month', 'year':
        in_df[x_col] = getattr(in_df['date'].dt, x_col)
    in_df.drop(['date'], 1, inplace = True)
    return in_df
def show_sales(in_df):
    sns.factorplot('elapsed_days', 'sales', hue = 'store', facet = 'item', data = in_df)
full_df = read_and_parse(os.path.join(base_dir, 'train.csv'))
full_df.sample(5)
fig, m_axs = plt.subplots(3, 3, figsize = (15, 25))
for c_ax, (item_id, c_df) in zip(m_axs.flatten(), 
                          full_df.groupby('item')):
    n_df = c_df
    out_pvt = c_df.pivot_table(values='sales', 
                        aggfunc='mean', 
                        index=['elapsed_days'], 
                        columns = 'store').reset_index()
    
    c_ax.matshow(out_pvt.values[:,1:])
    c_ax.set_title('Item: {}'.format(item_id))
    #c_ax.axis('off')
    c_ax.set_ylabel('Time')
    c_ax.set_aspect(0.025)
fig, m_axs = plt.subplots(3, 3, figsize = (15, 25))
for c_ax, (store_id, c_df) in zip(m_axs.flatten(), 
                          full_df.groupby('store')):
    n_df = c_df
    out_pvt = c_df.pivot_table(values='sales', 
                        aggfunc='mean', 
                        index=['elapsed_days'], 
                        columns = 'item').reset_index()
    
    c_ax.matshow(out_pvt.values[:,1:])
    c_ax.set_title('Store: {}'.format(store_id))
    #c_ax.axis('off')
    c_ax.set_ylabel('Time')
    c_ax.set_aspect(0.025)
test_out_df = read_and_parse(os.path.join(base_dir, 'test.csv'))
test_out_df.sample(5)
test_store = 9
test_item = 36
from sklearn.dummy import DummyRegressor

q_string = f'store=={test_store} and item=={test_item}'
sample_train_df = full_df.query(q_string).copy()
fig, ax1 = plt.subplots(1,1,figsize = (20, 10))
sample_train_df.plot(x = 'elapsed_days', y = 'sales', ax = ax1)
y_vec = sample_train_df.pop('sales')

# predict and show the test data
dummy_reg = DummyRegressor(strategy = 'median')
dummy_reg.fit(sample_train_df, y_vec)
sample_test_df = test_out_df.query(q_string).copy()
sample_test_df['sales'] = dummy_reg.predict(sample_test_df[sample_train_df.columns])
sample_test_df.plot(x = 'elapsed_days', y = 'sales', ax = ax1)
out_rows = []
for c_grp, c_train_df in full_df.groupby(['store', 'item']):
    y_vec = c_train_df.pop('sales')
    c_grp_string = f'store=={c_grp[0]} and item=={c_grp[1]}'
    out_df = test_out_df.query(c_grp_string).copy()
    dummy_reg = DummyRegressor(strategy = 'median')
    dummy_reg.fit(c_train_df, y_vec)
    out_df['sales'] = dummy_reg.predict(out_df[c_train_df.columns])
    out_rows += [out_df]
full_out_df = pd.concat(out_rows)
full_out_df.head(5)
full_out_df[['id', 'sales']].to_csv('prediction.csv', index = False)