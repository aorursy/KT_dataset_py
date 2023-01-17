import pandas as pd

train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

item_cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')



train['date'] = pd.to_datetime(train['date'], format="%d.%m.%Y")

train['year'] = train['date'].dt.year

train['month'] = train['date'].dt.month



train.head()
# (shop_id, item_id) sales for october 2015

october_2015 = train[(train['year'] == 2015) & (train['month'] == 10)]

october_2015 = october_2015.groupby(['shop_id', 'item_id'])['item_cnt_day'].sum()





def make_oct_2015(x):

    if (x['shop_id'], x['item_id']) in october_2015.index:

        return october_2015[x['shop_id']][x['item_id']]

    else:

        return 0
%%time

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')



test['oct_2015'] = test.apply(make_oct_2015, axis='columns')

test.head()
# Predicting same sales as in oct_2015 

# also prediction values should be clipped [0, 20]

# there actually are some very large numbers...





def make_pred(x):

    return max(0, min(20, x['oct_2015']))

    

    

test['pred'] = test.apply(make_pred, axis='columns')



output = test[['ID', 'pred']]

output.columns = ['ID', 'item_cnt_month']

output = output.set_index('ID')

output.to_csv('basic_averages')



# lb score --> 1.16777

# xD Well that's a start.