import pandas as pd

def prepare_train_data():
    train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')


    # Summing sales for each month and clipping into [0, 20]
    train = train.groupby(['shop_id', 'item_id','date_block_num'])['item_cnt_day'].sum()
    train = train.reset_index()
    train['item_cnt_day'] = train['item_cnt_day'].map(lambda x: max(0, min(20, x)))
    
   
    # add month and year
    train['month'] = train['date_block_num'].map(lambda x: (x % 12) + 1)
    train['year'] = train['date_block_num'].map(lambda x: 2013 + x // 12)

    
    # Dropping sales in Dec.
    train = train[(train['month'] != 1)]
    
    return train


def prepare_test_data():
    test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

    test['date_block_num'] = 34 
    test['month'] = 11
    test['year'] = 2015
    
    return test


train = prepare_train_data()
test = prepare_test_data()
def add_features(train, test):
    # Making flag is it is the first(release) month for (shop_id, item_id)
    first_month = train.groupby(['shop_id', 'item_id'])['date_block_num'].min()

    train['new_item'] = train.apply(lambda x: x['date_block_num'] == first_month[(x['shop_id'], x['item_id'])], axis='columns')
    test['new_item'] = test.apply(lambda x: not ((x['shop_id'], x['item_id']) in first_month.index), axis='columns')
    
    

    # Add prev month sales for (shop_id, item_id)
    shop_item_db = train.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].mean()

    def make_prev_month_sales(x):
        if (x['shop_id'], x['item_id'], x['date_block_num'] - 1) in shop_item_db.index:
            return shop_item_db[(x['shop_id'], x['item_id'], x['date_block_num'] - 1)]
        else:
            return 0

    train['prev_month_sales'] = train.apply(make_prev_month_sales, axis='columns')
    test['prev_month_sales'] = test.apply(make_prev_month_sales, axis='columns')
    
    
    # Add item category id
    items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
    train['cat_id'] = train['item_id'].map(lambda x: items.loc[x, 'item_category_id'])
    test['cat_id'] = test['item_id'].map(lambda x: items.loc[x, 'item_category_id'])


    # Shop categories by location
    def make_shop_loc(x):
        # Yakutsk
        if x in [0, 1, 57, 58]: 
            return 'Yakutsk 4'
        # Moscow area
        elif x in [3, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 54]:
            return 'Moscow 16'
        # Voronej
        elif x in [6, 7, 8]:
            return 'Voronej 3'
        # Online
        elif x in [9, 12, 55]:
            return 'Online 3'
        # Jukovski
        elif x in [10, 11]:
            return 'Jukovski 2'
        # Kazan
        elif x in [13, 14]:
            return 'Kazan 2'
        # Krasnoyarsk
        elif x in [17, 18]:
            return 'Krasnoyarsk 2'
        # NNovgorod
        elif x in [35, 36]:
            return 'NNovgorod 2'
        # Novosib
        elif x in [36, 37]:
            return 'Novosib 2'
        # Rostov
        elif x in [39, 40, 41]:
            return 'Rostov 2'
        # Spb
        elif x in [42, 43]:
            return 'Spb 2'
        # Samara
        elif x in [44, 45]:
            return 'Samara 2'
        # Tumen
        elif x in [49, 50, 51]:
            return 'Tumen 2'
        # Ufa
        elif x in [52, 53]:
            return 'Ufa 2'
        else:
            return 'no_group'

    train['shop_loc'] = train['shop_id'].map(make_shop_loc)
    test['shop_loc'] = test['shop_id'].map(make_shop_loc)
    
    
    # Adding flag if shop could not be meaningfully grouped
    train['no_loc_group'] = (train['shop_loc'] == 'no_group')
    test['no_loc_group'] = (test['shop_loc'] == 'no_group')
    
    
    # adding seasons
    def make_season(x):
        if x in [6, 7, 8]:
            return 'summer'
        elif x in [9, 10, 11]:
            return 'autumn'
        elif x in [3, 4, 5]:
            return 'spring'
        else:
            return 'winter'

    train['season'] = train['month'].map(make_season)
    test['season'] = test['month'].map(make_season)
    
    
    # seasonal sales for (shop_id, item_id)
    seasonal_sales = train.groupby(['shop_id', 'item_id', 'season'])['item_cnt_day'].sum()

    def make_seasonal_sales(x):
        if (x['shop_id'], x['item_id'], x['season']) in seasonal_sales.index:
            return seasonal_sales[(x['shop_id'], x['item_id'], x['season'])] / 9
        else:
            return 0

    train['seasonal_sales_shop_id'] = train.apply(make_seasonal_sales, axis='columns')
    test['seasonal_sales_shop_id'] = test.apply(make_seasonal_sales, axis='columns')
    
    
    # add prev month sales for (shop_loc, item_id)
    loc_item_db = train.groupby(['shop_loc', 'item_id', 'date_block_num'])['item_cnt_day'].mean()

    def make_prev_month_sales_loc(x):
        if x['shop_loc'] == 'no_group':
            return x['prev_month_sales']
        elif (x['shop_loc'], x['item_id'], x['date_block_num'] - 1) in loc_item_db.index:
            num = x['shop_loc'].split()
            num = int(num[1])
            return loc_item_db[(x['shop_loc'], x['item_id'], x['date_block_num'] - 1)] / num
        else:
            return 0

    train['prev_month_sales_loc'] = train.apply(make_prev_month_sales_loc, axis='columns')
    test['prev_month_sales_loc'] = test.apply(make_prev_month_sales_loc, axis='columns')
    
    
    # add seasonaladd_features() sales for (shop_loc, item_id)
    loc_item_seasonal = train.groupby(['shop_loc', 'item_id', 'season'])['item_cnt_day'].sum()

    def make_loc_item_seasonal(x):
        if x['shop_loc'] == 'no_group':
            return x['seasonal_sales_shop_id']
        if (x['shop_loc'], x['item_id'], x['season']) in loc_item_seasonal.index:
            num = x['shop_loc'].split()
            num = int(num[1])
            return loc_item_seasonal[(x['shop_loc'], x['item_id'], x['season'])] / num
        else:
            return 0

    train['loc_item_seasonal'] = train.apply(make_prev_month_sales, axis='columns')
    test['loc_item_seasonal'] = test.apply(make_prev_month_sales, axis='columns')
    
    # add pair (shop_id, item_id)
    train['shop_id_item_id'] = train['shop_id'].astype(str) + '->' + train['item_id'].astype(str)
    test['shop_id_item_id'] = test['shop_id'].astype(str) + '->' + test['item_id'].astype(str)
%%time
add_features(train, test)
train.head()
features = ['shop_id',
            'item_id',
            'shop_id_item_id',
            'new_item',
            'month',
            'year',
            'prev_month_sales',
            'cat_id',
            'shop_loc',
            'no_loc_group',
            'season', 
            'seasonal_sales_shop_id',
            'prev_month_sales_loc',
            'loc_item_seasonal',
           ]

cats = ['shop_id',
        'item_id',
        'shop_id_item_id',
        'new_item',
        'month',
        'year',
        'cat_id',
        'shop_loc',
        'no_loc_group', 
        'season', 
        ]

X = train[features]
y = train['item_cnt_day']
model_ctb = CatBoostRegressor(iterations=3000, loss_function='RMSE',
                              learning_rate=0.06,
                              depth=8,
                              l2_leaf_reg=11,
                              random_seed=17,
                              silent=True,
                              )

model_ctb.fit(X, y,
              cat_features=cats,
              plot=True,
              )
predictions = model_ctb.predict(test[features])
predictions = pd.Series(predictions)
# Clipping predictions into [0, 20].
predictions = predictions.map(lambda x: max(0, min(20, x)))

test['pred'] = predictions
output = test[['ID', 'pred']]
output.columns = ['ID', 'item_cnt_month']
output = output.set_index('ID')
output.to_csv('catboost_baseline')

output.head()