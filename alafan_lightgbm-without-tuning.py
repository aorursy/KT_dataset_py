import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import re



from lightgbm import LGBMRegressor

import lightgbm



from sklearn.model_selection import learning_curve,train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder



from itertools import product



import matplotlib.pylab as plt

plt.rcParams["figure.figsize"] = (20,4)



import seaborn as sns

%matplotlib inline





import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
folder = '/kaggle/input/competitive-data-science-predict-future-sales/'

#folder = './files/'



train = pd.read_csv(folder+'sales_train.csv')

shops = pd.read_csv(folder+'shops.csv')

items = pd.read_csv(folder+'items.csv')

itemcat = pd.read_csv(folder+'item_categories.csv')

test = pd.read_csv(folder+'test.csv')
train.head()
shops.head()
items.head()
itemcat.head()
fig, ax = plt.subplots(1,2,figsize = (20,4))

sns.distplot(train.item_price, ax = ax[0])

sns.boxplot(train.item_price, ax = ax[1])
plt.scatter(x = train.item_id,y = train.item_price, alpha = .2)
train[['item_id','item_price','shop_id']].drop_duplicates().sort_values(by = 'item_price',ascending = False).head(3)
items[items.item_id==6066]
test[test.item_id==6066]
train[train.item_id==6066]
train.drop(train[train.item_id==6066].index, inplace = True)
fig, ax = plt.subplots(1,2,figsize = (20,4))

sns.distplot(train.item_price, ax = ax[0])

sns.boxplot(train.item_price, ax = ax[1])
expensive = train[train.item_price>10000][['item_id']].drop_duplicates()

test_expensive = expensive.merge(test[['item_id']].drop_duplicates(), on = 'item_id', how = 'inner')

print('There are {} items expensiver than 10000 in train.\n{} of them are in test'.format(

len(expensive),len(test_expensive)))

fig, ax = plt.subplots(1,1,figsize = (20,4))

sns.boxplot(data = train[['item_id','item_price']].merge(test_expensive, on = 'item_id', how = 'inner'),

            x = 'item_id',y = 'item_price', ax = ax)

fig,ax = plt.subplots(figsize = (20,4))

sns.boxplot(data = train[['shop_id','item_cnt_day']],

            x = 'shop_id',y = 'item_cnt_day', ax = ax)
train[train.item_id==11365].merge(items, on = 'item_id',how = 'left').sort_values(by = 'item_price',ascending = False).head()
items[items.item_category_id==9]
train = train.merge(items, on = 'item_id', how = 'left')
sns.distplot(train[train.item_category_id==9].item_price.clip(0,3000))
np.percentile(train[train.item_category_id==9].item_price,99.9)
shops[shops.shop_id==12]
# for every text column: deleting spec symbols and extra spaces, lower case

def str_clean(x):

    return re.sub(re.compile('\W'),' ',x.lower()).strip()

shops['shop_name'] = shops['shop_name'].apply(str_clean)

itemcat['item_category_name'] = itemcat['item_category_name'].apply(str_clean)
items['item_name'] = items['item_name'].apply(str_clean)
#fix duplicates:

duplicate_shops = {57:0,58:1,40:39,11:10}

def fix_shop_id(x):

    if x not in duplicate_shops.keys():

        return x

    return duplicate_shops[x]

train['shop_id'] = train['shop_id'].apply(fix_shop_id)

test['shop_id'] = test['shop_id'].apply(fix_shop_id)

shops.drop(shops[shops.shop_id.isin(duplicate_shops.keys())].index,inplace = True)
shops['city'] = shops.shop_name.apply(lambda x: x.split()[0])
le = LabelEncoder()

shops['city_id'] = le.fit_transform(shops['city'])
#mean price for id:

prices = train.groupby(['item_id','shop_id','date_block_num'],as_index=False).agg(

    {

        'item_price':{'mean_price':'mean','price_std':'std','price_count':'count'},

        'item_cnt_day':{'sold_total':'sum'}

    }

).fillna(0)
prices.columns = ['item_id','shop_id','date_block_num','mean_price','price_std','price_count','sold_total']
prices['date_block_num']+=1
prices.head()
#prices['mean_price'] = prices['mean_price'].clip(0,np.percentile(prices.mean_price,99))
X = train.groupby(['date_block_num','shop_id','item_id'],as_index = False

                 ).item_cnt_day.sum()

X.rename(columns = {'item_cnt_day':'target'}, inplace=True)

index_cols = ['shop_id', 'item_id', 'date_block_num']

grid = [] 

for block_num in train['date_block_num'].unique():

    cur_shops = train[train['date_block_num']==block_num]['shop_id'].unique()

    cur_items = train[train['date_block_num']==block_num]['item_id'].unique()

    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

X = pd.merge(grid,X,how='left',on=index_cols).fillna(0)

#sort the data

X.sort_values(['date_block_num','shop_id','item_id'],inplace=True)
X.head()
shops[(shops.shop_id==55) |(shops.shop_id==12)]
fig,ax = plt.subplots(figsize = (20,4))

sns.boxplot(x = 'date_block_num',y = 'target',data = X[X.target>0])
fit,ax=plt.subplots(figsize = (20,4))

sns.distplot(X[X.target>0].target)
#previous months

X.sort_values(by = ['shop_id','item_id','date_block_num'],inplace=True)

month_ago = 3

for i in range(1, month_ago+1):

    colname = 'prev' + str(i)

    X[colname] = X.target.shift(i)

    mask1 = X.shop_id != X.shop_id.shift(i)  

    mask2 = X.item_id!= X.item_id.shift(i)

    mask = mask1|mask2

    X[colname][mask] = 0



#test_data prev columns forming:

train_columns = ['shop_id','item_id','target']+['prev'+str(i) for i in range(1,month_ago+1)]

test = test.merge(X[X.date_block_num==33][train_columns],on = ['shop_id','item_id'],how = 'left')
test.rename(columns = {'prev3':'prev4','prev2':'prev3','prev1':'prev2','target':'prev1'}, inplace = True)

del test['prev'+str(month_ago+1)]

X_target = X.target.values
del X['target']
test['date_block_num']=34

X['ID'] = X.index
X = X.append(test)
X = X.merge(items[['item_id','item_category_id', 'item_name']], on = 'item_id', how = 'left')

X = X.merge(shops[['shop_id','city_id','shop_name']], on = 'shop_id', how = 'left')
X = X.merge(itemcat, on = 'item_category_id', how = 'left')
X = X.merge(prices, on = ['item_id','shop_id','date_block_num'], how = 'left')
X['month'] = X['date_block_num']%12 + 1
# is it internet shop?

X['is_internet'] = X['shop_id'].apply(lambda x: 1 if x in [12,55] else 0)
X['target_block_num'] = X['date_block_num']-1

X['target_month'] = X['target_block_num']%12+1
X[(~X.item_id.isin([20949]))&(~X.item_category_id.isin(['9']))].sort_values(

    by = 'prev1', ascending = False).head().T
X['is_bag'] = X.item_id.apply(lambda x: 1 if x == 20949 else 0)
fig,ax = plt.subplots(figsize = (20,4))

sns.boxplot(data = X[X.is_bag==1], x = 'target_block_num',y = 'prev1')
popular_games = X[(X.item_category_id==31)&(X.prev1>100)].sort_values(by = 'prev1', ascending = False).item_id.drop_duplicates().head(10).to_list()
popular_games
game = 3730

X[X.item_id==game].groupby('target_block_num',as_index = False)['prev1'].sum()
fig, ax = plt.subplots(figsize = (20,6))

for game, color in zip(popular_games,sns.cubehelix_palette(10,start=.5, rot=-1.75)):

    print(game,color)

    sns.barplot(x = 'target_block_num',y = 'prev1',

            data = X[X.item_id==game].groupby('target_block_num',as_index = False)['prev1'].sum(),

            color = color,

            alpha = .5, 

            label = 'game'

           )

    


X[X.prev1>100].item_id.nunique()
col = 'item_category_id'

for col in ['item_category_id','item_id','shop_id','city_id']:

    X = X.merge(

        X.groupby(col, as_index=False)['prev1'].agg({col+'_mean':'mean',col+'_median':'median',col+'_std':'std'}),

        on = col,

        how = 'left'

    )
ftr = [

    'month',

    'city_id',

    'item_id',

    'shop_id',

    'date_block_num',

    'prev1','prev2','prev3',

    'item_category_id',

    'mean_price',

    'price_count',

    'sold_total',

    'is_internet',

    'is_bag'

]
#clip for outliers

for i in [x for x in ftr if x not in ['prev1','prev2','prev3']]:

    X[i] = X[i].clip(0,np.percentile(X[i],99.9))
for i in ['prev1','prev2','prev3']:

    X[i] = X[i].clip(0,20)
lgbm = LGBMRegressor(random_state = 42, 

                     max_depth = 8,

                     objective = 'rmse',

           

                    )
months = pd.DataFrame()

for m in range(32):

    lgbm.fit(X[(X.date_block_num>m)&(X.date_block_num<33)][ftr],

         X_target[X[(X.date_block_num>m)&(X.date_block_num<33)].index].clip(0,20))

    pred = lgbm.predict(X[X.date_block_num==33][ftr])

    months.loc[m,'score'] = mean_squared_error(

        pred.clip(0,20),X_target[X[X.date_block_num==33].index].clip(0,20))**0.5

    print(m,": ",months.loc[m,'score'])

    
plt.plot(months)
begin_month = 10

lgbm.fit(X[(X.date_block_num>begin_month)&(X.date_block_num<33)][ftr],

         X_target[X[(X.date_block_num>begin_month)&(X.date_block_num<33)].index])

pred = lgbm.predict(X[X.date_block_num==33][ftr])

score = mean_squared_error(pred.clip(0,20),X_target[X[X.date_block_num==33].index].clip(0,20))**0.5

print(score)
from lightgbm import plot_importance

fig, ax = plt.subplots(figsize = (12,8))

plot_importance(lgbm, ax = ax)
fig,ax = plt.subplots(figsize = (20,15))

lightgbm.plot_tree(lgbm, ax = ax)
lgbm.fit(X[(X.date_block_num>begin_month)&(X.date_block_num<34)][ftr],

         X_target[X[(X.date_block_num>begin_month)&(X.date_block_num<34)].index].clip(0,20))

pred = lgbm.predict(X[X.date_block_num==34][ftr]).clip(0,20)

res = pd.DataFrame()

res['ID'] = test['ID']

res['item_cnt_month'] = pred

nt = test.merge(items[['item_id','item_category_id']], on = 'item_id',how = 'left')

res.loc[nt[(nt.item_category_id==9)&(nt.shop_id!=12)].index,'item_cnt_month'] = 0
res.to_csv('submission.csv', index = False)