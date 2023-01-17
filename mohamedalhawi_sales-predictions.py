import pandas as pd

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

plt.style.use('ggplot')

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/competitive-data-science-predict-future-sales"))



# Any results you write to the current directory are saved as output.

#from learntools.pandas.data_types_and_missing_data import *

#! pip install tensorflow==2.0.0-rc0 

print("Setup Complete")
start = '../input/competitive-data-science-predict-future-sales'

items  = pd.read_csv(start+'/items.csv')

train = pd.read_csv(start+'/sales_train.csv')

test = pd.read_csv(start+'/test.csv')

item_category = pd.read_csv(start+'/item_categories.csv')

shops = pd.read_csv(start+'/shops.csv')

sample = pd.read_csv(start+'/sample_submission.csv')
def disc(data):

    print("----------head----------")

    print(data.head(7))

    print("-----------info-----------")

    print(data.info())

    print("----------Missing value-----------")

    print(data.isnull().sum())

    print("----------Null value-----------")

    print(data.isna().sum())

    print("----------shape----------")

    print(data.shape)

    print('----------Number of duplicates----------')

    print(len(data[data.duplicated()]))

def graph_insight(data):

    print(set(data.dtypes.tolist()))

    df_num = data.select_dtypes(include = ['float64', 'int64'])

    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);

    

def drop_duplicate(data, subset):

    print('Before drop shape:', data.shape)

    before = data.shape[0]

    data.drop_duplicates(subset,keep='first', inplace=True) #subset is list where you have to put all column for duplicate check

    data.reset_index(drop=True, inplace=True)

    print('After drop shape:', data.shape)

    after = data.shape[0]

    print('Total Duplicate:', before-after)

    

def discripe_dataset(data):

    disc(data)

    graph_insight(data)

    

def pre(data):

    for _ in [1,2] :

        for col in data.keys(): 

            if data[col].dtype in ['int64'] :

                data[col] = data[col].astype('int16')

            elif data[col].dtype in ['float64'] :

                data[col] = data[col].astype('float16')

            elif data[col].dtype in ['object'] :

                data[col] = LabelEncoder().fit_transform(data[col])
disc(train)
disc(items)
disc(shops)
disc(item_category)
trained = train.groupby(['shop_id', 'item_id','date_block_num']).item_cnt_day.count().reset_index(name='items_month')

trained = trained.sort_values(by=['date_block_num','item_id'])
trained.head()
disc(test)
tested = test.copy().drop("ID" ,axis=1)

tested['date_block_num'] = 34 

tested['items_month'] = 0
tested.head()
con = [tested, trained]

train_test = pd.concat(con)
disc(train_test)
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops['shop_center'] = shops['shop_name'].str.split(' ').map(lambda x: x[1])

shops = shops[['shop_id','city','shop_center']]

pre(shops)
disc(shops)
item_category['type'] = item_category['item_category_name'].str.split('-').map(lambda x: x[0].strip())

item_category['subtype'] = item_category['item_category_name'].str.split('-').map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

item_category = item_category[['item_category_id','type','subtype']]

pre(item_category)

pre(items)
item_category.head()
items.head()
items_icat = pd.merge(items,  item_category, on=['item_category_id'], how='left')
items_icat.head()
tr_te_sh = pd.merge(shops,train_test, on=['shop_id'], how='right', sort=False)
tr_te_sh.head()
tr_te_sh_it_icat = pd.merge(tr_te_sh, items_icat, on=['item_id'], how='left', sort=False)
tr_te_sh_it_icat.head()
tr_te_sh_it_icat.shape
final = tr_te_sh_it_icat.copy()
pre(final)
final.shape
X_train = final[final.date_block_num < 33].drop(['items_month'], axis=1)

Y_train = final[final.date_block_num < 33]['items_month']

X_valid = final[final.date_block_num == 33].drop(['items_month'], axis=1)

Y_valid = final[final.date_block_num == 33]['items_month']

X_test  = final[final.date_block_num == 34].drop(['items_month'], axis=1)
model = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)



model.fit(

    X_train, 

    Y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    verbose=True, 

    early_stopping_rounds = 10)
# Y_pred = model.predict(X_valid).clip(0, 20)

Y_test = model.predict(X_test).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": Y_test

})

submission.to_csv('xgb_sub.csv', index=False)