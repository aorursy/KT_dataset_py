# !pip install optuna
# !pip install GML
# !pip install sweetviz
import warnings

warnings.filterwarnings("ignore")
import pandas as pd

import numpy as np



# import sweetviz as sv



# from GML.Ghalat_Machine_Learning import Ghalat_Machine_Learning



from sklearn.metrics import mean_squared_log_error

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.model_selection import StratifiedKFold

from sklearn.cluster import KMeans



import tqdm



# import optuna



import multiprocessing
def eval_metric(y_true,y_pred):

    return 100*np.sqrt(mean_squared_log_error(y_true,y_pred))
train = pd.read_csv('/kaggle/input/train_0irEZ2H.csv')

test = pd.read_csv('/kaggle/input/test_nfaJ3J5.csv')

sample = pd.read_csv('/kaggle/input/sample_submission_pzljTaX.csv')
train['week'] = pd.to_datetime(train['week'])

test['week'] = pd.to_datetime(test['week'])
train['day'] = train['week'].dt.day

train['month'] = train['week'].dt.month

train['year'] = train['week'].dt.year

# train['dow'] = train['week'].dt.dayofweek



test['day'] = test['week'].dt.day

test['month'] = test['week'].dt.month

test['year'] = test['week'].dt.year

# test['dow'] = test['week'].dt.dayofweek





train.drop(['week'],axis=1,inplace=True)

test.drop(['week'],axis=1,inplace=True)
# r = sv.compare([train,'train'],[test,'test'],'units_sold')
# r.show_html('report_train_test.html')
train['b_m_t'] = (train['base_price'] - train['total_price'])/train['base_price']

test['b_m_t'] = (test['base_price'] - test['total_price'])/test['base_price']
def price_bin(s):

    if 100 <= s <= 250:

        return 1

    else:

        return 0



# train['p_bin'] = train['total_price'].apply(price_bin)

# test['p_bin'] = test['total_price'].apply(price_bin)
"""

mp = train.groupby('store_id')['units_sold'].mean()

train['store_avg'] = train['store_id'].map(mp)

test['store_avg'] = test['store_id'].map(mp)



mp = train.groupby('store_id')['units_sold'].min()

train['store_min'] = train['store_id'].map(mp)

test['store_min'] = test['store_id'].map(mp)



mp = train.groupby('store_id')['units_sold'].max()

train['store_max'] = train['store_id'].map(mp)

test['store_max'] = test['store_id'].map(mp)

"""

pass
sku = np.unique(train['sku_id'])
"""

for sid in stores:

    gml = Ghalat_Machine_Learning()

    t = train[train['store_id']==sid].copy()

    X = t.drop('units_sold',axis=1)

    y = t['units_sold'].copy()

    gml.GMLRegressor(X, y, metric = metric, neural_net = 'yes', verbose = False)

"""

pass
test['units_sold'] = np.nan
stores = np.unique(train['store_id'])
def st_cat(s):

    return int(s/100)

def r_st_cat(s):

    return int(s%10)



train['st_cat'] = train['store_id'].apply(st_cat)

test['st_cat'] = test['store_id'].apply(st_cat)
"""

for sk in tqdm.tqdm(sku):

    for store in stores:

        tr = train[(train['sku_id']==sk) & (train['store_id']==store)].copy()

        ts = test[(test['sku_id']==sk) & (test['store_id']==store)].copy()

        

        if tr.empty or ts.empty:

            continue

            

        mp = tr.groupby('store_id')['units_sold'].mean()

        tr['store_avg'] = tr['store_id'].map(mp)

        ts['store_avg'] = ts['store_id'].map(mp)



        mp = tr.groupby('store_id')['units_sold'].std()

        tr['store_std'] = tr['store_id'].map(mp)

        ts['store_std'] = ts['store_id'].map(mp)



        mp = tr.groupby('store_id')['units_sold'].min()

        tr['store_min'] = tr['store_id'].map(mp)

        ts['store_min'] = ts['store_id'].map(mp)



        mp = tr.groupby('store_id')['units_sold'].max()

        tr['store_max'] = tr['store_id'].map(mp)

        ts['store_max'] = ts['store_id'].map(mp)



        s_y = tr.groupby('year')['units_sold'].mean()

        tr['sales_yearly'] = tr['year'].map(s_y)

        ts['sales_yearly'] = ts['year'].map(s_y)



        X = tr.drop(['units_sold'],axis=1)

        X.fillna(0,inplace=True)

        y = tr['units_sold'].copy()

        tes = ts.drop(['units_sold'],axis=1)



        model = ExtraTreesRegressor()

        model.fit(X, y)

        preds = model.predict(tes)



        test.at[ts.index.values,'units_sold'] = preds

"""

pass
from sklearn.linear_model import BayesianRidge, LinearRegression

from scipy import stats
def rounding(num): # magic function. used it during blending

    round_num, round_num2 = 0, 0

    uniques = np.unique(train['units_sold'])

    for i,n in enumerate(uniques):

        if n > num:

            break

        round_num = n

        round_num2 = uniques[i+1]

    return (round_num+round_num2)/2
def partition(price,val):

    if price <= val:

        return 1

    else:

        return 0



def partition_val(price):

    return np.mean(price)
def change_shop(s):

    if np.abs(s['total_price']-s['base_price']) > 0 and s['is_featured_sku'] == 0 :

        return 1

    else:

        return 0
for store in tqdm.tqdm(sku):

    tr = train[train['sku_id']==store].copy()

    ts = test[test['sku_id']==store].copy()

    

    tr.sort_values('store_id',inplace=True)

    ts.sort_values('store_id',inplace=True)

    

    mp = tr.groupby('store_id')['units_sold'].mean()

    tr['store_avg'] = tr['store_id'].map(mp)

    ts['store_avg'] = ts['store_id'].map(mp)

    

    mp = tr.groupby('store_id')['units_sold'].std()

    tr['store_std'] = tr['store_id'].map(mp)

    ts['store_std'] = ts['store_id'].map(mp)



    mp = tr.groupby('store_id')['units_sold'].min()

    tr['store_min'] = tr['store_id'].map(mp)

    ts['store_min'] = ts['store_id'].map(mp)



    mp = tr.groupby('store_id')['units_sold'].max()

    tr['store_max'] = tr['store_id'].map(mp)

    ts['store_max'] = ts['store_id'].map(mp)

    

    s_y = tr.groupby('year')['units_sold'].mean()

    tr['sales_yearly'] = tr['year'].map(s_y)

    ts['sales_yearly'] = ts['year'].map(s_y)

    

    tr['pct_change'] = tr['total_price'].pct_change()

    ts['pct_change'] = ts['total_price'].pct_change()

    

    tr['is_B'] = (tr['is_featured_sku'] & tr['is_display_sku'])

    ts['is_B'] = (ts['is_featured_sku'] & ts['is_display_sku'])

    

    tr['avg_change'] = (tr['total_price'] + tr['base_price'])/2

    ts['avg_change'] = (ts['total_price'] + ts['base_price'])/2

    

    tr['ExtraTax'] = tr.apply(change_shop,axis=1)

    ts['ExtraTax'] = ts.apply(change_shop,axis=1)

    

    lr = BayesianRidge()

    

    X = tr.drop(['units_sold','sku_id'],axis=1)

    X.fillna(0,inplace=True)

    y = tr['units_sold'].copy()

    lr.fit(X, y)

    X['pred'] = lr.predict(X)

    # X['pred'] = X['pred'].apply(rounding)

    tes = ts.drop(['units_sold','sku_id'],axis=1)

    tes.fillna(0,inplace=True)

    tes['pred'] = lr.predict(tes)

    # tes['pred'] = tes['pred'].apply(rounding)

    

    model1 = RandomForestRegressor(n_estimators=500,min_samples_leaf=5,n_jobs=-1)

    model2 = ExtraTreesRegressor(n_jobs=-1,min_samples_leaf=3)

    model1.fit(X, y)

    model2.fit(X, y)

    preds = (model1.predict(tes) + model2.predict(tes))/2 

    

    test.at[ts.index.values,'units_sold'] = preds
# test['units_sold'] -= 0.5
test[['record_ID','units_sold']].to_csv('submission.csv',index=False)
train['units_sold'].describe()
test['units_sold'].describe()
imp = pd.DataFrame()

imp['f'] = X.columns

imp['i'] = model1.feature_importances_
imp.sort_values('i',ascending = False)