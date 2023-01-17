""" GENERAL IMPORTS """
import numpy as np 
import pandas as pd
pd.options.display.max_columns = 50

from  datetime import datetime, timedelta
# import gc
# from tqdm.notebook import tqdm

""" CHARTING IMPORTS """
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (18.0, 4)
%matplotlib inline 
plt.rcParams.update({'figure.max_open_warning': 0})
import seaborn as sns; sns.set()
""" LOADING DATA """
cal = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv", parse_dates=['date'])
# prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv" )
sales = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")
""" helpful lists """
stores_list = sales.store_id.unique().tolist() + ['ALL']
dept_list = sales.dept_id.unique().tolist() + ['ALL']
bom_days  = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365] # beginnings of months

id_cols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
weeks_all = cal.wm_yr_wk.unique().tolist()
weeks_train = weeks_all[:-8]
""" create SALES_WEEKLY DataFrame"""
sales_w = sales.iloc[:,:6].astype(str) # copy headers
for w in weeks_all:
    dw_list = cal[cal.wm_yr_wk == w].d.tolist()
    dw_list = [d for d in dw_list if d in sales.columns] 
    len_dw_list = len(dw_list)
    if len_dw_list > 0:
        sales_w[w] = sales.loc[:,dw_list].sum(axis=1)
print (sales_w.shape)
sales_w.sample(3)
def plot_multiyear_series(my_series, ax, title):
    sns.lineplot(data=my_series, x='doy', y='nsold', hue='year',
                 palette='rainbow', legend='full', ax=ax)
    ax.set_xticks(bom_days, minor=False)
    ax.set_xticklabels(range(13))
    ax.set_xlim(0, 366)
    ax.set_title(title, fontsize=18)
    ax.legend(bbox_to_anchor=(1.1, 1))
    ax.set(xlabel=None, ylabel='units sold')


def plot_single_multiyear_weekly(wseries, title='weeky volumes by year'):
    my_series = wseries_to_multiyear(wseries)
    fig, ax = plt.subplots(figsize=(16, 4))
    plot_multiyear_series(my_series, ax, title)


def wseries_to_multiyear(wseries):
    my_series = wseries.to_frame().reset_index()
    my_series.columns = ['wm_yr_wk', 'nsold']
    my_series = my_series.merge(cal[['date', 'year', 'wm_yr_wk']], on='wm_yr_wk'). \
        groupby('wm_yr_wk').agg({'date': 'min', 'nsold': 'median'})
    my_series['year'] = my_series.date.dt.year
    my_series['woy'] = (my_series.date+timedelta(2)).dt.weekofyear
    my_series['doy'] = (my_series.date+timedelta(2)).dt.dayofyear
    return my_series
wseries = sales_w[weeks_train[:-1]].sum(axis=0)
plot_single_multiyear_weekly(wseries, 'All sales')
""" importing magical components: """
import ipywidgets as widgets
from ipywidgets import interact, interact_manual 
@interact  # note the decorator
def show_store_sales_by_dept(sid=stores_list):  # default argument value as list
    if sid == 'ALL':
        store_w = sales_w.fillna(0).groupby(['item_id','dept_id'])[weeks_train[:-1]].sum().reset_index().copy()
    else:
        store_w = sales_w[sales_w.store_id==sid].copy()

    fig, axs = plt.subplots(4, 2, figsize=(28, 14), constrained_layout=True)

    for ax, did in zip (axs.ravel(), dept_list[:]):
        if did == 'ALL':
            wseries = store_w.loc[:, weeks_train[:-1]].sum(axis=0)
        else:
            wseries = store_w.loc[store_w.dept_id == did, weeks_train[:-1]].sum(axis=0)
        my_series = wseries_to_multiyear (wseries)
        title = f"Weekly sales by year for store_id={sid}, dept_id={did}" 
        plot_multiyear_series(my_series, ax, title)
@interact_manual  # requires pressing button for manual update
def show_store_sales(sid=stores_list, did=dept_list):  # arguments as lists
    title = f"Weekly sales by year for store_id={sid}, dept_id={did}" 
    if sid == 'ALL':
        sid = stores_list[1:]
    else:
        sid = [sid]
    if did == 'ALL':
        did=dept_list[1:] # if  else  did = [did]
    else:
        did = [did]
    wseries = sales_w.loc[(sales_w.store_id.isin(sid))&(sales_w.dept_id.isin(did)), weeks_train[:-1]].sum(axis=0)
    plot_single_multiyear_weekly(wseries, title)
dept_ = widgets.Dropdown(options=dept_list[:-1])
store_ = widgets.Dropdown(options=stores_list[-1:]+stores_list[:-1]) 
items = widgets.Dropdown(options=sales_w[sales_w.dept_id==dept_.value].item_id.unique())

def update_items(*args):
    items.options = sales_w[sales_w.dept_id==dept_.value].item_id.unique().tolist()

dept_.observe(update_items, 'value')
store_.observe(update_items, 'value')
    
def show_item_sales_by_store(did, iid, sid):
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)

    if sid == 'ALL':
        wseries = sales_w.loc[sales_w.item_id==iid, weeks_train[:-1]].sum(axis=0)
    else:
        wseries = sales_w.loc[(sales_w.item_id==iid)&(sales_w.store_id == sid), weeks_train[:-1]].sum(axis=0)
    my_series = wseries_to_multiyear (wseries)
    title = f"Weekly sales by year for store_id={sid}, dept_id={did}, item_id={iid}" 
    plot_multiyear_series(my_series, ax, title)

_ = interact(show_item_sales_by_store, did=dept_, iid=items, sid=store_)