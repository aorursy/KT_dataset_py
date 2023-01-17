

import sklearn.datasets as datasets

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

sns.set_palette("husl")

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
sp = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

cl = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

stv = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
def stvt(id):

    ids = id.split('_')

    item_id = ids[0]+'_'+ids[1]+'_'+ids[2]

    store_id = ids[3]+'_'+ids[4]

    stv[stv['id']==id]

    result = sp[(sp['item_id']==item_id) & (sp['store_id']==store_id)]

    result = pd.merge(result,cl,on='wm_yr_wk')

    result['date'] = pd.to_datetime(result['date'])

    st = stv[(stv['item_id']==item_id) & (stv['store_id']==store_id)].T

    st = st[st.index.str.startswith('d_')]

    st['d'] = st.index

    st['sales'] = st.iloc[:,0].astype('float')

    stvcl =  pd.merge(st,result,on=['d'])

    stvcl['sales_amount'] = stvcl['sales'] * stvcl['sell_price']

    return stvcl

    

stv1 = stvt('HOBBIES_1_001_CA_1_validation')

stv1
def show_plots(stv1):

    fig, axes = plt.subplots(3, figsize=(20,10))

    sns.lineplot(data=stv1,x='date',y='sales',label='sales', ax=axes[0] )

    sns.lineplot(data=stv1,x='date',y='sell_price',label='sell_price', ax=axes[1] )

    sns.lineplot(data=stv1,x='date',y='sales_amount',label='sales_amount', ax=axes[2] )

    

show_plots(stv1)
show_plots(stvt('HOBBIES_1_001_CA_1'))

show_plots(stvt('HOBBIES_1_001_CA_2'))

show_plots(stvt('HOBBIES_1_001_CA_3'))
def show_rolling_plots(stv1):

    fig, axes = plt.subplots(6, figsize=(20,20))

    stv1['sales_amount_r7'] = stv1['sales_amount'].rolling(7).mean()

    stv1['sales_amount_r30'] = stv1['sales_amount'].rolling(30).mean()

    stv1['sales_amount_r90'] = stv1['sales_amount'].rolling(90).mean()

    sns.lineplot(data=stv1,x='date',y='sales',label='sales', ax=axes[0] )

    sns.lineplot(data=stv1,x='date',y='sell_price',label='sell_price', ax=axes[1] )

    sns.lineplot(data=stv1,x='date',y='sales_amount',label='sales_amount', ax=axes[2] )

    sns.lineplot(data=stv1,x='date',y='sales_amount_r7',label='sales_amount_r7', ax=axes[3] )

    sns.lineplot(data=stv1,x='date',y='sales_amount_r30',label='sales_amount_r30', ax=axes[4] )

    sns.lineplot(data=stv1,x='date',y='sales_amount_r90',label='sales_amount_r90', ax=axes[5] )

    

show_rolling_plots(stvt('HOBBIES_1_001_CA_1'))