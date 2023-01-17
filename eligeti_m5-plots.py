import numpy as np 

import pandas as pd 

import os

from matplotlib import pyplot as plt

import plotly.graph_objects as go



import warnings

warnings.filterwarnings('ignore')

from IPython.display import display

import ipywidgets as widgets
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%matplotlib inline

pd.set_option('precision', 2)

pd.set_option('display.float_format', '{:.2f}'.format)

pd.set_option('display.max_columns', 200)
def reduce_memory(df, verbose=False):

    numerics = ['int16', 'int32', 'int64', 

#                 'float16', 'float32', 'float64'

               ]

    if verbose:

        start_mem = df.memory_usage().sum() / 1024**2    

    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            col_min = df[col].min()

            col_max = df[col].max()

            if str(col_type)[:3]=='int':

                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)                   

                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif col_min > np.iinfo(np.int64).min and col_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

                    

                else:

                    

                    if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:

                        df[col] = df[col].astype(np.float16)

                    elif col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:

                        df[col] = df[col].astype(np.float32)

                    else:

                        df[col] = df[col].astype(np.float64) 

        

        

    if verbose:

        end_mem = df.memory_usage().sum() / 1024**2

        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    

    return df  
df_calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

df_sell_prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")

df_sample_submission = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")

df_sales_train_validation = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")
# California

stv_ca = df_sales_train_validation[df_sales_train_validation['state_id']=='CA']
stv_ca.memory_usage().sum() / 1024**2 
# # don't use when there are large number of columns

# stv_ca = reduce_memory(stv_ca, verbose=True)
stv_ca = pd.melt(stv_ca, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
stv_ca.memory_usage().sum()/1024**2
stv_ca.head()
stv_ca.memory_usage().sum()/1024**2
df_calendar.rename(columns={'d':'day'}, inplace=True)

stv_ca_m = stv_ca.merge(df_calendar, how='left', on='day')

stv_ca_m = stv_ca_m.merge(df_sell_prices[df_sell_prices['store_id'].isin(['CA_1','CA_2','CA_3'])], how='left', on=['store_id','item_id', 'wm_yr_wk'])
stv_ca_m.memory_usage().sum()/1024**2
stv_ca_m.head()
cat_id_map=dict(zip(stv_ca_m.cat_id.unique(),np.arange(1,1+stv_ca_m.cat_id.nunique(),dtype=np.int8)))

store_id_map=dict(zip(stv_ca_m.store_id.unique(),np.arange(1,1+stv_ca_m.store_id.nunique(),dtype=np.int8)))

state_id_map=dict(zip(stv_ca_m.state_id.unique(),np.arange(1,1+stv_ca_m.state_id.nunique(),dtype=np.int8)))

dept_id_map=dict(zip(stv_ca_m.dept_id.unique(),np.arange(1,1+stv_ca_m.dept_id.nunique(),dtype=np.int8)))



stv_ca_m['dept_id'] = stv_ca_m['dept_id'].map(dept_id_map)

stv_ca_m['cat_id'] = stv_ca_m['cat_id'].map(cat_id_map)

stv_ca_m['store_id'] = stv_ca_m['store_id'].map(store_id_map)

stv_ca_m['state_id'] = stv_ca_m['state_id'].map(state_id_map)





event_name_1_map = dict(zip(stv_ca_m.event_name_1.unique(), np.arange(stv_ca_m.event_name_1.nunique(),dtype=np.int8)))

event_type_1_map = dict(zip(stv_ca_m.event_type_1.unique(), np.arange(stv_ca_m.event_type_1.nunique(),dtype=np.int8)))

event_name_2_map = dict(zip(stv_ca_m.event_name_2.unique(), np.arange(stv_ca_m.event_name_2.nunique(),dtype=np.int8)))

event_type_2_map = dict(zip(stv_ca_m.event_type_2.unique(), np.arange(stv_ca_m.event_type_2.nunique(),dtype=np.int8)))



stv_ca_m['event_name_1'] = stv_ca_m['event_name_1'].map(event_name_1_map)

stv_ca_m['event_type_1'] = stv_ca_m['event_type_1'].map(event_type_1_map)

stv_ca_m['event_name_2'] = stv_ca_m['event_name_2'].map(event_name_2_map)

stv_ca_m['event_type_2'] = stv_ca_m['event_type_2'].map(event_type_2_map)
stv_ca_m.memory_usage().sum()/1024**2
stv_ca_m = reduce_memory(stv_ca_m, verbose=True)
stv_ca_m['lag_t7'] = stv_ca_m.groupby(['id'])['demand'].transform(lambda x: x.shift(7))

stv_ca_m['lag_t14'] = stv_ca_m.groupby(['id'])['demand'].transform(lambda x: x.shift(14))

stv_ca_m['lag_t28'] = stv_ca_m.groupby(['id'])['demand'].transform(lambda x: x.shift(28))

stv_ca_m['lag_t365'] = stv_ca_m.groupby(['id'])['demand'].transform(lambda x: x.shift(365))

stv_ca_m['rolling_mean_t7'] = stv_ca_m.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())

stv_ca_m['rolling_std_t7'] = stv_ca_m.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())

stv_ca_m['rolling_mean_t14'] = stv_ca_m.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(14).mean())

stv_ca_m['rolling_std_t14'] = stv_ca_m.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(14).std())
stv_ca_m.head()
df = stv_ca_m.drop(columns=['wm_yr_wk','weekday'])
stv_ca_m = stv_ca_m[stv_ca_m['date']>='2015-01-01']
def demand_plot(item_id):



    fig = go.Figure(layout={'height':500, 'width':1000, 

                            'hoverlabel':{'namelength':0}})



    fig.add_trace(go.Scatter(x =stv_ca_m[stv_ca_m['item_id']==item_id]['date'],

                                        y = stv_ca_m[stv_ca_m['item_id']==item_id]['demand'],

                             mode='lines',

                             line=dict(color='black', width=3),

                             name='demand'))



    fig.add_trace(go.Scatter(x =stv_ca_m[stv_ca_m['item_id']==item_id]['date'],

                                        y = stv_ca_m[stv_ca_m['item_id']==item_id]['lag_t7'],

                            mode='lines',

#                             opacity=0.5,

                            name='lag_t7'))



    fig.add_trace(go.Scatter(x =stv_ca_m[stv_ca_m['item_id']==item_id]['date'],

                                        y = stv_ca_m[stv_ca_m['item_id']==item_id]['lag_t14'],

                            mode='lines',

#                             opacity=0.5,

                            name='lag_t14'))



    fig.add_trace(go.Scatter(x =stv_ca_m[stv_ca_m['item_id']==item_id]['date'],

                                        y = stv_ca_m[stv_ca_m['item_id']==item_id]['lag_t28'],

                            mode='lines',

#                             opacity=0.5, 

                            name='lag_t28'))



    fig.add_trace(go.Scatter(x =stv_ca_m[stv_ca_m['item_id']==item_id]['date'],

                                        y = stv_ca_m[stv_ca_m['item_id']==item_id]['lag_t365'],

                            mode='lines',

#                             opacity=0.5, 

                            name='lag_t365'))



    # fig.add_trace(go.Scatter(x =stv_ca_m[stv_ca_m['item_id']==item_id]['date'],

    #                                     y = stv_ca_m[stv_ca_m['item_id']==item_id]['rolling_mean_t7']))





    # fig.add_trace(go.Scatter(x =stv_ca_m[stv_ca_m['item_id']==item_id]['date'],

    #                                     y = stv_ca_m[stv_ca_m['item_id']==item_id]['rolling_mean_t14']))





    # fig.add_trace(go.Scatter(x =stv_ca_m[stv_ca_m['item_id']==item_id]['date'],

    #                                     y = stv_ca_m[stv_ca_m['item_id']==item_id]['rolling_std_t7']))



    # fig.add_trace(go.Scatter(x =stv_ca_m[stv_ca_m['item_id']==item_id]['date'],

    #                                     y = stv_ca_m[stv_ca_m['item_id']==item_id]['rolling_std_t14']))





    fig.update_layout(title=item_id,

                      title_x=0.45,           

                      title_font_size=25,

                      hovermode='x',

                    xaxis_title='date',       

                    yaxis_title='demand metric')



    return fig.show()
demand_plot('HOBBIES_1_016')
#ipywidget



item_dropdown = widgets.Dropdown(options = stv_ca_m.item_id.unique().tolist())

output_plot = widgets.Output()



def item_eventhandler(change):

    output_plot.clear_output(wait=True)

    with output_plot:    

        if change.name=='HOBBIES_1_001':

            display(demand_plot('HOBBIES_1_001'))

        else :

            display(demand_plot(change.new))

item_dropdown.observe(item_eventhandler, type='change',names=['value'])

display(item_dropdown)

display(output_plot)