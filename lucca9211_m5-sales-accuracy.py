import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pylab as plt

plt.style.use('fivethirtyeight')



type_colors = sns.color_palette("hls", 16)

# import tensorflow as tf

# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# from keras.callbacks import ModelCheckpoint

# from keras.models import Sequential

# from keras.layers import Dense, Activation, Flatten, Dropout



# from matplotlib import pyplot as plt



pd.options.display.max_columns = None

pd.options.display.max_rows = None
cal_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

print("Shape of calendar dataset:- ", cal_df.shape)

cal_df.head()
cal_df.isna().sum()
sales_df = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")

print("Shape of sales_train_validation dataset:- ", sales_df.shape)

sales_df.head()
sales_df.isna().sum()
sell_price_df = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")

print("Shape of sell_price dataset:- ", sell_price_df.shape)

sell_price_df.head()
sell_price_df.isna().sum()
cal_df["event_name_1"].fillna("no_event_1", inplace = True) 

cal_df["event_name_2"].fillna("no_event_2", inplace = True) 

cal_df["event_type_1"].fillna("no_type_1", inplace = True) 

cal_df["event_type_2"].fillna("no_type_2", inplace = True) 

cal_df.isna().sum()
# Check different items in sales_train_validation Dataset

sales_df.item_id.value_counts(normalize=False)

# Check different department in sales_train_validation Dataset

sales_df.dept_id.value_counts(normalize=False)

# Check different items category in sales_train_validation Dataset

sales_df.cat_id.value_counts(normalize=False)

# Check no of stores in sales_train_validation Dataset

sales_df.store_id.value_counts(normalize=False)

# Check no of states in sales_train_validation Dataset

sales_df.state_id.value_counts(normalize=False)

dd = sales_df.groupby(['state_id']).sum()

dd.head()
dd.index
dd.index[0]
# Check different items in sales_train_validation Dataset

sales_df.state_id.value_counts(normalize=False)

sd_plot = sales_df.dept_id.value_counts()

ax = sd_plot.plot(kind='barh', figsize=(7, 5),   # barh -> for Horizontal rectangles plot & bar -> Vertical rectangles plot

          title='Department wise Items contribution',

          color=type_colors)

for i, (p, pr) in enumerate(zip(sd_plot.index, sd_plot.values)):

    

    plt.text(s=str(pr), x=pr-5, y=i, color="w",

             verticalalignment="center", horizontalalignment="right", size=12)

ax.set_xlabel("Total count")

plt.show()
sd_plot = sales_df.dept_id.value_counts()

ax = sd_plot.plot(kind='bar', figsize=(10, 5),   # barh -> for Horizontal rectangles plot & bar -> Vertical rectangles plot

          title='Department wise Items contribution',

          color=type_colors)

for i, (p, pr) in enumerate(zip(sd_plot.index, sd_plot.values)):

    

    plt.text(s=str(pr), y=pr-5, x=i, color="b",

             horizontalalignment='center', verticalalignment='top',

              size=14)

ax.set_xlabel("Department Group")

ax.set_ylabel("Count")

plt.xticks(rotation= 45) 

plt.show()
with sns.axes_style('white'):

    g = sns.catplot(x="dept_id", data=sales_df.sort_values("dept_id"), aspect=2,

                       kind="count", palette=type_colors, order=sales_df['dept_id'].value_counts().index)

    

    g.set_xlabels('Department Group')

    g.fig.suptitle("Department wise Items contribution")

    g.set_xticklabels(rotation=45, horizontalalignment='right')

sc_plot = sales_df.cat_id.value_counts()

ax = sc_plot.plot(kind='bar', figsize=(10, 5),   # barh -> for Horizontal rectangles plot & bar -> Vertical rectangles plot

          title='Category wise Items contribution',

          color=type_colors)

for i, (p, pr) in enumerate(zip(sc_plot.index, sc_plot.values)):

    

    plt.text(s=str(pr), y=pr-5, x=i, color="b",

             horizontalalignment='center', verticalalignment='top',

              size=14)

ax.set_xlabel("Category Group")

ax.set_ylabel("Count")

plt.xticks(rotation= 45) 

plt.show()
ss_plot = sales_df.state_id.value_counts()

ax = ss_plot.plot(kind='bar', figsize=(10, 5),   # barh -> for Horizontal rectangles plot & bar -> Vertical rectangles plot

          title='State wise Items contribution',

          color=type_colors)

for i, (p, pr) in enumerate(zip(ss_plot.index, ss_plot.values)):

    

    plt.text(s=str(pr), y=pr-5, x=i, color="b",

             horizontalalignment='center', verticalalignment='top',

              size=14)

ax.set_xlabel("State Group")

ax.set_ylabel("Count")

plt.xticks(rotation= 45) 

plt.show()
dd = sales_df.groupby(['state_id']).sum()

dd
dd.columns
dd_cols = [c for c in dd.columns if 'd_' in c] # sales data columns
dd.loc[dd.index[0]][dd_cols].T.plot(figsize=(15, 5),

          title='Sales per day in California State (CA)'

                                   )

plt.legend('')

plt.show()
dd.loc[dd.index[1]][dd_cols].T.plot(figsize=(15, 5),   # T -> is for transpose

          title='Sales per day in Texas State (TA)'

                                   )

plt.legend('')

plt.show()
dd.loc[dd.index[2]][dd_cols].T.plot(figsize=(15, 5),   # T -> is for transpose

          title='Sales per day in Wisconsin State (WI)'

                                   )

plt.legend('')

plt.show()
sdd = sales_df.groupby(['state_id', 'store_id']).sum()

sdd
sdd.index
sdd.index[2]        # Store No. 3 (CA_3) in CA State
sdddd_cols = [c for c in sdd.columns if 'd_' in c] # sales data columns
sdd.loc[sdd.index[2]][sdddd_cols].T.plot(figsize=(15, 5),   # T -> is for transpose

          title='Sales per day in Store (CA_3) in California State (CA)'

                                   )

plt.legend('')

plt.show()
sdd.loc[sdd.index[3]][sdddd_cols].T.plot(figsize=(15, 5),   # T -> is for transpose

          title='Sales per day in Store (CA_4) in California State (CA)'

                                   )

plt.legend('')

plt.show()
sdd.index[4]
sdd.loc[sdd.index[4]][sdddd_cols].T.plot(figsize=(15, 5),   # T -> is for transpose

          title='Sales per day in Store (TX_1) in Texas State (CA)'

                                   )



plt.legend('')

plt.show()
sdd.loc[sdd.index[4]][sdddd_cols].T.plot(figsize=(15, 5))

sdd.loc[sdd.index[5]][sdddd_cols].T.plot(figsize=(15, 5))

sdd.loc[sdd.index[6]][sdddd_cols].T.plot(figsize=(15, 5),

                title='Sales per day per Store in Texas State (TX)')



plt.legend('')

plt.show()
ddd = sales_df.groupby(['state_id', 'store_id', 'cat_id']).sum()

ddd
ddddd_cols = ddd.columns
ddd.index[0]
ddd.loc[ddd.index[0]][ddddd_cols].T.plot(figsize=(15, 5),

                         title='Foods Sales per day by Store (CA_1) in California State (CA)')

# sdd.loc[sdd.index[5]][sdddd_cols].T.plot(figsize=(15, 5))

# sdd.loc[sdd.index[6]][sdddd_cols].T.plot(figsize=(15, 5),

#                 title='Sales per day per Store in Texas State (TX)')



plt.legend('')

plt.show()
sales_df.groupby(['state_id',  'cat_id']).size() 
ss_plot = sales_df.groupby(['state_id',  'cat_id']).size()#.groupby(level=1).max()

ax = ss_plot.plot(kind='bar', figsize=(10, 5),   # barh -> for Horizontal rectangles plot & bar -> Vertical rectangles plot

          title='Category wise sales by Sates',

          color=type_colors)

for i, (p, pr) in enumerate(zip(ss_plot.index, ss_plot.values)):

    

    plt.text(s=str(pr), y=pr-5, x=i, color="b",

             horizontalalignment='center', verticalalignment='top',

              size=14)

ax.set_xlabel("States Group")

ax.set_ylabel("Count")

plt.xticks(rotation= 45) 

plt.show()
ss_plot = sales_df.groupby(['state_id',  'cat_id', 'dept_id']).size()#.groupby(level=1).max()

ax = ss_plot.plot(kind='bar', figsize=(10, 5),   # barh -> for Horizontal rectangles plot & bar -> Vertical rectangles plot

          title='Department & Category wise sales by Sates',

          color=type_colors)

for i, (p, pr) in enumerate(zip(ss_plot.index, ss_plot.values)):

    

    plt.text(s=str(pr), y=pr-5, x=i, color="b",

             horizontalalignment='center', verticalalignment='top',

              size=14)

ax.set_xlabel("States Group")

ax.set_ylabel("Count")

plt.xticks(rotation= 45) 

plt.show()
# sales_df= pd.melt(sales_df, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],

#                        var_name = 'd', value_name = 'sales')





del cal_df

del dd

del sd_plot

del ax

del sc_plot

del ss_plot

del dd_cols

del sdd

del sdddd_cols

del ddd

del ddddd_cols

del g
import gc

gc.collect()
from IPython import get_ipython

get_ipython().magic('reset -sf') # pd given cal
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pylab as plt

plt.style.use('fivethirtyeight')



type_colors = sns.color_palette("hls", 16)



pd.options.display.max_columns = None

pd.options.display.max_rows = None
cal_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

sales_df = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")
# Convert string to datetime64

cal_df['date'] = cal_df['date'].apply(pd.to_datetime,format='%Y-%m-%d')

cal_df["event_name_1"].fillna("no_event_1", inplace = True) 

cal_df["event_name_2"].fillna("no_event_2", inplace = True) 

cal_df["event_type_1"].fillna("no_type_1", inplace = True) 

cal_df["event_type_2"].fillna("no_type_2", inplace = True) 

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
cal_df = reduce_mem_usage(cal_df)

sales_df = reduce_mem_usage(sales_df)
given_cal_df = cal_df[0:1913]

comming_cal_df = cal_df[1914:1941]

pred_cal_df = cal_df[1942:1969]

sales_df= pd.melt(sales_df, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],

                       var_name = 'd', value_name = 'sales')
sales_df = reduce_mem_usage(sales_df)
# data_with_sales_date = pd.concat([given_cal_df, sales_df]) 

# data_with_sales_date.head()



data_with_sales_date = pd.merge(given_cal_df, sales_df, how = 'left', left_on = ['d'], right_on = ['d'])

del given_cal_df
data_with_sales_date = reduce_mem_usage(data_with_sales_date)
data_with_sales_date.head()