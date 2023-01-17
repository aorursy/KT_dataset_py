import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.subplots import make_subplots

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from tqdm.notebook import tqdm as tqdm

import statsmodels.api as sm

import gc

plt.style.use('fivethirtyeight')

from pylab import rcParams

import random

import seaborn as sns

from lightgbm import LGBMRegressor

# to display all the columns in the dataset

pd.pandas.set_option('display.max_columns', None)
train_sales = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_evaluation.csv")

calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

sell_prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
train_sales.shape, calendar.shape,sell_prices.shape
train_sales.info()
calendar.info()
sell_prices.info()
train_sales.head()
calendar.head()
sell_prices.head()
train_sales.isnull().sum().sort_values(ascending = False)
sell_prices.isnull().sum().sort_values(ascending = False)
calendar.isnull().sum().sort_values(ascending = False)
# memory usage reduction

def downcast(df):

    cols = df.dtypes.index.tolist()

    types = df.dtypes.values.tolist()

    for i,t in enumerate(types):

        if 'int' in str(t):

            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:

                df[cols[i]] = df[cols[i]].astype(np.int8)

            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:

                df[cols[i]] = df[cols[i]].astype(np.int16)

            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:

                df[cols[i]] = df[cols[i]].astype(np.int32)

            else:

                df[cols[i]] = df[cols[i]].astype(np.int64)

        elif 'float' in str(t):

            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:

                df[cols[i]] = df[cols[i]].astype(np.float16)

            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:

                df[cols[i]] = df[cols[i]].astype(np.float32)

            else:

                df[cols[i]] = df[cols[i]].astype(np.float64)

        elif t == np.object:

            if cols[i] == 'date':

                df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')

            else:

                df[cols[i]] = df[cols[i]].astype('category')

    return df  
# calling memory reduction function for each data set

train_sales = downcast(train_sales)

sell_prices = downcast(sell_prices)

calendar = downcast(calendar)
# let's save the list of date variables to a list

d_cols = [c for c in train_sales.columns if 'd_' in c]
# lets save top 3 selling items to be analysed later

top3 = train_sales.set_index("id")[d_cols].sum(1).sort_values(ascending  = False)[:3].index
grid_df = pd.melt(train_sales, 

                  id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 

                  var_name = 'd', 

                  value_name = "sales")
group = grid_df.groupby(['state_id','store_id','cat_id','dept_id'],as_index=False)['sales'].sum().dropna()

group['USA'] = 'United States of America'

group.rename(columns={'state_id':'State','store_id':'Store','cat_id':'Category','dept_id':'Department','item_id':'sales'},inplace=True)

fig = px.treemap(group, path=['USA','State', 'Store', 'Category', 'Department'], values='sales',

                  color='sales',

                  title='Sum of sales across whole USA/different States/Stores/Categories/Departments')

fig.update_layout(template='seaborn')

fig.show()
del train_sales

gc.collect()
# lets drop the columns we are not going to use for EDA

calendar.drop(['wm_yr_wk','weekday','wday','month','year','event_name_1','event_type_1', 'event_name_2','event_type_2'],1,inplace=True)
master = pd.merge(grid_df,calendar, on = "d")

master.head()
del grid_df

gc.collect()
def sales(feat,param):

    sales_df = master.loc[master[feat] == param]

    sales_df['date'] = pd.to_datetime(sales_df['date'])

    sales_df =sales_df.groupby('date')['sales'].sum().reset_index()

    sales_df = sales_df.set_index('date')

    return sales_df
from itertools import cycle, islice

def decompose(y):

    rcParams['figure.figsize'] = 18, 8

    decomposition = sm.tsa.seasonal_decompose(y, model='additive')

    fig = decomposition.plot()

    plt.show()
def random_color():

    colors = ["blue","black","brown","red","yellow","green","orange","turquoise","magenta","cyan"]

    random.shuffle(colors)

    return colors[0]
# list of unique states

master.state_id.unique()
CA = sales("state_id","CA") # create a dataframe for the state CA

y_ca = CA['sales'].resample('MS').mean() # taking monthly average

colour = random_color()

y_ca.plot(figsize=(15, 6),color = colour,title = ("Sales for the state of CA"))

plt.ylabel = ("Sales")

plt.show()
decompose(y_ca)
WI = sales("state_id","WI")

y_wi = WI['sales'].resample('MS').mean()

colour = random_color()

y_wi.plot(figsize=(15, 6),color = colour,title = ("Sales for the state of WI"))

plt.ylabel = ("Sales")

plt.show()
decompose(y_wi)
TX = sales("state_id","TX")

y_tx = TX['sales'].resample('MS').mean()

colour = random_color()

y_tx.plot(figsize=(15, 6),color = colour,title = ("Sales for the state of TX"))

plt.show()
decompose(y_tx)
del CA,WI,TX

gc.collect()
# list of unique categories

master.cat_id.unique()
foods = sales("cat_id","FOODS")

y_f = foods['sales'].resample('MS').mean()

colour = random_color()

y_f.plot(figsize=(15, 6),color = colour,title = ("Sales for the category:FOODS"))

plt.show()
decompose(y_f)
hobbies = sales("cat_id","HOBBIES")

y_hb = hobbies['sales'].resample('MS').mean()

colour = random_color()

plt.ylabel = ("Sales")

y_hb.plot(figsize=(15, 6),color = colour,title = ("Sales for the category:HOBBIES"))

plt.show()
decompose(y_hb)
household = sales("cat_id","HOUSEHOLD")

y_hh = household['sales'].resample('MS').mean()

colour = random_color()

y_hh.plot(figsize=(15, 6),color = colour,title = ("Sales for the category:HOUSEHOLD"))

plt.show()
decompose(y_hh)
del foods,hobbies,household,y_f,y_hb,y_hh

gc.collect()
master.store_id.unique
CA_1 = sales("store_id","CA_1")

y_CA1 = CA_1['sales'].resample('MS').mean()

colour = random_color()

y_CA1.plot(figsize=(15, 6),color = colour,title = ("Sales for the store:CA_1"))

plt.show()
decompose(y_CA1)
CA_2 = sales("store_id","CA_2")

y_CA2 = CA_2['sales'].resample('MS').mean()

colour = random_color()

y_CA2.plot(figsize=(15, 6),color = colour,title = ("Sales for the store:CA_2"))

plt.show()
decompose(y_CA2)
CA_3 = sales("store_id","CA_3")

y_CA3 = CA_3['sales'].resample('MS').mean()

colour = random_color()

y_CA3.plot(figsize=(15, 6),color = colour,title = "Sales for the store:CA_3")

plt.show()
decompose(y_CA3)
CA_4 = sales("store_id","CA_4")

y_CA4 = CA_4['sales'].resample('MS').mean()

colour = random_color()

y_CA4.plot(figsize=(15, 6),color = colour,title = ("Sales for the store:CA_4"))

plt.show()
decompose(y_CA4)
TX_1 = sales("store_id","TX_1")

y_TX1 = TX_1['sales'].resample('MS').mean()

colour = random_color()

y_TX1.plot(figsize=(15, 6),color = colour,title = ("Sales for the store:TX_1"))

plt.show()
decompose(y_TX1)
TX_2 = sales("store_id","TX_2")

y_TX2 = TX_2['sales'].resample('MS').mean()

colour = random_color()

plt.ylabel = ("Sales")

y_TX2.plot(figsize=(15, 6),color = colour,title = ("Sales for the store:TX_2"))

plt.show()
decompose(y_TX2)
TX_3 = sales("store_id","TX_3")

y_TX3 = TX_3['sales'].resample('MS').mean()

colour = random_color()

plt.ylabel = ("Sales")

y_TX3.plot(figsize=(15, 6),color = colour,title = ("Sales for the store:TX_3"))

plt.show()
decompose(y_TX3)
WI_1 = sales("store_id","WI_1")

y_WI1 = WI_1['sales'].resample('MS').mean()

colour = random_color()

plt.ylabel = ("Sales")

y_WI1.plot(figsize=(15, 6),color = colour,title = ("Sales for the store:WI_1"))

plt.show()
decompose(y_WI1)
WI_2= sales("store_id","WI_2")

y_WI2 = WI_2['sales'].resample('MS').mean()

colour = random_color()

plt.ylabel = ("Sales")

y_WI2.plot(figsize=(15, 6),color = colour,title = ("Sales for the store:WI_2"))

plt.show()
decompose(y_WI2)
WI_3= sales("store_id","WI_3")

y_WI3 = WI_3['sales'].resample('MS').mean()

colour = random_color()

plt.ylabel = ("Sales")

y_WI3.plot(figsize=(15, 6),color = colour,title = ("Sales for the store:WI_3"))

plt.show()
decompose(y_WI3)
del CA_1,CA_2,CA_3,CA_4,TX_1,TX_2,TX_3,WI_1,WI_2,WI_3

gc.collect()
top = sales("id",top3[0])

y_top = top['sales'].resample('MS').mean()

colour = random_color()

y_top.plot(figsize=(15, 6),color = colour,title = ("Sales for the Product:" + top3[0]))

plt.show()
top = sales("id",top3[1])

y_top = top['sales'].resample('MS').mean()

colour = random_color()

y_top.plot(figsize=(15, 6),color = colour,title = ("Sales for the Product:" + top3[1]))

plt.show()
top = sales("id",top3[2])

y_top = top['sales'].resample('MS').mean()

colour = random_color()

y_top.plot(figsize=(15, 6),color = colour,title = ("Sales for the Product:" + top3[2]))

plt.show()
del top3,y_top

gc.collect()
colour = random_color()

sns.distplot(sell_prices["sell_price"],color = colour).set_title("Price Distribution")
colour = random_color()

CA_1= sell_prices[sell_prices["store_id"] == "CA_1"]

sns.distplot(CA_1["sell_price"],color = colour).set_title("Price Distribution for CA_1")
colour = random_color()

CA_2= sell_prices[sell_prices["store_id"] == "CA_2"]

sns.distplot(CA_2["sell_price"],color = colour).set_title("Price Distribution for CA_2")
colour = random_color()

CA_3= sell_prices[sell_prices["store_id"] == "CA_3"]

sns.distplot(CA_3["sell_price"],color = colour).set_title("Price Distribution for CA_3")
colour = random_color()

CA_4= sell_prices[sell_prices["store_id"] == "CA_4"]

sns.distplot(CA_4["sell_price"],color = colour).set_title("Price Distribution for CA_4")
colour = random_color()

TX_1= sell_prices[sell_prices["store_id"] == "TX_1"]

sns.distplot(TX_1["sell_price"],color = colour).set_title("Price Distribution for TX_1")
colour = random_color()

TX_2= sell_prices[sell_prices["store_id"] == "TX_2"]

sns.distplot(TX_2["sell_price"],color = colour).set_title("Price Distribution for TX_2")
colour = random_color()

TX_3= sell_prices[sell_prices["store_id"] == "TX_3"]

sns.distplot(TX_3["sell_price"],color = colour).set_title("Price Distribution for TX_3")
colour = random_color()

WI_1= sell_prices[sell_prices["store_id"] == "WI_1"]

sns.distplot(WI_1["sell_price"],color = colour).set_title("Price Distribution for WI_1")
colour = random_color()

WI_2= sell_prices[sell_prices["store_id"] == "WI_2"]

sns.distplot(WI_2["sell_price"],color = colour).set_title("Price Distribution for WI_2")
colour = random_color()

WI_3= sell_prices[sell_prices["store_id"] == "WI_3"]

sns.distplot(WI_3["sell_price"],color = colour).set_title("Price Distribution for WI_3")
del sell_prices,CA_1,CA_2,CA_3,CA_4,TX_1,TX_2,TX_3,WI_1,WI_2,WI_3

gc.collect()
import gc

gc.collect()