# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# make calendar maps
!pip install calmap
import calmap
# Main libraries that we will use in this kernel
import datetime
import numpy as np
import pandas as pd

# # garbage collector: free some memory is needed
import gc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# pip install squarify (algorithm for treemap) if missing
import squarify

# statistical package and some useful functions to analyze our timeseries
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.stattools as stattools

import time

from xgboost import XGBRegressor
from string import punctuation
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

def print_files():
    import os
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")
# let's correct the shops df and also generate a few more features
def fix_shops(shops):
    '''
    This function modifies the shops df inplace.
    It correct's 3 shops that we have found to be 'duplicates'
    and also creates a few more features: extracts the city and encodes it using LabelEncoder
    '''
    
    d = {0:57, 1:58, 10:11, 23:24}
    
    # this 'tricks' allows you to map a series to a dictionary, but all values that are not in the dictionary won't be affected
    # it's handy since if we blindly map the values, the missings values will be replaced with nan
    shops["shop_id"] = shops["shop_id"].apply(lambda x: d[x] if x in d.keys() else x)
    
    # replace all the punctuation in the shop_name columns
    shops["shop_name_cleaned"] = shops["shop_name"].apply(lambda s: "".join([x for x in s if x not in punctuation]))
    
    # extract the city name
    shops["city"] = shops["shop_name_cleaned"].apply(lambda s: s.split()[0])
    
    # encode it using a simple LabelEncoder
    shops["city_id"] = LabelEncoder().fit_transform(shops['city'])
# a simple function that creates a global df with all joins and also shops corrections
def create_df():
    '''
    This is a helper function that creates the train df.
    '''
    # import all df
    shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")
    fix_shops(shops) # fix the shops as we have seen before
    
    items_category = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")
    items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
    sales = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
    
    # fix shop_id in sales so that we can leater merge the df
    d = {0:57, 1:58, 10:11, 23:24}
    sales["shop_id"] = sales["shop_id"].apply(lambda x: d[x] if x in d.keys() else x)
    
    # create df by merging the previous dataframes
    df = pd.merge(items, items_category, left_on = "item_category_id", right_on = "item_category_id")
    df = pd.merge(sales, df, left_on = "item_id", right_on = "item_id")
    df = pd.merge(df, shops, left_on = "shop_id", right_on = "shop_id")
    
    # convert to datetime and sort the values
#     df["date"] = pd.to_datetime(df["date"], format = "%d.%m.%Y")
    df.sort_values(by = ["shop_id", "date"], ascending = True, inplace = True)
    
    # reduce memory usage
#     df = reduce_mem_usage(df)
    
    return df
df = create_df()
df['date']=pd.to_datetime(df['date'], format="%d.%m.%Y")
df_ma7d = df[["date", "item_cnt_day"]]
# Ponemos como indice el campo fecha (date)
df_ma7d.set_index("date", inplace = True)
df_ma7d.head()
# Como solo tengo una variable en este dataframe, podemos hacer el resample agrupando por dias asi:
df_ma7d = df_ma7d.resample("D").sum()
# Si tuvieramos mas variables en el dataframe, lo podriamos hacer asi: 
# df_ma7d = df_ma7d.resample("D")['item_cnt_day'].sum().to_frame()
# Vemos que han quedado agrupados por dia con el sumatorio de las ventas:
df_ma7d.head()
df_ma7d['MA7D']=df_ma7d['item_cnt_day'].rolling(window = 7).mean()
# Vemos que han quedado agrupados por dia y su correspondiente MA:
df_ma7d.head(10)
# calculate the intra day variation between total sales
df_ma7d["Variation"] = df_ma7d["MA7D"].diff()/df_ma7d["MA7D"].shift(1)
# Vemos que han quedado agrupados por dia, su correspondiente MA7D y su variacion diaria:
df_ma7d.head(10)
# Borro los NaN
df_ma7d.dropna(axis=0,inplace=True)
# Preparamos la figura
fig = plt.figure(figsize = (15, 10))
ax = fig.add_subplot(111)

# pintamos las medias moviles a 7 dias
plot1 = ax.plot(df_ma7d["MA7D"], label = "MA7D sales", color = "blue", alpha = 0.5)

# creamos un eje secundario y pintamos la variacion diaira de las medias moviles a 7 dias:
ax_bis = ax.twinx()
plot2 = ax_bis.plot(df_ma7d["Variation"], label = "Intra - day variation MA7D", color = "red", alpha = 0.5)

# create a common legend for both plots
lns = plot1 + plot2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc = "upper left")

# añadimos el titulo
ax.set_title("Total MA7D sales and day variation");
df.groupby(['city_id']).size().sort_values(ascending=False).to_frame().head()
df[ df['city_id']==13]['city'].unique()
df_timeindex_city=df.set_index('date').groupby('city_id').resample("W")["item_cnt_day"].sum().to_frame()
df_timeindex_city.head()
# Lo formateamos un poco:
df_timeindex_city.reset_index(inplace=True)
df_timeindex_city.set_index('date',inplace=True)
# Vemos los datos para Moscu:
df_timeindex_city[df_timeindex_city['city_id']==13]
# decompose the series using stats module
# results in this case is a special class 
# whose attributes we can acess
result = seasonal_decompose(df_timeindex_city[df_timeindex_city['city_id']==13]["item_cnt_day"])

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
# make the subplots share teh x axis
fig, axes = plt.subplots(ncols = 1, nrows = 4, sharex = True, figsize = (12,10))

# ----------------------------------------------------------------------------------------------------
# plot the data
# using this cool thread:
# https://stackoverflow.com/questions/45184055/how-to-plot-multiple-seasonal-decompose-plots-in-one-figure
# This allows us to have more control over the plots

# plot the original data
result.observed.plot(ax = axes[0], legend = False)
axes[0].set_ylabel('Observed')
axes[0].set_title("Decomposition of a series")

# plot the trend
result.trend.plot(ax = axes[1], legend = False)
axes[1].set_ylabel('Trend')

# plot the seasonal part
result.seasonal.plot(ax = axes[2], legend = False)
axes[2].set_ylabel('Seasonal')

# plot the residual
result.resid.plot(ax = axes[3], legend = False)
axes[3].set_ylabel('Residual')

# ----------------------------------------------------------------------------------------------------
# prettify the plot

# get the xticks and the xticks labels
xtick_location = df_timeindex_city.index.tolist()

# set x_ticks
axes[0].set_xticks(xtick_location);
df_items=df[['item_category_name', 'item_cnt_day']]
df_items.head()
df_items=df_items.groupby(['item_category_name'])['item_cnt_day'].sum().sort_values(ascending=False).to_frame()
# Cambiamos el nombre ,quitamos el dia y ponemos toital:
df_items.columns=['item_cnt_total']
df_items.head(10)
# get the x and y values
my_values = df_items["item_cnt_total"]
my_pct = df_items["item_cnt_total"]/df_items["item_cnt_total"].sum()
labels = ['{} - Sales :{}k \n {}% of total'.format(item, sales/1000, round(pct, 2)*100) 
          if (pct >= 0.01) else '' for item, sales, pct in zip(df_items.index, my_values, my_pct)]

plt.figure(figsize = (30, 8))
squarify.plot(sizes = my_values, label = labels,  alpha = 0.8)
plt.title("Sales by city item_category % over total sales",fontsize = 23, fontweight = "bold")

plt.axis('off')
plt.tight_layout()

# create a color palette, mapped to the previous values
cmap = matplotlib.cm.Blues

# we want to normalize our values, otherwise a city will have the darkest collor and all the others will pale
mini = min(my_values)
maxi= np.percentile(my_values, q = 99)
norm = matplotlib.colors.Normalize(vmin = mini, vmax = maxi)
colors = [cmap(norm(value)) for value in my_values]
# instanciate the figure
plt.figure(figsize = (30, 8))
# we can pass colors but Moscow is way too big and most of the cities are pale blue
squarify.plot(sizes = my_values, label = labels,  alpha = 0.8, color  = colors)

# Remove our axes, set a title and display the plot
plt.title("Sales by city item_category % over total sales", fontsize = 23, fontweight = "bold")
plt.axis('off')
plt.tight_layout()
