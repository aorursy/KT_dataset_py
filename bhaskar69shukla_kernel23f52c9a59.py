

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import squarify





from statsmodels.tsa.stattools import pacf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.seasonal import seasonal_decompose

import statsmodels.tsa.stattools as stattools



import time



from xgboost import XGBRegressor

from string import punctuation

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

# import the df

shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")

shops.shape
shops
# We don't have any duplicates in the shop_name field

shops.shape[0] == len(shops["shop_name"].unique())
# No missing values in the shops df

shops.isnull().sum().sum()
 #let's correct the shops df and also generate a few more features

def fix_shops(shops):

    '''

    This function modifies the shops df inplace.

    It correct's 3 shops that we have found to be 'duplicates'

    and also creates a few more features: extracts the city and encodes it using LabelEncoder

    '''

    d = {0:57, 1:58, 10:11, 23:24}

    shops["shop_id"] = shops["shop_id"].apply(lambda x: d[x] if x in d.keys() else x)

    # replace all the punctuation in the shop_name columns

    shops["shop_name_cleaned"] = shops["shop_name"].apply(lambda s: "".join([x for x in s if x not in punctuation]))

    # extract the city name

    shops["city"] = shops["shop_name_cleaned"].apply(lambda s: s.split()[0])

    # encode it using a simple LabelEncoder

    shops["city_id"] = LabelEncoder().fit_transform(shops['city'])
# apply our function to the shops_df

fix_shops(shops)
shops.sample(10)
# import df

items_category = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")

items_category.shape
items_category.sample(10)
# We don't have any duplicates in the item_category_name field

items_category.shape[0] == len(items_category["item_category_name"].unique())
pd.options.display.max_rows = items_category.shape[0]
items_category
items_category["PS_flag"] = items_category["item_category_name"].apply(lambda x: True if "PS" in x else False)

items_category[items_category["PS_flag"] == True]
# No missing values in the items_category df

items_category.isnull().sum().sum()
# import df

items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")

items.shape
# allow pandas to show all the rows from this df

pd.options.display.max_rows = items.shape[0]

items.head(10)
 #No missing values in the items category

items.isnull().sum().sum()
# Let's see the top 10 and bottom 10 item categories

items_gb = items.groupby("item_category_id").size().to_frame()
items_gb.rename(columns = {0:"counts"}, inplace = True)
top_10 = items_gb.sort_values("counts", ascending=False)[:10]

top_10
bottom_10 = items_gb.sort_values("counts", ascending=True)[:10]

bottom_10
op_10 = top_10.append(bottom_10)

top_10 = top_10.sort_values("counts", ascending = False)
top_10.reset_index()
pd.merge(top_10, items_category, left_on = "item_category_id", right_on = "item_category_id")
# import df

sales = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

sales.shape
sales.sample(10)
# No null values in the sales df

sales.isnull().sum().sum()
# useful function that manipulates the df and casts all the values to a lower numeric type and saves memory

def reduce_mem_usage(df, verbose=True):

    '''

    Reduces the space that a DataFrame occupies in memory.

    '''

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
# a simple function that creates a global df with all joins and also shops corrections

def create_df():

   

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

    df["date"] = pd.to_datetime(df["date"], format = "%d.%m.%Y")

    df.sort_values(by = ["shop_id", "date"], ascending = True, inplace = True)

    

    # reduce memory usage

    df.memory_usage()/1014**2

    

    return df
df = create_df()

df.head()
# calculate the monthly sales

x = df[["date", "item_cnt_day"]].set_index("date").resample("M").sum()







# plot the data using matplotlib

plt.figure(figsize = (20, 10))

plt.plot(x.index, x, color = "blue", label = "Monthly sales")



plt.title("Monthly sales ")

plt.legend();