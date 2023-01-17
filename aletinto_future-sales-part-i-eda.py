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
# Let's see how many different files we are dealing with
print_files()
# import the df
shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")
shops.shape
shops
# We don't have any duplicates in the shop_name field
shops.shape[0] == len(shops["shop_name"].unique())
# However inspecting the df by name, we can see that shop_id 10 and 11 are very similar. Later we will try and group them once we inspect the sales per shop
shops[shops["shop_id"].isin([10, 11])]
# The same happens with the shops with shop_id 23 and 24
shops[shops["shop_id"].isin([23, 24])]
# No missing values in the shops df
shops.isnull().sum().sum()
shops.sample(10)
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
# apply our function to the shops_df
fix_shops(shops)
shops.sample(10)
# import df
items_category = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")
items_category.shape
items_category.sample(10)
# We don't have any duplicates in the item_category_name field
items_category.shape[0] == len(items_category["item_category_name"].unique())
# allow pandas to show all the rows from this df
pd.options.display.max_rows = items_category.shape[0]
# If we take a closer look, we can see that we have a lot of Play Station categories: like accesories, games and so on. We have the same categories for XBOX and also for PC Games.
# A lot of categories have to deal with books, presents and computer software and music (CD).
# We will generate later some features by parsing the names and making groupedby features.
items_category
# If we apply a simple lambda function and extract the everything that contains PS, we will get 16 different categories for PlayStation
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
# We have a lot of items_id, and as we can see some of them are very familiar.
items[items["item_id"].isin([69, 70])]["item_name"].iloc[0]
items[items["item_id"].isin([69, 70])]["item_name"].iloc[1]
# No missing values in the items category
items.isnull().sum().sum()
# Let's see the top 10 and bottom 10 item categories
items_gb = items.groupby("item_category_id").size().to_frame()
# a sample of our groupby dataframe
items_gb.sample(10)
items_gb.rename(columns = {0:"counts"}, inplace = True)
items_gb.sort_values("counts", ascending = False, inplace = True)
top_10 = items_gb[:10]
top_10
bottom_10 = items_gb[-10:]
bottom_10
top_10 = top_10.append(bottom_10)
top_10 = top_10.sort_values("counts", ascending = False)
top_10.reset_index()
# We can notice that in the top 10 most popular items products we have PS3
# At the same time, in the bottom 10 products, we can find 2 PS2.
# This means, that we have to be careful while generating features like PS
pd.merge(top_10, items_category, left_on = "item_category_id", right_on = "item_category_id")
# import df
sales = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
sales.shape
sales.sample(10)
#sales block num hace refenrencia a ka cantidad de es va de 0 a un numero
sales.info()
# No null values in the sales df

# Is this True?

sales.isnull().sum().sum()
sales.info()
list(sales['item_cnt_day'].unique())
del shops, items_category, items, sales
gc.collect()
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
df.shape
df.head()
#Aqui te dan el día la venta y el item y si ha occurrido una venta...tiene items que se han vendido durante un solo periodo de tiempo, las ventas no reportadas tienen que estar a 0
# It seems that there are no null values, however this is not fully true. 
# As we will see in the next section, when we groupby and plot the data, there are a lot of months where there have been no sales so basically it's a null value, and we have to impute zero sales for that month.
df.isnull().sum().sum()
# Let's group by Month and see all the sales

# resample in timeseries is the same as groupby
# in order it to work, we must set the date column as index, and it must be a datetime format (strings are not valid)
# when we resample it, we can pass D: daily, W: weekly or M: monthly
# we can then perform operation on the 'resampled' columns like
# sum, mean and others.

# calculate the monthly sales
df["date"] = pd.to_datetime(df["date"], format = "%d.%m.%Y")
x = df[["date", "item_cnt_day"]].set_index("date").resample("M").sum()
x.head()
df['Year'] = df['date'].dt.year
df['Year'] = df['date'].dt.month
# get the percentile 5 and 95 to plot them on the figure

# plot the data using matplotlib
plt.figure(figsize = (20, 10))
plt.plot(x, color = "blue", label = "Monthly sales")
plt.title("Monthly sales")
plt.legend();
# perform the same operations but on a weekly basis
x = df[["date", "item_cnt_day"]].set_index("date").resample("W").sum()

plt.figure(figsize = (20, 10))
plt.plot(x.index, x, color = "blue", label = "Weekly sales")
plt.title("Weekly sales")
plt.legend();
x = df[["date", "item_cnt_day"]]
x["YEAR"] = x["date"].dt.year
x = x.set_index("date").groupby("YEAR").resample("M")["item_cnt_day"].sum().to_frame().reset_index()

x_95_2013 = [np.percentile(x[x["YEAR"] == 2013]["item_cnt_day"], q = 95) for i in range(x[x["YEAR"] == 2013].shape[0])]

x_05_2013 = [np.percentile(x[x["YEAR"] == 2013]["item_cnt_day"], q = 5) for i in range(x[x["YEAR"] == 2013].shape[0])]

x_95_2014 = [np.percentile(x[x["YEAR"] == 2014]["item_cnt_day"], q = 95) for i in range(x[x["YEAR"] == 2014].shape[0])]

x_05_2014 = [np.percentile(x[x["YEAR"] == 2014]["item_cnt_day"], q = 5) for i in range(x[x["YEAR"] == 2014].shape[0])]

x_95_2015 = [np.percentile(x[x["YEAR"] == 2015]["item_cnt_day"], q = 95) for i in range(x[x["YEAR"] == 2015].shape[0])]

x_05_2015 = [np.percentile(x[x["YEAR"] == 2015]["item_cnt_day"], q = 5) for i in range(x[x["YEAR"] == 2015].shape[0])]
plt.figure(figsize = (15, 7.5))
plt.plot(x["date"], x["item_cnt_day"], color = "blue", label = "Monthly sales")

# extact the dates for year 2013 and use them as x for the plot
plt.plot(x[x["YEAR"] == 2013]["date"], x_95_2013, color = "green")
plt.plot(x[x["YEAR"] == 2013]["date"], x_05_2013, color = "red")

plt.plot(x[x["YEAR"] == 2014]["date"], x_95_2014, color = "green")
plt.plot(x[x["YEAR"] == 2014]["date"], x_05_2014, color = "red")

plt.plot(x[x["YEAR"] == 2015]["date"], x_95_2015, color = "green", label = "Percentile 95 of Monthly sales in Year 2015")
plt.plot(x[x["YEAR"] == 2015]["date"], x_05_2015, color = "red", label = "Percentile 5 of Monthly sales in Year 2015")
plt.title("Monthly sales with percentile 5 and 95 calculated per year")
plt.tight_layout()
plt.legend();
# perform the same operation on a weekly basis
x = df[["date", "item_cnt_day"]]
x["YEAR"] = x["date"].dt.year
x = x.set_index("date").groupby("YEAR").resample("W")["item_cnt_day"].sum().to_frame().reset_index()

x_95_2013 = [np.percentile(x[x["YEAR"] == 2013]["item_cnt_day"], q = 95) for i in range(x[x["YEAR"] == 2013].shape[0])]

x_05_2013 = [np.percentile(x[x["YEAR"] == 2013]["item_cnt_day"], q = 5) for i in range(x[x["YEAR"] == 2013].shape[0])]

x_95_2014 = [np.percentile(x[x["YEAR"] == 2014]["item_cnt_day"], q = 95) for i in range(x[x["YEAR"] == 2014].shape[0])]

x_05_2014 = [np.percentile(x[x["YEAR"] == 2014]["item_cnt_day"], q = 5) for i in range(x[x["YEAR"] == 2014].shape[0])]

x_95_2015 = [np.percentile(x[x["YEAR"] == 2015]["item_cnt_day"], q = 95) for i in range(x[x["YEAR"] == 2015].shape[0])]

x_05_2015 = [np.percentile(x[x["YEAR"] == 2015]["item_cnt_day"], q = 5) for i in range(x[x["YEAR"] == 2015].shape[0])]
plt.figure(figsize = (15, 7.5))
plt.plot(x["date"], x["item_cnt_day"], color = "blue", label = "Weekly sales")
plt.plot(x[x["YEAR"] == 2013]["date"], x_95_2013, color = "green")
plt.plot(x[x["YEAR"] == 2013]["date"], x_05_2013, color = "red")

plt.plot(x[x["YEAR"] == 2014]["date"], x_95_2014, color = "green")
plt.plot(x[x["YEAR"] == 2014]["date"], x_05_2014, color = "red")

plt.plot(x[x["YEAR"] == 2015]["date"], x_95_2015, color = "green", label = "Percentile 95 of Weekly sales in Year 2015")
plt.plot(x[x["YEAR"] == 2015]["date"], x_05_2015, color = "red", label = "Percentile 5 of Weekly sales in Year 2015")
plt.tight_layout()
plt.legend();
russian_holidays_start = [
datetime.datetime(2013, 1, 1),
datetime.datetime(2013, 2, 23),
datetime.datetime(2013, 3, 8),
datetime.datetime(2013, 5, 1),
datetime.datetime(2013, 5, 9),
datetime.datetime(2013, 6, 12),
datetime.datetime(2013, 11, 4),

datetime.datetime(2014, 1, 1),
datetime.datetime(2014, 2, 23),
datetime.datetime(2014, 3, 8),
datetime.datetime(2014, 5, 1),
datetime.datetime(2014, 5, 9),
datetime.datetime(2014, 6, 12),
datetime.datetime(2014, 11, 4),

datetime.datetime(2015, 1, 1),
datetime.datetime(2015, 2, 23),
datetime.datetime(2015, 3, 8),
datetime.datetime(2015, 5, 1),
datetime.datetime(2015, 5, 9),
datetime.datetime(2015, 6, 12),
datetime.datetime(2015, 11, 4)
]
russian_holidays_end = [
datetime.datetime(2013, 1, 8),
datetime.datetime(2013, 2, 23),
datetime.datetime(2013, 3, 8),
datetime.datetime(2013, 5, 1),
datetime.datetime(2013, 5, 9),
datetime.datetime(2013, 6, 12),
datetime.datetime(2013, 11, 4),

datetime.datetime(2014, 1, 8),
datetime.datetime(2014, 2, 23),
datetime.datetime(2014, 3, 8),
datetime.datetime(2014, 5, 1),
datetime.datetime(2014, 5, 9),
datetime.datetime(2014, 6, 12),
datetime.datetime(2014, 11, 4),

datetime.datetime(2015, 1, 8),
datetime.datetime(2015, 2, 23),
datetime.datetime(2015, 3, 8),
datetime.datetime(2015, 5, 1),
datetime.datetime(2015, 5, 9),
datetime.datetime(2015, 6, 12),
datetime.datetime(2015, 11, 4)
]
#for iterable in sorted(list(df["shop_name"].unique())):
#
#   # create the size of the figure
#   plt.figure(figsize = (20, 10))
#
#   # create the subplot for Monthly sales of the each shop
#   plt.subplot(1, 2, 1)
#   
#   # calculate the Monthly sales of each shop
#   short_df = df[df["shop_name"] == iterable][["date","item_cnt_day"]]
#    short_df["date"] = pd.to_datetime(short_df["date"], format = "%d.%m.%Y")
#    short_df["YEAR"] = short_df["date"].dt.year
#    short_df = short_df.set_index("date").groupby("YEAR").resample("M")["item_cnt_day"].sum()
#    short_df = short_df.reset_index()
#    
#    # adding moving average
#    short_df["MA3M"] = short_df["item_cnt_day"].rolling(window = 3).mean()
#    short_df["MA4M"] = short_df["item_cnt_day"].rolling(window = 4).mean()
#    short_df["MA5M"] = short_df["item_cnt_day"].rolling(window = 5).mean()
#    
#    # assing the data to plot
#    sales = short_df["item_cnt_day"]
#    dates = short_df["date"]
#    sales_95_global = [np.percentile(sales, q = 95) for i in range(len(sales))]
#    sales_5_global = [np.percentile(sales, q = 5) for i in range(len(sales))]
#    
#    average_3_months = short_df["MA3M"]
#    average_4_months = short_df["MA4M"]
#    average_5_months = short_df["MA5M"]
#    
#    # percentile 5 and 95 of year 2013
#    sales_95_2013 = [np.percentile(short_df[short_df["YEAR"] == 2013]["item_cnt_day"], q = 95) for i in range(short_df[short_df["YEAR"] == 2013].shape[0])]
#    sales_5_2013 = [np.percentile(short_df[short_df["YEAR"] == 2013]["item_cnt_day"], q = 5) for i in range(short_df[short_df["YEAR"] == 2013].shape[0])]
#    dates_2013 = short_df[short_df["YEAR"] == 2013]["date"]
#    
#    # percentile 5 and 95 of year 2014
#    sales_95_2014 = [np.percentile(short_df[short_df["YEAR"] == 2014]["item_cnt_day"], q = 95) for i in range(short_df[short_df["YEAR"] == 2014].shape[0])]
#    sales_5_2014 = [np.percentile(short_df[short_df["YEAR"] == 2014]["item_cnt_day"], q = 5) for i in range(short_df[short_df["YEAR"] == 2014].shape[0])]
#    dates_2014 = short_df[short_df["YEAR"] == 2014]["date"]
#    
#    # percentile 5 and 95 of year 2015
#    sales_95_2015 = [np.percentile(short_df[short_df["YEAR"] == 2015]["item_cnt_day"], q = 95) for i in range(short_df[short_df["YEAR"] == 2015].shape[0])]
#    sales_5_2015 = [np.percentile(short_df[short_df["YEAR"] == 2015]["item_cnt_day"], q = 5) for i in range(short_df[short_df["YEAR"] == 2015].shape[0])]
#    dates_2015 = short_df[short_df["YEAR"] == 2015]["date"]
#
#    # plot the data and add label
#    plt.plot(dates, sales, 'o-', label = "Monthly sales")
#    
#    plt.plot(dates, average_3_months, '.-', label = "Average sales of the last 3 months")
#    
#    plt.plot(dates, sales_95_global, '-', color = "black", label = "P95 of Monthly sales over all years")
#    plt.plot(dates, sales_5_global, '-', color = "magenta", label = "P5 of Monthly sales over all years")
#    
#    plt.plot(dates_2013, sales_95_2013, "--", color = "green")
#    plt.plot(dates_2013, sales_5_2013, ":", color = "red")
#    
#    plt.plot(dates_2014, sales_95_2014, "--", color = "green")
#    plt.plot(dates_2014, sales_5_2014, ":", color = "red")
#    
#    plt.plot(dates_2015, sales_95_2015, "--", color = "green", label = "P95 of Monthly sales by year")
#    plt.plot(dates_2015, sales_5_2015, ":", color = "red", label = "P5 of Monthly sales by year")
#
#    # get current axis and plot the areas
#    ax = plt.gca()
#    alpha = 0.2
#    
#    for start_date, end_date in zip(russian_holidays_start, russian_holidays_end):
#        
#        # add shaded areas for holidays 2013
#        ax.axvspan(start_date, end_date, alpha = alpha, color = 'red')    
#       
#    # add title and show legend    
#    plt.title('Monthly sales of shop {}'.format(iterable))
#    plt.ylabel('Total Monthly sales of shop {}'.format(iterable))
#    plt.xlabel("Time grouped by month")
#    plt.legend()
#    
#    #######################################################################################
#    # Weekly sales
#    #######################################################################################
#    
#    plt.subplot(1, 2, 2)
#    
#      # calculate the Weekly sales of each shop
#    short_df = df[df["shop_name"] == iterable][["date","item_cnt_day"]]
#    short_df["date"] = pd.to_datetime(short_df["date"], format = "%d.%m.%Y")
#    short_df["YEAR"] = short_df["date"].dt.year
#    short_df = short_df.set_index("date").groupby("YEAR").resample("W")["item_cnt_day"].sum()
#    short_df = short_df.reset_index()
#    
#    # adding moving average
#    short_df["MA3W"] = short_df["item_cnt_day"].rolling(window=3).mean()
#    short_df["MA4W"] = short_df["item_cnt_day"].rolling(window=4).mean()
#    short_df["MA5W"] = short_df["item_cnt_day"].rolling(window=5).mean()
#    
#    # assing the data to plot
#    
#    # general sales
#    sales = short_df["item_cnt_day"]
#    dates = short_df["date"]
#    sales_95_global = [np.percentile(sales, q = 95) for i in range(len(sales))]
#    sales_5_global = [np.percentile(sales, q = 5) for i in range(len(sales))]
#    
#    average_3_weeks = short_df["MA3W"]
#    average_4_weeks = short_df["MA4W"]
#    average_5_weeks = short_df["MA5W"]
#    
#    # percentile 5 and 95 of year 2013
#    sales_95_2013 = [np.percentile(short_df[short_df["YEAR"] == 2013]["item_cnt_day"], q = 95) for i in range(short_df[short_df["YEAR"] == 2013].shape[0])]
#    sales_5_2013 = [np.percentile(short_df[short_df["YEAR"] == 2013]["item_cnt_day"], q = 5) for i in range(short_df[short_df["YEAR"] == 2013].shape[0])]
#    dates_2013 = short_df[short_df["YEAR"] == 2013]["date"]
#    
#    # percentile 5 and 95 of year 2014
#    sales_95_2014 = [np.percentile(short_df[short_df["YEAR"] == 2014]["item_cnt_day"], q = 95) for i in range(short_df[short_df["YEAR"] == 2014].shape[0])]
#    sales_5_2014 = [np.percentile(short_df[short_df["YEAR"] == 2014]["item_cnt_day"], q = 5) for i in range(short_df[short_df["YEAR"] == 2014].shape[0])]
#    dates_2014 = short_df[short_df["YEAR"] == 2014]["date"]
#    
#    # percentile 5 and 95 of year 2015
#    sales_95_2015 = [np.percentile(short_df[short_df["YEAR"] == 2015]["item_cnt_day"], q = 95) for i in range(short_df[short_df["YEAR"] == 2015].shape[0])]
#    sales_5_2015 = [np.percentile(short_df[short_df["YEAR"] == 2015]["item_cnt_day"], q = 5) for i in range(short_df[short_df["YEAR"] == 2015].shape[0])]
#    dates_2015 = short_df[short_df["YEAR"] == 2015]["date"]
#
#    # plot the data and add label
#    plt.plot(dates, sales, 'o-', label = "Weekly sales")
#    plt.plot(dates, average_3_weeks, '.-', label = "Average sales of the last 3 weeks")
#    
#    plt.plot(dates, sales_95_global, '-', color = "black", label = "P95 of Weekly sales over all years")
#    plt.plot(dates, sales_5_global, '-', color = "magenta", label = "P5 of Weekly sales over all years")
#    
#    plt.plot(dates_2013, sales_95_2013, "--", color = "green")
#    plt.plot(dates_2013, sales_5_2013, ":", color = "red")
#    
#    plt.plot(dates_2014, sales_95_2014, "--", color = "green")
#    plt.plot(dates_2014, sales_5_2014, ":", color = "red")
#    
#    plt.plot(dates_2015, sales_95_2015, "--", color = "green", label = "P95 of Weekly sales by year")
#    plt.plot(dates_2015, sales_5_2015, ":", color = "red", label = "P5 of Weekly sales by year")
#    
#    # get current axis and plot the areas
#    ax = plt.gca()
#    
#    for start_date, end_date in zip(russian_holidays_start, russian_holidays_end):
#        
#        # add shaded areas for holidays 2013
#        ax.axvspan(start_date, end_date, alpha = alpha, color = 'red')
#    
#    # add title and show legend
#    plt.title('Weekly sales of shop {}'.format(iterable))
#    plt.ylabel('Total Weekly sales of shop {}'.format(iterable))
#    plt.xlabel("Time grouped by week")
#    plt.legend()
#    
#    # general sales
#    plt.show()
#
#for iterable in sorted(list(df["item_category_name"].unique())):
#
#    # create the size of the figure
#    plt.figure(figsize = (20, 10))
#
#    # create the subplot for Monthly sales of the each shop
#    plt.subplot(1, 2, 1)
#    
#    # calculate the Monthly sales of each shop
#    short_df = df[df["item_category_name"] == iterable][["date","item_cnt_day"]]
#    short_df["date"] = pd.to_datetime(short_df["date"], format = "%d.%m.%Y")
#    short_df["YEAR"] = short_df["date"].dt.year
#    short_df = short_df.set_index("date").groupby("YEAR").resample("M")["item_cnt_day"].sum()
#    short_df = short_df.reset_index()
#    
#    # adding moving average
#    short_df["MA3M"] = short_df["item_cnt_day"].rolling(window=3).mean()
#    short_df["MA4M"] = short_df["item_cnt_day"].rolling(window=4).mean()
#    short_df["MA5M"] = short_df["item_cnt_day"].rolling(window=5).mean()
#    
#    # assing the data to plot
#    sales = short_df["item_cnt_day"]
#    dates = short_df["date"]
#    sales_95_global = [np.percentile(sales, q = 95) for i in range(len(sales))]
#    sales_5_global = [np.percentile(sales, q = 5) for i in range(len(sales))]
#    
#    average_3_months = short_df["MA3M"]
#    average_4_months = short_df["MA4M"]
#    average_5_months = short_df["MA5M"]
#    
#    # percentile 5 and 95 of year 2013
#    sales_95_2013 = [np.percentile(short_df[short_df["YEAR"] == 2013]["item_cnt_day"], q = 95) for i in range(short_df[short_df["YEAR"] == 2013].shape[0])]
#    sales_5_2013 = [np.percentile(short_df[short_df["YEAR"] == 2013]["item_cnt_day"], q = 5) for i in range(short_df[short_df["YEAR"] == 2013].shape[0])]
#    dates_2013 = short_df[short_df["YEAR"] == 2013]["date"]
#    
#    # percentile 5 and 95 of year 2014
#    sales_95_2014 = [np.percentile(short_df[short_df["YEAR"] == 2014]["item_cnt_day"], q = 95) for i in range(short_df[short_df["YEAR"] == 2014].shape[0])]
#    sales_5_2014 = [np.percentile(short_df[short_df["YEAR"] == 2014]["item_cnt_day"], q = 5) for i in range(short_df[short_df["YEAR"] == 2014].shape[0])]
#    dates_2014 = short_df[short_df["YEAR"] == 2014]["date"]
#    
#    # percentile 5 and 95 of year 2015
#    sales_95_2015 = [np.percentile(short_df[short_df["YEAR"] == 2015]["item_cnt_day"], q = 95) for i in range(short_df[short_df["YEAR"] == 2015].shape[0])]
#    sales_5_2015 = [np.percentile(short_df[short_df["YEAR"] == 2015]["item_cnt_day"], q = 5) for i in range(short_df[short_df["YEAR"] == 2015].shape[0])]
#    dates_2015 = short_df[short_df["YEAR"] == 2015]["date"]
#
#    # plot the data and add label
#    plt.plot(dates, sales, 'o-', label = "Monthly sales")
#    
#    plt.plot(dates, average_3_months, '.-', label = "Average sales of the last 3 months")
#    
#    plt.plot(dates, sales_95_global, '-', color = "black", label = "P95 of Monthly sales over all years")
#    plt.plot(dates, sales_5_global, '-', color = "magenta", label = "P5 of Monthly sales over all years")
#    
#    plt.plot(dates_2013, sales_95_2013, "--", color = "green")
#    plt.plot(dates_2013, sales_5_2013, ":", color = "red")
#    
#    plt.plot(dates_2014, sales_95_2014, "--", color = "green")
#    plt.plot(dates_2014, sales_5_2014, ":", color = "red")
#    
#    plt.plot(dates_2015, sales_95_2015, "--", color = "green", label = "P95 of Monthly sales by year")
#    plt.plot(dates_2015, sales_5_2015, ":", color = "red", label = "P5 of Monthly sales by year")
#
#    # get current axis and plot the areas
#    ax = plt.gca()
#    alpha = 0.2
#    
#    for start_date, end_date in zip(russian_holidays_start, russian_holidays_end):
#        
#        # add shaded areas for holidays 2013
#        ax.axvspan(start_date, end_date, alpha = alpha, color = 'red')   
#    
#    # add title and show legend
#    plt.title('Monthly sales of item category {}'.format(iterable))
#    plt.ylabel('Total Monthly sales of item category {}'.format(iterable))
#    plt.xlabel("Time grouped by month")
#    plt.legend()
#    
#
#    #######################################################################################
#    # Weekly sales
#    #######################################################################################
#    
#    plt.subplot(1, 2, 2)
#    
#      # calculate the Weekly sales of each shop
#    short_df = df[df["item_category_name"] == iterable][["date","item_cnt_day"]]
#    short_df["date"] = pd.to_datetime(short_df["date"], format = "%d.%m.%Y")
#    short_df["YEAR"] = short_df["date"].dt.year
#    short_df = short_df.set_index("date").groupby("YEAR").resample("W")["item_cnt_day"].sum()
#    short_df = short_df.reset_index()
#    
#    # adding moving average
#    short_df["MA3W"] = short_df["item_cnt_day"].rolling(window = 3).mean()
#    short_df["MA4W"] = short_df["item_cnt_day"].rolling(window = 4).mean()
#    short_df["MA5W"] = short_df["item_cnt_day"].rolling(window = 5).mean()
#    
#    # assing the data to plot
#    
#    # general sales
#    sales = short_df["item_cnt_day"]
#    dates = short_df["date"]
#    sales_95_global = [np.percentile(sales, q = 95) for i in range(len(sales))]
#    sales_5_global = [np.percentile(sales, q = 5) for i in range(len(sales))]
#    
#    average_3_weeks = short_df["MA3W"]
#    average_4_weeks = short_df["MA4W"]
#    average_5_weeks = short_df["MA5W"]
#    
#    # percentile 5 and 95 of year 2013
#    sales_95_2013 = [np.percentile(short_df[short_df["YEAR"] == 2013]["item_cnt_day"], q = 95) for i in range(short_df[short_df["YEAR"] == 2013].shape[0])]
#    sales_5_2013 = [np.percentile(short_df[short_df["YEAR"] == 2013]["item_cnt_day"], q = 5) for i in range(short_df[short_df["YEAR"] == 2013].shape[0])]
#    dates_2013 = short_df[short_df["YEAR"] == 2013]["date"]
#    
#    # percentile 5 and 95 of year 2014
#    sales_95_2014 = [np.percentile(short_df[short_df["YEAR"] == 2014]["item_cnt_day"], q = 95) for i in range(short_df[short_df["YEAR"] == 2014].shape[0])]
#    sales_5_2014 = [np.percentile(short_df[short_df["YEAR"] == 2014]["item_cnt_day"], q = 5) for i in range(short_df[short_df["YEAR"] == 2014].shape[0])]
#    dates_2014 = short_df[short_df["YEAR"] == 2014]["date"]
#    
#    # percentile 5 and 95 of year 2015
#    sales_95_2015 = [np.percentile(short_df[short_df["YEAR"] == 2015]["item_cnt_day"], q = 95) for i in range(short_df[short_df["YEAR"] == 2015].shape[0])]
#    sales_5_2015 = [np.percentile(short_df[short_df["YEAR"] == 2015]["item_cnt_day"], q = 5) for i in range(short_df[short_df["YEAR"] == 2015].shape[0])]
#    dates_2015 = short_df[short_df["YEAR"] == 2015]["date"]
#
#    # plot the data and add label
#    plt.plot(dates, sales, 'o-', label = "Weekly sales")
#    plt.plot(dates, average_3_weeks, '.-', label = "Average sales of the last 3 weeks")
#    
#    plt.plot(dates, sales_95_global, '-', color = "black", label = "P95 of Weekly sales over all years")
#    plt.plot(dates, sales_5_global, '-', color = "magenta", label = "P5 of Weekly sales over all years")
#    
#    plt.plot(dates_2013, sales_95_2013, "--", color = "green")
#    plt.plot(dates_2013, sales_5_2013, ":", color = "red")
#    
#    plt.plot(dates_2014, sales_95_2014, "--", color = "green")
#    plt.plot(dates_2014, sales_5_2014, ":", color = "red")
#    
#    plt.plot(dates_2015, sales_95_2015, "--", color = "green", label = "P95 of Weekly sales by year")
#    plt.plot(dates_2015, sales_5_2015, ":", color = "red", label = "P5 of Weekly sales by year")
#    
#    # get current axis and plot the areas
#    ax = plt.gca()
#    
#    for start_date, end_date in zip(russian_holidays_start, russian_holidays_end):
#        
#        # add shaded areas for holidays 2013
#        ax.axvspan(start_date, end_date, alpha = alpha, color = 'red')
#        
#    # add title and show legend
#    plt.title('Weekly sales of item category {}'.format(iterable))
#    plt.ylabel('Total Weekly sales of item category {}'.format(iterable))
#    plt.xlabel("Time grouped by week")
#    plt.legend()
#    # general sales
#    plt.show()
# we can observe a general trend of decrasing sales.
# let's add a second axis to see the variation of intradays sales

# select the columns of interest
df_var = df[["date", "item_cnt_day"]]

# convert to datetime
df_var["date"] = pd.to_datetime(df["date"], format = "%d.%m.%Y")

# set date as index
df_var.set_index("date", inplace = True)

# resample/groupby by date and convert to frame the total daily sales
df_var = df_var.resample("w")["item_cnt_day"].sum().to_frame()

# calculate the intra week variation between total sales
df_var["Variation"] = df_var["item_cnt_day"].diff()/df_var["item_cnt_day"].shift(1)

df_var.head()
# separate x and y
y_sales = df_var["item_cnt_day"]
y_variation = df_var["Variation"]

# instanciate the figure
fig = plt.figure(figsize = (15, 10))
ax = fig.add_subplot(111)

# plot the total sales
plot1 = ax.plot(y_sales, label = "Total weekly sales", color = "blue", alpha = 0.5)

# create a secondary axis and plot the variation data
ax_bis = ax.twinx()
plot2 = ax_bis.plot(y_variation, label = "Intra - week variation of sales", color = "red", alpha = 0.5)

# create a common legend for both plots
lns = plot1 + plot2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc = "upper left")

# add a custom title to the plot
ax.set_title("Total weekly sales and variation");
# start with the regular df
df_for_question_1 = create_df()
df_for_question_1.head()
df_for_question_1['date'] = pd.to_datetime(df_for_question_1['date'], format = "%d.%m.%Y")
df_for_question_1.info()
df_for_question_1.set_index('date',inplace = True)
df_for_question_1.head()
df_resampled_question_1 = df_for_question_1.resample('D')['item_cnt_day'].sum().to_frame()
df_resampled_question_1.head()
df_resampled_question_1['MA_sales'] = df_resampled_question_1['item_cnt_day'].rolling(window = 7).mean()
df_resampled_question_1.head(8)
df_resampled_question_1["Variation"] = df_resampled_question_1["item_cnt_day"].diff()/df_resampled_question_1["item_cnt_day"].shift(1)
df_resampled_question_1.head(9)
fig  = plt.figure(figsize = [20,10])
ax = fig.add_subplot(1,1,1)

plot1 = ax.plot(df_resampled_question_1['MA_sales'], label = "Total diary sales", color = "blue", alpha = 0.5)

ax_bis = ax.twinx()

plot2 = ax_bis.plot(df_resampled_question_1['Variation'], label = "Intra - diary variation of sales", color = "red", alpha = 0.5)

lns = plot1 + plot2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc = "upper left")

# add a custom title to the plot
ax.set_title("Total weekly sales and variation")
plt.show()

# calendar heatmaps are really useful to see the overall activity for a certain period of time per day and per month.
# let's build one using python.
# we will be using the calmap package for this, because it makes it extremenly easy to plot this data
# select the columns
df_calendar = df[["date", "item_cnt_day"]]

# set date as index and resample
df_calendar.set_index("date", inplace = True)
# notice that this time, we don't convert it to_frame()
# df_calendar is a pandas series
# THIS IS IMPORTANT since calmap expects a series
# with a datetime index and the values to plot
df_calendar = df_calendar.resample("D")["item_cnt_day"].sum()

# ----------------------------------------------------------------------------------------------------
# plot the data using calmap
calmap.calendarplot(df_calendar, # pass the series
                    fig_kws = {'figsize': (16,10)}, 
                    yearlabel_kws = {'color':'black', 'fontsize':14}, 
                    subplot_kws = {'title':'Total sales per year'}
                   );
# This plot are fundamental in timeseries analysis.
# Basically here we compare the a series again itself but with some lags.
# These are plots that graphically summarize the strength of a relationship with an observation in a time series with observations at prior time steps.

# More info: 
# https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# ----------------------------------------------------------------------------------------------------
# plot the data using the built in plots from the stats module

# The AutoCorrelation plot: compares a value v with the value v but n times in the past.
plot_acf(df.set_index("date").resample("D")["item_cnt_day"].sum(), ax = ax1, lags = 7)

# The Parcial AutoCorrelation plot: partial autocorrelation at lag k is the correlation that results after removing the effect of any correlations due to the terms at shorter lags.
plot_pacf(df.set_index("date").resample("D")["item_cnt_day"].sum(), ax = ax2, lags = 7);
# This code snippets show you have to calculate the Partial Autocorrelation
# Partial Autocorrelation can be very counter intuitive since in some of our steps we are fitting a linear model
# to predict the values of t - 2 using t - 1
# Wait, what? Why we use values from yesterday to predict values before yesterday?
# Basically because we assume that our timeseries is auto regressive. This means that the data at point t captures
# all the variance/information from all the previuos data points.
# This way, t - 1, must have captured all the variance from previous points, thus t - 2, and so t - 1 becomes
# a good predictor for values from t - 2.
# create a dataframe with total sales per day (all shops and all items)
df_total_sales = df.set_index("date").resample("D")["item_cnt_day"].sum().to_frame()

# rename the column item_cnt_day to total_sales
df_total_sales.columns = ["total_sales"]

# create a few features that we need in order to calculate the parcial autocorrelation
df_total_sales["T-1"] = df_total_sales["total_sales"].shift(1)
df_total_sales["T-2"] = df_total_sales["total_sales"].shift(2)

# we have a few nan for the first 2 rows so we must drop them
print(df_total_sales.shape)
df_total_sales.dropna(axis = "rows", inplace = True)
print(df_total_sales.shape)
# instanciate the Linear model
model = LinearRegression()

# separate X and y
X = df_total_sales[["T-1"]]
y = df_total_sales["total_sales"]

# fit and predict with the model
model.fit(X, y)
predictions = model.predict(X)

# save our predictions to the total_sales df
df_total_sales["total_sales_from_T-1"] = predictions
# instanciate the Linear model
model = LinearRegression()

# separate X and y
X = df_total_sales[["T-1"]]
y = df_total_sales["T-2"]

# fit and predict with the model
model.fit(X, y)
predictions = model.predict(X)

# save our predictions to the total_sales df
df_total_sales["T-2_from_T-1"] = predictions
# calculate the residual
# this means: total_sales - total_sales_from_T-1
# and: T-2 - "T-2_from_T-1"
df_total_sales["Residual_total_sales_T-1"] = df_total_sales["total_sales"] - df_total_sales["total_sales_from_T-1"]

# this step is very important based on the asumptions we have about many of the timeseries
# for more information I recommend this read
# https://towardsdatascience.com/understanding-partial-auto-correlation-fa39271146ac
df_total_sales["Residual_T-2_T-1"] = df_total_sales["T-2"] - df_total_sales["T-2_from_T-1"]
# calculathe the parcial autocorrelation using manual method
manual_pacf = df_total_sales.corr(method = "pearson")["Residual_total_sales_T-1"]["Residual_T-2_T-1"]
print("Manual parcial autocorrelation method {}".format(round(manual_pacf, 5)))

# calculate the parcial autocorrelation using statsmodel package
stats_pacf = pacf(df_total_sales['total_sales'], nlags = 2)[2]
print("Parcial autocorrelation method using stats package {}".format(round(stats_pacf, 5)))
df_total_sales.head()
# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# ----------------------------------------------------------------------------------------------------
# plot the data using the built in plots from the stats module
plot_acf(df.set_index("date").resample("W")["item_cnt_day"].sum(), ax = ax1, lags = 8)
plot_pacf(df.set_index("date").resample("W")["item_cnt_day"].sum(), ax = ax2, lags = 8);
# ----------------------------------------------------------------------------------------------------
# instanciate the figure
fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# ----------------------------------------------------------------------------------------------------
# plot the data using the built in plots from the stats module
plot_acf(df.set_index("date").resample("M")["item_cnt_day"].sum(), ax = ax1, lags = 13)
plot_pacf(df.set_index("date").resample("M")["item_cnt_day"].sum(), ax = ax2, lags = 13);
# Useful for:
# The theory behind timeseries, says that a series can be decomposed into 3 parts
# The trend
# The seasonal part
# And the residual
# This plots shows how to do this

# More info: 
# https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/

df_timeindex = df.set_index("date").resample("D")["item_cnt_day"].sum().to_frame()

# decompose the series using stats module
# results in this case is a special class 
# whose attributes we can acess
result = seasonal_decompose(df_timeindex["item_cnt_day"])

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
xtick_location = df_timeindex.index.tolist()

# set the xticks to be every 6'th entry
# every 6 months
ax.set_xticks(xtick_location);
result
df_timeindex = df.set_index("date").resample("W")["item_cnt_day"].sum().to_frame()

# decompose the series using stats module
# results in this case is a special class 
# whose attributes we can acess
result = seasonal_decompose(df_timeindex["item_cnt_day"])

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
xtick_location = df_timeindex.index.tolist()

# set x_ticks
ax.set_xticks(xtick_location);
df_timeindex = df.set_index("date").resample("M")["item_cnt_day"].sum().to_frame()

# decompose the series using stats module
# results in this case is a special class 
# whose attributes we can acess
result = seasonal_decompose(df_timeindex["item_cnt_day"])

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
xtick_location = df_timeindex.index.tolist()

# set x_ticks
ax.set_xticks(xtick_location);
# start with the regular df, #HACER LA DESCOMPOSICIÓN SOLAMENTE PARA UNA CIUDAD, HAY QUE HACER TODOS LOS PASOS
df_for_question_2 = create_df()
df_for_question_2 = df_for_question_2.loc[df_for_question_2['city']=='Москва']
df_for_question_2.date = pd.to_datetime(df_for_question_2.date)
df_for_question_2.info()
df_timeindex = df_for_question_2.set_index("date").resample("W")["item_cnt_day"].sum().to_frame()
df_timeindex = df_for_question_2.set_index("date").resample("W")["item_cnt_day"].sum().to_frame()

result = seasonal_decompose(df_timeindex["item_cnt_day"])

# ----------------------------------------------------------------------------------------------------
# instanciate the figure
# make the subplots share teh x axis
fig, axes = plt.subplots(ncols = 1, nrows = 4, sharex = True, figsize = (12,10))


# plot the original data
result.observed.plot(ax = axes[0], legend = False)
axes[0].set_ylabel('Observed')
axes[0].set_title("Decomposition of a series solamente en Moscú")

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
xtick_location = df_timeindex.index.tolist()

# set x_ticks
ax.set_xticks(xtick_location);
df.head()
# prepare the data

# extract each year using dt.year
df["YEAR"] = df["date"].dt.year

# create a smaller df for year 2013
short_df = df[df["YEAR"] == 2013][["item_cnt_day", "city"]]

# groupby by city and sum all the sales
short_df = short_df.groupby("city")["item_cnt_day"].sum().to_frame()

# sort the values in the smaller df inplace
short_df.sort_values("item_cnt_day", ascending = False, inplace = True)

# get the x and y values
my_values = short_df["item_cnt_day"]
my_pct = short_df["item_cnt_day"]/short_df["item_cnt_day"].sum()

# create custom labels for each city with their total sales and overall contribution
labels = ['{} - Sales :{}k \n {}% of total'.format(city, sales/1000, round(pct, 2)*100) for city, sales, pct in zip(short_df.index, my_values, my_pct)]

# create a color palette, mapped to the previous values
cmap = matplotlib.cm.Blues

# we want to normalize our values, otherwise a city will have the darkest collor and all the others will pale
mini = min(my_values)
maxi= np.percentile(my_values, q = 99)
norm = matplotlib.colors.Normalize(vmin = mini, vmax = maxi)
colors = [cmap(norm(value)) for value in my_values]

# instanciate the figure
plt.figure(figsize = (30, 10))
# we can pass colors but Moscow is way too big and most of the cities are pale blue
squarify.plot(sizes = my_values, label = labels,  alpha = 0.8, color  = colors)

# Remove our axes, set a title and display the plot
plt.title("Sales by city and their % over total sales in 2013", fontsize = 23, fontweight = "bold")
plt.axis('off')
plt.tight_layout()
# prepare the data

# extract each year using dt.year
df["YEAR"] = df["date"].dt.year

# create a smaller df for year 2013
short_df = df[df["YEAR"] == 2013][["item_cnt_day", "city"]]

# groupby by city and sum all the sales
short_df = short_df.groupby("city")["item_cnt_day"].sum().to_frame()

# sort the values in the smaller df inplace
short_df.sort_values("item_cnt_day", ascending = False, inplace = True)

# get the x and y values
my_values = short_df["item_cnt_day"]
my_pct = short_df["item_cnt_day"]/short_df["item_cnt_day"].sum()

# create custom labels for each city with their total sales and overall contribution
labels = ['{} - Sales :{}k \n {}% of total'.format(city, sales/1000, round(pct, 2)*100) for city, sales, pct in zip(short_df.index, my_values, my_pct)]

# create a color palette, mapped to the previous values
cmap = matplotlib.cm.Blues

# we want to normalize our values, otherwise a city will have the darkest collor and all the others will pale
mini = min(my_values)
maxi= np.percentile(my_values, q = 99)
norm = matplotlib.colors.Normalize(vmin = mini, vmax = maxi)
colors = [cmap(norm(value)) for value in my_values]

# instanciate the figure
plt.figure(figsize = (30, 10))
# we can pass colors but Moscow is way too big and most of the cities are pale blue
squarify.plot(sizes = my_values, label = labels,  alpha = 0.8)#, color  = 'RgYl')

# Remove our axes, set a title and display the plot
plt.title("Sales by city and their % over total sales in 2013", fontsize = 23, fontweight = "bold")
plt.axis('off')
plt.tight_layout()
# we will do the same plot as before but without custom colors
# Moscow is a big outlier so it pales the rest of the cities

short_df = df[df["YEAR"] == 2014][["item_cnt_day", "city"]]
short_df = short_df.groupby("city")["item_cnt_day"].sum().to_frame()
short_df.sort_values("item_cnt_day", ascending = False, inplace = True)

my_values = short_df["item_cnt_day"]
my_pct = short_df["item_cnt_day"]/short_df["item_cnt_day"].sum()
labels = ['{} - Sales :{}k \n {}% of total'.format(city, sales/1000, round(pct, 2)*100) for city, sales, pct in zip(short_df.index, my_values, my_pct)]

plt.figure(figsize = (30, 10))
squarify.plot(sizes = my_values, label = labels,  alpha = 0.8)
plt.title("Sales by city and their % over total sales in 2014",fontsize = 23, fontweight = "bold")

plt.axis('off')
plt.tight_layout()
# we will do the same plot as before but without custom colors
# Moscow is a big outlier so it pales the rest of the cities

short_df = df[df["YEAR"] == 2015][["item_cnt_day", "city"]]
short_df = short_df.groupby("city")["item_cnt_day"].sum().to_frame()
short_df.sort_values("item_cnt_day", ascending = False, inplace = True)

my_values = short_df["item_cnt_day"]
my_pct = short_df["item_cnt_day"]/short_df["item_cnt_day"].sum()
labels = ['{} - Sales :{}k \n {}% of total'.format(city, sales/1000, round(pct, 2)*100) for city, sales, pct in zip(short_df.index, my_values, my_pct)]

plt.figure(figsize = (30, 10))
squarify.plot(sizes = my_values, label = labels,  alpha = 0.8)
plt.title("Sales by city and their % over total sales in 2015",fontsize = 23, fontweight = "bold")

plt.axis('off')
plt.tight_layout()
df[["city", "city_id"]].drop_duplicates()
# treemaps are very useful to see the difference and the weights of categories
# but they don't give us that much of information about the distribution of each category
# let's use boxplot to see the distribution of Moscow city

# we can see huge outliers for Moscow city.
plt.figure(figsize = (10, 10))
sns.boxplot(x = "city",
            y = "item_cnt_day", 
            data = df[(df["YEAR"] == 2013) & (df["city_id"] == 13)]
           );
# start with the regular df
df_for_question_3 = create_df()
df_for_question_3.head()
df_agrupado_3 = df_for_question_3.groupby("item_category_name")["item_cnt_day"].sum().to_frame()
df_agrupado_3.sort_values('item_cnt_day',ascending = False, inplace = True)

df_agrupado_3.head()
my_pct = df_agrupado_3['item_cnt_day']/df_agrupado_3['item_cnt_day'].sum()
my_values = df_agrupado_3["item_cnt_day"]
df_agrupado_3 = df_agrupado_3[my_pct>=0.020]
my_values = df_agrupado_3["item_cnt_day"]
my_pct = my_pct[my_pct>=0.020]
df_agrupado_3.head()
sns.boxplot(my_pct)
labels = ['{} - Sales :{}k \n {}% of total'.format(item, sales/1000, round(pct, 2)*100) for item, sales, pct in zip(df_agrupado_3.index, my_values, my_pct)]
my_values
labels[:2]
cmap = matplotlib.cm.Blues

# we want to normalize our values, otherwise a city will have the darkest collor and all the others will pale
mini = min(my_values)
maxi= np.percentile(my_values, q = 99)
norm = matplotlib.colors.Normalize(vmin = mini, vmax = maxi)
colors = [cmap(norm(value)) for value in my_values]

# instanciate the figure
plt.figure(figsize = (30, 10))
# we can pass colors but Moscow is way too big and most of the cities are pale blue
squarify.plot(sizes = my_values, label = labels,  alpha = 0.8, color  = colors)

# Remove our axes, set a title and display the plot
plt.title("Sales by item_category_name and their % over total sales", fontsize = 23, fontweight = "bold")
plt.axis('off')
plt.tight_layout()
# start with the regular df
df_for_question_3 = create_df()
df_for_question_3 = create_df()
df_for_question_3 = df_for_question_3.groupby("item_category_name")["item_cnt_day"].sum().to_frame()
df_for_question_3.sort_values('item_cnt_day',ascending = False, inplace = True)
my_pct = df_for_question_3['item_cnt_day']/df_for_question_3['item_cnt_day'].sum()
my_values = df_for_question_3["item_cnt_day"]
my_values = df_for_question_3["item_cnt_day"]
my_pct = my_pct[my_pct>=0.010]
my_values.head()
my_pct.head(7)
df_agrupado_3.head()
labels = ['{} - Sales :{}k \n {}% of total'.format(item, sales/1000, round(pct, 2)*100) for item, sales, pct in zip(df_for_question_3.index, my_values, my_pct)]

labels
cmap = matplotlib.cm.Blues

# we want to normalize our values, otherwise a city will have the darkest collor and all the others will pale
mini = min(my_values)
maxi= np.percentile(my_values, q = 99)
norm = matplotlib.colors.Normalize(vmin = mini, vmax = maxi)
colors = [cmap(norm(value)) for value in my_values]

# instanciate the figure
plt.figure(figsize = (30, 10))
# we can pass colors but Moscow is way too big and most of the cities are pale blue
squarify.plot(sizes = my_values, label = labels,  alpha = 0.8, color  = colors)

# Remove our axes, set a title and display the plot
plt.title("Sales by item_category_name and their % over total sales", fontsize = 23, fontweight = "bold")
plt.axis('off')
plt.tight_layout()
# This plot will help us visualize the missing values for each datetime and item_id
# This is the most granular plots possible, since we will be seeing individual sales by day and item_id
# This plot can be very consufing, but the main point is to show all the "missing values" we have
# We have seen previously in our EDA, that when we groupby and resamples our sales, we might think
# that we don't have any missing values. But its not true, we only have the reported sales
# This means that, if we have a shop or item_id that only had 3 sales per year, when we resample
# our df by day, pandas will generate additional days with null sales.
# those null sales is what we want to plot here

plt.figure(figsize = (20, 10))
plot = sns.heatmap(df.pivot_table(index = ["date"], columns = ['item_id'], values = "item_cnt_day", aggfunc = sum).isnull(), cbar = True, cmap = "inferno")
plot.set_title("Null sales by item_id and day");
gc.collect()
# create a dataframe with True and False if there are missing values
gb_df_ = df.pivot_table(index = ["date"], columns = ['item_id'], values = "item_cnt_day", aggfunc = sum).isnull()

# generate a custom list with the colums name sorte by values
order_of_columns = list(gb_df_.sum().sort_values().index)

# change the order of the df from the lowest amount of missing values to the highest
gb_df_.columns = order_of_columns

# plo the data using seaborn heatmap
plt.figure(figsize = (20, 10))
plot = sns.heatmap(gb_df_, cbar = True, cmap = "inferno")
plot.set_title("Null sales by item_id and day");
gc.collect()
del gb_df_
gb_df_ = df.pivot_table(index = ["date"], columns = ['item_id'], values = "item_cnt_day", aggfunc = sum).isnull()
order_of_columns = list(gb_df_.sum().sort_values().index)
gb_df_ = gb_df_[order_of_columns]
plt.figure(figsize = (20, 10))
plot = sns.heatmap(gb_df_, cbar = True, cmap = "inferno")
plot.set_title("Null sales by item_id and day");
gc.collect()
del gb_df_
# This is a similar plot to the previous one, but here instead of item_id we will be plotting shop_id and their total sales
# We expect to have fewer missing values for each shop.
gb_df_ = df.pivot_table(index = ["date"], columns = ['shop_id'], values = "item_cnt_day", aggfunc = sum).isnull()
order_of_columns = list(gb_df_.sum().sort_values().index)
gb_df_ = gb_df_[order_of_columns]
plt.figure(figsize = (20, 10))
plot = sns.heatmap(gb_df_, cbar = True, cmap = "inferno")
plot.set_title("Null sales by shop and day");
df.head()
gc.collect()
# this will allow us to see a all the columns of the df
pd.options.display.max_columns = 999
# create a smaller df
short_df = df[["date", "item_cnt_day", "shop_name"]]
# set the date to be the index (to resample later)
short_df.set_index("date", inplace = True)
# groupby by shop_name
gb = short_df.groupby("shop_name")
# resample the df by month sales (resample = groupby by months in timeseries)
gbr = gb.resample("M")["item_cnt_day"].sum()
# unstack the gbr to have columns name
gbr = gbr.unstack(level = -1).T
# sort the values, from no nulls to more null values
order_of_columns = list(gbr.isnull().sum().sort_values().index)
# change the order of the df
gbr = gbr[order_of_columns]
gbr.head()
# change all nulls to 1 and sales to 0
gbr.head().isnull()*1
# let's plot the null values for each shop
plt.figure(figsize=(20, 10))
# this lines gbr.unstack(level = -1).T.isnull()*1
# converts any null to 1 and the rest will be 0
sns.heatmap(gbr.isnull()*1, cmap = "inferno", cbar = True).set_title("Null values by shop and Month");
# create a smaller df
short_df = df[["date", "item_cnt_day", "item_category_name"]]

# set the date to be the index (to resample later)
short_df.set_index("date", inplace = True)

# groupby by shop_name
gb = short_df.groupby("item_category_name")

# resample the df by month sales (resample = groupby by months in timeseries)
gbr = gb.resample("M")["item_cnt_day"].sum()

# unstack the gbr to have columns name
gbr = gbr.unstack(level = -1).T

# sort the values, from no nulls to more null values
order_of_columns = list(gbr.isnull().sum().sort_values().index)

# change the order of the df
gbr = gbr[order_of_columns]
gbr.head()
# change all nulls to 1 and sales to 0
gbr.head().isnull()*1
# let's plot the null values for each shop
plt.figure(figsize=(20, 10))

# this lines gbr.unstack(level = -1).T.isnull()*1
# converts any null to 1 and the rest will be 0
sns.heatmap(gbr.isnull()*1, cmap = "inferno", cbar = True).set_title("Null values by item category and Month");
# let's look at outliers for item sales
# We will use boxplots because they are very useful to see the distribution of values
plt.figure(figsize = (10,4))
sns.boxplot(x = df["item_cnt_day"]);
# let's look at outliers for item price
plt.figure(figsize = (10,4))
plt.xlim(df["item_price"].min(), df["item_price"].max()*1.1)
sns.boxplot(x = df["item_price"]);
# joint plot is another very convenient way to plot the relationship between 2 variables
# but because we have huge outliers, we don't see them 
# https://seaborn.pydata.org/generated/seaborn.jointplot.html
plt.figure(figsize = (10,4))
sns.jointplot(x = "item_price", y = "item_cnt_day", data = df);
# let's filter the outliers and make the same joint plot
df = df[(df["item_price"] < np.percentile(df["item_price"], q = 99)) & (df["item_cnt_day"] >= 0) & (df["item_cnt_day"] < np.percentile(df["item_cnt_day"], q = 99))]
# we have removed the outliers and now 
plt.figure(figsize = (10, 10))
sns.jointplot(x = "item_price", y = "item_cnt_day", data = df);
