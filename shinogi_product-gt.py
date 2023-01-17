import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from ipywidgets import widgets, interactive, interact

import ipywidgets as widgets

from IPython.display import display



import os



import pickle



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
base_paht = "/kaggle/input/"

train_sales = pd.read_csv(base_paht+"m5-forecasting-accuracy/sales_train_validation.csv")

## samepling

id_list = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] + list(map(lambda x: "d_"+str(x),list(range(1183, 1914))))

train_sales = train_sales[id_list]

#

calendar_df = pd.read_csv(base_paht+"m5-forecasting-accuracy/calendar.csv")

calendar_slim_df = calendar_df[["d", "date", "weekday", "wm_yr_wk"]]

calendar_slim_df.rename(columns={"d": "day_count"}, inplace=True)

del calendar_df

#

sell_prices = pd.read_csv(base_paht+"m5-forecasting-accuracy/sell_prices.csv")
id_list = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

days_list = list(filter(lambda x: x not in id_list, list(train_sales.columns)))

# unpivot

temp_unpivot_list      = ["id"] + days_list

temp_unpivot_df        = train_sales[temp_unpivot_list]

train_sales_df         = pd.melt(temp_unpivot_df, id_vars=["id"], value_vars=days_list)

train_sales_df.columns = ["id", "day_count", "quantity"]

del days_list, temp_unpivot_list, temp_unpivot_df

eda_base_table = train_sales_df.merge(train_sales[id_list], how="left", on="id")

del id_list, train_sales_df, train_sales

## join calendar

eda_base_table = eda_base_table.merge(calendar_slim_df, how="left", on="day_count")

del calendar_slim_df

## join calendar

eda_base_table = eda_base_table.merge(sell_prices, how="left", on=["store_id", "item_id","wm_yr_wk"])

del sell_prices

#

eda_base_table = eda_base_table[["date", "weekday", "wm_yr_wk", "quantity" ,"item_id", "dept_id", "cat_id", "store_id", "state_id"]]

# eda_base_table.to_pickle("/kaggle/output/eda_base_table.pkl")
eda_base_table.to_pickle("/kaggle/working/eda_base_table.pkl")
eda_base_table = pd.read_pickle("eda_base_table.pkl")

eda_base_table["yyyymm"] = eda_base_table.date.apply(lambda x:x[0:7])

product_columes = list(eda_base_table.item_id.unique())
# product_columes = ["HOUSEHOLD_2_282"]

count = 0

for col in product_columes:

    eda_base_table_one = eda_base_table[eda_base_table.item_id==col]

    eda_base_table_one = eda_base_table_one[["yyyymm", "weekday", "state_id", "store_id","item_id", "quantity"]]

    agg_base_one_df = eda_base_table_one.groupby(["yyyymm", "weekday", "state_id", "store_id","item_id"]).sum().reset_index()

    # ---------------

    one_df = agg_base_one_df.groupby(["yyyymm", "weekday", "state_id"]).sum().reset_index()

    del agg_base_one_df

    print(col)

    print(count)

    count = count + 1

    fig, axes = plt.subplots(1, 3, figsize=(36, 6))



    subcount = 0

    for area in ["CA", "TX", "WI"]:

        axes[subcount].bar(one_df[one_df.state_id==area]["yyyymm"], one_df[one_df.state_id==area]["quantity"], color="blue", alpha=1.0)

        axes[subcount].set_ylim(0, max(one_df.quantity))

        axes[subcount].legend(["quantity"])

        fig.autofmt_xdate(rotation=45)

        axes[subcount].set_title(col+"-"+area, fontsize=18)

        subcount = subcount + 1

    plt.show()

    # del one_df

    print("-------------------------")