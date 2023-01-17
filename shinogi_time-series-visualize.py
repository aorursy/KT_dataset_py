from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from ipywidgets import widgets, interactive, interact

import ipywidgets as widgets

from IPython.display import display



import os
pd.set_option("display.max_column", 2000)
# for dirname, _, filenames in os.walk("./"):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
base_paht = "/kaggle/input/"
train_sales = pd.read_csv(base_paht+"m5-forecasting-accuracy/sales_train_validation.csv")
calendar_df = pd.read_csv(base_paht+"m5-forecasting-accuracy/calendar.csv")

calendar_slim_df = calendar_df[["d", "date", "weekday", "wm_yr_wk"]]

calendar_slim_df.rename(columns={"d": "day_count"}, inplace=True)

del calendar_df

# calendar_slim_df
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
eda_base_table = eda_base_table[["date", "weekday", "wm_yr_wk", "quantity" ,"item_id", "dept_id", "cat_id", "store_id", "state_id"]]
eda_base_table.head()
quantity_state_df = eda_base_table.groupby(["date", "state_id"]).sum()

quantity_state_df = quantity_state_df["quantity"].reset_index()

quantity_state_df = pd.pivot_table(quantity_state_df, index="date", columns="state_id", values="quantity")

plt.figure()

quantity_state_df.plot(figsize=(36, 6))

del quantity_state_df
quantity_store_df = eda_base_table.groupby(["date", "store_id"]).sum()

quantity_store_df = quantity_store_df["quantity"].reset_index()

quantity_store_df = pd.pivot_table(quantity_store_df, index="date", columns="store_id", values="quantity")

plt.figure()

quantity_store_df[["CA_1", "CA_2", "CA_3", "CA_4"]].plot(figsize=(36, 6))

quantity_store_df[["TX_1", "TX_2", "TX_3"]].plot(figsize=(36, 6))

quantity_store_df[["WI_1", "WI_2", "WI_3"]].plot(figsize=(36, 6))

del quantity_store_df
quantity_cat_df = eda_base_table.groupby(["date", "cat_id"]).sum()

quantity_cat_df = quantity_cat_df["quantity"].reset_index()

quantity_cat_df = pd.pivot_table(quantity_cat_df, index="date", columns="cat_id", values="quantity")

plt.figure()

quantity_cat_df.plot(figsize=(36, 6))

del quantity_cat_df
quantity_dept_df = eda_base_table.groupby(["date", "dept_id"]).sum()

quantity_dept_df = quantity_dept_df["quantity"].reset_index()

quantity_dept_df = pd.pivot_table(quantity_dept_df, index="date", columns="dept_id", values="quantity")

plt.figure()

# quantity_dept_df.plot(figsize=(36, 6))

quantity_dept_df[["FOODS_1", "FOODS_2", "FOODS_3"]].plot(figsize=(36, 6))

quantity_dept_df[["HOBBIES_1", "HOBBIES_2"]].plot(figsize=(36, 6))

quantity_dept_df[["HOUSEHOLD_1", "HOUSEHOLD_2"]].plot(figsize=(36, 6))

del quantity_dept_df
quantity_item_df = eda_base_table.groupby(["date", "dept_id", "item_id"]).sum()

quantity_item_df = quantity_item_df["quantity"].reset_index()

quantity_item_df = pd.pivot_table(quantity_item_df, index="date", columns=["dept_id", "item_id"], values="quantity")

plt.figure()

# quantity_item_df.plot(figsize=(36, 6))

quantity_item_df[["FOODS_1", "FOODS_2", "FOODS_3"]].plot(figsize=(36, 6))

quantity_item_df[["HOBBIES_1", "HOBBIES_2"]].plot(figsize=(36, 6))

quantity_item_df[["HOUSEHOLD_1", "HOUSEHOLD_2"]].plot(figsize=(36, 6))

del quantity_item_df
quantity_state_weekday_df = eda_base_table.groupby(["weekday", "state_id"]).sum()

quantity_state_weekday_df = quantity_state_weekday_df["quantity"].reset_index()

quantity_state_weekday_df = pd.pivot_table(quantity_state_weekday_df, index="weekday", columns="state_id", values="quantity")

quantity_state_weekday_df["temp_no"] = [6,2,7,1,5,3,4]

quantity_state_weekday_df.sort_values("temp_no", inplace=True)

del quantity_state_weekday_df["temp_no"]

plt.figure()

quantity_state_weekday_df.plot(figsize=(36, 6))

del quantity_state_weekday_df
quantity_state_store_df = eda_base_table.groupby(["weekday", "store_id"]).sum()

quantity_state_store_df = quantity_state_store_df["quantity"].reset_index()

quantity_state_store_df = pd.pivot_table(quantity_state_store_df, index="weekday", columns="store_id", values="quantity")

quantity_state_store_df["temp_no"] = [6,2,7,1,5,3,4]

quantity_state_store_df.sort_values('temp_no', inplace=True)

quantity_state_store_df.sort_values("temp_no", inplace=True)

del quantity_state_store_df["temp_no"]

plt.figure()

quantity_state_store_df[["CA_1", "CA_2", "CA_3", "CA_4"]].plot(figsize=(36, 6))

quantity_state_store_df[["TX_1", "TX_2", "TX_3"]].plot(figsize=(36, 6))

quantity_state_store_df[["WI_1", "WI_2", "WI_3"]].plot(figsize=(36, 6))

del quantity_state_store_df
quantity_cat_weekday_df = eda_base_table.groupby(["weekday", "cat_id"]).sum()

quantity_cat_weekday_df = quantity_cat_weekday_df["quantity"].reset_index()

quantity_cat_weekday_df = pd.pivot_table(quantity_cat_weekday_df, index="weekday", columns="cat_id", values="quantity")

quantity_cat_weekday_df["temp_no"] = [6,2,7,1,5,3,4]

quantity_cat_weekday_df.sort_values('temp_no', inplace=True)

quantity_cat_weekday_df.sort_values("temp_no", inplace=True)

del quantity_cat_weekday_df["temp_no"]

plt.figure()

quantity_cat_weekday_df.plot(figsize=(36, 6))

del quantity_cat_weekday_df
quantity_dept_id_weekday_df = eda_base_table.groupby(["weekday", "dept_id"]).sum()

quantity_dept_id_weekday_df = quantity_dept_id_weekday_df["quantity"].reset_index()

quantity_dept_id_weekday_df = pd.pivot_table(quantity_dept_id_weekday_df, index="weekday", columns="dept_id", values="quantity")

quantity_dept_id_weekday_df["temp_no"] = [6,2,7,1,5,3,4]

quantity_dept_id_weekday_df.sort_values('temp_no', inplace=True)

quantity_dept_id_weekday_df.sort_values("temp_no", inplace=True)

del quantity_dept_id_weekday_df["temp_no"]

plt.figure()

quantity_dept_id_weekday_df[["FOODS_1", "FOODS_2", "FOODS_3"]].plot(figsize=(36, 6))

quantity_dept_id_weekday_df[["HOBBIES_1", "HOBBIES_2"]].plot(figsize=(36, 6))

quantity_dept_id_weekday_df[["HOUSEHOLD_1", "HOUSEHOLD_2"]].plot(figsize=(36, 6))

del quantity_dept_id_weekday_df
quantity_item_id_weekday_df = eda_base_table.groupby(["weekday", "dept_id", "item_id"]).sum()

quantity_item_id_weekday_df = quantity_item_id_weekday_df["quantity"].reset_index()

quantity_item_id_weekday_df = pd.pivot_table(quantity_item_id_weekday_df, index="weekday", columns=["dept_id", "item_id"], values="quantity")

quantity_item_id_weekday_df["temp_no"] = [6,2,7,1,5,3,4]

quantity_item_id_weekday_df.sort_values('temp_no', inplace=True)

quantity_item_id_weekday_df.sort_values("temp_no", inplace=True)

del quantity_item_id_weekday_df["temp_no"]

plt.figure()

# quantity_item_id_weekday_df.plot(figsize=(36, 6))

quantity_item_id_weekday_df[["FOODS_1", "FOODS_2", "FOODS_3"]].plot(figsize=(36, 6))

quantity_item_id_weekday_df[["HOBBIES_1", "HOBBIES_2"]].plot(figsize=(36, 6))

quantity_item_id_weekday_df[["HOUSEHOLD_1", "HOUSEHOLD_2"]].plot(figsize=(36, 6))

del quantity_item_id_weekday_df