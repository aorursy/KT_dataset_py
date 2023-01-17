import os

import numpy as np

import pandas as pd

import seaborn as sns

from bokeh.io import show, output_notebook

from bokeh.models import ColumnDataSource, FactorRange, HoverTool

from bokeh.plotting import figure

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from tqdm import tqdm

from IPython.display import display

from random import randint



output_notebook()



%matplotlib inline

%config InlineBackend.figure_format = 'retina'
input_files_path = '../input/'



files = ["train", "test", "transactions", "stores", "oil", "items", "holidays_events"]

datasets_path = [input_files_path + f + ".csv" for f in files]
print('# File sizes')

for f in datasets_path:

    print(f.ljust(30) + str(round(os.path.getsize(input_files_path + f) / 1000000, 2)) + 'MB')
chunksize = 25_000_000 # ~20% of the training set

data_df = {}

for filename, filepath in tqdm(zip(files, datasets_path), total=len(files)):

    if chunksize:

        data_df[filename] = pd.read_csv(filepath, chunksize=chunksize, low_memory=False).get_chunk()

    else:

        data_df[filename] = pd.read_csv(filepath, low_memory=False)
for chunk, df in data_df.items():

    print("Dataset {} of size {} with fields:\n {}".format(chunk, len(df), df.dtypes))

    display(df.tail())

    print(df.describe(), end='\n\n\n')
train_df = data_df["train"]

train_df["date"] = pd.to_datetime(train_df["date"])

test_df = data_df["test"]

test_df["date"] = pd.to_datetime(test_df["date"])

oil_df = data_df["oil"]

oil_df["date"] = pd.to_datetime(oil_df["date"])

items_df = data_df["items"]

stores_df = data_df["stores"]

transactions_df = data_df["transactions"]

transactions_df["date"] = pd.to_datetime(transactions_df["date"])

holidays_events_df = data_df["holidays_events"]

holidays_events_df["date"] = pd.to_datetime(holidays_events_df["date"])



print("Train set date range: {} to {}".format(train_df["date"].min(), train_df["date"].max()))

print("Test set date range: {} to {}".format(test_df["date"].min(), test_df["date"].max()))

print("Transactions date range: {} to {}".format(transactions_df["date"].min(), transactions_df["date"].max()))

print("Promotions count on train set: {}".format(len(train_df["onpromotion"]) - 

                                                 train_df["onpromotion"].isnull().sum()))

print("Promotions count on test set: {}".format(len(test_df["onpromotion"]) - 

                                                len(test_df[test_df["onpromotion"] == True])))

print("Number of different stores: {}".format(len(stores_df)))

print("Number of different stores clusters: {}".format(stores_df["cluster"].max()))

print("Oil price index range: {} - {}".format(oil_df["dcoilwtico"].min(), oil_df["dcoilwtico"].max()))

print("Oil unknown price count: {} ".format(oil_df["dcoilwtico"].isnull().sum()))

print("Items {} uniques families: {}\n".format(len(items_df["family"].unique()), items_df["family"].unique()))

print("Different types of holiday events: {}\n".format(holidays_events_df["type"].unique()))

print("Different holiday locales: {}\n".format(holidays_events_df["locale"].unique()))

print("Region list where the holidays applies: {}\n".format(holidays_events_df["locale_name"].unique()))

print("All different kind of holidays: {}".format(holidays_events_df["description"].unique()))

print("Example of transfered holidays: ")

display(holidays_events_df[holidays_events_df["transferred"] == True].head())
train_df['onpromotion'].fillna(False, inplace=True)

test_df['onpromotion'].fillna(False, inplace=True)



final_df = train_df.append(test_df)

assert len(final_df) == len(train_df) + len(test_df)

final_df['onpromotion'] = final_df['onpromotion'].astype(bool)

print("Final df size: {}".format(len(final_df)))

#assert len(final_df) == len(final_df["item_nbr"].unique())

final_df.head()
# id	date	store_nbr	item_nbr	unit_sales	onpromotion

sns.countplot(x="onpromotion", data=final_df);
unit_df = train_df.groupby("date").sum().reset_index()



source = ColumnDataSource(unit_df)

hover = HoverTool(

    tooltips=[

        ("date", "@date{%F}"),

        ("unit_sales", "@unit_sales{0.00 a}"),

    ], 

# Kaggle docker image is not up to date for bokeh

#     formatters={

#         'date': 'datetime'

#     },

)





p = figure(x_axis_type="datetime", tools=[hover, 'pan', 'box_zoom', 'wheel_zoom', 'reset'], 

           title="Unit sales by date", plot_width=900, plot_height=400)

p.xgrid.grid_line_color=None

p.ygrid.grid_line_alpha=0.5

p.xaxis.axis_label = 'Time'

p.yaxis.axis_label = 'Value'



p.line(x="date", y="unit_sales", line_color="gray", source=source)



show(p)
on_prom_df = train_df[train_df["onpromotion"] == True].groupby("date").sum()

on_prom_df.sort_values("onpromotion", inplace=True)

ax = sns.lmplot(x="onpromotion", y="unit_sales", data=on_prom_df)

ax.set(xlabel='onpromotion count by date', ylabel='unit_sales count');
train_df['weekday'] = pd.DatetimeIndex(train_df['date']).weekday

train_df['month'] = pd.DatetimeIndex(train_df['date']).month

r = train_df.pivot_table(values='unit_sales', index='weekday', columns='month', aggfunc=np.mean)

r.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

r.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

f, ax = plt.subplots(figsize=(15, 6))

sns.heatmap(r, linewidths=.5, ax=ax, cmap='rainbow')
# store_nbr	city	state	type	cluster

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))



sns.countplot(x="city", data=stores_df, ax=ax1)

sns.countplot(x="state", data=stores_df, ax=ax2)

sns.countplot(x="cluster", data=stores_df, ax=ax3)



plt.setp(ax1.xaxis.get_majorticklabels(), rotation=70)

plt.setp(ax2.xaxis.get_majorticklabels(), rotation=70)

plt.tight_layout()
perish_items = final_df.merge(items_df, on='item_nbr')



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.set_title("Perishable items relative to all different items")

ax2.set_title("Perishable items relative to all items in train/test set")

sns.countplot(x="perishable", data=items_df, ax=ax1)

sns.countplot(x="perishable", data=perish_items, ax=ax2);
f, ax = plt.subplots(figsize=(16, 8))

sns.countplot(x="family", data=items_df, ax=ax)

plt.setp(ax.xaxis.get_majorticklabels(), rotation=80);
trans_per_date = transactions_df.groupby(["date"], as_index=False).sum()



source = ColumnDataSource(trans_per_date)

hover = HoverTool(

    tooltips=[

        ("date", "@date{%F}"),

        ("transactions", "@transactions{0.00 a}"),

    ], 

# Kaggle docker image is not up to date for bokeh

#     formatters={

#         'date': 'datetime'

#     },

)





p = figure(x_axis_type="datetime", tools=[hover, 'pan', 'box_zoom', 'wheel_zoom', 'reset'], 

           title="Transactions", plot_width=900, plot_height=400)

p.xgrid.grid_line_color=None

p.ygrid.grid_line_alpha=0.5

p.xaxis.axis_label = 'Time'

p.yaxis.axis_label = 'Transactions'



p.line(x="date", y="transactions", line_color="gray", source=source)



show(p)
source = ColumnDataSource(oil_df)

hover = HoverTool(

    tooltips=[

        ("date", "@date{%F}"),

        ("dcoilwtico", "@dcoilwtico{0.00 a}"),

    ], 

# Kaggle docker image is not up to date for bokeh

#     formatters={

#         'date': 'datetime'

#     },

)





p = figure(x_axis_type="datetime", tools=[hover, 'pan', 'box_zoom', 'wheel_zoom', 'reset'], 

           title="Oil price index", plot_width=900, plot_height=400)

p.xgrid.grid_line_color=None

p.ygrid.grid_line_alpha=0.5

p.xaxis.axis_label = 'Time'

p.yaxis.axis_label = 'Index'



p.line(x="date", y="dcoilwtico", line_color="gray", source=source)



show(p)
transac = transactions_df.groupby("date").sum().reset_index()

transac = transac.merge(oil_df, on='date')

sns.jointplot(x='dcoilwtico', y='transactions', data=transac);
holidays_events_df.head()
f, ax = plt.subplots(1, 3, figsize=(16, 4))

ax = ax.ravel()



sns.countplot(x="type", data=holidays_events_df, ax=ax[0])

sns.countplot(x="locale", data=holidays_events_df, ax=ax[1])

sns.countplot(x="transferred", data=holidays_events_df, ax=ax[2])

plt.tight_layout()



f, axf = plt.subplots(figsize=(14, 8))

sns.countplot(x="locale_name", data=holidays_events_df, ax=axf)

plt.setp(axf.xaxis.get_majorticklabels(), rotation=80)

plt.tight_layout()