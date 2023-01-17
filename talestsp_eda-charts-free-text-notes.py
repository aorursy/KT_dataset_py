import pandas as pd



import my_dao

import process

import pretties

import time_utils

import stats

import plotter



import warnings

from bokeh.plotting import show, output_notebook
pretties.max_data_frame_columns()

pretties.decimal_notation()

output_notebook()

warnings.filterwarnings('ignore')
train = my_dao.load_dataset("train")

train = train.groupby("store_dept").apply(process.train_sales_semantic_enrichment)



feat = my_dao.load_features()

feat = process.features_semantic_enrichment(feat)



stores = my_dao.load_stores()
train = train.merge(feat, how="left", left_on=["Store", "Date"], right_on=["Store", "Date"], suffixes=["", "_y"])

del train["IsHoliday_y"]

del train["timestamp_y"]

train = train.merge(stores, how="left", left_on=["Store"], right_on=["Store"])
cols = ['Date', 'Store', 'Dept', 'Weekly_Sales', 'pre_holiday', 'IsHoliday', 'pos_holiday', 'Fuel_Price', 

        'CPI', 'Unemployment', 'celsius', 'datetime', 'Type', 'sales_diff', 'sales_diff_p',

        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 

        'Size', 'Temperature', 'timestamp', 'store_dept', "day_n", "week_n", "month_n", "year", "wm_date", "up_diff", "celsius_diff"]



train = train[cols].sort_values("timestamp")

train.sample(6)
train["Date"].head(1).append(train["Date"].tail(1))
grouped_sales = train.groupby("Date")["Weekly_Sales"].median()

p = plotter.plot_time_series_count(grouped_sales.index, grouped_sales, color="navy", title="Weekly_Sales median vs Datetime", legend="All depts sales median", relative_y_axis=True, height=300)

p = plotter.time_series_count_painted_holidays(train, p=p, color="cyan", alpha=0.9)

show(p)
p = plotter.time_series_count_painted(train, title="Weekly_Sales median vs Datetime - Diamonds repesents Holidays", height=300, width=900)

p = plotter.time_series_count_painted_holidays(train, p=p, color="cyan", alpha=0.9)

show(p)
from statsmodels.tsa.filters.hp_filter import hpfilter



gdp_cycle, gdp_trend = hpfilter(grouped_sales, lamb=10)

p = plotter.plot_time_series_count(grouped_sales.index, grouped_sales, color="navy", title="Weekly_Sales median vs Datetime", legend="All depts sales median", relative_y_axis=True, height=300)



grouped_sales = train.groupby("Date")["Weekly_Sales"].median()

p = plotter.plot_time_series_count(grouped_sales.index, gdp_trend, color="magenta", title="Weekly_Sales vs Datetime", legend="Hodrick-Prescott filter", relative_y_axis=True, height=300, p=p)

show(p)
p = plotter.plot_error_values(train, "week_n", "sales_diff_p", drop_quantile=0.15, 

                           title="Weekly_Sales errors grouped by week_n")

show(p)
p = plotter.plot_error_values(train, "wm_date", "Weekly_Sales", drop_quantile=0.25, 

                           title="Weekly_Sales errors grouped by wm_date", width=1200)

show(p)
p = plotter.plot_error_values(train, "wm_date", "sales_diff_p", drop_quantile=0.25, 

                           title="sales_diff_p errors grouped by wm_date", width=1400)

show(p)
train.groupby("Store")["Weekly_Sales"].mean().sort_values().plot.bar(title="Sales amout per store", figsize=(10, 3))
train.drop_duplicates(["Store", "Date"]).groupby("Store")["IsHoliday"].value_counts().plot.bar(title="Holidays count by store",figsize=(18,3))
stats.freq(train.drop_duplicates(["Store", "Date"])["IsHoliday"])
train.groupby("IsHoliday")["Weekly_Sales"].median().plot.bar(title="Weekly_Sales grouped by Holidays")
p = plotter.plot_error_values(train, "IsHoliday", "Weekly_Sales", drop_quantile=0.25, 

                           title="Weekly_Sales errors grouped by IsHoliday")

show(p)
train.groupby("pre_holiday")["Weekly_Sales"].median().plot.bar(title="Weekly_Sales BEFORE Holidays", figsize=(5,2))
p = plotter.plot_error_values(train, "pre_holiday", "Weekly_Sales", drop_quantile=0.25, 

                           title="Weekly_Sales errors grouped by pre_holiday", width=350, height=200)

show(p)
train.groupby("pos_holiday")["Weekly_Sales"].median().plot.bar(title="Weekly_Sales AFTER Holidays", figsize=(5,2))
p = plotter.plot_error_values(train, "pos_holiday", "Weekly_Sales", drop_quantile=0.25, 

                           title="Weekly_Sales errors grouped by pos_holiday", width=350, height=200)

show(p)
train.groupby(["Store", "wm_date"]).apply(lambda g : g["Fuel_Price"].corr(g["Weekly_Sales"])).hist(bins=20)
train.groupby(["Store", "wm_date"]).apply(lambda g : g["CPI"].corr(g["Weekly_Sales"])).hist(bins=20)
train.groupby(["Store", "wm_date"]).apply(lambda g : g["Unemployment"].corr(g["Weekly_Sales"])).hist(bins=20)
train.plot.scatter("celsius", "Weekly_Sales")
grouped_sales = train.groupby("Date")["celsius"].median()

p = plotter.plot_time_series_count(grouped_sales.index, grouped_sales, color="magenta", title="Temperature vs Datetime", legend="celsius", relative_y_axis=True, height=200)

p.legend.location = 'bottom_center'

show(p)
grouped_sales = train.groupby("Date")["Weekly_Sales"].median()

p = plotter.plot_time_series_count(grouped_sales.index, grouped_sales, color="navy", title="Weekly_Sales vs Datetime", legend="overall median", relative_y_axis=True, height=200)

show(p)
train["celsius"].corr(train["Weekly_Sales"])
train["celsius_diff"].corr(train["Weekly_Sales"])
size_sales = train.groupby("Size")["Weekly_Sales"].median().reset_index()

print(size_sales["Size"].corr(size_sales["Weekly_Sales"]))

size_sales.plot.scatter("Size", "Weekly_Sales", title="Weekly_Sales median vs Size")
mds = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
for md in mds:

    print(md+":", round(train[md].corr(train["Weekly_Sales"]), 4))
from statsmodels.tsa.filters.hp_filter import hpfilter
store_dept = train["store_dept"].sample().iloc[0]

store_dept
store_dept_sales = train[train["store_dept"] == store_dept].set_index("Date")

years = store_dept_sales["year"].drop_duplicates().to_list()[0:2]

years
dy1 = store_dept_sales[store_dept_sales["year"] == years[0]]

dy2 = store_dept_sales[store_dept_sales["year"] == years[1]].reset_index()

dy2["Date"] = dy2["Date"].str.slice(4,10).apply(lambda dt : str(years[0]) + str(dt))

dy2 = dy2.set_index("Date")
pretties.display_md("#### Store-Dept: {}".format(store_dept))
p = plotter.plot_time_series_count(dy1.index, dy1["Weekly_Sales"], color="navy", title="Weekly_Sales vs Datetime for store_dept {}".format(store_dept), 

                                relative_y_axis=True, height=300, legend=str(years[0]), p=None)

p = plotter.plot_time_series_count(dy2.index, dy2["Weekly_Sales"], color="magenta", title="Weekly_Sales vs Datetime for store_dept {}".format(store_dept), 

                                relative_y_axis=True, height=300, legend=str(years[1]), p=p)



show(p)
cycle1, trend1 = hpfilter(dy1["Weekly_Sales"], lamb=0.5)

cycle2, trend2 = hpfilter(dy2["Weekly_Sales"], lamb=0.5)



p = plotter.plot_time_series_count(dy1.index, trend1, color="cyan", title="Weekly_Sales vs Datetime", 

                                relative_y_axis=True, height=300, line_width=3, legend="hp " + str(years[0]), p=p)

p = plotter.plot_time_series_count(dy2.index, trend2, color="#FFC0C8", title="Weekly_Sales vs Datetime", 

                                relative_y_axis=True, height=300, line_width=3, legend="hp " + str(years[1]), p=p)



p.legend.location = 'top_center'

show(p)