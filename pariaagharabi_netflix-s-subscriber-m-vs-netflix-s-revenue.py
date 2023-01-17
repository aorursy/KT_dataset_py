from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Import the necessary packages

import numpy as np

import pandas as pd



import warnings

warnings.simplefilter(action ="ignore")



# Data visualization

import matplotlib.pyplot as plt

import seaborn as sns

from bokeh.io import output_file, show

from bokeh.models import ColumnDataSource

from bokeh.plotting import figure

from bokeh.transform import dodge
# Load Dataset

df_subscribers = pd.read_csv("../input/netflix2020/NetflixSubscribersbyCountryfrom2018toQ2_2020.csv")



df_revenue  = pd.read_csv("../input/netflix2020/NetflixsRevenue2018toQ2_2020.csv")



df_subscribers_V2 = pd.read_csv("../input/netflix2020/DataNetflixSubscriber2020_V2.csv")



df_revenue_V2 = pd.read_csv("../input/netflix2020/DataNetflixRevenue2020_V2.csv")
df_subscribers.head()
df_revenue.head()
df_subscribers_V2.head()
df_revenue_V2.head()
df_subscribers.describe()
df_revenue.describe()
df_subscribers_V2.describe()
df_revenue_V2.describe()
df_subscribers.info()
df_revenue.info()
df_subscribers_V2.info()
df_revenue_V2.info()
df_subscribers.columns
df_revenue.columns
df_subscribers_V2.columns
df_revenue_V2.columns
print(f"The subscribers data size: {df_subscribers.shape}")

print(f"The revenue data size: {df_revenue.shape}")

print(f"The subscribers data size (V2): {df_subscribers_V2.shape}")

print(f"The revenue data size (V2): {df_revenue_V2.shape}")
df_subscribers_V2.groupby(["Area", "Years"])["Subscribers"].sum().unstack("Years").plot(figsize=(18, 8), kind="bar")

plt.title("Netflix's subscribers (M) growth from 2018 to Q2_2020 by region")

plt.xlabel("Region")

plt.ylabel("Netflix's subscribers (M)")
years = ["Q1 - 2018", "Q2 - 2018", "Q3 - 2018", "Q4 - 2018", "Q1 - 2019", "Q2 - 2019", "Q3 - 2019", "Q4 - 2019", "Q1 - 2020", "Q2 - 2020"]



data = {"Period" : years,

        "United States and Canada": [60909000, 61870000, 63010000, 64757000, 66633000, 66501000, 67114000, 67662000, 69969000,                 72904000],

        "Europe, Middle East, and Africa": [29339000, 31317000, 33836000, 37818000, 42542000, 44229000, 47355000, 51778000, 58734000,          61483000],

        "Latin America": [21260000, 22795000, 24115000, 26077000, 27547000, 27890000, 29380000, 31417000, 34318000, 36068000],

        "Asia-Pacific": [7394000, 8372000, 9461000, 10607000, 12141000, 12942000, 14485000, 16233000, 19835000, 22492000]

        }



source = ColumnDataSource(data=data)



p = figure(x_range=data["Period"], y_range=(0, 100000000), plot_height=500, plot_width=1200, title="Netflix's subscriber from Q1-2018 to Q2-2020 by region", toolbar_location=None, tools="")



p.vbar(x=dodge("Period", -0.30, range=p.x_range), top="United States and Canada", width=0.15, color="#8B0000", source=source, legend_label="United States and Canada")



p.vbar(x=dodge("Period", -0.15,  range=p.x_range), top="Europe, Middle East, and Africa", width=0.15, color="#B22222", source=source, legend_label="Europe, Middle East, and Africa")



p.vbar(x=dodge("Period", -0.00, range=p.x_range), top="Latin America", width=0.15, color="#CD5C5C", source=source, legend_label="Latin America")



p.vbar(x=dodge("Period", 0.15, range=p.x_range), top="Asia-Pacific", width=0.15, color="#F08080", source=source, legend_label="Asia-Pacific")



p.x_range.range_padding = 0.1

p.xgrid.grid_line_color = None

p.legend.location = "top_left"

p.legend.orientation = "horizontal"

p.title.text_font_size = "16pt"



show(p)
df_revenue_V2.groupby(["Area", "Years"])["Revenue"].sum().unstack("Years").plot(figsize=(18, 8), kind="bar")

plt.title("Netflix's revenue ($) growth from 2018 to Q2_2020 by region")

plt.xlabel("Region")

plt.ylabel("Netflix's revenue ($)")
years = ["Q1 - 2018", "Q2 - 2018", "Q3 - 2018", "Q4 - 2018", "Q1 - 2019", "Q2 - 2019", "Q3 - 2019", "Q4 - 2019", "Q1 - 2020", "Q2 - 2020"]



data = {"Period" : years,

        "United States and Canada": [1976157000, 2049546000, 2094850000, 2160979000, 2256851000, 2501199000, 2621250000, 2671908000,           2702776000, 2839670000],

        "Europe, Middle East, and Africa": [886649000, 975497000, 1004749000, 1096812000, 1233379000, 1319087000, 1428040000,                  1562561000, 1723474000, 1892537000],

        "Latin America": [540182000, 568071000, 562307000, 567137000, 630472000, 677136000, 741434000, 746392000, 793453000,                   785368000],

        "Asia-Pacific": [199117000, 221252000, 248691000, 276756000, 319602000, 349494000, 382304000, 418121000, 483660000, 569140000]

        }



source = ColumnDataSource(data=data)



p = figure(x_range=data["Period"], y_range=(0, 3000000000), plot_height=500, plot_width=1200, title="Netflix's revenue from Q1-2018 to Q2-2020 by region", toolbar_location=None, tools="")



p.vbar(x=dodge("Period", -0.30, range=p.x_range), top="United States and Canada", width=0.15, color="#8B0000", source=source, legend_label="United States and Canada")



p.vbar(x=dodge("Period", -0.15,  range=p.x_range), top="Europe, Middle East, and Africa", width=0.15, color="#B22222", source=source, legend_label="Europe, Middle East, and Africa")



p.vbar(x=dodge("Period", -0.00, range=p.x_range), top="Latin America", width=0.15, color="#CD5C5C", source=source, legend_label="Latin America")



p.vbar(x=dodge("Period", 0.15, range=p.x_range), top="Asia-Pacific", width=0.15, color="#F08080", source=source, legend_label="Asia-Pacific")



p.x_range.range_padding = 0.1

p.xgrid.grid_line_color = None

p.legend.location = "top_left"

p.legend.orientation = "horizontal"

p.title.text_font_size = "16pt"



show(p)