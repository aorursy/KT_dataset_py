import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import plotly.offline as pyo
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
init_notebook_mode(connected=True)

import plotly.graph_objs as go
print(os.listdir(".."))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/all_stocks_5yr.csv")
df["year"] = pd.DatetimeIndex(df["date"]).year
df["month"] = pd.DatetimeIndex(df["date"]).month
#df = df[df["date"] < '2018-01-01']
print(df.head(5))
df_year_end_date = df.groupby(["year"]).agg({"date":{"date":["max"]}})
df_year_begin_date = df.groupby(["year"]).agg({"date":{"date":["min"]}})
df_year_end_date = df_year_end_date.reset_index()
df_year_end_date.columns = df_year_end_date.columns.map(lambda x: x[0])
df_year_begin_date = df_year_begin_date.reset_index()
df_year_begin_date.columns = df_year_begin_date.columns.map(lambda x: x[0])
df_year_end_data = pd.merge(df, df_year_end_date, on=["year","date"], how="inner")
df_year_begin_data = pd.merge(df, df_year_begin_date, on=["year","date"], how="inner")
df_year_volume_data = df.groupby(["year","Name"]).agg({"volume":{"volume":["sum"]}})
df_year_volume_data = df_year_volume_data.reset_index()
df_year_volume_data.columns = df_year_volume_data.columns.map(lambda x: x[0])
df_year_data = pd.merge(df_year_volume_data, df_year_end_data[["year","Name","close"]], on=["year","Name"], how="inner")
df_year_data = pd.merge(df_year_data, df_year_begin_data[["year","Name","open"]], on=["year","Name"], how="inner")

df_year_data["incr price"] = df_year_data["close"] - df_year_data["open"]
df_year_data["% incr price"] = (df_year_data["incr price"] * 100) / df_year_data["open"]
df_year_data.head(5)
df_name = df.groupby("Name").agg({"volume":{"mean volume":["mean"]},
                                  "close":{"mean close":["mean"]}})
df_name = df_name.reset_index()
df_name.columns = df_name.columns.map(lambda x: x[0])
df = pd.merge(df, df_name, on="Name")
df["diff from mean volume"] = df["volume"] - df["mean volume"]
df["diff price"] = df["close"] - df["open"]
df = df.sort_values(["diff from mean volume"])
top_10_by_diff_from_mean_volume = df.tail(10)
top_10_name_by_diff_from_mean_volume = list(top_10_by_diff_from_mean_volume["Name"])
top_10_by_diff_from_mean_volume
df["% diff from mean volume"] = (df["diff from mean volume"] * 100 ) / df["mean volume"]
df["% diff price"] = (df["diff price"] * 100 ) / df["open"]
df = df.sort_values(["% diff from mean volume"])
top_10_by_perc_diff_from_mean_volume = df.tail(10)
top_10_by_perc_diff_from_mean_volume
for name in top_10_by_perc_diff_from_mean_volume["Name"]:
    if name in top_10_name_by_diff_from_mean_volume:
        name_data = df[df["Name"]==name]["volume"]
        data = [go.Histogram(x=name_data)]
        layout = go.Layout(title=name + " - Histogram")
        fig = go.Figure(data=data,layout=layout)
        iplot(fig)
for name in top_10_by_perc_diff_from_mean_volume["Name"]:
    if name in top_10_name_by_diff_from_mean_volume:
        name_data = df[df["Name"]==name]["volume"]
        data = [go.Box(y=name_data)]
        layout = go.Layout(title=name + " - Box Chart")
        fig = go.Figure(data=data,layout=layout)
        iplot(fig)
for name in top_10_by_perc_diff_from_mean_volume["Name"]:
    if name in top_10_name_by_diff_from_mean_volume:
        df_name = df[df["Name"]==name].sort_values(["date"])
        y_data = df_name["volume"]
        x_data = df_name["date"]
        data = [go.Scatter(x=x_data, y=y_data, mode="lines")]
        layout = go.Layout(title=name + " - Line Chart")
        fig = go.Figure(data=data,layout=layout)
        iplot(fig)
for name in ["MAA"]:
    df_name = df[df["Name"]==name].sort_values(["date"])
    y_data = df_name["volume"]
    x_data = df_name["date"]
    data = [go.Scatter(x=x_data, y=y_data, mode="lines")]
    layout = go.Layout(title=name + " - Line Chart")
    fig = go.Figure(data=data,layout=layout)
    iplot(fig)
num_rec = 500
df = df.sort_values(["diff from mean volume"])
list_volume = set(df.tail(num_rec)["Name"])
df = df.sort_values(["% diff from mean volume"])
list_perc_volume = set(df.tail(num_rec)["Name"])
df = df.sort_values(["diff price"])
list_close = set(df.tail(num_rec)["Name"])
df = df.sort_values(["% diff price"])
list_perc_close = set(df.tail(num_rec)["Name"])
print(list_volume & list_perc_volume & list_close & list_perc_close)
df_Stock = df[df["Name"]=="NVDA"]
print(df_Stock.sort_values(["diff from mean volume"]).tail(2))
print(df_Stock.sort_values(["% diff from mean volume"]).tail(2))
print(df_Stock.sort_values(["diff price"]).tail(2))
print(df_Stock.sort_values(["% diff price"]).tail(2))
for name in ["NVDA"]:
    df_name = df[df["Name"]==name].sort_values(["date"])
    y_data1 = df_name["volume"]/500000
    y_data2 = df_name["close"]
    x_data = df_name["date"]
    data = [go.Scatter(x=x_data, y=y_data1, mode="lines", name="Volume"), go.Scatter(x=x_data, y=y_data2, mode="lines", name="Closing Price")]
    layout = go.Layout(title=name + " - Line Chart")
    fig = go.Figure(data=data,layout=layout)
    iplot(fig)
df_Stock = df[df["Name"]=="NFLX"][["Name","open","close","diff from mean volume","% diff from mean volume","diff price","% diff price","date"]]
print(df_Stock.sort_values(["diff from mean volume"]).tail(2))
print(df_Stock.sort_values(["% diff from mean volume"]).tail(2))
print(df_Stock.sort_values(["diff price"]).tail(2))
print(df_Stock.sort_values(["% diff price"]).tail(2))
for name in ["NFLX"]:
    df_name = df[df["Name"]==name].sort_values(["date"])
    y_data1 = df_name["volume"]/500000
    y_data2 = df_name["close"]
    x_data = df_name["date"]
    data = [go.Scatter(x=x_data, y=y_data1, mode="lines", name="Volume"), go.Scatter(x=x_data, y=y_data2, mode="lines", name="Closing Price")]
    layout = go.Layout(title=name + " - Line Chart")
    fig = go.Figure(data=data,layout=layout)
    iplot(fig)
df["tran value"] = df["volume"] * ((df["close"] + df["open"])/2)
df_tran_value = df.groupby(["Name"]).agg({"tran value":{"tot tran value":["sum"]}})
df_tran_value = df_tran_value.reset_index()
df_tran_value.columns = df_tran_value.columns.map(lambda x: x[0])
df_tran_value = df_tran_value.sort_values(["tot tran value"])
print(df_tran_value.tail(5))
def plot_macd(name):
    df_stock = df[df["Name"] == name].sort_values(["date"])
    df_stock['26 ema'] = df_stock["close"].ewm(span=26, adjust=False).mean()
    df_stock['12 ema'] = df_stock["close"].ewm(span=12, adjust=False).mean()
    df_stock['MACD'] = df_stock['12 ema'] - df_stock['26 ema']
    df_stock['9 ema'] = df_stock["MACD"].ewm(span=9, adjust=False).mean()
    y_data1 = df_stock["MACD"]
    y_data2 = df_stock["9 ema"]
    x_data = df_stock["date"]
    data = [go.Scatter(x=x_data, y=y_data1, mode="lines", name="MACD"), go.Scatter(x=x_data, y=y_data2, mode="lines", name="9 day EMA")]
    layout = go.Layout(title=name + " - Moving Average Convergance Divergence")
    fig = go.Figure(data=data,layout=layout)
    iplot(fig)
plot_macd("AAPL")
plot_macd("FB")
plot_macd("AMZN")
plot_macd("MSFT")
plot_macd("BAC")
