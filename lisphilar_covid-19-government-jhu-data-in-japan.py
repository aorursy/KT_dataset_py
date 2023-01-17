from datetime import datetime

time_format = "%d%b%Y %H:%M"

datetime.now().strftime(time_format)
import os

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import matplotlib

from matplotlib.ticker import ScalarFormatter

import numpy as np

import pandas as pd

import seaborn as sns

# Matplotlib

%matplotlib inline

plt.style.use("seaborn-ticks")

plt.rcParams["xtick.direction"] = "in"

plt.rcParams["ytick.direction"] = "in"

plt.rcParams["font.size"] = 11.0

plt.rcParams["figure.figsize"] = (9, 6)

# Pandas

pd.set_option("display.max_colwidth", 1000)
def line_plot(df, title, xlabel=None, ylabel="Cases",

              h=None, v=None, xlim=(None, None), ylim=(0, None),

              math_scale=True, x_logscale=False, y_logscale=False, y_integer=False,

              show_legend=True, bbox_to_anchor=(1.02, 0),  bbox_loc="lower left"):

    """

    Show chlonological change of the data.

    """

    ax = df.plot()

    # Scale

    if math_scale:

        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        ax.ticklabel_format(style="sci",  axis="y",scilimits=(0, 0))

    if x_logscale:

        ax.set_xscale("log")

        if xlim[0] == 0:

            xlim = (None, None)

    if y_logscale:

        ax.set_yscale("log")

        if ylim[0] == 0:

            ylim = (None, None)

    if y_integer:

        fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)

        fmt.set_scientific(False)

        ax.yaxis.set_major_formatter(fmt)

    # Set metadata of figure

    ax.set_title(title)

    ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)

    ax.set_xlim(*xlim)

    ax.set_ylim(*ylim)

    if show_legend:

        ax.legend(bbox_to_anchor=bbox_to_anchor, loc=bbox_loc, borderaxespad=0)

    else:

        ax.legend().set_visible(False)

    if h is not None:

        ax.axhline(y=h, color="black", linestyle=":")

    if v is not None:

        if not isinstance(v, list):

            v = [v]

        for value in v:

            ax.axvline(x=value, color="black", linestyle=":")

    plt.tight_layout()

    plt.show()
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
gov_raw = pd.read_csv("/kaggle/input/covid19-dataset-in-japan/covid_jpn_total.csv")

gov_raw.tail()
# https://www.kaggle.com/lisphilar/eda-of-japan-dataset

df = gov_raw.copy()

df.dropna(how="all", inplace=True)

df["Date"] = pd.to_datetime(df["Date"])

df = df.groupby("Location").apply(

    lambda x: x.set_index("Date").resample("D").interpolate(method="linear")

)

df = df.drop("Location", axis=1).reset_index()

df = df.sort_values("Date").reset_index(drop=True)

sel = df.columns.isin(["Location", "Date"])

df.loc[:, ~sel] = df.loc[:, ~sel].fillna(0).astype(np.int64)

# Select Confirmed/Recovered/Fatal

df = df.loc[:, ["Location", "Date", "Positive", "Fatal", "Discharged"]]

df = df.rename({"Positive": "Confirmed", "Discharged": "Recovered"}, axis=1)

# Show

gov_df = df.copy()

gov_df.tail(9)
gov_total_df = gov_df.groupby("Date").sum()

gov_total_df.tail()
df = gov_df.copy()

df = df.loc[df["Location"] == "Domestic", :].drop("Location", axis=1)

df = df.groupby("Date").last()

gov_dom_df = df.copy()

gov_dom_df.tail()
jhu_raw = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

jhu_raw.loc[jhu_raw["Country/Region"] == "Japan", :]
df = jhu_raw.copy()

df = df.rename({"ObservationDate": "Date", "Deaths": "Fatal"}, axis=1)

df = df.loc[df["Country/Region"] == "Japan", ["Date", "Confirmed", "Fatal", "Recovered"]]

df["Date"] = pd.to_datetime(df["Date"])

df = df.groupby("Date").sum()

df = df.astype(np.int64)

df = df.reset_index()

jhu_df = df.copy()

jhu_df.tail()
df = pd.merge(

    gov_total_df, gov_dom_df,

    left_index=True, right_index=True,

    suffixes=["/Total", "/Domestic"]

)

df = pd.merge(

    df.add_suffix("/GOV"), jhu_df.set_index("Date").add_suffix("/JHU"),

    left_index=True, right_index=True

)

comp_df = df.copy()

comp_df.tail()
c_df = comp_df.loc[:, comp_df.columns.str.startswith("Confirmed")]

c_df.tail(10)
df = c_df.copy()

df.columns = df.columns.str.replace("Confirmed/", "")

line_plot(df, "Confirmed cases in Japan: Comparison of datasets", y_integer=True)
df = c_df.copy()

df.columns = df.columns.str.replace("Confirmed/", "")

series = df["JHU"] - df["Total/GOV"]

line_plot(

    series,

    "Confirmed cases in Japan: JHU minus Total/GOV",

    y_integer=True, ylim=(None, None), show_legend=False,

    h=0

)
d_df = comp_df.loc[:, comp_df.columns.str.startswith("Fatal")]

d_df.tail(10)
df = d_df.copy()

df.columns = df.columns.str.replace("Fatal/", "")

line_plot(df, "Fatal cases in Japan: Comparison of datasets", y_integer=True)
df = d_df.copy()

df.columns = df.columns.str.replace("Fatal/", "")

series = df["JHU"] - df["Total/GOV"]

line_plot(

    series,

    "Fatal cases in Japan: JHU minus Total/GOV",

    y_integer=True, ylim=(None, None), show_legend=False,

    h=0

)
r_df = comp_df.loc[:, comp_df.columns.str.startswith("Recovered")]

r_df.tail(10)
df = r_df.copy()

df.columns = df.columns.str.replace("Recovered/", "")

line_plot(df, "Recovered cases in Japan: Comparison of datasets", y_integer=True)
df = r_df.copy()

df.columns = df.columns.str.replace("Recovered/", "")

series = df["JHU"] - df["Total/GOV"]

line_plot(

    series,

    "Recovered cases in Japan: JHU minus Total/GOV",

    y_integer=True, ylim=(None, None), show_legend=False,

    h=0

)