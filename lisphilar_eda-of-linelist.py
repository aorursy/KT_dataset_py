from datetime import datetime

time_format = "%d%b%Y %H:%M"

datetime.now().strftime(time_format)
import collections

from datetime import timedelta

from dateutil.relativedelta import relativedelta

from pprint import pprint

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from matplotlib.ticker import ScalarFormatter

%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns
plt.style.use("seaborn-ticks")

plt.rcParams["xtick.direction"] = "in"

plt.rcParams["ytick.direction"] = "in"

plt.rcParams["font.size"] = 12

plt.rcParams["figure.figsize"] = (9, 6)
def line_plot(df, title, ylabel="Cases", h=None, v=None, xlim=(None, None), ylim=(0, None), math_scale=True):

    """

    Show chlonological change of the data.

    """

    ax = df.plot()

    if math_scale:

        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        ax.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))

    ax.set_title(title)

    ax.set_xlabel(None)

    ax.set_ylabel(ylabel)

    ax.set_xlim(*xlim)

    ax.set_ylim(*ylim)

    ax.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)

    if h is not None:

        ax.axhline(y=h, color="black", linestyle="--")

    if v is not None:

        ax.axvline(x=v, color="black", linestyle="--")

    plt.tight_layout()

    plt.show()
import os

for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        print(os.path.join(dirname, filename))
raw = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

raw.tail()
raw.describe(include="all")
pd.DataFrame(raw.isnull().sum()).T
raw.columns
pd.DataFrame(raw["If_onset_approximated"].value_counts())
pd.DataFrame(raw["symptom"].value_counts()).T
raw["death"].unique()
raw["recovered"].unique()
pd.to_datetime(raw["recovered"].replace({"0": "", "1": "", "12/30/1899": "12/30/2019"})).unique()
case_df = raw.loc[:, ~raw.columns.str.startswith("Unnamed:")]

case_df = case_df.drop(["id", "case_in_country", "summary", "source", "link"], axis=1)

# Date

case_date_dict = {

    "reporting date": "Confirmed_date",

    "exposure_start": "Exposed_date",

    "exposure_end": "Quarantined_date",

    "hosp_visit_date": "Hospitalized_date",

    "symptom_onset": "Onset_date",

    "death": "Deaths_date",

    "recovered": "Recovered_date"    

}

case_df["death"] = case_df["death"].replace({"0": "", "1": ""})

case_df["recovered"] = case_df["recovered"].replace({"0": "", "1": "", "12/30/1899": "12/30/2019"})

for (col, _) in case_date_dict.items():

    case_df[col] = pd.to_datetime(case_df[col])

case_df = case_df.rename(case_date_dict, axis=1)

# Location

case_df["Country"] = case_df["country"].fillna("-")

case_df["Province"] = case_df["location"].fillna("-")

case_df["Province"] = case_df[["Country", "Province"]].apply(lambda x: "-" if x[0] == x[1] else x[1], axis=1)

# Personal

case_df["Gender"] = case_df["gender"].fillna("-")

case_df["Age"] = case_df["age"].fillna(-1).astype(int)

case_df["From_Wuhan"] = case_df["from Wuhan"]

case_df["To_Wuhan"] = case_df["visiting Wuhan"]

# Medical

case_df["Events"] = case_df["symptom"].fillna("-")

# Order of columns

case_df = case_df.loc[

    :,

    [

        "Country", "Province",

        "Exposed_date", "Onset_date", "Hospitalized_date", "Confirmed_date", "Quarantined_date", "Deaths_date", "Recovered_date",

        "Events",

        "Gender", "Age", "From_Wuhan", "To_Wuhan"

    ]

]

case_df.tail()
case_df.describe(include="all")
case_df.info()
_list = case_df["Events"].tolist()

_dict = collections.Counter(",".join(_list).replace(" ", "").split(","))

_dict.pop("-")

event_df = pd.DataFrame.from_dict(_dict, orient="index").sort_values(0, ascending=False)

event_df.columns = ["Event"]

event_df[:10].plot.bar()

plt.title("Events linked with COVID-19 (Top 10)")

plt.show()
case_df[["Exposed_date", "Confirmed_date", "Deaths_date", "Recovered_date"]]
case_period_df = pd.DataFrame(columns=["Name", "Start", "End"])



for (sta, end) in zip(["Exposed", "Confirmed", "Confirmed"], ["Confirmed", "Deaths", "Recovered"]):

    df = pd.DataFrame(

        {

            "Name": f"{sta}_to_{end}",

            "Start": case_df[f"{sta}_date"],

            "End": case_df[f"{end}_date"]

        }

    )

    case_period_df = pd.concat([case_period_df, df], axis=0)



case_period_df = case_period_df.dropna()

case_period_df["Length [day]"] = (case_period_df["End"] - case_period_df["Start"]).dt.days

case_period_df.tail()
case_period_df.groupby("Name").describe()
case_period_df.hist(column="Length [day]", by="Name")

plt.show()
case_period_est_df = case_period_df.groupby("Name").median()

case_period_est_df["1/Length [1/day]"] = 1 / case_period_est_df["Length [day]"]

case_period_est_df