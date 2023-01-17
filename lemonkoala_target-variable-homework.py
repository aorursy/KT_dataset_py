import numpy  as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (20, 7)
df = pd.read_csv("../input/train.csv")
params = ["param_1", "param_2", "param_3"]

df[params] = df[params].fillna("")
df[params].isnull().any()
item_types = ["parent_category_name", "category_name"] + params

dfg = df.groupby(item_types + ["deal_probability"]) \
        .size() \
        .reset_index(name="count")
dfg.head()
services = dfg[dfg["parent_category_name"] == "Услуги"].copy()
services["deal_probability_diff"] = services.groupby(item_types)["deal_probability"].diff()
services[services["param_2"] != ""].head(n=15)
dpdiff_stats = services.groupby(item_types)["deal_probability_diff"].agg(["mean", "size"]).reset_index()
dpdiff_stats.rename(columns={"size": "N", "mean": "dp_diff"}, inplace=True)
dpdiff_stats[dpdiff_stats["param_2"] != ""]
dpdiff_stats[dpdiff_stats["param_2"] == "Ремонт часов"]
dfg[dfg["param_2"] == "Ремонт часов"]
dpdiff_stats[dpdiff_stats["param_2"] == ""]
dfg["deal_probability_diff"] = dfg.groupby(item_types)["deal_probability"].diff()
dfg.head()
dp_diff_stats = dfg.groupby(item_types)["deal_probability_diff"].agg(["mean", "size"]).reset_index()
dp_diff_stats.rename(columns={"size": "N", "mean": "dp_diff"}, inplace=True)
dp_diff_stats.head(n=15)
dp_diff_stats.dropna(inplace=True)
dp_diff_stats["expected_df_diff"] = 1 / (dp_diff_stats["N"]  - 1)
dp_diff_stats["expected_N"] = (1 / dp_diff_stats["dp_diff"]) + 1
dp_diff_stats.head(n=15)
(dp_diff_stats["expected_N"] >= dp_diff_stats["N"]).all()
example = dfg[(dfg["category_name"] == "Игры, приставки и программы") & (dfg["param_1"] == "")]
example
def calculate_bins(N):
    return [
        round(n / float(N - 1), 5)
        for n in range(N)
    ]

def print_bins(N):
    print(f"For N={N}: {calculate_bins(N)}")

print_bins(5)
dp_diff_stats[
    (dp_diff_stats["category_name"] == "Игры, приставки и программы") &
    (dp_diff_stats["param_1"]       == "")
]
print_bins(7)
print_bins(8)
import math

def gcd_arr(arr):
    diff = arr[0]
    for num in arr[1:]:
        diff = math.gcd(diff, num)
    return diff

diff = gcd_arr([0, 12, 20, 48, 84, 96, 100])
diff
%pprint
print(f"N = {int(100 / diff)}")
print(f"Series: {[ num for num in range(0, 101, diff)]}")
from tqdm import tqdm

def brute_force_N(probs):
    for N in tqdm(range(2, 100000)):
        bins = calculate_bins(N)
        
        # this `in` comparison could fail because of rounding issues
        # use math.isclose instead?
        if all([ prob in bins for prob in probs]):
            return N

brute_force_N([0, 0.06322, 0.18389, 0.34615, 0.55800, 0.76786])
aggs = []
for col in [
    "region",
    "city",
    "parent_category_name",
    "category_name",
    "param_1"
]:
    agg = df.groupby(col)["deal_probability"].agg(["nunique", "count"]).reset_index(drop=True)
    agg["column"] = col
    aggs.append(agg)

aggs = pd.concat(aggs)
aggs.head()
sns.lmplot(
    data=aggs,
    x="count", y="nunique", col="column",
    fit_reg=False, sharex=False, sharey=False
);
agg = df.groupby(["region", "parent_category_name"])["deal_probability"].agg(["nunique", "count"]).reset_index()
agg.head()
sns.lmplot(
    data=agg,
    x="count", y="nunique", col="parent_category_name",
    fit_reg=False, sharex=False, sharey=False
);












