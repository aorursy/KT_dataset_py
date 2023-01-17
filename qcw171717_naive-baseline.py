# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

from tqdm import tqdm
df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
price_df = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
df.head()
price_df.head()
cal_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
cal_df.head()
cal_df["d"]=cal_df["d"].apply(lambda x: int(x.split("_")[1]))

price_df["id"] = price_df["item_id"] + "_" + price_df["store_id"] + "_validation"
for day in tqdm(range(1858, 1886)):

    wk_id = list(cal_df[cal_df["d"]==day]["wm_yr_wk"])[0]

    wk_price_df = price_df[price_df["wm_yr_wk"]==wk_id]

    df = df.merge(wk_price_df[["sell_price", "id"]], on=["id"], how='inner')

    df["unit_sales_" + str(day)] = df["sell_price"] * df["d_" + str(day)]

    df.drop(columns=["sell_price"], inplace=True)
df["dollar_sales"] = df[[c for c in df.columns if c.find("unit_sales")==0]].sum(axis=1)
df.drop(columns=[c for c in df.columns if c.find("unit_sales")==0], inplace=True)
df["weight"] = df["dollar_sales"] / df["dollar_sales"].sum()
df.drop(columns=["dollar_sales"], inplace=True)
for d in range(1886, 1914):

    df["F_" + str(d)] = 0
agg_df = pd.DataFrame(df[[c for c in df.columns if c.find("d_") == 0 or c.find("F_") == 0]].sum()).transpose()

agg_df["level"] = 1

agg_df["weight"] = 1/12

column_order = agg_df.columns
agg_df
level_groupings = {2: ["state_id"], 3: ["store_id"], 4: ["cat_id"], 5: ["dept_id"], 

              6: ["state_id", "cat_id"], 7: ["state_id", "dept_id"], 8: ["store_id", "cat_id"], 9: ["store_id", "dept_id"],

              10: ["item_id"], 11: ["item_id", "state_id"]}
df.groupby(by=level_groupings[11]).sum()
for level in tqdm(level_groupings):

    temp_df = df.groupby(by=level_groupings[level]).sum().reset_index(drop=True)

    temp_df["level"] = level

    temp_df["weight"] /= 12

    agg_df = agg_df.append(temp_df[column_order])



del temp_df
df["weight"] /= 12
print(df.shape[0], agg_df.shape[0], df.shape[0] + agg_df.shape[0])
agg_df["weight"].sum() + df["weight"].sum()
h = 28

n = 1885

def rmsse(ground_truth, forecast, train_series, axis=1):

    # assuming input are numpy array or matrices

    assert axis == 0 or axis == 1

    assert type(ground_truth) == np.ndarray and type(forecast) == np.ndarray and type(train_series) == np.ndarray

    

    if axis == 1:

        # using axis == 1 we must guarantee these are matrices and not arrays

        assert ground_truth.shape[1] > 1 and forecast.shape[1] > 1 and train_series.shape[1] > 1

    

    numerator = ((ground_truth - forecast)**2).sum(axis=axis)

    if axis == 1:

        denominator = 1/(n-1) * ((train_series[:, 1:] - train_series[:, :-1]) ** 2).sum(axis=axis)

    else:

        denominator = 1/(n-1) * ((train_series[1:] - train_series[:-1]) ** 2).sum(axis=axis)

    return (1/h * numerator/denominator) ** 0.5
train_series_cols = [c for c in df.columns if c.find("d_") == 0][:-28]

ground_truth_cols = [c for c in df.columns if c.find("d_") == 0][-28:]

forecast_cols = [c for c in df.columns if c.find("F_") == 0]
df["rmsse"] = rmsse(np.array(df[ground_truth_cols]), 

                   np.array(df[forecast_cols]), np.array(df[train_series_cols]))

agg_df["rmsse"] = rmsse(np.array(agg_df[ground_truth_cols]), 

                   np.array(agg_df[forecast_cols]), np.array(agg_df[train_series_cols]))
# for row_idx in range(len(df)):

#     row_df = pd.DataFrame(df.iloc[row_idx]).transpose()

#     train_series = np.array(row_df[train_series_cols].transpose()[row_idx])

#     ground_truth_series = np.array(row_df[ground_truth_cols].transpose()[row_idx])

#     forecast_series = np.array(row_df[forecast_cols].transpose()[row_idx])

#     print(rmsse(ground_truth_series, forecast_series, train_series, axis=0))
df["wrmsse"] = df["weight"] * df["rmsse"]

agg_df["wrmsse"] = agg_df["weight"] * agg_df["rmsse"]
df["wrmsse"].sum() + agg_df["wrmsse"].sum()