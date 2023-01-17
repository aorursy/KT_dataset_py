import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

train_df.head(3)
test_df = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')

test_df.head(3)
cc_by_country = train_df[train_df['Date'] == '2020-03-18'].groupby('Country/Region').sum()['ConfirmedCases'].reset_index()
fig = plt.figure()

ax = fig.add_subplot(111)

n, bins, batches = ax.hist(cc_by_country['ConfirmedCases'])
cc_less_than_10000 = cc_by_country[cc_by_country['ConfirmedCases'] <= 10000]['ConfirmedCases']



fig = plt.figure()

ax = fig.add_subplot(111)

n, bins, batches = ax.hist(cc_less_than_10000)
cc_less_than_2000 = cc_by_country[cc_by_country['ConfirmedCases'] <= 2000]['ConfirmedCases']



fig = plt.figure()

ax = fig.add_subplot(111)

n, bins, batches = ax.hist(cc_less_than_2000)
cc_less_than_500 = cc_by_country[cc_by_country['ConfirmedCases'] <= 500]['ConfirmedCases']



fig = plt.figure()

ax = fig.add_subplot(111)

n, bins, batches = ax.hist(cc_less_than_500)
train_df_jp = train_df[train_df['Country/Region'] == 'Japan']



fig = plt.figure(figsize=(18,6))

ax = fig.add_subplot(111)

ax.bar(train_df_jp['Date'].str.slice(start=-5), train_df_jp['ConfirmedCases'])
train_df_jp['CcDiff'] = train_df_jp['ConfirmedCases'].diff().fillna(0)



fig = plt.figure(figsize=(18,6))

ax = fig.add_subplot(111)

ax.bar(train_df_jp['Date'].str.slice(start=-5), train_df_jp['CcDiff'])
# https://qiita.com/MToyokura/items/8a58cb43e634e6421834

import statsmodels.graphics.api as smg

from statsmodels.graphics.tsaplots import plot_acf



plot_acf(train_df_jp['CcDiff'], lags=29, alpha=None)

plt.show()
train_df_ch = train_df[train_df['Country/Region'] == 'China'].groupby('Date').sum()['ConfirmedCases'].reset_index()



fig = plt.figure(figsize=(18,6))

ax = fig.add_subplot(111)

ax.bar(train_df_ch['Date'].str.slice(start=-7), train_df_ch['ConfirmedCases'])
train_df_ch['CcDiff'] = train_df_ch['ConfirmedCases'].diff().fillna(0)



fig = plt.figure(figsize=(18,6))

ax = fig.add_subplot(111)

ax.bar(train_df_ch['Date'].str.slice(start=-5), train_df_ch['CcDiff'])
# https://qiita.com/MToyokura/items/8a58cb43e634e6421834

import statsmodels.graphics.api as smg

from statsmodels.graphics.tsaplots import plot_acf



plot_acf(train_df_ch['CcDiff'], lags=29, alpha=None)

plt.show()
train_df_ch = train_df[train_df['Country/Region'] == 'China']

prv_lst = train_df_ch['Province/State'].unique().tolist()



fig, axes = plt.subplots(11,3)

fig.set_figheight(23)

fig.set_figwidth(18)



for i in range(0, len(prv_lst)):

    prv = prv_lst[i]

    prv_df = train_df_ch[train_df_ch['Province/State'] == prv]

    axes[int(i / 3)][i % 3].bar(prv_df['Date'].str.slice(start=-7), prv_df['ConfirmedCases'])

    axes[int(i / 3)][i % 3].set_title(prv)

    axes[int(i / 3)][i % 3].set_xticks([])

    axes[int(i / 3)][i % 3].set_xticklabels([])
tmp_geo_df = train_df.apply(

    lambda x: x['Country/Region'] if pd.isna(x['Province/State']) else x['Country/Region'] + '-' + x['Province/State'],

    axis=1).to_frame('GeoName')

train_df_geo = pd.concat([train_df, tmp_geo_df], axis=1)

prv_lst = train_df_geo['GeoName'].unique().tolist()



fig, axes = plt.subplots(95,3)

fig.set_figheight(200)

fig.set_figwidth(18)



for i in range(0, len(prv_lst)):

    prv = prv_lst[i]

    prv_df = train_df_geo[train_df_geo['GeoName'] == prv]

    axes[int(i / 3)][i % 3].bar(prv_df['Date'].str.slice(start=-7), prv_df['ConfirmedCases'])

    axes[int(i / 3)][i % 3].set_title(prv)

    axes[int(i / 3)][i % 3].set_xticks([])

    axes[int(i / 3)][i % 3].set_xticklabels([])