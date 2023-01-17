import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
PinDuoDuo = pd.read_csv("../input/apple-store-selected-apps/PinDuoDuo.csv", index_col=['DATE'], parse_dates=['DATE'])

BiXin = pd.read_csv("../input/apple-store-selected-apps/BiXin.csv", index_col=['DATE'], parse_dates=['DATE'])

Du = pd.read_csv("../input/apple-store-selected-apps/Du.csv", index_col=['DATE'], parse_dates=['DATE'])

DuoShan = pd.read_csv("../input/apple-store-selected-apps/DuoShan.csv", index_col=['DATE'], parse_dates=['DATE'])

QingYan = pd.read_csv("../input/apple-store-selected-apps/QingYan.csv", index_col=['DATE'], parse_dates=['DATE'])

QuTouTiao = pd.read_csv("../input/apple-store-selected-apps/QuTouTiao.csv", index_col=['DATE'], parse_dates=['DATE'])

ShuaBao = pd.read_csv("../input/apple-store-selected-apps/ShuaBao.csv", index_col=['DATE'], parse_dates=['DATE'])

Soul = pd.read_csv("../input/apple-store-selected-apps/Soul.csv", index_col=['DATE'], parse_dates=['DATE'])

XiaoHongShu = pd.read_csv("../input/apple-store-selected-apps/XiaoHongShu.csv", index_col=['DATE'], parse_dates=['DATE'])

ZuiYou = pd.read_csv("../input/apple-store-selected-apps/ZuiYou.csv", index_col=['DATE'], parse_dates=['DATE'])
PinDuoDuo['应用总榜 (免费)'].plot(legend=True, figsize=(18, 10), label='PinDuoDuo')
PinDuoDuo['daily-return'] = PinDuoDuo['应用总榜 (免费)'].pct_change()

PinDuoDuo['daily-return'].plot(legend=True, figsize=(18, 10), linestyle='--', marker='o')
plt.figure(figsize=(18, 10))

sns.distplot(PinDuoDuo['daily-return'].dropna(), bins=100, color='purple')

plt.show()
df = pd.concat([PinDuoDuo['应用总榜 (免费)'], BiXin['应用总榜 (免费)'], Du['应用总榜 (免费)'], DuoShan['应用总榜 (免费)'], QingYan['应用总榜 (免费)'],

               QuTouTiao['应用总榜 (免费)'], ShuaBao['应用总榜 (免费)'], Soul['应用总榜 (免费)'], XiaoHongShu['应用总榜 (免费)'], ZuiYou['应用总榜 (免费)']], axis=1)

df.columns = ['PinDuoDuo', 'BiXin', 'Du', 'DuoShan', 'QingYan', 'QuTouTiao', 'ShuaBao', 'Soul','XiaoHongShu', 'ZuiYou']

df
df_daily_return = df.pct_change()

df_daily_return.head()
df.plot(legend=True, figsize=(18, 10))
ma_day = [5, 30, 60]

for ma in ma_day:

    column_name = "MA for %s days" % (ma)

    PinDuoDuo[column_name] = PinDuoDuo['应用总榜 (免费)'].rolling(ma).mean()

PinDuoDuo[['应用总榜 (免费)', 'MA for 5 days', 'MA for 30 days', 'MA for 60 days']].plot(figsize=(18, 10))

plt.show()
sns.jointplot('PinDuoDuo', 'BiXin', df, kind='scatter')

plt.show()
returns_fig = sns.PairGrid(df_daily_return.dropna())

returns_fig.map_upper(plt.scatter, color='purple')

returns_fig.map_lower(sns.kdeplot, cmap='cool_d')

returns_fig.map_diag(plt.hist, bins=30)

plt.show()
df_ret = df.dropna()

# 过去平均值

df_ret.mean()
# 波动程度

df_ret.std()
plt.figure(figsize=(18, 10))

plt.scatter(df_ret.mean(), df_ret.std())

plt.xlabel("Expected Return")

plt.ylabel("Risk")

for label, x, y in zip(df_ret.columns, df_ret.mean(), df_ret.std()):

    plt.annotate(label, xy=(x, y), xytext=(100, 0), textcoords="offset points", ha="right", va="bottom",

                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3"))

plt.show()
df_daily_return['PinDuoDuo'].quantile(0.05)
df_daily_return['XiaoHongShu'].quantile(0.05)
df_daily_return['ZuiYou'].quantile(0.05)
df_daily_return['QuTouTiao'].quantile(0.05)