import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

import datetime as dt

import os





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_dji = pd.read_csv(os.path.join(dirname, 'djia.csv'))

df_comp = pd.read_csv(os.path.join(dirname, 'stock_prices.csv'))
df_comp.head()
df_dji.head()
df_dji.info()

print('-'*50)

df_comp.info()
df_comp['Date'] = pd.to_datetime(df_comp['Date'], dayfirst=True)

df_dji['Date'] = pd.to_datetime(df_dji['Date'], dayfirst=True)

df_comp = df_comp[df_comp.Date > dt.datetime(2019,3,19)].reset_index(drop=True)

df_dji = df_dji[df_dji.Date > dt.datetime(2019,3,19)].reset_index(drop=True)
df_dji
df_comp
from pandas.plotting import register_matplotlib_converters





register_matplotlib_converters()

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15,30))

fig.suptitle('Stock price dynamics', y=0.9, fontsize=24)

comp = np.random.choice(df_comp.columns[1:], 4, replace=False).tolist()

for i, ax in enumerate(axes):

    ax.plot(df_comp.loc[:, 'Date'].values, df_comp.loc[:, comp[i]].values)

    ax.set_title(comp[i], loc='left', fontsize=18)

    ax.grid(linestyle = '-.')

    ax.set_xlim(df_comp.loc[:, 'Date'].values[0], df_comp.loc[:, 'Date'].values[-1])

# plt.subplots_adjust(hspace=0.4)

plt.show()
plt.figure(figsize=(15,5))

plt.plot(df_dji.loc[:,'Date'].values, df_dji.loc[:,'DJIA'].values)

plt.xlim(df_dji.loc[:,'Date'].values[0], df_dji.loc[:,'Date'].values[-1])

plt.title('DJIA dynamics', fontsize=20)

plt.grid(linestyle = '-.')

plt.show()
cols = ['Apple', 'IBM', 'Intel', 'Cisco', 'Microsoft', 'Visa',

        'Boeing', 'Chevron', 'ExxonMobile', 'Travelers', 'Home Depot', 

        'Johnson&Johnson', 'Coca-Cola', 'McDonalds', 'Procter&Gamble', 'Walmart']

# cols = np.random.choice(df_comp.columns[1:], 15, replace=False).tolist()

cm = np.corrcoef(df_comp.loc[:, cols].values.T)

plt.figure(figsize=(20,12))

sns.set(font_scale=1.6)

hm = sns.heatmap(cm,

                 annot=True,

                 fmt='.2f',

                 cmap='BuGn',

                 annot_kws={'size': 18},

                 yticklabels=cols,

                 xticklabels=cols)

plt.show()
sc = StandardScaler()

X = df_comp.iloc[:,1:].values

y = df_dji.iloc[:,1].values

X_std = sc.fit_transform(X)

y_std = sc.fit_transform(y.reshape(-1, 1))

y_std.shape
all_comp = df_comp.columns[1:].values

val = np.vstack((df_dji.iloc[:,1].values, X_std.T))

print('Stock prices - DJIA correlation')

print('-'*30)

corr_coef = {}

for i in range(30):

    print('%i)  %.3f  %s'  % (i+1, np.corrcoef(val)[0,1:][i], all_comp[i]))

    corr_coef[all_comp[i]] = np.corrcoef(val)[0,1:][i]
sort_val = sorted(corr_coef.items(), key=lambda kv: kv[1])

all_comp_sorted = [sort_val[i][0] for i in range(len(sort_val))]

corr_coef_sorted = [sort_val[i][1] for i in range(len(sort_val))]
plt.figure(figsize=(20,25))

rects = plt.barh(all_comp_sorted, corr_coef_sorted)

plt.xlim(0,1.2)



for i, rect in enumerate(rects):

    plt.text(rect.get_width(), rect.get_y() + 0.2, round(corr_coef_sorted[i], 2))

plt.show()

sns.reset_orig()
plt.figure(figsize=(15,5))

plt.plot(df_dji.loc[:,'Date'].values, X_std[:,6], label='Cisco')

plt.plot(df_dji.loc[:,'Date'].values, X_std[:,10], label='Goldman Sachs', c='g')

plt.plot(df_dji.loc[:,'Date'].values, y_std, label='DJIA', c='r')

plt.xlim(df_dji.loc[:,'Date'].values[0], df_dji.loc[:,'Date'].values[-1])

plt.title('Max and min correlation', fontsize=20)

plt.legend(fontsize=12)

plt.grid(linestyle = '-.')

plt.show()
df_comp_2020 = df_comp[df_comp.Date > dt.datetime(2020,1,1)].reset_index(drop=True)

df_dji_2020 = df_dji[df_dji.Date > dt.datetime(2020,1,1)].reset_index(drop=True)

df_comp_2020.describe()
df_comp_2020.head()
sc = StandardScaler()

X = df_comp_2020.iloc[:,1:].values

y = df_dji_2020.iloc[:,1].values

X_std = sc.fit_transform(X)

y_std = sc.fit_transform(y.reshape(-1, 1))

val = np.vstack((df_dji_2020.iloc[:,1].values, X_std.T))

print('Stock prices - DJIA correlation (Feb - Mar 2020)')

print('-'*40)

corr_coef = {}

for i in range(30):

    print('%i) %.3f  %s'  % (i+1, np.corrcoef(val)[0,1:][i], all_comp[i]))

    corr_coef[all_comp[i]] = np.corrcoef(val)[0,1:][i]
sort_val = sorted(corr_coef.items(), key=lambda kv: kv[1])

all_comp_sorted = [sort_val[i][0] for i in range(len(sort_val))]

corr_coef_sorted = [sort_val[i][1] for i in range(len(sort_val))]



plt.figure(figsize=(10,15))

rects = plt.barh(all_comp_sorted, corr_coef_sorted)

plt.xlim(0,1.2)



for i, rect in enumerate(rects):

    plt.text(rect.get_width(), rect.get_y() + 0.2, round(corr_coef_sorted[i], 2))

plt.show()

sns.reset_orig()
plt.figure(figsize=(12,8))

dates = pd.to_datetime(df_dji_2020.loc[:,'Date'].values).strftime('%Y-%m-%d')

ticks = [dates[i] for i in range(0,54,10)]

plt.plot(dates, X_std[:,23], label='United Technologies')

plt.plot(dates, X_std[:,7], label='Coca-cola')

plt.plot(dates, y_std, label='DJIA', c='r', linewidth=5)

plt.xlim(ticks[0], ticks[-1])

plt.ylim((-4,3))

plt.xticks(ticks)

plt.title('Correlations', fontsize=20)

plt.legend(fontsize=12)

plt.grid(linestyle = '-.')

plt.show()
plt.figure(figsize=(12,8))

plt.plot(dates, X_std[:,28], label='Walmart')

plt.plot(dates, X_std[:,27], label='Walgreens Boots Alliance')

plt.plot(dates, y_std, label='DJIA', c='r', linewidth=5)

plt.xlim(ticks[0], ticks[-1])

plt.ylim((-4,3))

plt.xticks(ticks)

plt.title('Correlations', fontsize=20)

plt.legend(fontsize=12)

plt.grid(linestyle = '-.')

plt.show()