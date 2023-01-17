# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport



import matplotlib as mplot

from matplotlib import pyplot as plt

from mpl_toolkits import mplot3d

from matplotlib import colors



base_colors_values = list(colors.BASE_COLORS.values())



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# https://www.kaggle.com/fabiendaniel/elo-world

def reduce_mem_usage(df, verbose=True):

    prefixes = ['int', 'float']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = str(df[col].dtype)

        if not col_type.startswith('int') and not col_type.startswith('float'):

            print('col_type:', col_type, 'not compressed')

            continue

        c_min = df[col].min()

        c_max = df[col].max()

        if col_type.startswith('int'):

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                df[col] = df[col].astype(np.int8)

            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                df[col] = df[col].astype(np.int16)

            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                df[col] = df[col].astype(np.int32)

            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                df[col] = df[col].astype(np.int64)  

        elif col_type.startswith('float'):

            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                df[col] = df[col].astype(np.float16)

            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                df[col] = df[col].astype(np.float32)

            else:

                df[col] = df[col].astype(np.float64)    

    if verbose:

        end_mem = df.memory_usage().sum() / 1024**2

        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

from pathlib import Path

INPUT = Path('../input/elo-merchant-category-recommendation/')

train = pd.read_csv(INPUT/'train.csv')

test = pd.read_csv(INPUT/'test.csv')

train.head()
print('Columns:', *train.columns, sep=' | ')
df_train = train.copy()
# https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-elo

target_col = "target"



plt.figure(figsize=(8,6))

plt.scatter(range(df_train.shape[0]), np.sort(df_train[target_col].values))

plt.xlabel('index', fontsize=12)

plt.ylabel('Loyalty Score', fontsize=12)

plt.show()
fig, axs = plt.subplots()



axs.hist(df_train.target, range(-34, 19), color='r')

axs.set_title('Hist of Loyalty', color='g');
print("Extremely low scores (< -30):", (df_train.target < -30).sum())

print("Extremely HIGH scores(> 10):", (df_train.target > 10).sum())
fig, ax = plt.subplots()

ax.hist(df_train.target, range(-10, 10), color='purple')

ax.set_title('[-10, 10] Hist of Loyalty', color='g');
fig, ax = plt.subplots()

ax.hist(df_train.target, range(-3, 4), color='cyan', density=True)

ax.set_title('[-3, 3] Hist of Loyalty', color='blue',);
first_mount = df_train.first_active_month.value_counts()

srt = first_mount.sort_index()

years = srt.index.str[:4].unique()



# WHITE color doesn't well... appear

MY_BASE_COLORS = colors.BASE_COLORS.copy()

del MY_BASE_COLORS['w']

MY_BASE_COLORS['purple'] = 'purple'



year_cmap = dict(zip(years, MY_BASE_COLORS))



cmap_seq = srt.index.map(lambda x: year_cmap[x[:4]])





fig = plt.figure(figsize=(14, 10))



plt.bar(

    srt.index,

    srt.values,

    color=cmap_seq,

)

plt.xticks(rotation='vertical');

plt.xlabel('First month');

plt.title('First card active month', color='g');



ax = plt.gca()

ax.set_facecolor((.6, .44, .98, .6))

# fig.patch.set_facecolor('xkcd:mint green')

for i, t in enumerate(plt.gca().get_xticklabels()):

    t.set_color( cmap_seq[i] )

plt.show()
first_month = df_train.first_active_month.value_counts()

srt = first_month.sort_index()

years = srt.index.str[:4].unique()

year_cmap = dict(zip(years, MY_BASE_COLORS))



vc = df_train.first_active_month.str[:4].value_counts()

srt=vc.sort_index()



indices = np.arange(len(srt))

fig, ax = plt.subplots(figsize=(12.8, 12.6))

ax.pie(

    srt.values,

    labels=srt.index

)

ax.set_title('First-month YEAR active', fontdict={'color': 'red'});

plt.show()
_=pd.cut(

    df_train.target,

    range(-34, 19)

)

vc = _.value_counts()

srt=vc.sort_index()



indices = np.arange(len(srt))

fig, ax = plt.subplots(figsize=(12.8, 12.6))

ax.pie(

    srt.values,

    labels=srt.index

)

ax.set_title('Loyalty Score Bins distribution', fontdict={'color': 'red'});
fts = [f'feature_{i}' for i in range(1,4)]

fig, ax = plt.subplots()



for ft in fts:

    _ = train.groupby(ft).size().sort_index()

    ax.bar(_.index, _.values)    
fts = [f'feature_{i}' for i in range(1,4)]

fig, axs = plt.subplots(nrows=1, ncols=len(fts), figsize=(len(fts) * 5 + 1, 8),

                        edgecolor='black',

                       frameon=True

)



# fig = plt.figure()

# ax = fig.add_subplot()

for i, ft in enumerate(fts):

#     _ = train.groupby(ft).size().sort_index()

    _ = train.groupby(ft).size().sort_values()

    

#     ax.bar(_.index, _.values, tick_label=_.index, color=base_colors_values[i])    

    axs[i].pie(_.values, labels=_.index, autopct='%1.1f%%')

    axs[i].set_title(ft, color=base_colors_values[i])

#     axs[i].bar(_.index, _.values, tick_label=_.index, color=base_colors_values[i])

df_historical = pd.read_csv(INPUT/'historical_transactions.csv')
df_historical.columns
df_historical.describe()
cat_cols = [

    'city_id', 'category_1', 'installments',

    'merchant_category_id', 'merchant_id', 'month_lag',

    'category_2', 'state_id', 'subsector_id'

]



for c in cat_cols:

    print(c, df_historical[c].unique())

base_colors_values = list(colors.BASE_COLORS.values())



n = 4

fig, axs = plt.subplots(4, 1, figsize=(15, n * 5 + 1))

for i, cat in enumerate(['category_2', 'installments', 'state_id', 'category_1']):

    

    _ = df_historical.groupby(by=cat).size().sort_values()

    if cat == 'installments':

        _.drop(999, inplace=True)

    c=base_colors_values[i]

    axs[i].bar(_.index, _.values, color=c)

    axs[i].set_title(cat, color=c, loc='right', size='large', style='italic', weight='heavy')

    maxval = max(_.values)

    for idx, xval in enumerate(_.values):

        rotation = 45 * (len(_.index) > 20)

        rotation += 45        

        axs[i].text(

            _.index[idx], 1.07 * xval, str(xval),

            rotation = rotation if xval < maxval else 0,

            size='medium',

            weight='bold',

            color='black',

        )

    

    
df_historical.groupby('installments').size().drop(999)
r=df_historical.groupby('installments').size().drop(999)

# plt.bar(r.index[:3], r.values[:3]);

plt.bar(r.index, r.values, color='red');
df_historical.head()