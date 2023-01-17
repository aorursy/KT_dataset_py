import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%reload_ext autoreload

%autoreload 2

%matplotlib inline
import warnings

warnings.filterwarnings('ignore')



import seaborn as sns

from matplotlib import rcParams



from fastai.imports import *



from pandas_summary import DataFrameSummary

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype



from IPython.display import display



from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
df_raw = pd.read_csv('../input/avocado-prices/avocado.csv', parse_dates=["Date"])



df_raw.drop(['Unnamed: 0'], axis = 1, inplace=True) # Removing the un-known column
df_raw.sort_values(by=['Date'], inplace=True, ascending=True)

df_raw.reset_index(drop=True, inplace=True)

df_raw.head() # Look at the Data
df_raw.info()
df_raw.describe(include='all').T
df_raw.boxplot(notch=True, column='AveragePrice', by='type', figsize=(7, 5), vert=False, color='cyan', patch_artist=True);
sns.swarmplot('Date','AveragePrice', data=df_raw, hue='type'); # this will take a minute to run
df_tmp = df_raw.groupby('type').sum()

plt.pie(df_tmp['Total Volume'], data = df_tmp, labels = ['conventional','organic'])

plt.show()
sns.lineplot(x = 'year', y = 'Total Volume', data = df_raw, palette= 'Blues');
fig = plt.figure(figsize = (26, 5))

ax = fig.add_subplot(111)

plt.scatter(df_raw['Date'], df_raw['AveragePrice'], c=df_raw['AveragePrice']);

fig.suptitle('Hass Avacados (Avg. Price)');

ax.set_xlabel('Date');

ax.set_ylabel('Average Price');
fig = plt.figure(figsize = (26, 5))

ax = fig.add_subplot(111)

plt.scatter(df_raw[df_raw['type']=='conventional']['Date'], 

            df_raw[df_raw['type']=='conventional']['AveragePrice'], 

            c=df_raw[df_raw['type']=='conventional']['AveragePrice']);

fig.suptitle('Hass Conventional Avacados (Avg. Price)');

ax.set_xlabel('Date');

ax.set_ylabel('Average Price');
fig = plt.figure(figsize = (26, 5))

ax = fig.add_subplot(111)

plt.scatter(df_raw[df_raw['type']=='organic']['Date'], 

            df_raw[df_raw['type']=='organic']['AveragePrice'], 

            c=df_raw[df_raw['type']=='organic']['AveragePrice']);

fig.suptitle('Hass Organic Avacados (Avg. Price)');

ax.set_xlabel('Date');

ax.set_ylabel('Average Price');
df_test = df_raw.groupby(["region"]).mean()

df_test = pd.DataFrame(df_test['AveragePrice'].sort_values())[:15]

exp_ava_city = df_test.iloc[-1]

cheapest_ava_city = df_test.iloc[0]

df_test.plot(y='AveragePrice', kind='barh', figsize=(10, 5), color=(0.9, 0.2, 0.1, 0.6));
df_test = df_raw.groupby(["region"]).mean()

df_test['Total Volume'] = df_test['Total Volume'] / 100

df_test = pd.DataFrame(df_test['Total Volume'].sort_values())[:15]

df_test.plot(y = 'Total Volume', kind = 'barh', figsize = (10, 5), color=(0.1, 0.7, 0.8, 0.6));
cheapest_ava_city, exp_ava_city
df_cat = df_raw.groupby(["region"], as_index=False).sum()

df_cat.rename(columns={'4046':'Small', '4225':'Large', '4770':'XLarge'}, inplace=True)

df_test = df_cat[['region', 'Small','Large', 'XLarge']][:20]



df_test = pd.melt(df_test, id_vars='region', var_name="hass_avacado_type", value_name="quantity sold")

df_test['quantity sold'] = df_test['quantity sold'] / df_test['quantity sold'].min(); df_test

sns.catplot(x='region', y='quantity sold', hue='hass_avacado_type', data=df_test, kind='bar',  height=3, aspect=8);
df_test = df_cat[['region', 'Small','Large', 'XLarge']][20:40]



df_test = pd.melt(df_test, id_vars='region', var_name="hass_avacado_type", value_name="quantity sold")

df_test['quantity sold'] = df_test['quantity sold'] / df_test['quantity sold'].min(); df_test

sns.catplot(x='region', y='quantity sold', hue='hass_avacado_type', data=df_test, kind='bar',  height=3, aspect=8);
def add_datepart(df, fldnames, drop=True, time=False, errors="raise"):  

    if isinstance(fldnames,str): 

        fldnames = [fldnames]

    for fldname in fldnames:

        fld = df[fldname]

        fld_dtype = fld.dtype

        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):

            fld_dtype = np.datetime64



        if not np.issubdtype(fld_dtype, np.datetime64):

            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)

        targ_pre = re.sub('[Dd]ate$', '', fldname)

        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',

                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']

        if time: attr = attr + ['Hour', 'Minute', 'Second']

        for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())

        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9

        if drop: df.drop(fldname, axis=1, inplace=True)
add_datepart(df_raw, 'Date')
def train_cats(df):

    for n,c in df.items():

        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()
y = df_raw['AveragePrice']

df = df_raw.drop('AveragePrice', axis=1)
rcParams['figure.figsize'] = 11.7,8.27

corr = df.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns);