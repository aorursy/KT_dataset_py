%reload_ext autoreload
%autoreload 2
from fastai.basics import *
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
PATH = Config().data_path()/Path('rossmann/')
#!mkdir {PATH}
#mkdir: cannot create directory ‘/root/.fastai/data/rossmann’: No such file or directory
!mkdir /root/.fastai/data/
!mkdir /root/.fastai/data/rossmann
!cp /kaggle/input/rossmann/*.csv {PATH}
table_names = ['train', 'store', 'store_states', 'state_names', 'googletrend', 'weather', 'test']
tables = [pd.read_csv(PATH/f'{fname}.csv', low_memory=False) for fname in table_names]
train, store, store_states, state_names, googletrend, weather, test = tables
googletrend.tail()
store.head()
store_states.head()
weather.tail()
len(train), len(test)
import warnings
warnings.filterwarnings('ignore')
train.StateHoliday = train.StateHoliday != '0'
test.StateHoliday = test.StateHoliday != '0'
def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if(right_on is None):
        right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, suffixes=("", suffix))
weather = join_df(weather, state_names, "file", "StateName" )
weather.head()
googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0]
googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]

googletrend.loc[googletrend.State=='NI', "State"] = 'HB,NI'

def add_datepart(df, fldname, drop=True, time=False):
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: 
        attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr:
        df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop :
        df.drop(fldname, axis=1, inplace=True)
            
add_datepart(weather, "Date", drop=False)
add_datepart(googletrend, "Date", drop=False)
add_datepart(train, "Date", drop=False)
add_datepart(test, "Date", drop=False)
len(train), len(test)
trend_germany = googletrend[googletrend.file == "Rossmann_DE"]
store = join_df(store, store_states, "Store")
len(store[store.State.isnull()])

# is isnull is zero it means the rows are consistent
combined  = join_df(train, store, "Store")
combined_test = join_df(test, store, "Store")
len(combined[combined.StoreType.isnull()]), len(combined_test[combined_test.StoreType.isnull()])
#len(combined), len(combined_test)
combined  = join_df(combined, googletrend, ["State", "Year", "Week"])
combined_test = join_df(combined_test, googletrend, ["State", "Year", "Week"])
len(combined[combined.trend.isnull()]), len(combined_test[combined_test.trend.isnull()])
#len(combined), len(combined_test)
combined = combined.merge(trend_germany, 'left', ["Year", "Week"], suffixes=('', '_DE'))
combined_test = combined_test.merge(trend_germany, 'left', ["Year", "Week"], suffixes=('', '_DE'))
len(combined[combined.trend_DE.isnull()]),len(combined_test[combined_test.trend_DE.isnull()])
#len(combined), len(combined_test)
combined = join_df(combined, weather, ["State","Date"])
combined_test = join_df(combined_test, weather, ["State","Date"])
len(combined[combined.Mean_TemperatureC.isnull()]),len(combined_test[combined_test.Mean_TemperatureC.isnull()])
#len(combined), len(combined_test)
for df in (combined, combined_test):
    for c in df.columns:
        if c.endswith('_y'):
            if c in df.columns:
                df.drop(c, inplace=True, axis=1)
#len(combined), len(combined_test)
for df in (combined, combined_test):
    df['CompetitionOpenSinceYear'] = df.CompetitionOpenSinceYear.fillna(1900).astype(np.int32)
    df['CompetitionOpenSinceMonth'] = df.CompetitionOpenSinceMonth.fillna(1).astype(np.int32)
    df['Promo2SinceYear'] = df.Promo2SinceYear.fillna(1900).astype(np.int32)
    df['Promo2SinceWeek'] = df.Promo2SinceWeek.fillna(1).astype(np.int32)
#len(combined), len(combined_test)
for df in (combined,combined_test):
    df["CompetitionOpenSince"] = pd.to_datetime(dict(year=df.CompetitionOpenSinceYear, 
                                                     month=df.CompetitionOpenSinceMonth, day=15))
    df["CompetitionDaysOpen"] = df.Date.subtract(df.CompetitionOpenSince).dt.days
#len(combined), len(combined_test)
for df in (combined, combined_test):
    df.loc[df.CompetitionDaysOpen<0, "CompetitionDaysOpen"] = 0
    df.loc[df.CompetitionOpenSinceYear<1990, "CompetitionDaysOpen"] = 0
for df in (combined,combined_test):
    df["CompetitionMonthsOpen"] = df["CompetitionDaysOpen"]//30
    df.loc[df.CompetitionMonthsOpen>24, "CompetitionMonthsOpen"] = 24
combined.CompetitionMonthsOpen.unique()
!pip install isoweek
from isoweek import Week
for df in (combined,combined_test):
    df["Promo2Since"] = pd.to_datetime(df.apply(lambda x: Week(
        x.Promo2SinceYear, x.Promo2SinceWeek).monday(), axis=1))
    df["Promo2Days"] = df.Date.subtract(df["Promo2Since"]).dt.days
for df in (combined,combined_test):
    df.loc[df.Promo2Days<0, "Promo2Days"] = 0
    df.loc[df.Promo2SinceYear<1990, "Promo2Days"] = 0
    df["Promo2Weeks"] = df["Promo2Days"]//7
    df.loc[df.Promo2Weeks<0, "Promo2Weeks"] = 0
    df.loc[df.Promo2Weeks>25, "Promo2Weeks"] = 25
    df.Promo2Weeks.unique()
#len(combined), len(combined_test)
combined.to_pickle(PATH/'combined')
combined_test.to_pickle(PATH/'combined_test')
def get_elapsed(fld, pre):
    day1 = np.timedelta64(1, 'D')
    last_date = np.datetime64()
    last_store = 0
    res = []

    for s,v,d in zip(df.Store.values,df[fld].values, df.Date.values):
        if s != last_store:
            last_date = np.datetime64()
            last_store = s
        if v: last_date = d
        res.append(((d-last_date).astype('timedelta64[D]') / day1))
    df[pre+fld] = res
columns = ["Date", "Store", "Promo", "StateHoliday", "SchoolHoliday"]
df = train[columns].append(test[columns])
fld = 'SchoolHoliday'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')
df.head()
# for 2 more fiellds

fld = 'StateHoliday'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')
df.head()
fld = 'Promo'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')
df.tail()
# setting active index to DAte

df = df.set_index("Date")

# setting null values from elapsed field to 0
columns = ['SchoolHoliday', 'StateHoliday', 'Promo']
for o in ['Before', 'After']:
    for p in columns:
        a = o+p
        df[a] = df[a].fillna(0).astype(int)

bwd = df[['Store']+columns].sort_index().groupby("Store").rolling(7, min_periods=1).sum()
fwd = df[['Store']+columns].sort_index(ascending=False
                                      ).groupby("Store").rolling(7, min_periods=1).sum()

bwd.drop('Store',1,inplace=True)
bwd.reset_index(inplace=True)
fwd.drop('Store',1,inplace=True)
fwd.reset_index(inplace=True)
df.reset_index(inplace=True)
# we will merge these values into the dif
df = df.merge(bwd, 'left', ['Date', 'Store'], suffixes=['', '_bw'])
df = df.merge(fwd, 'left', ['Date', 'Store'], suffixes=['', '_fw'])
df.drop(columns,1,inplace=True)
df.head()

df.to_pickle(PATH/'df')
df["Date"] = pd.to_datetime(df.Date)
df.columns
joined = pd.read_pickle(PATH/'combined')
joined_test = pd.read_pickle(PATH/f'combined_test')
joined = join_df(joined, df, ['Store', 'Date'])
joined_test = join_df(joined_test, df, ['Store', 'Date'])
#len(joined), len(joined_test)
joined = joined[joined.Sales!=0]


len(joined), len(joined_test)
joined.reset_index(inplace=True)
joined_test.reset_index(inplace=True)

joined.to_pickle(PATH/'train_clean')
joined_test.to_pickle(PATH/'test_clean')


len(joined), len(joined_test)
#PATH
train_df = pd.read_pickle(PATH/'train_clean')
train_df.head().T
n = len(train_df)
print(n)
from fastai.tabular import *
idx = np.random.permutation(range(n))[:2000]
idx.sort()
small_train_df = train_df.iloc[idx[:1000]]
small_test_df = train_df.iloc[idx[1000:]]
small_cont_vars = ['CompetitionDistance', 'Mean_Humidity']
small_cat_vars =  ['Store', 'DayOfWeek', 'PromoInterval']
small_train_df = small_train_df[small_cat_vars + small_cont_vars + ['Sales']]
small_test_df = small_test_df[small_cat_vars + small_cont_vars + ['Sales']]
small_train_df.head()
small_test_df.head()
categorify = Categorify(small_cat_vars, small_cont_vars)
categorify(small_train_df)
categorify(small_test_df, test=True)
small_test_df.head()
small_train_df.PromoInterval.cat.categories
small_train_df['PromoInterval'].cat.codes[:5]
fill_missing = FillMissing(small_cat_vars,small_cont_vars)
fill_missing(small_train_df)
fill_missing(small_test_df, test=True)
small_train_df[small_train_df['CompetitionDistance_na'] == True]

train_df = pd.read_pickle(PATH/'train_clean')
test_df = pd.read_pickle(PATH/'test_clean')
#len(train_df), len(test_df)
cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen',
    'Promo2Weeks', 'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear',
    'State', 'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw',
    'SchoolHoliday_fw', 'SchoolHoliday_bw']

cont_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
   'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 
   'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
   'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday']
procs = [FillMissing, Categorify, Normalize]
dep_var = 'Sales'
df = train_df[cat_vars + cont_vars + [dep_var,'Date']].copy()
test_df['Date'].min(), test_df['Date'].max()
#print(len(train_df), len(test_df))
cut = train_df['Date'][(train_df['Date'] == train_df['Date'][len(test_df)])].index.max()


valid_idx = range(cut)
df[dep_var].head()
data = (TabularList.from_df(df, path=PATH, cat_names=cat_vars, cont_names=cont_vars, procs=procs,)
                .split_by_idx(valid_idx)
                .label_from_df(cols=dep_var, label_cls=FloatList, log=True)
                .add_test(TabularList.from_df(test_df, path=PATH, cat_names=cat_vars, cont_names=cont_vars))
                .databunch())
max_log_y = np.log(np.max(train_df['Sales']) * 1.2)
y_range = torch.tensor([0, max_log_y],
                      device = defaults.device)

learn = tabular_learner(data, layers= [1000,500], ps=[0.001,0.01], emb_drop=0.04, y_range=y_range, metrics =exp_rmspe)
learn.model
len(data.train_ds.cont_names)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, 1e-3, wd=0.2)
learn.save('1')
learn.recorder.plot_losses(skip_start=1000)
learn.load('1')
learn.fit_one_cycle(5, 3e-4)
learn.fit_one_cycle(5, 3e-4)
test_preds = learn.get_preds(DatasetType.Test)
test_df["Sales"] = np.exp(test_preds[0].data).numpy().T[0]
test_df[["Id", "Sales"]] = test_df[["Id", "Sales"]].astype("int")
test_df[["Id", "Sales"]].to_csv("rossmann_submission.csv", index=False)