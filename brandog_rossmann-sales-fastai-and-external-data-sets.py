from fastai.tabular import *

from isoweek import Week

#import tarfile



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# path to external datasets

tar = tarfile.open('/kaggle/input/external-datasets/rossmann.tgz', "r:gz")
# place holders

path = "/kaggle/input/rossmann-store-sales/"

base_path="../output"
# paths to kaggle data sets

train = "/kaggle/input/rossmann-store-sales/train.csv"

test = "/kaggle/input/rossmann-store-sales/test.csv"

store = "/kaggle/input/rossmann-store-sales/store.csv"



# paths to external tar file datasets

store_states = tar.extractfile('store_states.csv')

state_names = tar.extractfile('state_names.csv')

googletrend = tar.extractfile('googletrend.csv')

weather = tar.extractfile('weather.csv')
# read in kaggle and external datasets as dataframes

table_names = [train, store, store_states, state_names, googletrend, weather, test]

tables = [pd.read_csv(fpath, low_memory=False) for fpath in table_names]

train, store, store_states, state_names, googletrend, weather, test = tables

len(train),len(test)
print(train.shape)

train.head()
print(test.shape)

test.head()
print(store.shape)

store.head()
print(store_states.shape)

store_states.head()
print(googletrend.shape)

googletrend.head()
print(weather.shape)

weather.head()
print(train.StateHoliday.unique())

print(test.StateHoliday.unique())
train.StateHoliday = train.StateHoliday!='0'

test.StateHoliday = test.StateHoliday!='0'
def join_df(left, right, left_on, right_on=None, suffix='_y'):

    if right_on is None: right_on = left_on

    return left.merge(right, how='left', left_on=left_on, right_on=right_on,

                     suffixes=("",suffix))
weather = join_df(weather, state_names, "file", "StateName")

weather.head(3)
googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0]

googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]

googletrend.loc[googletrend.State=='NI', "State"] = 'HB,NI'
googletrend.head(3)
def add_datepart(df, fldname, drop=True, time=False):

    "Helper function that adds columns relevant to a date."

    fld = df[fldname]

    fld_dtype = fld.dtype

    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):

        fld_dtype = np.datetime64



    if not np.issubdtype(fld_dtype, np.datetime64):

        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)

    targ_pre = re.sub('[Dd]ate$', '', fldname)

    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',

            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']

    if time: attr = attr + ['Hour', 'Minute', 'Second']

    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())

    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9

    if drop: df.drop(fldname, axis=1, inplace=True)
add_datepart(googletrend,"Date", drop=False)

googletrend.head(3)
# continue with all other tables

add_datepart(weather, "Date", drop=False)

add_datepart(train, "Date", drop=False)

add_datepart(test, "Date", drop=False)
trend_de = googletrend[googletrend.file == 'Rossmann_DE']

trend_de.head(3)
store = join_df(store, store_states, "Store")

len(store[store.State.isnull()])
joined = join_df(train, store, "Store")

joined_test = join_df(test, store, "Store")

len(joined[joined.StoreType.isnull()]), len(joined_test[joined_test.StoreType.isnull()])
# join the joined df with googletrend with ["State","Year","Week"] as the index

# this way the non matching day dates do not create issues.

joined = join_df(joined, googletrend, ["State","Year", "Week"])

joined_test = join_df(joined_test, googletrend, ["State","Year", "Week"])

len(joined[joined.trend.isnull()]),len(joined_test[joined_test.trend.isnull()])
# now join the overal germany trend

joined = joined.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))

joined_test = joined_test.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))

len(joined[joined.trend_DE.isnull()]),len(joined_test[joined_test.trend_DE.isnull()])
# finally join the weather data

joined = join_df(joined, weather, ["State","Date"])

joined_test = join_df(joined_test, weather, ["State","Date"])

len(joined[joined.Mean_TemperatureC.isnull()]),len(joined_test[joined_test.Mean_TemperatureC.isnull()])
# now we can drop duplicated columns

for df in (joined, joined_test):

    for c in df.columns:

        if c.endswith('_y'):

            if c in df.columns: df.drop(c, inplace=True, axis=1)
for df in (joined,joined_test):

    df['CompetitionOpenSinceYear'] = df.CompetitionOpenSinceYear.fillna(1900).astype(np.int32)

    df['CompetitionOpenSinceMonth'] = df.CompetitionOpenSinceMonth.fillna(1).astype(np.int32)

    df['Promo2SinceYear'] = df.Promo2SinceYear.fillna(1900).astype(np.int32)

    df['Promo2SinceWeek'] = df.Promo2SinceWeek.fillna(1).astype(np.int32)
for df in (joined, joined_test):

    df['CompetitionOpenSince'] = pd.to_datetime(dict(year=df.CompetitionOpenSinceYear,

                                                     month=df.CompetitionOpenSinceMonth, 

                                                     day=15))

    df['CompetitionDaysOpen'] = df.Date.subtract(df.CompetitionOpenSince).dt.days
for df in (joined, joined_test):

    df.loc[df.CompetitionDaysOpen<0, "CompetitionDaysOpen"] = 0

    df.loc[df.CompetitionOpenSinceYear<1990, "CompetitionDaysOpen"] = 0
for df in (joined,joined_test):

    df["CompetitionMonthsOpen"] = df["CompetitionDaysOpen"]//30

    df.loc[df.CompetitionMonthsOpen>24, "CompetitionMonthsOpen"] = 24

joined.CompetitionMonthsOpen.unique()
for df in (joined, joined_test):

    df["Promo2Since"] = pd.to_datetime(df.apply(

        lambda x: Week(x.Promo2SinceYear, x.Promo2SinceWeek).monday(), axis=1))

    df["Promo2Days"] = df.Date.subtract(df['Promo2Since']).dt.days
for df in (joined,joined_test):

    df.loc[df.Promo2Days<0, "Promo2Days"] = 0

    df.loc[df.Promo2SinceYear<1990, "Promo2Days"] = 0

    df["Promo2Weeks"] = df["Promo2Days"]//7

    df.loc[df.Promo2Weeks<0, "Promo2Weeks"] = 0

    df.loc[df.Promo2Weeks>25, "Promo2Weeks"] = 25

    df.Promo2Weeks.unique()
#joined.to_pickle(PATH/'joined')

#joined_test.to_pickle(PATH/'joined_test')
def get_elapsed(fld, pre):

    day1 = np.timedelta64(1, 'D')

    last_date = np.datetime64()

    last_store = 0

    res = []

    

    for s,v,d in zip(df.Store.values, df[fld].values, df.Date.values):

        if s != last_store:

            last_date = np.datetime64()

            last_store = s

        if v: last_date = d

        res.append(((d-last_date).astype('timedelta64[D]') / day1))

    df[pre+fld] = res
columns = ["Date", "Store", "Promo", "StateHoliday", "SchoolHoliday"]
df = train[columns].append(test[columns])

df.head(3)
fld = 'SchoolHoliday'

df = df.sort_values(['Store', 'Date'])

get_elapsed(fld, 'After')

df = df.sort_values(['Store', 'Date'], ascending=[True, False])

get_elapsed(fld, 'Before')
fld = 'StateHoliday'

df = df.sort_values(['Store', 'Date'])

get_elapsed(fld, 'After')

df = df.sort_values(['Store', 'Date'], ascending=[True, False])

get_elapsed(fld, 'Before')
fld = 'Promo'

df = df.sort_values(['Store', 'Date'])

get_elapsed(fld, 'After')

df = df.sort_values(['Store', 'Date'], ascending=[True, False])

get_elapsed(fld, 'Before')
df = df.set_index('Date')
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
df = df.merge(bwd, 'left', ['Date', 'Store'], suffixes=['', '_bw'])

df = df.merge(fwd, 'left', ['Date', 'Store'], suffixes=['', '_fw'])
df.drop(columns,1,inplace=True)
df.head()
#df.to_pickle(PATH/'df')
df["Date"] = pd.to_datetime(df.Date)
df.columns
#joined = pd.read_pickle(PATH/'joined')

#joined_test = pd.read_pickle(PATH/f'joined_test')
joined = join_df(joined, df, ['Store', 'Date'])
joined_test = join_df(joined_test, df, ['Store', 'Date'])
joined = joined[joined.Sales!=0]
joined.reset_index(inplace=True)

joined_test.reset_index(inplace=True)
#joined.to_pickle(path/'train_clean')

#joined_test.to_pickle(path/'test_clean')
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
train_df = joined

test_df = joined_test
train_df.head().T
print(test_df.shape)

test_df.head()
n = len(train_df); n
idx = np.random.permutation(range(n))[:2000]

idx.sort()

small_train_df = train_df.iloc[idx[:1000]]

small_test_df = train_df.iloc[idx[1000:]]

small_cont_vars = ['CompetitionDistance','Mean_Humidity']

small_cat_vars = ['Store','DayOfWeek','PromoInterval']

small_train_df = small_train_df[small_cat_vars + small_cont_vars + ['Sales']]

small_test_df = small_test_df[small_cat_vars + small_cont_vars + ['Sales']]
small_train_df.head()
small_test_df.head()
categorify = Categorify(small_cat_vars, small_cont_vars)

categorify(small_train_df)

categorify(small_test_df, test=True)
small_train_df.head()
small_train_df.PromoInterval.cat.categories
# we convert to categories then add 1 to -1 (NaNs) to turn it to zero because you can not look up 1 in an embedding matrix

small_train_df['PromoInterval'].cat.codes[:5]
fill_missing = FillMissing(small_cat_vars, small_cont_vars)

fill_missing(small_train_df)

fill_missing(small_test_df, test=True)
# find any missing values, create a column called "_na" and set it to True any time it is missing

# then replace the empty value with the median of CompetitionDistance because it needs to be a continues varaiable

small_train_df[small_train_df['CompetitionDistance_na'] == True]
len(train_df),len(test_df)
# as seen above, create pre processers fill missing, categorify 

# and normalize (normalize: for any continous var subtract the mean and divide by std)

procs=[FillMissing, Categorify, Normalize]
# name your category variables, keep some continues variables like "day" as cat because

# as a cat var it will create an embedding matrix and the different days of the month will create different behavors

cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen',

    'Promo2Weeks', 'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear',

    'State', 'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw',

    'SchoolHoliday_fw', 'SchoolHoliday_bw']



# name your continues variables

cont_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',

   'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 

   'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',

   'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday']
# dependant var

dep_var = 'Sales'



# the final df to pass in will be the cat_vars, cont_vars, dep_var, and date, date will be used to create the validation set, 

#it will be the same number of records at the end of the time period as the test set from kaggle

df = train_df[cat_vars + cont_vars + [dep_var,'Date']].copy()
df.head()
test_df['Date'].min(), test_df['Date'].max()
cut = train_df['Date'][(train_df['Date'] == train_df['Date'][len(test_df)])].index.max()

cut
valid_idx = range(cut)
# finally, lets look 

df[dep_var].head()
# create databunch

data = (TabularList.from_df(df, path='.', cat_names=cat_vars, cont_names=cont_vars, procs=procs,)

                .split_by_idx(valid_idx)

                .label_from_df(cols=dep_var, label_cls=FloatList, log=True)

                .add_test(TabularList.from_df(test_df, path=path, cat_names=cat_vars, cont_names=cont_vars))

                .databunch())
max_log_y = np.log(np.max(train_df['Sales'])*1.2)

y_range = torch.tensor([0, max_log_y], device=defaults.device)
# Learner

learn = tabular_learner(data, layers=[1000,500], ps=[0.001,0.01], emb_drop=0.04,

                       y_range=y_range, metrics=exp_rmspe)
learn.model
len(data.train_ds.cont_names)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, 1e-3, wd=0.2)
learn.save('1')
learn.recorder.plot_losses(skip_start=10000)
learn.load('1');
learn.fit_one_cycle(5, 3e-4)
test_preds=learn.get_preds(DatasetType.Test)

test_df["Sales"]=np.exp(test_preds[0].data).numpy().T[0]

test_df[["Id","Sales"]]=test_df[["Id","Sales"]].astype("int")

test_df[["Id","Sales"]].to_csv("rossmann_submission.csv",index=False)