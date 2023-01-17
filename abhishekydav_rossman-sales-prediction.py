import sys
!curl -s https://course.fast.ai/setup/colab | bash
!git clone https://github.com/yabhi0807/libml1.git /kaggle/tmp/fastai # This is my repo with all the fastai(updated) libraries 
sys.path.append('/kaggle/tmp/fastai')
!mkdir /kaggle/tmp/data/
!ln -s /kaggle/tmp/* /kaggle/working/
# !wget -O /kaggle/tmp/fastai/weights.tgz http://files.fast.ai/models/weights.tgz
# !tar xvfz /kaggle/tmp/fastai/weights.tgz -C /kaggle/tmp/fastai
%matplotlib inline
%reload_ext autoreload
%autoreload 2
from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)

PATH='/kaggle/tmp/data/rossmann/'
!mkdir /kaggle/tmp/data/rossmann/
!cp -r ../input/rossmansalesdata/* /kaggle/tmp/data/rossmann/
table_names = ['train', 'store', 'store_states', 'state_names', 
               'googletrend', 'weather', 'test']
%time tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]
from IPython.display import HTML, display
for t in tables: display(t.head())
for t in tables: display(DataFrameSummary(t).summary())
train, store, store_states, state_names, googletrend, weather, test = tables
len(train),len(test)
train.StateHoliday = train.StateHoliday!='0'
test.StateHoliday = test.StateHoliday!='0'
def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, suffixes=("", suffix))
weather = join_df(weather, state_names, "file", "StateName")
googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0]
googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]
googletrend.loc[googletrend.State=='NI', "State"] = 'HB,NI'
add_datepart(weather, "Date", drop=False)
add_datepart(googletrend, "Date", drop=False)
add_datepart(train, "Date", drop=False)
add_datepart(test, "Date", drop=False)
trend_de = googletrend[googletrend.file == 'Rossmann_DE']
store = join_df(store, store_states, "Store")
len(store[store.State.isnull()])
joined = join_df(train, store, "Store")
joined_test = join_df(test, store, "Store")
len(joined[joined.StoreType.isnull()]),len(joined_test[joined_test.StoreType.isnull()])
googletrend.dtypes
joined = join_df(joined, googletrend, ["State","Year", "Week"])
joined_test = join_df(joined_test, googletrend, ["State","Year", "Week"])
len(joined[joined.trend.isnull()]),len(joined_test[joined_test.trend.isnull()])
joined = joined.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))
joined_test = joined_test.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))
len(joined[joined.trend_DE.isnull()]),len(joined_test[joined_test.trend_DE.isnull()])
joined = join_df(joined, weather, ["State","Date"])
joined_test = join_df(joined_test, weather, ["State","Date"])
len(joined[joined.Mean_TemperatureC.isnull()]),len(joined_test[joined_test.Mean_TemperatureC.isnull()])
for df in (joined, joined_test):
    for c in df.columns:
        if c.endswith('_y'):
            print(c)
            if c in df.columns: df.drop(c, inplace=True, axis=1)
for df in (joined,joined_test):
    df['CompetitionOpenSinceYear'] = df.CompetitionOpenSinceYear.fillna(1900).astype(np.int32)
    df['CompetitionOpenSinceMonth'] = df.CompetitionOpenSinceMonth.fillna(1).astype(np.int32)
    df['Promo2SinceYear'] = df.Promo2SinceYear.fillna(1900).astype(np.int32)
    df['Promo2SinceWeek'] = df.Promo2SinceWeek.fillna(1).astype(np.int32)
for df in (joined,joined_test):
    df["CompetitionOpenSince"] = pd.to_datetime(dict(year=df.CompetitionOpenSinceYear, 
                                                     month=df.CompetitionOpenSinceMonth, day=15))
    df["CompetitionDaysOpen"] = df.Date.subtract(df.CompetitionOpenSince).dt.days
for df in (joined,joined_test):
    df.loc[df.CompetitionDaysOpen<0, "CompetitionDaysOpen"] = 0
    df.loc[df.CompetitionOpenSinceYear<1990, "CompetitionDaysOpen"] = 0
for df in (joined,joined_test):
    df["CompetitionMonthsOpen"] = df["CompetitionDaysOpen"]//30
    df.loc[df.CompetitionMonthsOpen>24, "CompetitionMonthsOpen"] = 24
joined.CompetitionMonthsOpen.unique()
for df in (joined,joined_test):
    df["Promo2Since"] = pd.to_datetime(df.apply(lambda x: Week(x.Promo2SinceYear, x.Promo2SinceWeek).monday(), axis=1))
    df["Promo2Days"] = df.Date.subtract(df["Promo2Since"]).dt.days
for df in (joined,joined_test):
    df.loc[df.Promo2Days<0, "Promo2Days"] = 0
    df.loc[df.Promo2SinceYear<1990, "Promo2Days"] = 0
    df["Promo2Weeks"] = df["Promo2Days"]//7
    df.loc[df.Promo2Weeks<0, "Promo2Weeks"] = 0
    df.loc[df.Promo2Weeks>25, "Promo2Weeks"] = 25
    df.Promo2Weeks.unique()
joined.to_feather(f'{PATH}joined')
joined_test.to_feather(f'{PATH}joined_test')
def get_elapsed(fld, pre):
    day1 = np.timedelta64(1, 'D')
    last_date = np.datetime64()
    last_store = 0
    res = []

    for s,v,d in zip(df.Store.values,df[fld].values, df.Date.values): # .values changes the series to numpy array and its much faster(300x) than df.iterrows()
        if s != last_store:
            last_date = np.datetime64()
            last_store = s
        if v: last_date = d
        res.append(((d-last_date).astype('timedelta64[D]') / day1).astype(int))
    df[pre+fld] = res
columns = ["Date", "Store", "Promo", "StateHoliday", "SchoolHoliday"]
#df = train[columns]
df = train[columns].append(test[columns]) # along axis = 1
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
df = df.set_index("Date")
columns = ['SchoolHoliday', 'StateHoliday', 'Promo']
for o in ['Before', 'After']:
    for p in columns:
        a = o+p
        df[a] = df[a].fillna(0).astype(int)
bwd = df[['Store']+columns].sort_index(ascending=True).groupby("Store").rolling(7, min_periods=1).sum() # moving average
fwd = df[['Store']+columns].sort_index(ascending=False).groupby("Store").rolling(7, min_periods=1).sum()
bwd
bwd.drop('Store',1,inplace=True)
bwd.reset_index(inplace=True)
fwd.drop('Store',1,inplace=True)
fwd.reset_index(inplace=True)
df.reset_index(inplace=True)
df = df.merge(bwd, 'left', ['Date', 'Store'], suffixes=['', '_bw'])
df = df.merge(fwd, 'left', ['Date', 'Store'], suffixes=['', '_fw'])
df.drop(columns,1,inplace=True)
df.head()
df.to_feather(f'{PATH}df')
df = pd.read_feather(f'{PATH}df')
df["Date"] = pd.to_datetime(df.Date)
df.columns
joined = join_df(joined, df, ['Store', 'Date'])
joined_test = join_df(joined_test, df, ['Store', 'Date'])
joined = joined[joined.Sales!=0]
joined.reset_index(inplace=True)
joined_test.reset_index(inplace=True)
joined.to_feather(f'{PATH}joined')
joined_test.to_feather(f'{PATH}joined_test')
joined = pd.read_feather(f'{PATH}joined')
joined_test = pd.read_feather(f'{PATH}joined_test')
joined.head().T.head(40)
cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen',
    'Promo2Weeks', 'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear',
    'State', 'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw',
    'SchoolHoliday_fw', 'SchoolHoliday_bw']

contin_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
   'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 
   'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
   'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday']

n = len(joined); n
dep = 'Sales'
joined = joined[cat_vars+contin_vars+[dep, 'Date']].copy()
joined_test[dep] = 0
joined_test = joined_test[cat_vars+contin_vars+[dep, 'Date', 'Id']].copy()
for v in cat_vars: joined[v] = joined[v].astype('category').cat.as_ordered()
apply_cats(joined_test, joined)
for v in contin_vars:
    joined[v] = joined[v].fillna(0).astype('float32')
    joined_test[v] = joined_test[v].fillna(0).astype('float32')
idxs = get_cv_idxs(n, val_pct=150000/n)
joined_samp = joined.iloc[idxs].set_index("Date")
samp_size = len(joined_samp); samp_size
samp_size = n
joined_samp = joined.set_index("Date")
joined_samp.head(2)
df, y, nas, mapper = proc_df(joined_samp, 'Sales', do_scale=True)# do_scale=True regularizes the data, (input-mean) * std
yl = np.log(y)
joined_test = joined_test.set_index("Date")
df_test, _, nas, mapper = proc_df(joined_test, 'Sales', do_scale=True, skip_flds=['Id'],
                                  mapper=mapper, na_dict=nas)
df.head(2)
train_ratio = 0.75
# train_ratio = 0.9
train_size = int(samp_size * train_ratio); train_size
val_idx = list(range(train_size, len(df)))
val_idx = np.flatnonzero((df.index<=datetime.datetime(2014,9,17)) & (df.index>=datetime.datetime(2014,8,1)))
val_idx=[0]
def inv_y(a, rf=False): return np.exp(a.cpu() if not rf else a)

def exp_rmspe(y_pred, targ, rf=False):
    if not rf:
        y_pred=y_pred.cpu()
        targ=targ.cpu()
    targ = inv_y(targ, rf)
    pct_var = (targ - inv_y(y_pred, rf))/targ
    return math.sqrt((pct_var**2).mean())

max_log_y = np.max(yl)
y_range = (0, max_log_y*1.2)
md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl.astype(np.float32), cat_flds=cat_vars, bs=128, test_df=df_test)
cat_sz = [(c, len(joined_samp[c].cat.categories)+1) for c in cat_vars]
cat_sz
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
emb_szs
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars), 0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
m.summary()
lr = 1e-3
m.lr_find()
m.sched.plot(100)
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars), 0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3
m.fit(lr, 3, metrics=[exp_rmspe])
m.fit(lr, 5, metrics=[exp_rmspe], cycle_len=1)
m.fit(lr, 2, metrics=[exp_rmspe], cycle_len=4)
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars), 0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3
m.fit(lr, 1, metrics=[exp_rmspe])
m.fit(lr, 3, metrics=[exp_rmspe])
m.fit(lr, 3, metrics=[exp_rmspe], cycle_len=1)
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars), 0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3
m.fit(lr, 3, metrics=[exp_rmspe])
m.fit(lr, 3, metrics=[exp_rmspe], cycle_len=1)
m.save('val0')
m.load('val0')
x,y=m.predict_with_targs()
exp_rmspe(x,y)
pred_test=m.predict(True)
pred_test = np.exp(pred_test)
joined_test['Sales']=pred_test
csv_fn=f'{PATH}tmp/sub.csv'
joined_test[['Id','Sales']].to_csv(csv_fn, index=False)
FileLink(csv_fn)
from sklearn.ensemble import RandomForestRegressor
((val,trn), (y_val,y_trn)) = split_by_idx(val_idx, df.values, yl)
m = RandomForestRegressor(n_estimators=40, max_features=0.99, min_samples_leaf=2, n_jobs=-1, oob_score=True)
m.fit(trn, y_trn);
preds = m.predict(val)
m.score(trn, y_trn), m.score(val, y_val), m.oob_score_, exp_rmspe(preds, y_val, True)
