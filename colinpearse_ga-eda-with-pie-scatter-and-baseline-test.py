import os
import sys
import warnings
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import json
import re
import time
from math import ceil
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

warnings.filterwarnings('ignore')
orig_max_rows = pd.options.display.max_rows
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_colwidth', 90)

do_for_loop_pies = True
do_for_loop_scatters = True

UNDERLINE = '\033[4m'
BOLD = '\033[1m'
END = '\033[0m'

#newdata_dir = "../input"
newdata_dir = "../input/ga-analytics-with-json-columns"
data_dir = '../input/ga-customer-revenue-prediction'
working_dir = "."
#working_dir = newdata_dir
newtrain_path = newdata_dir+"/newtrain.csv"
newtest_path  = newdata_dir+"/newtest.csv"
submission_path  = working_dir+"/test_results_lgb.csv"
!ls -ld $newdata_dir/*.csv
#%%javascript
#IPython.OutputArea.prototype._should_scroll = function(lines) {
#    return false;
#}
#// Kaggle fails on the above lines
newtrain = pd.read_csv(newtrain_path, dtype={'fullVisitorId': 'str', 'visitId': 'str'})
newtest = pd.read_csv(newtest_path, dtype={'fullVisitorId': 'str', 'visitId': 'str'})
def summary(df, info="describe", cols=None):
    if info == "describe":
        headings=['Null','dType','Type','MinMax','Mean','Std','Skew','Unique','Examples']
    elif info == "revenue":
        headings=['Null','nNull','nRevs','nNull%','nRevs%','dType','Type','Unique','Examples']    
        nrevs = df['totals.transactionRevenue'].count()
    else:
        headings=['Null','dType','Type','Unique','Examples']

    if cols == None:
        cols = df.columns

    nrows = df.shape[0]
    if (nrows > orig_max_rows):
        pd.set_option('display.max_rows', nrows)

    print('DataFrame shape',df.shape)
    sdf = pd.DataFrame(index=cols, columns=headings)
    for col in cols:
        sys.stdout.write('.')
        sys.stdout.flush()
        sdf['Null'][col]     = df[col].isna().sum()
        sdf['Unique'][col]   = len(list(df[col].unique()))
        sdf['dType'][col]    = df[col].dtypes
        sdf['Type'][col]     = "-" if df[col].notna().sum() == 0 else type(df[col].dropna().iloc[0])
        sdf['Examples'][col] = "-" if df[col].notna().sum() == 0 else list(df[col].unique())
        if info == "describe":
            if 'float' in str(df[col].dtype) or 'int' in str(df[col].dtype):
                sdf['MinMax'][col] = str(round(df[col].min(),2))+'/'+str(round(df[col].max(),2))
                sdf['Mean'][col]   = df[col].mean()
                sdf['Std'][col]    = df[col].std()
                sdf['Skew'][col]   = df[col].skew()
        elif info == "revenue":
            sdf['nNull'][col] = df[col].count()
            sdf['nRevs'][col] = df.groupby(col)['totals.transactionRevenue'].count().sum()
            sdf['nNull%'][col] = round(sdf['nNull'][col] / (nrows/100), 1)
            sdf['nRevs%'][col] = round(sdf['nRevs'][col] / (nrevs/100), 1)
    return sdf.fillna('-')
def get_unused(df):
    rows = df.shape[0]
    ddf = df.isna().sum()
    return list(ddf[ddf >= rows].index)

droplist = get_unused(newtrain)
for dropcol in get_unused(newtest):
    if dropcol not in droplist:
        droplist.append(dropcol)

print ("dropping these columns from both newtrain and newtest:")
print (droplist)
newtrain.drop(columns=droplist, inplace=True)
newtest.drop(columns=droplist, inplace=True)
# Don't bother shuffling new column around
#def move_last_col(df, col, insert_after):
#    ordercols = list(df.columns[:-1].insert(insert_after, col))
#    return df[ordercols]

def add_visitStartTimeHH(df):
    df['visitStartTimeHH'] = pd.to_datetime(df['visitStartTime'], unit='s').apply(
        lambda x: re.sub(r'^[^ \s]+ (\d\d):\d\d:\d\d',r'\1', str(x))).astype(int)
    return df

def add_dateMm(df):
    df['dateMm'] = df['date'].apply(lambda x: re.sub(r'^\d\d\d\d(\d\d)\d\d',r'\1', str(x))).astype(int)
    return df

#def str2strlen(df, col):
#    df['tmp.'+col] = df[col].str.count(".")
#    df.drop(columns=[col], inplace=True)
#    df.rename(columns={'tmp.'+col: col}, inplace=True)
#    return df
print ("train/test: add visitStartTimeHH")
newtrain = add_visitStartTimeHH(newtrain)
newtest  = add_visitStartTimeHH(newtest)
print ("train/test: add dateMm")
newtrain = add_dateMm(newtrain)
newtest  = add_dateMm(newtest)
#print ("train/test: change unique strings to len(strings) in ...gclId")
#newtrain = str2strlen(newtrain, 'trafficSource.adwordsClickInfo.gclId')
#newtest  = str2strlen(newtest, 'trafficSource.adwordsClickInfo.gclId')
summary(newtrain, info="revenue")
summary(newtest, info="basic")
def mybar_rows(data, labels=[], title="title"):
    drange = list(range(1,len(data)+1))
    plt.bar(drange, data)
    plt.ylabel('Rows', fontsize=15)
    plt.xticks(drange, labels, fontsize=15, rotation=0)
    plt.title(title, fontsize=15);

nrows = newtrain['totals.transactionRevenue'].size
nrevs = newtrain['totals.transactionRevenue'].dropna().size
zrevs = nrows - nrevs

print ("nrows = {:d}; NaN = {:d}; >0 = {:d}; {:1.2f}%".format(nrows,zrevs,nrevs,nrevs/(nrows/100)))
mybar_rows([nrows, zrevs, nrevs], labels=["nrows", "NaN", ">0"], title="totals.transactionRevenue")
def ifsort(df, sorty=True):
    return df.sort_values(ascending=True) if sorty is True else df

def col_by_size(df, col, threshold=0, sorty=True):
    return ifsort(df.fillna({col: -1}).groupby(col).size().apply(lambda x: (x if x>threshold else np.nan)).dropna(), sorty=sorty)

def col_by_col_count(df, col1, col2, threshold=0, sorty=True):
    return ifsort(df.fillna({col1: -1}).groupby(col1)[col2].count().apply(lambda x: (x if x>threshold else np.nan)).dropna(), sorty=sorty)

def col_by_col_sum(df, col1, col2, threshold=0, sorty=True):
    return ifsort(df.fillna({col1: -1}).groupby(col1)[col2].sum(numeric_only=True).apply(lambda x: (x if x>threshold else np.nan)).dropna(), sorty=sorty)

def myautopct(pct):
    return ('%.2f' % pct) if pct > 2 else ''

def get_colors(df):
    colors = []
    for i in df.index:
        s = str(abs(hash(str(i))))
        colors.append( ( float('0.'+s[0]), float('0.'+s[1]), float('0.'+s[2]) ) )
    return colors

def get_labels(df):
    labels = []
    total = df.values.sum()
    for i,v in dict(df).items():
        pct = v / (total/100)
        labels.append( ( i if pct > 1 else '' ) )
    return labels

def mypie(df, title, angle=0, autocol=True, autolab=True):
    colors = get_colors(df) if autocol is True else None
    labels = get_labels(df) if autolab is True else None
    # autopct='%1.1f%%'
    # textprops={'size': 'small'}  (Kaggle python (3.6.6) + libs didn't recognise this)
    df.plot(kind='pie', radius=1.2, startangle=angle, autopct=myautopct, pctdistance=0.8,
        figsize=(5, 5), rotatelabels=False, legend=True, colors=colors, labels=labels, explode=[0.02]*df.size);
    plt.title(title, weight='bold', size=14, x=2.0, y=-0.01);
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(bbox_to_anchor=(2.5, 1.0), ncol=2, fontsize=10, fancybox=True, shadow=True);
    
def make_revenue_pie(df, col, unique=500, minperc=0, threshold=0, angle=0, sorty=True):
    print ('\n\n'+BOLD+UNDERLINE+'Making a pie for',col,"..."+END)
    uentries = df[col].astype(str).unique().size
    percvalid = df[col].notna().sum() / (df[col].size / 100)
    if percvalid < minperc:
        print ("NO PIE because {:s} has only {:1.1f}% non-Null entries (min={:d}%)".format(col, percvalid, minperc))
    elif uentries > unique:
        print ("NO PIE because there are over",unique,"unique",col,"entries")
    elif uentries <= 1:
        print ("NO PIE because all entries are identical")
    else:
        dfrev = col_by_col_count(df, col, 'totals.transactionRevenue', threshold=threshold, sorty=sorty)
        if not dfrev.empty:
            dfsum = col_by_col_sum(df, col, 'totals.transactionRevenue', threshold=threshold, sorty=sorty)
            dfgen = col_by_size(df, col, threshold=threshold, sorty=sorty)
            mypie(dfgen, col+' prevalence',           angle=angle, autocol=True); plt.show()
            mypie(dfrev, col+' by revenue instances', angle=angle, autocol=True); plt.show()
            mypie(dfsum, col+' by revenue sum',       angle=angle, autocol=True); plt.show()
        else:
            print ("NO PIE because",col," does not have a unique entry with more than",threshold,"revenue instances")
print (BOLD+'EXAMPLE:'+END)
make_revenue_pie(newtrain, 'dateMm', unique=1000, threshold=100, angle=0)
#make_revenue_pie(newtrain, 'device.operatingSystem', unique=1000, threshold=100, angle=0, sorty=False)
#make_revenue_pie(newtrain, 'geoNetwork.subContinent', unique=1000, threshold=100, angle=0)
#make_revenue_pie(newtrain, 'geoNetwork.region', unique=1000, threshold=100, angle=0)
#make_revenue_pie(newtrain, 'socialEngagementType', unique=1000, threshold=100, angle=0)
#make_revenue_pie(newtrain, 'visitStartTimeHH', unique=1000, threshold=100, angle=0)
if do_for_loop_pies:
    for col in newtrain.columns:
        make_revenue_pie(newtrain, col, unique=1000, threshold=100, angle=0)
else:
    print ("skipping pie chart for loop")
def truncate(avals, tlen):
    tstrs = []
    tstrs[:] = (re.sub(r'(.{'+str(tlen)+'})(.*)', r'\1......', str(val)) for val in avals)
    return tstrs

def get_xticks(df, maxtick=39):
    rows = df.shape[0]
    xti = list(range(0, rows, ceil(rows/maxtick)))
    #xtl = list(df.index)[0::ceil(rows/maxtick)]
    xtl = truncate(list(df.index)[0::ceil(rows/maxtick)], 30)
    if xtl[-1] != df.index[-1]:
        xti.append(rows)
        xtl.append(df.index[-1])
    return xti, xtl

def myscat(df, col1, col2, log1p=True, sorty=True, threshold=0):
    ylabel = "ln("+col2+")"      if log1p is True else col2
    ytmp   = np.log1p(df.values) if log1p is True else df.values
    y      = np.sort(ytmp)       if sorty is True else ytmp
    x      = range(0, len(y))
    #print ("NOTE:",len(y),"unique entries found where each value >",threshold,"(threshold) ; meaning",len(y),"data points for both x and y")
    plt.figure(figsize=(15,5))
    plt.scatter(x, y)
    plt.xlabel(col1, fontsize=12)
    plt.ylabel(ylabel+" > "+str(threshold), fontsize=12)
    xti, xtl = get_xticks(df)
    plt.xticks(xti, xtl, rotation=90)
    plt.show()

def myscat_size(df, col, log1p=True, threshold=0, sorty=True):
    dfsize = col_by_size(newtrain, col, threshold=threshold, sorty=sorty)
    myscat(dfsize, col, 'number of rows', log1p=log1p, sorty=sorty, threshold=threshold)

def myscat_countrev(df, col, log1p=True, threshold=0, sorty=True):
    dfcount = col_by_col_count(newtrain, col, 'totals.transactionRevenue', threshold=threshold, sorty=sorty)
    myscat(dfcount, col, 'totals.transactionRevenue', log1p=log1p, sorty=sorty, threshold=threshold)

def myscat_sumrev(df, col, log1p=True, threshold=0, sorty=True):
    dfsum = col_by_col_sum(newtrain, col, 'totals.transactionRevenue', threshold=threshold, sorty=sorty)
    myscat(dfsum, col, 'totals.transactionRevenue', log1p=log1p, sorty=sorty, threshold=threshold)

def scatter_revenue(df, col, unique=100, threshold=0, sorty=True):
    print ('\n\n'+BOLD+UNDERLINE+'Scatter graphs for',col,"(a) prevalence (b) by revenue instances (c) by revenue sum ("+('sorted' if sorty is True else 'unsorted')+")"+END)
    uentries = df[col].dropna().astype(str).unique().size
    if uentries < unique:
        print ("NO SCATTER because there are under",unique,"unique",col,"entries")
    else:
        myscat_size    (newtrain, col, log1p=True, threshold=threshold, sorty=sorty)
        myscat_countrev(newtrain, col, log1p=True, threshold=threshold, sorty=sorty)
        myscat_sumrev  (newtrain, col, log1p=True, threshold=threshold, sorty=sorty)
print (BOLD+'EXAMPLE:'+END)
scatter_revenue(newtrain, 'totals.hits', unique=0, threshold=0, sorty=False)
#scatter_revenue(newtrain, 'fullVisitorId', unique=0, threshold=0, sorty=True)
if do_for_loop_scatters:
    for col in newtrain.columns:
        sorty = True if newtrain[col].dtypes == 'object' else False
        scatter_revenue(newtrain, col, unique=100, threshold=0, sorty=sorty)
else:
    print ("skipping scatter graph for loop")
def check_unique_values(df1, df2, col):
    l1 = list(df1[col].dropna().astype(str).unique()) 
    l2 = list(df2[col].dropna().astype(str).unique()) 
    return list(set(l1).intersection(l2))

def check_unique_values_with_revenue(df1, df2, col):
    l1 = col_by_col_count(df1, col, 'totals.transactionRevenue', threshold=0, sorty=False).index
    l2 = list(df2[col].dropna().astype(str).unique()) 
    return list(set(l1).intersection(l2))

trows = newtest.shape[0]
print ("rows in newtest:", trows)
for col in ['fullVisitorId']:
    ul1 = check_unique_values(newtrain, newtest, col)
    ul2 = check_unique_values_with_revenue(newtrain, newtest, col)
    print ("{:4d} common entries ({:1.2f}% of newtest), {:4d} rev>0 ({:1.2f}% of 1st value) for {:s}".format(
        len(ul1), len(ul1)/(trows/100), len(ul2), len(ul2)/(len(ul1)/100), col))
useful_test_cols = [
    'fullVisitorId',
    'date',
    'dateMm',
    'visitStartTimeHH',
    'channelGrouping',
    'visitNumber',
    'device.browser',
    'device.deviceCategory',
    'device.isMobile',
    'device.operatingSystem',
    'geoNetwork.city',
    'geoNetwork.continent',
    'geoNetwork.country',
    'geoNetwork.networkDomain',
    'geoNetwork.region',
    'geoNetwork.subContinent',
    'geoNetwork.metro',
    'totals.hits',
    'totals.pageviews',
    'totals.bounces',
    'totals.newVisits',
    'trafficSource.keyword',
    'trafficSource.referralPath',
    'trafficSource.medium',
    'trafficSource.source',
    'trafficSource.isTrueDirect',
    ]

# totals.transactionRevenue will be removed later from train and validation data
useful_train_cols = list(set(useful_test_cols + ['totals.transactionRevenue']))
    
# still not sure about these, play around with them later on...
# 'trafficSource.adwordsClickInfo.gclId',
# 'trafficSource.adContent',
# 'trafficSource.adwordsClickInfo.adNetworkType',
# 'trafficSource.adwordsClickInfo.isVideoAd',
# 'trafficSource.adwordsClickInfo.page',
# 'trafficSource.adwordsClickInfo.slot',
# 'trafficSource.campaign',
def split_data(df, cutoff=None, test_size=0.1, shuffle=True):
    if cutoff is None:
        tx, vx, ty, vy = train_test_split(df[useful_train_cols],
                                          df['totals.transactionRevenue'],
                                          test_size=test_size, shuffle=shuffle)
        ty = ty.fillna(0).values
        vy = vy.fillna(0).values
    else:
        tx = df[useful_train_cols].loc[(newtrain['date'] <  cutoff)]
        vx = df[useful_train_cols].loc[(newtrain['date'] >= cutoff)]
        ty = tx['totals.transactionRevenue'].fillna(0).values
        vy = vx['totals.transactionRevenue'].fillna(0).values
    return tx, vx, ty, vy

#trainx, validx, trainy, validy = split_data(newtrain, cutoff=20170601)
trainx, validx, trainy, validy = split_data(newtrain, cutoff=None, test_size=0.1, shuffle=False)

testx     = newtest[useful_test_cols]
validvids = validx['fullVisitorId'].values
testvids  = newtest['fullVisitorId'].values
tnrows = trainx['totals.transactionRevenue'].size
vnrows = validx['totals.transactionRevenue'].size
tnrevs = trainx['totals.transactionRevenue'].dropna().size
vnrevs = validx['totals.transactionRevenue'].dropna().size
tzrevs = tnrows - tnrevs
vzrevs = vnrows - vnrevs

print ("trainx: nrows = {:6d}; NaN = {:4d}; >0 = {:4d}; {:1.2f}%".format(tnrows,tzrevs,tnrevs,tnrevs/(tnrows/100)))
print ("validx: nrows = {:6d}; NaN = {:4d}; >0 = {:4d}; {:1.2f}%".format(vnrows,vzrevs,vnrevs,vnrevs/(vnrows/100)))
mybar_rows([tnrows, tzrevs, tnrevs, vnrows, vzrevs, vnrevs], labels=["tnrows", "tNaN", "t>0", "vnrows", "vNaN", "v>0"], title="REVENUE: train vs validation")
# not managing to get the row number using panda functions; I need this if I use shuffle=True above
def get_row_number(arr, value):
    row = 0
    for v in arr:
        if v == value:
            return row
        else:
            row = row + 1

def get_testrow(df, dfx, arry, col, value):
    dfnames = sys._getframe(1).f_code.co_names[1:4]  # [0] is the function name
    i = df.index[(df[col] == value)].tolist()[0]
    r = get_row_number(dfx[col], value)
    print ("using revenue value = {:d} to check trainx/trainy".format(value))
    print ("{:8s}: value found at index {:d}, which is actually row {:d}".format(dfnames[0], i, r))
    print ("{:8s}: value found at index {:6d} is: {:1.0f}".format(dfnames[1], i, dfx.loc[i][col]))
    print ("{:8s}: value found at row   {:6d} is: {:1.0f}".format(dfnames[1], r, dfx.iloc[r][col]))
    print ("{:8s}: value found at row   {:6d} is: {:1.0f}".format(dfnames[2], r, arry[r]))
    assert dfx.loc[i][col] == arry[r]
    assert dfx.iloc[r][col] == arry[r]

get_testrow(newtrain, trainx, trainy, 'totals.transactionRevenue', 306670000)
trainx.drop(columns=['totals.transactionRevenue'], inplace=True)
validx.drop(columns=['totals.transactionRevenue'], inplace=True)
assert set(trainx.columns) == set(validx.columns)
assert set(trainx.columns) == set(testx.columns)
summary(trainx, info="basic")
def get_unique_categories(df, dfx, dft):
    aunique = {}
    for col in df.columns:
        aunique[col] = list(set(list(df[col].astype(str).unique()) +
                                list(dfx[col].astype(str).unique()) +
                                list(dft[col].astype(str).unique())))
        aunique[col].sort()
    return aunique

def encode_columns(df, ucats, onehot=50):
    sys.stdout.write("encoding "+sys._getframe(1).f_code.co_names[1])  # [0] is function name
    le = LabelEncoder()
    for col,coltype in df.dtypes.items():
        sys.stdout.write('.')
        sys.stdout.flush()
        if coltype == 'object':
            if (len(ucats[col]) > onehot):
                le.fit(ucats[col])
                df[col] = le.transform(df[col].astype('str'))
    df = pd.get_dummies(df)

    for col,coltype in df.dtypes.items():
        if coltype != 'float':
            df[col] = df[col].astype(float)

    print ("returned new shape",df.shape)
    return df
unique_categories = get_unique_categories(trainx, validx, testx)

trainxe = encode_columns(trainx.copy(), unique_categories, onehot=0)
validxe = encode_columns(validx.copy(), unique_categories, onehot=0)
testxe  = encode_columns(testx.copy(), unique_categories, onehot=0)

trainxe,validxe = trainxe.align(validxe, join='inner', axis=1)
trainxe,testxe  = trainxe.align(testxe, join='inner', axis=1)
trainxe,validxe = trainxe.align(validxe, join='inner', axis=1)
print ("trainxe shape after align", trainxe.shape)
print ("validxe shape after align", validxe.shape)
print ("testxe shape after align", testxe.shape)
assert set(trainxe.columns) == set(validxe.columns)
assert set(trainxe.columns) == set(testxe.columns)
assert set(trainxe.dtypes) == set(validxe.dtypes)
assert set(trainxe.dtypes) == set(testxe.dtypes)
summary(trainxe, info="basic")
def train_nn(trainx, trainy, validx, validy, testx):
    params = {
        "objective"         : "regression",
        "metric"            : "rmse", 
        "num_leaves"        : 30,
        "min_child_samples" : 100,
        "learning_rate"     : 0.1,
        "bagging_fraction"  : 0.7,
        "feature_fraction"  : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed"      : 2018,
        "verbosity"         : -1
    }
    
    trainlgb = lgb.Dataset(trainxe, label=trainy)
    validlgb = lgb.Dataset(validxe, label=validy)
    return lgb.train(params, trainlgb, 5000, valid_sets=[validlgb], early_stopping_rounds=100, verbose_eval=100)

nn = train_nn(trainxe, np.log1p(trainy), validxe, np.log1p(validy), testxe)
predy = nn.predict(testxe, num_iteration=nn.best_iteration)
predvy = nn.predict(validxe, num_iteration=nn.best_iteration)
def rmse_log1p(df, col1, col2):
    return np.sqrt(metrics.mean_squared_error(np.log1p(df[col1].values), np.log1p(df[col2].values)))

predvy[predvy<0] = 0
validation = pd.DataFrame({'fullVisitorId':             validvids,
                           'totals.transactionRevenue': validy,
                           'totals.predictedRevenue':   np.expm1(predvy),
                           'totals.allZeros':           [0] * len(validy) })
                           #'totals.allZeros':       np.random.randint(0, 2, (1, len(validy)))[0] })
                           #'totals.allZeros':       np.random.random((1, len(validy)))[0] })

print("RMSE score where prediction always 0: ", rmse_log1p(validation, 'totals.transactionRevenue', 'totals.allZeros'))
print("RMSE score for validation (ungrouped):", rmse_log1p(validation, 'totals.transactionRevenue', 'totals.predictedRevenue'))
validation = validation.groupby('fullVisitorId')['totals.transactionRevenue','totals.predictedRevenue'].sum().reset_index()
print("RMSE score for validation (grouped):  ", rmse_log1p(validation, 'totals.transactionRevenue', 'totals.predictedRevenue'))
predy[predy<0] = 0
submission = pd.DataFrame({'fullVisitorId':       testvids,
                           'PredictedLogRevenue': np.expm1(predy)})
submission = submission.groupby('fullVisitorId')['PredictedLogRevenue'].sum().reset_index()
assert 617242 == submission['fullVisitorId'].nunique()  # Kaggle informs us that there can only be 617242 lines

submission["PredictedLogRevenue"] = np.log1p(submission["PredictedLogRevenue"])
submission.to_csv(submission_path, index=False)
