import os
import sys
import warnings
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
import time

warnings.filterwarnings('ignore')
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_colwidth', 90)

def set_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path

newdata_dir = "."
#newdata_dir = "../input/ga-analytics-with-json-columns"
data_dir1   = "../input/ga-customer-revenue-prediction"
data_dir2   = "../input"
newtrain_path = newdata_dir+"/newtrain.csv"
newtest_path  = newdata_dir+"/newtest.csv"
train_path    = set_path([data_dir1+"/train.csv", data_dir2+"/train.csv"])
test_path     = set_path([data_dir1+"/test.csv",  data_dir2+"/test.csv"])

!ls -ld $newdata_dir/*.csv $data_dir1/*.csv $data_dir2/*.csv
def load_new_or_orig(newpath, path):
    new = None
    orig = None
    if os.path.exists(newpath):
        new = pd.read_csv(newpath, dtype={'fullVisitorId': 'str', 'visitId': 'str'})
        print ("loaded",newpath)
    elif os.path.exists(path):
        orig = pd.read_csv(path, dtype={'fullVisitorId': 'str', 'visitId': 'str'})
        print ("loaded",path)
    else:
        print ("ERROR: loaded nothing")
    return new, orig

newtrain, train = load_new_or_orig(newtrain_path, train_path)
newtest,  test  = load_new_or_orig(newtest_path, test_path)
def summary(df, info="describe"):
    if info == "describe":
        headings=['Null','Unique','dType','Type','MinMax','Mean','Std','Skew','Examples']
    else:
        headings=['Null','Unique','dType','Type','Examples']
        
    print('DataFrame shape',df.shape)
    sdf = pd.DataFrame(index=df.columns, columns=headings)
    for col in df.columns:
        sdf['Null'][col]     = df[col].isna().sum()
        sdf['Unique'][col]   = df[col].astype(str).unique().size
        sdf['dType'][col]    = df[col].dtype
        sdf['Type'][col]     = "-" if df[col].notna().sum() == 0 else type(df[col].dropna().iloc[0])
        sdf['Examples'][col] = "-" if df[col].notna().sum() == 0 else df[col].astype(str).unique() #.dropna().values
        if info == "describe":
            if 'float' in str(df[col].dtype) or 'int' in str(df[col].dtype):
                sdf['MinMax'][col] = str(round(df[col].min(),2))+'/'+str(round(df[col].max(),2))
                sdf['Mean'][col]   = df[col].mean()
                sdf['Std'][col]    = df[col].std()
                sdf['Skew'][col]   = df[col].skew()
    return sdf.fillna('-')


def is_json(j):
    if re.match(r'^{\"', j):
        return True
    else:
        return False

def get_json(df, col):
    if is_json(df[col][0]) == False:
        return None
    jdf_lines = df[col].apply(json.loads)   # do normalize separately or it will use just one column
    jdf = pd.io.json.json_normalize(jdf_lines).add_prefix(col+'.')
    for jcol in jdf.columns:
        jdf[jcol].replace('not available in demo dataset', np.nan, inplace=True, regex=True)
        jdf[jcol].replace('(not provided)', np.nan, inplace=True, regex=True)
        jdf[jcol].replace('(not set)', np.nan, inplace=True, regex=True)
    return jdf

def expand_json(df):
    newdf = pd.concat([df, get_json(df, 'device'),
                           get_json(df, 'geoNetwork'),
                           get_json(df, 'totals'),
                           get_json(df, 'trafficSource')], axis=1, sort=False)
    newdf.drop(columns=['device', 'geoNetwork', 'totals', 'trafficSource'], inplace=True)
    return newdf
# Adhoc validation
#get_json(train, 'trafficSource')
#get_json(train, 'fullVisitorId')
#summary(train, info="basic")
#summary(get_json(train, 'device'),       info="basic")
#summary(get_json(train, 'geoNetwork'),    info="basic")
#summary(get_json(train, 'totals'),        info="basic")
#summary(get_json(train, 'trafficSource'), info="basic")
def expand_json_to_df(newdf, df):
    newname = sys._getframe(1).f_code.co_names[1]  # [0] is the function name
    if newdf is None:
        print (time.ctime(), newname, "will contain expanded json entries")
        newdf = expand_json(df)
        print (time.ctime(), newname, "finished: shape =", newdf.shape, "vs original shape =", df.shape)
    else:
        print (time.ctime(), newname, "already loaded")
    return newdf

newtrain = expand_json_to_df(newtrain, train)
newtest  = expand_json_to_df(newtest, test)
summary(newtrain, info="basic")
summary(newtest, info="basic")
# newtrain.csv is only 250MB (from 1.5GB) because we've removed the json repetition
# newtest.csv is only 230MB (from 1.3GB) because we've removed the json repetition

if os.access(newdata_dir, os.W_OK):
    if not os.path.exists(newtrain_path):
        newtrain.to_csv(newtrain_path, index=False, encoding="utf-8")
        print ("wrote", newtrain_path)
        newtrain = pd.read_csv(newtrain_path)
        print ("reloaded newtrain")
    if not os.path.exists(newtest_path):
        newtest.to_csv(newtest_path, index=False, encoding="utf-8")
        print ("wrote", newtest_path)
        newtest = pd.read_csv(newtest_path)
        print ("reloaded newtest")
else:
    print (newdata_dir, "is not writable")
!echo Quick validation of rows:
!sed -n '$=' $newtrain_path
!sed -n '$=' $train_path
!sed -n '$=' $newtest_path
!sed -n '$=' $test_path
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
summary(newtrain, info="basic")
summary(newtest, info="basic")
zrows = newtrain['totals.transactionRevenue'].size
nrows = newtrain['totals.transactionRevenue'].dropna().size
print ("NaN =", zrows, "; >0 =", nrows)
#plt.bar([1,2], [zrows, nrows])
#plt.ylabel('Rows', fontsize=15)
#plt.xticks([1,2], ["NaN", ">0"], fontsize=15, rotation=0)
#plt.title("totals.transactionRevenue", fontsize=15);
def col_by_col_count(df, col1, col2, threshold=0):
    return df.groupby([col1]).count()[col2].apply(lambda x: (x if x>threshold else np.nan)).dropna()

def col_by_col_sum(df, col1, col2, threshold=0):
    return df.groupby([col1]).sum(numeric_only=True)[col2].apply(lambda x: (x if x>threshold else np.nan)).dropna()

def myautopct(pct):
    return ('%.2f' % pct) if pct > 2 else ''

def mypie(df, title, angle=0):
    # autopct='%1.1f%%'
    # textprops={'size': 'small'}  (Kaggle python (3.6.6) + libs didn't recognise this)
    df.plot(kind='pie', figsize=(5, 5), radius=1.2, startangle=angle, autopct=myautopct, pctdistance=0.8,
        rotatelabels=False, legend=True, explode=[0.02]*df.size);
    plt.title(title, weight='bold', size=14, x=2.0, y=-0.01);
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(bbox_to_anchor=(2.5, 1.0), ncol=2, fontsize=10, fancybox=True, shadow=True);
newtrain['device.operatingSystem'].astype(str).unique()
df = col_by_col_count(newtrain, 'device.operatingSystem', 'sessionId', threshold=100)
mypie(df, 'OS prevalence', angle=100)
df = col_by_col_count(newtrain, 'device.operatingSystem', 'totals.transactionRevenue', threshold=100)
mypie(df, 'OS prevalence by revenue instances', angle=0)
df = col_by_col_sum(newtrain, 'device.operatingSystem', 'totals.transactionRevenue', threshold=100)
mypie(df, 'OS prevalence by revenue sum', angle=10)
