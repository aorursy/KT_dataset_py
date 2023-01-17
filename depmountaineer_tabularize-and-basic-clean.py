import pandas as pd
x=pd;print(x.__name__, "version:", x.__version__)
import numpy as np
x=np;print(x.__name__, "version:", x.__version__)
import matplotlib
import matplotlib.pyplot as plt
x=matplotlib;print(x.__name__, "version:", x.__version__)
import seaborn as sns
x=sns;print(x.__name__, "version:", x.__version__)
from sklearn.model_selection import train_test_split
from sklearn import linear_model, kernel_ridge, cluster, model_selection
import os, sys, math, datetime, shutil, pickle, itertools, json
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" # this causes all lines of a notebook cell to show output, not just last line
%matplotlib inline
pd.set_option('display.max_columns', 100) #default of 10 columns is just too small to see much
pd.set_option('display.max_rows', 10)
plt.style.use('seaborn-poster') #makes the graphs bigger; important for my 4K monitor
sns.set(palette='deep')
sns.set(context='poster')
input_prefix = '../input/'
val_prefix = './val_'
output_prefix = './'

id = "fullVisitorId"
df_train = pd.read_csv(input_prefix + "train.csv", index_col=id, dtype={id:str})
df_test = pd.read_csv(input_prefix + "test.csv", index_col=id, dtype={id:str})
json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']#from inspection of data

other_cols = list(set(df_train.columns).union(set(df_test.columns)) - set(json_cols))# all columns except json columns
for col in other_cols:
    if len(set(df_train[col].unique()).union(set(df_test[col].unique()))) <= 1:# if all values are the same value, the column is not needed
        val = df_train[col].iloc[0]
        del df_train[col]
        del df_test[col]
        print("Removing trivial column: %r with unique value: %r" % (col, val))
#just for reproduceability
np.random.seed(2718281828)

columns = json_cols #start processing with the json_columns known by inspection
while len(columns) > 0: #while there are json columns left to process
    additional_json_cols = []#if a newly-generated column is itself a json column (from nesting), we need to process it the same way
    for col in columns:
        print("Getting keys for column:", col)
        print("  Train column from string to dict:", col)
        d = df_train[col].apply(json.loads)#get the dicts of all entries (strings) of this column
        print("  Test column from string to dict:", col)
        t = df_test[col].apply(json.loads)

        print("  Union of %s train col keys:" % len(d), col)
        keys = {x for x in d[0]}# try to find all keys, considering that keys may be missing in some rows
        if col == 'totals':
            #make sure we don't miss this one!  It is the target.
            keys.add('transactionRevenue')
            #add keys found in previous run
            keys = keys.union({'newVisits', 'bounces', 'transactionRevenue', 'pageviews', 'visits', 'hits'})
        elif col == 'device':
            #add keys found in previous run
            keys = keys.union({'mobileInputSelector', 'mobileDeviceMarketingName', 'browserVersion', 'mobileDeviceModel', 'flashVersion', 'language', 'screenResolution', 'operatingSystem', 'mobileDeviceBranding', 
                               'browser', 'browserSize', 'isMobile', 'deviceCategory', 'operatingSystemVersion', 'mobileDeviceInfo', 'screenColors'})
        elif col == 'geoNetwork':
            #add keys found in previous run
            keys = keys.union({'cityId', 'longitude', 'networkLocation', 'latitude', 'country', 'metro', 'networkDomain', 'region', 'city', 'continent', 'subContinent'})
        elif col == 'trafficSource':
            #add keys found in previous run
            keys = keys.union({'isTrueDirect', 'medium', 'source', 'adwordsClickInfo', 'campaign', 'adContent', 'keyword', 'referralPath'})
        elif col == 'trafficSource_adwordsClickInfo':
            #add keys found in previous run
            keys = keys.union({'slot', 'gclId', 'targetingCriteria', 'isVideoAd', 'criteriaParameters', 'page', 'adNetworkType'})

        #random samples to make sure we got all the keys
        samples = np.random.randint(1, len(d) - 1, 1000)#increase this to ensure getting all keys
        if col == 'totals' or col == 'trafficSource' or col == 'trafficSource_adwordsClickInfo':
            samples = np.random.randint(1, len(d) - 1, 5000)#increase this to ensure getting all keys
        count = 0
        for i in samples:
            count += 1
            if count % 20000 == 0:
                print("    rows processed:", i)
            if len(keys) != len(d[i]):#may have missing keys to add
                old_keys = keys
                keys = keys.union({x for x in d[i]})
                if len(keys - old_keys) > 0:
                    print("%s new keys added" % len(keys - old_keys))
            else:
                for x in d[i]:
                    if x not in keys:#definitely have missing keys to add
                        old_keys = keys
                        keys = keys.union({x for x in d[i]})
                        print("%s new keys added" % len(keys - old_keys))
                        break

        print("  Union of %s test col keys:" % len(t), col)
        count = 0
        for i in np.random.randint(1, len(t) - 1, 100):#increase this to ensure getting all keys
            count += 1
            if count % 20000 == 0:
                print("    rows processed:", i)
            if len(keys) != len(t[i]):
                old_keys = keys
                keys = keys.union({x for x in t[i]})
                if len(keys - old_keys) > 0:
                    print("%s new keys added" % len(keys - old_keys))
            else:
                for x in t[i]:
                    if x not in keys:
                        old_keys = keys
                        keys = keys.union({x for x in t[i]})
                        print("%s new keys added" % len(keys - old_keys))
                        break

        print("Found %s keys for train/test column %s; replacing with keyed columns; keys = %r" % (len(keys), col, keys))
        keylist = list(keys)
        keylist.sort() #for reproduceability
        
        for key in keylist:#now, add new fields of type column_key and delete the original column.  For missing values, use 'XNA' string; this will be replaced with something better on a column-by-column basis in the next script
            new_field = col + "_" + key
            df_train[new_field] = d.apply(lambda x:x.get(key, 'XNA'))#use 'XNA' string for missing values
            df_test[new_field] = t.apply(lambda x:x.get(key, 'XNA'))

            #some json structures have extra recursive depth; make value into a string and re-process
            try:
                l = len(set(df_train[new_field].unique()).union(set(df_test[new_field].unique())))# l==1 means column is trivial
            except TypeError: # most likely: found a dict as a value, which "unique" doesn't like, so need to recurse
                def to_str(x):
                    if isinstance(x, dict):
                        return json.dumps(x)
                    else:
                        return json.dumps(dict())
                print("Recursively processing new column:", new_field)
                df_train[new_field] = df_train[new_field].apply(to_str)#convert to string first
                df_test[new_field] = df_test[new_field].apply(to_str)
                additional_json_cols.append(new_field)#add to columns that need to be processed again
                l = len(set(df_train[new_field].unique()).union(set(df_test[new_field].unique())))

            #make sure the column isn't trivial (just one constant value)
            if l <= 1:
                val = df_train[new_field].iloc[0]
                del df_train[new_field]
                del df_test[new_field]
                if new_field in additional_json_cols:
                    additional_json_cols.remove(new_field)
                print("Removing trivial column: %r with unique value: %r" % (new_field, val))
        del df_train[col]
        del df_test[col]
    columns = additional_json_cols
#check for equivalent columns 

equivalent_columns = set()
for i in range(len(df_train.columns)):
    for j in range(i):
        col_i = df_train.columns[i]
        col_j = df_train.columns[j]
        if (df_train[col_i].factorize()[0] == df_train[col_j].factorize()[0]).all():
            found = False
            for x in equivalent_columns:
                if col_i in x or col_j in x:
                    x.add(col_i)
                    x.add(col_j)
                    found = True
                    break
            if not found:
                equivalent_columns.add({col_i, col_j})

equivalent_columns#possibly-equivalent columns
df_train.to_csv(output_prefix + "train_clean.csv.zip", compression='zip')
df_test.to_csv(output_prefix + "test_clean.csv.zip", compression='zip')
split = int(df_train[['visitStartTime']].describe().loc['75%', 'visitStartTime'])
df_valtrain = df_train[df_train['visitStartTime'] < split]
df_valtest = df_train[df_train['visitStartTime'] >= split]
df_valtrain.to_csv(val_prefix + "train_clean.csv.zip", compression='zip')
df_valtest.to_csv(val_prefix + "test_clean.csv.zip", compression='zip')