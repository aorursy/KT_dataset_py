# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.tabular.transform import add_cyclic_datepart

import collections
import sys
import gc


import pickle # for saving and loading processed datasets and hyperparameters


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
PATH = '/kaggle/input/recsys-challenge-2015' 
def read_buys(limit=None):
    print("Loading buys")
    buys = pd.read_csv(f"{PATH}/yoochoose-buys.dat",
                    names=["session", "timestamp", "item", "price", "qty"],
                   parse_dates=["timestamp"])
    buys = buys.sort_values(by=["timestamp", "session"])
    if limit:
        buys = buys.iloc[:limit]
    print("Buys shape %s %s" % buys.shape)
    return buys

def export_df(X, filename='X'):
    try:
        with open('{}.pickle'.format(filename), 'wb') as fout:
            pickle.dump(X, fout)
        print('Preprocessed dataframe exported')
    except FileNotFoundError:
        print('File not found')
        
def find_item(item_bought_pct, x):
    if x in item_bought_pct:
        return item_bought_pct[x]
    else:
        return 0
def df_preprocessing(clicks_df,buys):
    
    
    types = {"category":'int64'}

    clicks_df = clicks_df.astype(types)    

#     clicks_cat = clicks_df.category.to_numpy()

#     count_cats = collections.Counter(clicks_cat)

    cats = np.array(clicks_df['category']) 
    
    print("Generating features from category... ")
    # Defining columns for new features based on "category"
    features_cols = []
    catsN_dict = dict()

    # The value "S" (encoded 999) indicates a special offer. Creating new binary column.
    spec_offer_array = np.where(cats==999,1,0).astype(int)
    catsN_dict.update({'spec_offer_array': spec_offer_array})

    # "0" indicates  a missing value. Creating new binary column for missing category.
    miss_cat_array = np.where(cats==0,1,0).astype(int)
    catsN_dict.update({'miss_cat_array': miss_cat_array})

    # number between 1 to 12 indicates a real category identifier. 
    #Creating new binary columns stored in dict for each real category. 

    catsN_names = list()

    for i in range(1,13):
        name  = 'cat'+str(i)+'_array'
        new_arr = np.where(cats==i,1,0).astype(int)
        catsN_dict.update({name: new_arr})
        catsN_names.append(name)

    features_cols += catsN_names

    # any other number indicates a brand. 

    brand_cat_array = np.where((cats > 12) & (cats != 999),1,0).astype(int)
    catsN_dict.update({'brand_cat_array': brand_cat_array})

    for key, value in catsN_dict.items():
        clicks_df[key] = value
        
    print("Generating features from category... Done ")
    print(f"Clicks_df size is {sys.getsizeof(clicks_df)}")
    

        # Releasing memory
    session_df = clicks_df.copy()
    del clicks_df
    gc.collect()
    
    
    session_df.drop(['item','category'],axis=1,inplace=True)
    
    print("Making final aggregation... ")
    # Final aggregation
    session_df = session_df.groupby('session').agg(
    
        start_ts=('timestamp','min'),
        end_ts=('timestamp','max'),

        spec_offer=('spec_offer_array','max'),
        cat1=('cat1_array','max'),
        cat2=('cat2_array','max'),
        cat3=('cat3_array','max'),
        cat4=('cat4_array','max'),
        cat5=('cat5_array','max'),
        cat6=('cat6_array','max'),
        cat7=('cat7_array','max'),
        cat8=('cat8_array','max'),
        cat9=('cat9_array','max'),
        cat10=('cat10_array','max'),
        cat11=('cat11_array','max'),
        cat12=('cat12_array','max'),

        brand_cat=('brand_cat_array','max'),
        miss_cat=('miss_cat_array','max'),

        cat_freq_mean=('cat_freq','mean'),
        cat_freq_min=('cat_freq','min'),
        cat_freq_max=('cat_freq','max'), 

        clicks_count_sum=('clicks_count','sum'),
        clicks_count_mean=('clicks_count','mean'),

        items_count_sum=('items_count','sum'),
        items_count_mean=('items_count','mean'),

        cat_count_sum=('cat_count','sum'),
        cat_count_mean=('cat_count','mean'),
        )
    
    print("Making final aggregation... Done")
    
    print("Calculating duration and click rate... ")
    
    session_df["total_duration_secs"] = (session_df["end_ts"] - session_df["start_ts"]).dt.seconds
    session_df["click_rate"] = session_df["clicks_count_sum"] / session_df["total_duration_secs"]
    session_df.click_rate = session_df.click_rate.replace(np.inf, np.nan)
    session_df.click_rate = session_df.click_rate.fillna(0)
    print("Calculating duration and click rate... Done")
    
    print("Adding cyclic dateparts... ")
    session_df = add_cyclic_datepart(session_df, "start_ts", drop=False)
    session_df = add_cyclic_datepart(session_df, "end_ts", drop=False)
    print("Adding cyclic dateparts... Done")
    
    session_df.drop(['start_ts', 'end_ts'], 1, inplace=True) #
        
    # Adding a destination feature - y
    session_df["buy"] = session_df.index.isin(buys["session"]).astype(int)
    
    
    return session_df
%timeit

filename = f"{PATH}/yoochoose-clicks.dat"

raw_df = pd.read_csv(filename,
                     names=["session", "timestamp", "item", "category"],
                    converters={"category": lambda c: 999 if c == "S" else c},
                     parse_dates=["timestamp"], nrows=100000, skiprows = 1000000)

print("Calculating frequencies and counts... ")
    

# Calculating overall category frequency
cat_freq = raw_df['category'].value_counts(normalize=True)
raw_df["cat_freq"] = raw_df["category"].map(lambda x: cat_freq[x])

# Count clicks per session
clicks_count = raw_df.groupby('session')['item'].count()
raw_df["clicks_count"] = raw_df["session"].map(lambda x: clicks_count[x])

# Count items per session
items_count = raw_df.groupby('session')['item'].nunique()
raw_df["items_count"] = raw_df["session"].map(lambda x: items_count[x])

# Count categories per session
cat_count = raw_df.groupby('session')['category'].nunique()
raw_df["cat_count"] = raw_df["session"].map(lambda x: cat_count[x])

print("Calculating frequencies and counts... Done")


buys = read_buys()

session_df = df_preprocessing(raw_df, buys)


session_df.head()
df = pd.DataFrame()

def load_saved_dataset(filename):
    try:
        with open('../input/recsys-preprocessed/{}.pickle'.format(filename), 'rb') as fin:
            Y = pickle.load(fin)
        print('Dataset loaded')
    except FileNotFoundError:
        print('File with saved dataset not found')
    return Y


filename = 'Processed_recsys'
raw_df = load_saved_dataset(filename)

raw_df.buy.value_counts(normalize=True)
raw_df.describe()
raw_df.info()