#!pip -q install --upgrade --ignore-installed numpy pandas scipy sklearn
#!pip -q install catboost
#!pip -q install lightgbm
#!pip -q install "dask[complete]"
#!pip -q install "dask-ml[complete]"
## https://stackoverflow.com/questions/49853303/how-to-install-pydot-graphviz-on-google-colab?rq=1
#!pip -q install graphviz 
#!apt-get install graphviz -qq
#!pip -q install pydot
# After this restart your kernel using Ctrl+M+. to reset all loaded libraries to latest ones, that we just installed.
import pandas as pd
import numpy as np
from multiprocessing import Pool
from pandas.io.json import json_normalize
from sklearn.pipeline import make_pipeline, Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from threading import Thread as trd
import queue
import json
import gc
gc.enable()
PATH = "../input/ga-customer-revenue-prediction/"
part = pd.read_csv(PATH+"train_v2.csv",dtype={"fullVisitorId":"str", "visitId":"str"}, nrows=10)
part.shape
columns = part.columns
part.head(2)
# From above output you will have a basic understanding of type of columns. Divide them so you
# can use same function on similar columns (for exploration etc).
id_columns = ["fullVisitorId", "visitId", "visitStartTime"]
num_columns = ["visitNumber"]
obj_columns = ["channelGrouping", "socialEngagementType", "date"]
dict_columns = [ "device", "geoNetwork", "totals", "trafficSource"]
complex_columns = ["customDimensions", "hits"]
col = "device"
df = pd.read_csv(PATH+"train_v2.csv", usecols = [3], converters={col: json.loads})

column_as_df = json_normalize(df[col])
column_as_df.head()
# Explore this column. Check if you want to keep it or not.
# If you want to scale this column or LableEncode this column, 
# do it and maintain a dictionary with all columns as keys and
# corresponding Scalar or LabelEncoder or both as values.
# Calculate Standard Deviation or check number of uniques, check if they are constant...
column_as_df.nunique()
drop_cols = [col for col in column_as_df.columns if col not in ['browser', 'deviceCategory', 'isMobile', 'operatingSystem']]
column_as_df.drop(drop_cols, axis=1, inplace=True)
column_as_df.head()
final_selected_columns = {
    "device": ["browser", "deviceCategory", "isMobile", "operatingSystem"],
    "geoNetwork":["city", "continent", "country", "metro", "networkDomain", "region", "subContinent"],
    "totals":["transactions", "sessionQualityDim", "hits", "pageviews", "totalTransactionRevenue", "timeOnSite", "transactionRevenue"],
    "trafficSource":["adContent", "adwordsClickInfo.adNetworkType", "adwordsClickInfo.page", "adwordsClickInfo.slot",
                     "campaign", "keyword",	"medium", "referralPath", "source"],
    "customDimensions": ["customDimensions"],
    "channelGrouping": ["channelGrouping"],
    "date":["date"],
    "visitNumber":["visitNumber"],
    "visitStartTime":["visitStartTime"]
}
columns_selected = []
for key in final_selected_columns.keys():
    if len(final_selected_columns[key]) == 1:
        columns_selected.append(key)
    else:
        for sub_col in final_selected_columns[key]:
            columns_selected.append(f"{key}.{sub_col}")
columns_selected
prep_pipeline = {col: [] for col in columns_selected}
for col_ in ["browser", "deviceCategory", "operatingSystem"]:
    lb = LabelEncoder()
    column_as_df[col_] = lb.fit_transform(column_as_df[col_].values)
    prep_pipeline[f"device.{col_}"].append(lb)
for col_ in ["browser", "deviceCategory", "operatingSystem"]:
    rsc = RobustScaler()
    column_as_df[col_] = rsc.fit_transform(column_as_df[col_].values.reshape(-1, 1)).reshape(1, -1)[0]
    prep_pipeline[f"device.{col_}"].append(rsc)
column_as_df.head()
import pickle
with open("pipeline.pickle", "wb") as fle:
    pickle.dump(prep_pipeline, fle)
!ls
del df, column_as_df
gc.collect()
PATH2 = "../input/googleanalbigfile/"
!ls {PATH2}
with open(PATH2+"preprocessing_pipeline (1).pickle", "rb") as fle:
    preprocessing_pipeline = pickle.load(fle)
final_selected_columns = {
    "device": ["browser", "deviceCategory", "isMobile", "operatingSystem"],
    "geoNetwork":["city", "continent", "country", "metro", "networkDomain", "region", "subContinent"],
    "totals":["transactions", "sessionQualityDim", "hits", "pageviews", "totalTransactionRevenue", "timeOnSite", "transactionRevenue"],
    "trafficSource":["adContent", "adwordsClickInfo.adNetworkType", "adwordsClickInfo.page", "adwordsClickInfo.slot",
                     "campaign", "keyword",	"medium", "referralPath", "source"],
    "customDimensions": ["customDimensions"],
    "channelGrouping": ["channelGrouping"],
    "date":["date"],
    "visitNumber":["visitNumber"],
    "visitStartTime":["visitStartTime"]
}
def browser_mapping(x):
    browsers = ['chrome','safari','firefox','explorer','edge','opera','coc coc','maxthon','iron']
    for br in browsers: 
      if br in x.lower(): return br
    if  ('android' in x) or ('samsung' in x) or ('mini' in x) or ('iphone' in x) or ('in-app' in x) or ('playstation' in x): return 'mobile browser'
    elif  ('mozilla' in x) or ('chrome' in x) or ('blackberry' in x) or ('nokia' in x) or ('browser' in x) or ('amazon' in x): return 'mobile browser'
    elif  ('lunascape' in x) or ('netscape' in x) or ('blackberry' in x) or ('konqueror' in x) or ('puffin' in x) or ('amazon' in x): return 'mobile browser'
    elif '(not set)' in x: return '(not set)'
    else: return 'others'

def adcontents_mapping(x):
    if 'google' in x: return 'google'
    elif ('placement' in x) | ('placememnt' in x): return 'placement'
    elif '(not set)' in x: return '(not set)'
    elif 'na' == x: return 'na'
    elif 'ad' in x: return 'ad'
    else: return 'others'
    
def keyword_mapping(x):
  if '(not provided)' in x: return '(not provided)'
  elif 'na' == x: return 'na'
  elif 'google' in x: return 'google'
  elif 'shart' in x: return 't-shirt'
  elif 'sticker' in x: return 'sticker'
  elif 'lav' in x: return 'lava-lamp'
  elif 'pen' in x: return 'pen'
  elif 'merc' in x: return 'merchandise'
  elif 'bag' in x: return 'bagpack'
  elif 'laptop' in x: return 'laptop'
  elif 'onesie' in x: return 'onesie'
  elif 'sunglass' in x: return 'sunglasses'
  elif 'cup' in x: return 'cup'
  elif 'cap' in x: return 'cap'
  elif 'arrel' in x: return 'apparel'
  elif 'arel' in x: return 'apparel'
  elif 'shirt' in x: return 't-shirt'
  elif 'jac' in x: return 'jacket'
  elif 'bank' in x: return 'powerbank'
  elif 'dre' in x: return 'dress'
  elif 'todd' in x: return 'toddler'
  elif 'bic' in x: return 'bicycle'
  elif 'earp' in x: return 'earphone'
  elif 'head' in x: return 'headphones'
  elif 'virt' in x: return 'virtual-reality'
  elif 'reali' in x: return 'virtual-reality'
  elif 'bott' in x: return 'bottle'
  elif 'book' in x: return 'notebook'
  elif 'men' in x: return 'men'
  elif 'women' in x: return 'women'
  elif 'wom' in x: return 'women'
  elif 'goo' in x: return 'google'
  elif 'gle' in x: return 'google'
  elif 'you' in x: return 'youtube'
  elif 'yu' in x: return 'youtube'
  elif 'yi' in x: return 'youtube'
  elif 'ube' in x: return 'youtube'
  elif 'tb' in x: return 'youtube'
  else: return 'others'
  
def referralPath_mapping(x):
  if 'intl' in x: return 'intl'
  elif 'yt' in x: return 'yt'
  elif 'golang' in x: return 'golang'
  elif 'document' in x: return 'document'
  elif '/r/' in x: return 'reddit'
  elif 'google' in x: return 'google'
  elif '/ads' in x: return 'ads'
  elif 'ads/' in x: return 'ads'
  elif 'adsense/' in x: return 'ads'
  elif 'mail' in x: return 'mail'
  elif 'hangouts' in x: return 'hangouts'
  elif 'iphone' in x: return 'iphone'
  elif 'webapps/' in x: return 'webapp'
  elif 'maps/' in x: return 'maps'
  elif 'webmasters/' in x: return 'webapp'
  elif 'android/' in x: return 'android'
  elif 'websearch/' in x: return 'websearch'
  elif 'youtube/' in x: return 'youtube'
  elif 'spreadsheets/' in x: return 'spreadsheets'
  elif 'presentation/' in x: return 'presentation'
  elif 'chromebook/' in x: return 'chromebook'
  elif 'bookmarks/' in x: return 'bookmark'
  elif 'drive/' in x: return 'drive'
  elif 'baiduid' in x: return 'baidu'
  elif 'calendar/' in x: return 'calendar'
  elif 'pin/' in x: return 'pinterest'
  elif 'entry/' in x: return 'entry'
  elif 'chrome/' in x: return 'chrome'
  elif '.html' in x: return 'html'
  elif '.pdf' in x: return 'pdf'
  elif 'wiki/' in x: return 'wiki'
  elif 'appserve/' in x: return 'webapp'
  elif 'web/' in x: return 'web'
  elif 'edit/' in x: return 'edit'
  elif 'feed/' in x: return 'feed'
  elif 'feedback' in x: return 'feedback'
  elif 'user/' in x: return 'user'
  elif '.htm' in x: return 'htm'
  elif '.php' in x: return 'php'
  elif 'class_section' in x: return 'class_section'
  elif 'messages/' in x: return 'message'
  elif 'na' == x: return 'na'
  else: return 'others'
  
def source_mapping(x):
    if 'google' in x:return 'google'
    elif 'youtube' in x:return 'youtube'
    elif 'yahoo' in x:return 'yahoo'
    elif 'facebook' in x:return 'facebook'
    elif 'reddit' in x:return 'reddit'
    elif 'bing' in x:return 'bing'
    elif 'quora' in x:return 'quora'
    elif 'outlook' in x:return 'outlook'
    elif 'linkedin' in x:return 'linkedin'
    elif 'pinterest' in x:return 'pinterest'
    elif 'ask' in x:return 'ask'
    elif 'siliconvalley' in x:return 'siliconvalley'
    elif 'lunametrics' in x:return 'lunametrics'
    elif 'amazon' in x:return 'amazon'
    elif 'mysearch' in x:return 'mysearch'
    elif 'qiita' in x:return 'qiita'
    elif 'messenger' in x:return 'messenger'
    elif 'twitter' in x:return 'twitter'
    elif 't.co' in x:return 't.co'
    elif 'vk.com' in x:return 'vk.com'
    elif 'search' in x:return 'search'
    elif 'edu' in x:return 'edu'
    elif 'mail' in x:return 'mail'
    elif 'ad' in x:return 'ad'
    elif 'golang' in x:return 'golang'
    elif 'direct' in x:return 'direct'
    elif 'dealspotr' in x:return 'dealspotr'
    elif 'sashihara' in x:return 'sashihara'
    elif 'phandroid' in x:return 'phandroid'
    elif 'baidu' in x:return 'baidu'
    elif 'mdn' in x:return 'mdn'
    elif 'duckduckgo' in x:return 'duckduckgo'
    elif 'seroundtable' in x:return 'seroundtable'
    elif 'metrics' in x:return 'metrics'
    elif 'sogou' in x:return 'sogou'
    elif 'businessinsider' in x:return 'businessinsider'
    elif 'github' in x:return 'github'
    elif 'gophergala' in x:return 'gophergala'
    elif 'yandex' in x:return 'yandex'
    elif 'msn' in x:return 'msn'
    elif 'dfa' in x:return 'dfa'
    elif 'feedly' in x:return 'feedly'
    elif 'arstechnica' in x:return 'arstechnica'
    elif 'squishable' in x:return 'squishable'
    elif 'flipboard' in x:return 'flipboard'
    elif 't-online.de' in x:return 't-online.de'
    elif 'sm.cn' in x:return 'sm.cn'
    elif 'wow' in x:return 'wow'
    elif 'baidu' in x:return 'baidu'
    elif 'partners' in x:return 'partners'
    elif '(not set)' in x: return "(not set)"
    elif 'nan' in x: return 'na'
    else: return 'others'
dict_columns = [ "device", "geoNetwork", "totals", "trafficSource"]
id_columns = ["fullVisitorId", "visitId"]
num_columns = ['totals.transactions', 'totals.sessionQualityDim', 'totals.hits',
               'totals.pageviews', 'totals.totalTransactionRevenue', 'totals.timeOnSite', 
              'trafficSource.adwordsClickInfo.page', 'totals.transactionRevenue', 'visitNumber']
process_columns = {'device.browser': browser_mapping, 
                   'trafficSource.adContent': adcontents_mapping, 
                   'trafficSource.source': source_mapping,
                   'trafficSource.keyword': keyword_mapping, 
                   'trafficSource.referralPath': referralPath_mapping}
drop_columns = ["hits", "socialEngagementType"]
all_columns = columns_selected
# Now that you have a dictionary which has all the columns you need as keys
# and corresponding methods you want to apply to that column sequentially,
# you can make a preprocesing method which will be used to clean data at
# every incremental step.

def preprocess(df):
  df.reset_index(drop=True, inplace=True)
  
  for col_ in drop_columns:
    if col_ in df.columns: df = df.drop([col_], axis=1)
  
  for col in dict_columns:
    col_df = json_normalize(df[col])
    col_df.columns = [f"{col}.{subcolumn}" for subcolumn in col_df.columns]
    
    to_drop = [temp for temp in col_df.columns if temp not in all_columns]
    col_df = col_df.drop(to_drop, axis=1)
    
    df = df.drop([col], axis=1).merge(col_df, right_index=True, left_index=True)
    
    for temp_col in col_df.columns:
      if temp_col in num_columns: df[temp_col] = df[temp_col].fillna(0.0).astype(float)
      elif temp_col == "device.isMobile": df[temp_col] = df[temp_col].astype(bool)
      else: df[temp_col] = df[temp_col].fillna('NA').astype(str)
        
      if temp_col in process_columns.keys(): df[temp_col] = df[temp_col].map(lambda x: process_columns[temp_col](str(x).lower())).astype('str')        
      
      if len(preprocessing_pipeline[temp_col]) >= 4:
        for i in range(4):
          if preprocessing_pipeline[temp_col][i] is None: continue
          elif i == 0: 
            lb = preprocessing_pipeline[temp_col][i]
            df[temp_col] = lb.transform(df[temp_col])
          elif i == 1:
            df[temp_col] = np.log1p(df[temp_col])
          elif i == 3:
            pass
            #rsc = preprocessing_pipeline[temp_col][i]
            #df[temp_col] = rsc.transform(df[temp_col].values.reshape(-1, 1)).reshape(1, -1)[0]
      elif len(preprocessing_pipeline[temp_col]) == 3:
        for i in range(3):
          if preprocessing_pipeline[temp_col][i] is None: continue
          elif i == 0: 
            lb = preprocessing_pipeline[temp_col][i]
            df[temp_col] = lb.transform(df[temp_col])
          elif i == 2:
            pass
            #rsc = preprocessing_pipeline[temp_col][i]
            #df[temp_col] = rsc.transform(df[temp_col].values.reshape(-1, 1)).reshape(1, -1)[0]
            
  df['visitNumber'] = df['visitNumber'].fillna(0.0).astype(float)
  df['channelGrouping'] = df['channelGrouping'].fillna('NA').astype(str)
  df['customDimensions'] = df['customDimensions'].fillna('NA').astype(str)
  
  for col_ in ["visitNumber", "customDimensions", "channelGrouping"]:
    for i in range(3):
      if preprocessing_pipeline[col_][i] is None: continue
      elif i==0:
        lb = preprocessing_pipeline[col_][i]
        df[col_] = lb.transform(df[col_])
      elif i == 2:
        pass
        #rsc = preprocessing_pipeline[col_][i]
        #df[col_] = rsc.transform(df[col_].values.reshape(-1, 1)).reshape(1, -1)[0]
  
  df['date_new'] = pd.to_datetime(df['visitStartTime'], unit='s')
  df.drop(['visitStartTime'], axis=1, inplace=True)
  df['sess_date_dow'] = df['date_new'].dt.dayofweek
  df['sess_date_hours'] = df['date_new'].dt.hour
  df['sess_month'] = df['date_new'].dt.month
  df['sess_year'] = df['date_new'].dt.year
  
  for col_ in ['totals.transactions', 'totals.totalTransactionRevenue']:
    if col_ in df.columns: df = df.drop([col_], axis=1)
  
  return df
import lightgbm as lgb
import xgboost as xgb
import catboost as cb # CatBoost is currently making its incremental learner: https://github.com/catboost/catboost/issues/464
dict_columns = [ "device", "geoNetwork", "totals", "trafficSource"]
id_columns = ["fullVisitorId", "visitId"]
incremental_dataframe = pd.read_csv(PATH + "train_v2.csv",
                                   dtype={"fullVisitorId":"str", "visitId":"str"},
                                   converters = {col_: json.loads for col_ in dict_columns},
                                   chunksize=100000) # Number of lines to read.
# This method will return a sequential file reader reading 'chunksize' lines every time.
# To read file from starting again, you will have to call this method again.
lgb_params = {
  "objective" : "regression",
  "n_estimators":3000,
  "max_depth": 4,
  "metric" : "rmse",
  "learning_rate" : 0.01,
  'subsample':.8,
  'colsample_bytree':.9
}
# First three are necessary for incremental learning.
xgb_params = {
  'update':'refresh',
  'process_type': 'update',
  'refresh_leaf': True,
  'silent': True,
  }
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids
drop_cols =['date', 'visitId', 'date_new', 'fullVisitorId', 'totals.transactionRevenue']
# For saving regressor for next use.
lgb_estimator = None

i=0
prev=None
for df in incremental_dataframe:
  print(i)
  df = preprocess(df)
  print("Preprocessed.")
  #folds = get_folds(df=df, n_splits=5)
  #for fold_, (trn_, val_) in enumerate(folds):
  #    trn_x, trn_y = df.drop(drop_cols, axis=1).iloc[trn_], df['totals.transactionRevenue'].iloc[trn_]
  #    val_x, val_y = df.drop(drop_cols, axis=1).iloc[val_], df['totals.transactionRevenue'].iloc[val_]
  if i==0:
    prev = df
    print("First prev")
    i += 1
    continue
  trn_x, trn_y = prev.drop(drop_cols, axis=1), prev['totals.transactionRevenue']
  val_x, val_y = df.drop(drop_cols, axis=1), df['totals.transactionRevenue']
  print("Starting to train...")
  lgb_estimator = lgb.train(lgb_params,
                         init_model=lgb_estimator, # Pass partially trained model
                         train_set=lgb.Dataset(trn_x, trn_y),
                         valid_sets=[lgb.Dataset(val_x, val_y)],
                         valid_names=["Valid"],
                         early_stopping_rounds = 50,
                         keep_training_booster=True, # For incremental learning
                         num_boost_round=70,
                         verbose_eval=50) # Output after each of 50th time
  print("Continuing...")
  prev = df
  del df, trn_x, trn_y, val_x, val_y
  gc.collect()
  i += 1

#trn_x, trn_y = prev.(drop_cols, axis=1), prev['totals.transactionRevenue']
#lgb_estimator = lgb.train(lgb_params,
#                         init_model=lgb_estimator, # Pass partially trained model
#                         train_set=lgb.Dataset(trn_x, trn_y),
#                         early_stopping_rounds = 50,
#                         keep_training_booster=True, # For incremental learning
#                         num_boost_round=70,
#                         verbose_eval=50) # Output after each of 50th time
del prev
gc.collect()
incremental_test = pd.read_csv(PATH + "test_v2.csv",
                                   dtype={"fullVisitorId":"str", "visitId":"str"},
                                   converters = {col_: json.loads for col_ in dict_columns},
                                   chunksize=100000)
test_df = pd.DataFrame()
for df in incremental_test:
  df = preprocess(df)
  test_df = pd.concat([test_df, df], axis=0, sort=False).reset_index(drop=True)

del df
gc.collect()
test_df.head()
from sklearn.metrics import mean_squared_error as mse
preds = lgb_estimator.predict(test_df.drop(drop_cols, axis=1))
#preds = preprocessing_pipeline['totals.transactionRevenue'][3].inverse_transform(preds.reshape(-1, 1)).reshape(1, -1)[0]
preds[preds<0] = 0

true = test_df['totals.transactionRevenue']
#true = preprocessing_pipeline['totals.transactionRevenue'][3].inverse_transform(true.values.reshape(-1, 1)).reshape(1, -1)[0]

mse(true, preds) ** 0.5
del preds, true, lgb_estimator, test_df
gc.collect()
%reset -f
import dask
import dask.dataframe as dd
from dask.distributed import Client
client = Client(processes=False, threads_per_worker=4, n_workers=4, memory_limit='4GB')
client
dict_cols = ["device", "geoNetwork", "totals", "trafficSource"]
df = dd.read_csv(PATH+"train_v2.csv",
                dtype={'fullVisitorId': 'str','date': 'str', 
                    **{c: 'str' for c in dict_cols}
                },
                parse_dates=['date'], blocksize=1e9)
df.npartitions
#df.visualize(size="15,15!")
df.head()
dict_cols = ["device", "geoNetwork", "totals", "trafficSource"]
drop_columns = ["hits", "socialEngagementType"]
# Convert string Series to dictionary Series
for col_ in dict_cols:
    df[col_] = df[col_].apply(lambda x: eval(x.replace('false', 'False')
                                                    .replace('true', 'True')
                                                    .replace('null', 'np.nan')), meta=('', 'object'))
final_selected_columns = {
    "device": ["browser", "deviceCategory", "isMobile", "operatingSystem"],
    "geoNetwork":["city", "continent", "country", "metro", "networkDomain", "region", "subContinent"],
    "totals":["sessionQualityDim", "hits", "pageviews", "timeOnSite", "transactionRevenue"],
    "trafficSource":["adContent", "adwordsClickInfo.adNetworkType", "adwordsClickInfo.page", "adwordsClickInfo.slot",
                     "campaign", "keyword",	"medium", "referralPath", "source"],
    "customDimensions": ["customDimensions"],
    "channelGrouping": ["channelGrouping"],
    "date":["date"],
    "visitNumber":["visitNumber"],
    "visitStartTime":["visitStartTime"]
}
gc.collect()
non_str_cols = ['isMobile', 'isTrueDirect', 'pageviews', 'hits', 'bounces', 'newVisits',
               'transactionRevenue', 'visits', 'timeOnSite','sessionQualityDim']
default = 0
for col_ in dict_cols:
    for key in final_selected_columns[col_]:
        if key in non_str_cols: 
            default = 0
            df[f'{col_}.{key}'] = df[col_].to_bag().pluck(key, default=default).to_dataframe().iloc[:,0]
            df[f'{col_}.{key}'] = df[f'{col_}.{key}'].astype(int)
        else: 
            default = "NaN"
            df[f'{col_}.{key}'] = df[col_].to_bag().pluck(key, default=default).to_dataframe().iloc[:,0]
            df[f'{col_}.{key}'] = df[f'{col_}.{key}'].astype(str)
    del df[col_]
    gc.collect()
df = df.drop(drop_columns, axis=1)
df.head()
#df.visualize(size="20,10!")
df.isnull().sum().compute()
# Necessary for converting dataframe to array. Takes specified length from each block.
lengths = []
for part in df.partitions:
  l = part.shape[0].compute()
  lengths.append(l)
  #print(l, part.shape[1])
drop_cols =['date', 'visitId', 'fullVisitorId', 'totals.transactionRevenue']
df.head()
lcols = ["customDimensions", "channelGrouping", "visitStartTime", "device.browser", "device.deviceCategory",
        "device.operatingSystem", "geoNetwork.city", "geoNetwork.continent", "geoNetwork.metro", "geoNetwork.networkDomain",
        "geoNetwork.region", "geoNetwork.subContinent", "trafficSource.adContent", "trafficSource.adwordsClickInfo.adNetworkType", 
         "trafficSource.adwordsClickInfo.page", "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
        "trafficSource.keyword", "trafficSource.medium", "trafficSource.referralPath", "trafficSource.source"]
lcols_index = []
for col_ in lcols:
    lcols_index.append(np.where(col_ == df.columns)[0][0])
# You cannot assign an array to a column of DataFrame, it has to be a Series.
# So, we are going to convert our data to arrays, scale and learn from array only.
X, y = df.drop(drop_cols, axis=1).to_dask_array(lengths=lengths) , df['totals.transactionRevenue'].to_dask_array(lengths=lengths)
del df
gc.collect()
#from dask_ml.preprocessing import RobustScaler, LabelEncoder
#Xo = dask.array.zeros((X.shape[0],1), chunks=(200000,1))
#for i in range(X.shape[1]):
#    if i in lcols_index:
#        lb = LabelEncoder()
#        temp = lb.fit_transform(X[:i])
#        Xo = dask.array.concatenate([Xo, temp], axis=1)
#X = Xo[:, 1:]
#Xo = dask.array.zeros((X.shape[0],1), chunks=(200000,1))
#for i in range(len(X.shape[1])):
#  if i == X.shape[1]-1:
#    rsc = RobustScaler()
#    y = rsc.fit_transform(y.reshape(-1, 1)).reshape(1, -1)[0]
#  else:
#    rsc = RobustScaler()
#    temp = rsc.fit_transform(X[:,i].reshape(-1, 1))
#    Xo = dask.array.concatenate([Xo, temp], axis=1)
del X
gc.collect()
#Xo = Xo[:, 1:]
#Xo = Xo.rechunk({1: Xo.shape[1]})
#Xo = Xo.rechunk({0: 200000})
#y = y.rechunk({0: 200000})
#tr_len = 0.8*Xo.shape[0]
#xtrain, ytrain = Xo[:tr_len], y[:tr_len]
#xvalid, yvalid = Xo[tr_len:], y[tr_len:]
#xtrain.shape, ytrain.shape, xvalid.shape, yvalid.shape
#from dask_ml.linear_model import LinearRegression
#est = LinearRegression()
#est.fit(xtrain, y=ytrain)
#preds = est.predict(xvalid)
#preds[0:10].compute()