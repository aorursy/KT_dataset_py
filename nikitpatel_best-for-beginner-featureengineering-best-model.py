# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import gc
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
import category_encoders as ce
import lightgbm as lgb
from catboost import CatBoostClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def dprint(*args, **kwargs):
    print("[{}] ".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + \
        " ".join(map(str,args)), **kwargs)

id_name = 'Id'
target_name = 'Target'


# Load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train['is_test'] = 0
test['is_test'] = 1
df_all = pd.concat([train, test], axis=0)

dprint('Clean features...')
cols = ['dependency']
for c in tqdm(cols):
    x = df_all[c].values
    strs = []
    for i, v in enumerate(x):
        try:
            val = float(v)
        except:
            strs.append(v)
            val = np.nan
        x[i] = val
    strs = np.unique(strs)

    for s in strs:
        df_all[c + '_' + s] = df_all[c].apply(lambda x: 1 if x == s else 0)

    df_all[c] = x
    df_all[c] = df_all[c].astype(float)
dprint("Done.")
dprint("Extracting features...")
def extract_features(df):
    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']
    df['rent_to_rooms'] = df['v2a1']/df['rooms']
    df['rent_to_bedrooms'] = df['v2a1']/df['bedrooms']
    df['tamhog_to_rooms'] = df['tamhog']/df['rooms'] # tamhog - size of the household
    df['tamhog_to_bedrooms'] = df['tamhog']/df['bedrooms']
    df['r4t3_to_tamhog'] = df['r4t3']/df['tamhog'] # r4t3 - Total persons in the household
    df['r4t3_to_rooms'] = df['r4t3']/df['rooms'] # r4t3 - Total persons in the household
    df['r4t3_to_bedrooms'] = df['r4t3']/df['bedrooms']
    df['rent_to_r4t3'] = df['v2a1']/df['r4t3']
    df['v2a1_to_r4t3'] = df['v2a1']/(df['r4t3'] - df['r4t1'])
    df['hhsize_to_rooms'] = df['hhsize']/df['rooms']
    df['hhsize_to_bedrooms'] = df['hhsize']/df['bedrooms']
    df['rent_to_hhsize'] = df['v2a1']/df['hhsize']
    df['qmobilephone_to_r4t3'] = df['qmobilephone']/df['r4t3']
    df['qmobilephone_to_v18q1'] = df['qmobilephone']/df['v18q1']
    

extract_features(train)
extract_features(test)
dprint("Done.")         
from sklearn.preprocessing import LabelEncoder

def encode_data(df):
   
    yes_no_map = {'no': 0, 'yes': 1}
    
    df['dependency'] = df['dependency'].replace(yes_no_map).astype(np.float32)
    
    df['edjefe'] = df['edjefe'].replace(yes_no_map).astype(np.float32)
    df['edjefa'] = df['edjefa'].replace(yes_no_map).astype(np.float32)
    
    df['idhogar'] = LabelEncoder().fit_transform(df['idhogar'])

dprint("Encoding Data....")
encode_data(train)
encode_data(test)
dprint("Done...")
def do_features(df):
    feats_div = [('children_fraction', 'r4t1', 'r4t3'), 
                 ('working_man_fraction', 'r4h2', 'r4t3'),
                 ('all_man_fraction', 'r4h3', 'r4t3'),
                 ('human_density', 'tamviv', 'rooms'),
                 ('human_bed_density', 'tamviv', 'bedrooms'),
                 ('rent_per_person', 'v2a1', 'r4t3'),
                 ('rent_per_room', 'v2a1', 'rooms'),
                 ('mobile_density', 'qmobilephone', 'r4t3'),
                 ('tablet_density', 'v18q1', 'r4t3'),
                 ('mobile_adult_density', 'qmobilephone', 'r4t2'),
                 ('tablet_adult_density', 'v18q1', 'r4t2'),
                 #('', '', ''),
                ]
    
    feats_sub = [('people_not_living', 'tamhog', 'tamviv'),
                 ('people_weird_stat', 'tamhog', 'r4t3')]

    for f_new, f1, f2 in feats_div:
        df['fe_' + f_new] = (df[f1] / df[f2]).astype(np.float32)       
    for f_new, f1, f2 in feats_sub:
        df['fe_' + f_new] = (df[f1] - df[f2]).astype(np.float32)
    
    # aggregation rules over household
    aggs_num = {'age': ['min', 'max', 'mean'],
                'escolari': ['min', 'max', 'mean']
               }
    aggs_cat = {'dis': ['mean']}
    for s_ in ['estadocivil', 'parentesco', 'instlevel']:
        for f_ in [f_ for f_ in df.columns if f_.startswith(s_)]:
            aggs_cat[f_] = ['mean', 'count']
    # aggregation over household
    for name_, df_ in [('18', df.query('age >= 18'))]:
        df_agg = df_.groupby('idhogar').agg({**aggs_num, **aggs_cat}).astype(np.float32)
        df_agg.columns = pd.Index(['agg' + name_ + '_' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
        df = df.join(df_agg, how='left', on='idhogar')
        del df_agg
    # do something advanced above...
    
    # Drop SQB variables, as they are just squres of other vars 
    df.drop([f_ for f_ in df.columns if f_.startswith('SQB') or f_ == 'agesq'], axis=1, inplace=True)
    # Drop id's
    df.drop(['Id', 'idhogar'], axis=1, inplace=True)
    # Drop repeated columns
    df.drop(['hhsize', 'female', 'area2'], axis=1, inplace=True)
    return df
    
dprint("Do_feature Engineering....")
train = do_features(train)
test = do_features(test)
dprint("Done....")
dprint("Fill Na value....")
train = train.fillna(0)
test = test.fillna(0)
dprint("Done....")
train.shape,test.shape
cols_to_drop = [
    id_name, 
    target_name,
]
X = train.drop(cols_to_drop, axis=1, errors='ignore')
y = train[target_name].values

X.shape,test.shape
import lightgbm as lgb
gc.collect()
def get_lgb_model():
    lgb_model = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.2, objective='multiclass',
                             silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced',
                            colsample_bytree= 0.93, 
                            min_child_samples= 56, num_leaves= 19, 
                                   subsample= 0.84) 
    return lgb_model
dprint("Prepared Model.....")
lgb_model = get_lgb_model()
lgb_model.fit(X, y)
dprint("Done Model.....")
dprint("Predict Test Value.....")
target_lgb = lgb_model.predict(test)
dprint("Done Prediction.....")
# def cat_model():
#     cat_model = CatBoostClassifier(verbose=False, loss_function='MultiClass')

    
#     return cat_model
# %%time
# dprint("Prepared Model.....")
# cat_model = cat_model()
# cat_model.fit(X, y)
# dprint("Done Model.....")
# dprint("Predict Test Value.....")
# target_cat = cat_model.predict(test).astype("int64")
# dprint("Done Prediction.....")
sub = pd.read_csv("../input/sample_submission.csv")
sub['Target'] = target_lgb
sub.to_csv("cat_boost.csv", index= False)
