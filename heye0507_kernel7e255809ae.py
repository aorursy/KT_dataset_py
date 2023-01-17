%load_ext autoreload
%autoreload 2
%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.ensemble import forest
from sklearn.tree import export_graphviz

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from IPython.display import display
from pandas_summary import DataFrameSummary
TRAIN_PATH = '../input/train_V2.csv'
TEST_PATH = '../input/test_V2.csv'

types = {
    'Id':'object',
    'groupId': 'object',
    'matchId': 'object',
    'assists': 'int8',
    'boosts': 'int8',
    'damageDealt': 'float32',
    'DBNOs': 'int8',
    'headshotKills': 'int8',
    'heals': 'int8',
    'killPlace': 'int8',
    'killPoints': 'int16',
    'kills': 'int8',
    'killStreaks': 'int8',
    'longestKill': 'float32',
    'matchDuration': 'int16',
    'matchType': 'object',
    'maxPlace': 'int8',
    'numGroups': 'int8',
    'rankPoints': 'int16',
    'revives': 'int8',
    'rideDistance': 'float32',
    'roadKills': 'int8',
    'swimDistance': 'float32',
    'teamKills': 'int8',
    'vehicleDestroys': 'int8',
    'walkDistance': 'float32',
    'weaponsAcqired': 'int16',
    'winPoints': 'int16',
    'winPlacePerc': 'float32' 
}

#%time df_train_raw = pd.read_csv(f'{TRAIN_PATH}')
#%time df_test_raw = pd.read_csv(f'{TEST_PATH}')

#display function will have hidden cols, like '...' 
#use fastai functions to display all for data exploration

############# fastai function ####################

def display_all(data_frame):
    with pd.option_context('display.max_rows',1000,'display.max_columns',1000):
        display(data_frame)

############# end of fastai function ################
%time df_all = pd.read_csv(f'{TRAIN_PATH}',dtype=types)
%time df_test = pd.read_csv(f'{TEST_PATH}')
display_all(df_all.describe(include='all').T)
#Pre-process the data using fastai
#Def some help functions that not avaliable in fastai 1.0

def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()

            
            

def apply_cats(df, trn):
    for n,c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name=='category'):
            df[n] = pd.Categorical(c, categories=trn[n].cat.categories, ordered=True)


            
  
def fix_missing(df, col, name, na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict



def numericalize(df, col, name, max_n_cat):
    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = col.cat.codes+1



def scale_vars(df, mapper):
    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)
    if mapper is None:
        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper



def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res




def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)




def set_rf_samples(n):
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

def reset_rf_samples():
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))

%%time
train_cats(df_all)
%%time
apply_cats(df_test,df_all)
df_all.loc[2744604]
#single player game, drop it
df_all.drop([2744604],inplace=True)
df_train,df_y,nas = proc_df(df_all,'winPlacePerc')
test_raw,_,_ = proc_df(df_test,na_dict=nas)
df_train.shape,df_y.shape,df_test.shape
from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y = train_test_split(df_train,df_y,test_size = 20000, random_state = 47)
train_X.shape,val_X.shape
display_all(train_X.describe(include='all'))
#def a function to print scores
from sklearn.metrics import mean_absolute_error

def print_scores(model):
    train_accuracy = model.score(train_X,train_y)
    val_accuracy = model.score(val_X,val_y)
    mae = mean_absolute_error(val_y,model.predict(val_X))
    print(f'Training set accuracy is: {train_accuracy}')
    print(f'Validataion set accuracy is: {val_accuracy}')
    print(f'Mean Absolute error is: {mae}')
    #print(f'OOB score is: {model.oob_score_}')
    
#train small model first
set_rf_samples(50000)
model_first = RandomForestRegressor(n_estimators=200,min_samples_leaf=3,n_jobs=-1)
%time model_first.fit(train_X,train_y)
%%time
print_scores(model_first)
fi = rf_feat_importance(model_first,df_train)
fi
fi[:15].plot('cols','imp','barh',figsize=(12,7))
to_drop = fi[fi.imp<0.001].cols
df_train.drop(to_drop,axis=1,inplace=True)
train_X,val_X,train_y,val_y = train_test_split(df_train,df_y,test_size = 20000, random_state = 47)
model_first = RandomForestRegressor(n_estimators=200,min_samples_leaf=3,n_jobs=-1)
%time model_first.fit(train_X,train_y)
%%time
print_scores(model_first)
fi = rf_feat_importance(model_first,df_train)
fi
#Start working on important features
df_all['total_distance'] = df_all.walkDistance + df_all.rideDistance + df_all.swimDistance
df_all.plot('walkDistance','winPlacePerc','scatter',figsize=(12,7))
df_all.plot('total_distance','winPlacePerc','scatter')
df_train,df_y,nas = proc_df(df_all,'winPlacePerc')
df_train.drop(to_drop,axis=1,inplace=True)
train_X,val_X,train_y,val_y = train_test_split(df_train,df_y,test_size = 20000, random_state = 47)
train_X.shape,val_X.shape
#train_X.total_distance.describe()
model_FE = RandomForestRegressor(n_estimators=200,min_samples_leaf=3,n_jobs=-1)
%time model_FE.fit(train_X,train_y)
%time print_scores(model_FE)
fi = rf_feat_importance(model_FE,df_train)
fi
#get joined player for each game
df_all['players_joined'] = df_all.groupby('matchId')['matchId'].transform('count') 
df_all['max_kill_place'] = df_all.groupby('matchId')['killPlace'].transform('max')
df_all[df_all.max_kill_place != df_all.players_joined][['matchId','max_kill_place','players_joined']]
display_all(df_all.loc[df_all.matchId.values=='fe57e25e37dbfd'].sort_values(by='killPlace',ascending=False))
df_train.matchDuration.describe()
predictions = model_first.predict(df_test)
print(predictions[0:10])
my_submission = pd.DataFrame({'Id':df_test_raw['Id'],'winPlacePerc':predictions})
my_submission.to_csv('submission_v_01.csv',index=False)
my_submission.head()
df_test_raw.Id.head()
