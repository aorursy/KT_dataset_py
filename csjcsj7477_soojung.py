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
mem_data = pd.read_csv('../input/mem_data.csv')
mem_tr = pd.read_csv('../input/transactions.csv')
song_info = pd.read_csv('../input/songs.csv')
mem_data.info()
features = []
mem_data["age"]=np.where(mem_data["age"]<=11,26,mem_data["age"])
mem_data["age"]=np.where(mem_data["age"]>=70,26,mem_data["age"])
num1 = (mem_data["age"] <= 8).sum()
num2 = (mem_data["age"] >= 70).sum()
num1,num2
f = pd.to_datetime(mem_data.reg_date, format='%Y%m%d')
f1 = pd.to_datetime(mem_data.ex_date, format='%Y%m%d')
f = (f1 - f).dt.days
mem_data['E_DAY'] = np.log(f)
mem_data.E_DAY.describe()
## log함수
def log(col_name):
        f = mem_data[col_name].where(mem_data[col_name]>=0, other=0)
        f = np.log(f+1)
        mem_data[col_name] = f
def drop(x):   
#     d_col = [x]x
    global mem_data
    mem_data = mem_data.drop(x, axis=1)
    mem_data.info()
#음원길이
fe = mem_tr.merge(song_info, how='left')
_mean_song_length = np.mean(fe['length'])
def smaller_song(x):
    if x < _mean_song_length:
        return 1
    return 0

mem_data['smaller_song'] = fe['length'].apply(smaller_song).astype(np.int8)
mem_data.head()
## 평균재생횟수
f = mem_tr.groupby('user_id')['listen'].agg({'total_listen':'sum'}).reindex().reset_index()
f = np.log(f)
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head()
# 추천을 받아들인 횟수
f = mem_tr.groupby('user_id')['listen'].agg({'rec_ratio':'count'}).reindex().reset_index()
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0)
mem_data['rec_ratio'] = (mem_data['total_listen'] / mem_data['rec_ratio'].values).fillna(0).astype('float32')
f = mem_tr.merge(song_info, how='left')
mem_tr['nation'] = f.isrc.str[0:2]
# mem_tr.head()
f = mem_tr.merge(song_info, how='left')
mem_tr['issue'] = f.isrc.str[2:5]
# mem_tr.head()
f = mem_tr.merge(song_info, how='left')
mem_tr['isrc_year'] = f.isrc.str[5:7]
# mem_tr.head()
mem_data.shape
f = pd.pivot_table(mem_tr, index='user_id', columns='nation', values='song_id', 
                   aggfunc=np.size, fill_value=0)
f.columns = f.columns.astype(str)
f.columns = [('nation_'+x) for x in f.columns]
f = f.reset_index()
f
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = pd.pivot_table(mem_tr, index='user_id', columns='issue', values='song_id', 
                   aggfunc=np.size, fill_value=0)
f.columns = f.columns.astype(str)
f.columns = [('issue_'+x) for x in f.columns]
f = f.reset_index()
f
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = pd.pivot_table(mem_tr, index='user_id', columns='rec_loc', values='song_id', 
                   aggfunc=np.size, fill_value=0)
f.columns = f.columns.astype(str)
f.columns = [('rec_loc_'+x) for x in f.columns]
f = f.reset_index()
f
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = pd.pivot_table(mem_tr, index='user_id', columns='rec_screen', values='song_id', 
                   aggfunc=np.size, fill_value=0)
f.columns = f.columns.astype(str)
f.columns = [('rec_screen_'+x) for x in f.columns]
f = f.reset_index()
f
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = pd.pivot_table(mem_tr, index='user_id', columns='entry', values='song_id', 
                   aggfunc=np.size, fill_value=0)
f.columns = f.columns.astype(str)
f.columns = [('entry_'+x) for x in f.columns]
f = f.reset_index()
f
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = pd.pivot_table(mem_tr, index='user_id', columns='listen', values='song_id', 
                   aggfunc=np.size, fill_value=0)
f.columns = f.columns.astype(str)
f.columns = [('listen_'+x) for x in f.columns]
f = f.reset_index()
f
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = mem_tr.groupby('user_id')['rec_loc'].agg([('rec_loc건수','nunique')]).reset_index()
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = mem_tr.groupby('user_id')['rec_loc'].agg([('rec_screen건수','nunique')]).reset_index()
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = mem_tr.groupby('user_id')['rec_loc'].agg([('entry건수','nunique')]).reset_index()
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = mem_tr.merge(song_info, how='left')
tr = f
f = pd.pivot_table(tr, index='user_id', columns='genre', values='song_id', 
                   aggfunc=np.size, fill_value=0)
f.columns = f.columns.astype(str)
f.columns = [('genre_'+x) for x in f.columns]
f = f.reset_index()
f
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = mem_tr.merge(song_info, how='left')
tr = f
f = pd.pivot_table(tr, index='user_id', columns='artist', values='song_id', 
                   aggfunc=np.size, fill_value=0)
f.columns = f.columns.astype(str)
f.columns = [('entry_'+x) for x in f.columns]
f = f.reset_index()
f
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = mem_tr.merge(song_info, how='left')
tr = f
f = pd.pivot_table(tr, index='user_id', columns='composer', values='song_id', 
                   aggfunc=np.size, fill_value=0)
f.columns = f.columns.astype(str)
f.columns = [('composer_'+x) for x in f.columns]
f = f.reset_index()
f
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = mem_tr.merge(song_info, how='left')
tr = f
f = pd.pivot_table(tr, index='user_id', columns='lyricist', values='song_id', 
                   aggfunc=np.size, fill_value=0)
f.columns = f.columns.astype(str)
f.columns = [('lyricist _'+x) for x in f.columns]
f = f.reset_index()
f
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = mem_tr.merge(song_info, how='left')
tr = f
f = pd.pivot_table(tr, index='user_id', columns='language', values='song_id', 
                   aggfunc=np.size, fill_value=0)
f.columns = f.columns.astype(str)
f.columns = [('language _'+x) for x in f.columns]
f = f.reset_index()
f
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = mem_tr.merge(song_info, how='left')
tr = f
f = tr.groupby('user_id')['genre'].agg([('genre건수','nunique')]).reset_index()
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = mem_tr.merge(song_info, how='left')
tr = f
f = tr.groupby('user_id')['artist'].agg([('artist건수','nunique')]).reset_index()
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = mem_tr.merge(song_info, how='left')
tr = f
f = tr.groupby('user_id')['composer'].agg([('composer건수','nunique')]).reset_index()
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = mem_tr.merge(song_info, how='left')
tr = f
f = tr.groupby('user_id')['lyricist'].agg([('lyricist건수','nunique')]).reset_index()
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = mem_tr.merge(song_info, how='left')
tr = f
f = tr.groupby('user_id')['language'].agg([('language건수','nunique')]).reset_index()
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head(); mem_data.shape
f = mem_tr.merge(song_info, how='left')
tr = f
f = mem_tr[['user_id','isrc_year']]
f.isrc_year = f.isrc_year.astype(int)
def f1(x):
    if  x < 17 :
        return(x+2000)
    else :
        return(x+1900)
f['isrc_real_year'] = f.isrc_year.apply(f1)
f = mem_tr.groupby('user_id')['isrc_year'].agg({'isrc_year_uq':'nunique'}).reset_index()
mem_data = mem_data.merge(f, how='left')
f = pd.pivot_table(mem_tr, index='user_id', columns='isrc_year', values='song_id', 
                   aggfunc=np.size, fill_value=0)
f
f.columns = f.columns.astype(str)
f.columns = [('year_'+x) for x in f.columns]
f = f.reset_index()
f
mem_data = mem_data.merge(f,how='left')
d_col = ['reg_date','ex_date']
mem_data = mem_data.drop(d_col, axis=1)
# mem_data.info()
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import glob
input_data = mem_data
output_data = 'prediction_dd.csv'
main= input_data
main.head()
main.shape
main= input_data
train = main[main.gender!='unknown']
train.gender = (train.gender=='male').astype(int)
target = train.gender.values
test = main[main.gender=='unknown']

test = test.sort_values('user_id')
t_final = test[['user_id', 'gender']]
test = test.drop(['gender','user_id'], axis=1)
train = train.drop(['gender','user_id'], axis=1)
main.shape, train.shape, target.shape, test.shape
from xgboost import XGBClassifier
import xgboost as xg
xgb_param = {'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 1.0, 'learning_rate': 0.05, 
              'min_child_weight': 5, 'silent': True, 'n_estimators': 200}
xgb = XGBClassifier(**xgb_param, random_state=0, n_jobs=-1)

from sklearn.model_selection import cross_val_score

# score=cross_val_score(xgb,train,target,cv=5,scoring='roc_auc')
# print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
train = train.replace([np.inf, -np.inf], np.nan) #무한대 값을 0으로 채움
train = pd.DataFrame(train).fillna(0)
test = test.replace([np.inf, -np.inf], np.nan) #무한대 값을 0으로 채움
test = pd.DataFrame(test).fillna(0)
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(random_state=0, max_depth=6)
from lightgbm import LGBMModel,LGBMClassifier
lgb = LGBMClassifier(n_estimators=200, silent=False, random_state =0, max_depth=3,num_leaves=31,objective='binary',metrics ='auc')
from sklearn.ensemble import VotingClassifier
votingC = VotingClassifier(estimators=[('xgb',xgb),('gbc', gbc),('lgb', lgb)], voting='soft')
score = cross_val_score(votingC, train, target, cv=5, scoring='roc_auc')
print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
train = train.replace([np.inf, -np.inf], np.nan) #무한대 값을 0으로 채움
train = pd.DataFrame(train).fillna(0)
test = test.replace([np.inf, -np.inf], np.nan) #무한대 값을 0으로 채움
test = pd.DataFrame(test).fillna(0)
train.shape, target.shape, test.shape
mode_xgb=votingC.fit(train, target)
t_final.gender = mode_xgb.predict_proba(test)[:,1]
pred = mode_xgb.predict_proba(test)[:,1]
# print('MODELING.............................................................................')
# mode_xgb=votingC.fit(train, target)
t_final.gender = mode_xgb.predict_proba(test)[:,1]
# t_final.to_csv(output_data, index=False)
# print('COMPLETE')