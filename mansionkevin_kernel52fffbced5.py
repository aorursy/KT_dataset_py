import pandas as pd

import numpy as np

import seaborn as sns

import gc

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

from xgboost import XGBRegressor

from xgboost import XGBClassifier

from sklearn.metrics import cohen_kappa_score

from sklearn.decomposition import PCA

from sklearn.metrics import make_scorer

import time
train=pd.read_csv('../input/data-science-bowl-2019/train.csv')

test=pd.read_csv('../input/data-science-bowl-2019/test.csv')

train_labels=pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

sample_submission=pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')

specs=pd.read_csv('../input/data-science-bowl-2019/specs.csv')

train_labels.drop(columns=['installation_id','title'],inplace=True)

train=train.merge(train_labels,on='game_session')
train['Date']=train.timestamp.str.split('T',expand=True).replace(np.nan,0)[0]

train['Hour']=train.timestamp.str.split('T',expand=True).replace(np.nan,0)[1].str.split('.',expand=True)[0]

train['hourofday']=pd.to_datetime(train['Hour']).dt.hour

train['dayofweek']=pd.to_datetime(train['Date']).dt.dayofweek

train['title_event_code']=list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
test['Date']=test.timestamp.str.split('T',expand=True).replace(np.nan,0)[0]

test['Hour']=test.timestamp.str.split('T',expand=True).replace(np.nan,0)[1].str.split('.',expand=True)[0]

test['hourofday']=pd.to_datetime(test['Hour']).dt.hour

test['dayofweek']=pd.to_datetime(test['Date']).dt.dayofweek

test['title_event_code']=list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
train['session_duration']=train.game_time[train.event_data.str.contains('session_duration')]

test['session_duration']=test.game_time[test.event_data.str.contains('session_duration')]
world_encode=set(list(train.world.unique())+list(test.world.unique()))

world_encode=dict(zip(world_encode,np.arange(len(world_encode))))

train['world']=train.world.map(world_encode)

test['world']=test.world.map(world_encode)
train_true=train[train.event_data.str.contains('true')].index

train['true_row']=0

train['true_row'].loc[train_true]=1

test_true=test[test.event_data.str.contains('true')].index

test['true_row']=0

test['true_row'].loc[test_true]=1
train_false=train[train.event_data.str.contains('false')].index

train['false_row']=0

train['false_row'].loc[train_false]=1

test_false=test[test.event_data.str.contains('false')].index

test['false_row']=0

test['false_row'].loc[test_false]=1
train['event_count_by_session']=train.groupby('game_session')['event_count'].transform('max')

test['event_count_by_session']=test.groupby('game_session')['event_count'].transform('max')
train=train.join(pd.get_dummies(train.title,prefix=('title_')))

train=train.join(pd.get_dummies(train.title_event_code,prefix=('title_event_')))

train=train.join(pd.get_dummies(train.event_code,prefix=('event_')))

train_first_column=pd.get_dummies(train.title,prefix=('title_')).columns[0]

train_last_column=train.columns[-1]

dummies_columns=list(train.loc[:,train_first_column:train_last_column].columns)
reduced_train=pd.DataFrame(train.groupby('game_session')['hourofday'].apply(lambda x: x.mode()[0]))

reduced_train=reduced_train.join(pd.DataFrame(train.groupby('game_session')['session_duration'].max()))

reduced_train=reduced_train.join(pd.DataFrame(train.groupby('game_session')['false_row'].apply(lambda x: x.sum()/len(x))))

reduced_train=reduced_train.join(pd.DataFrame(train.groupby('game_session')['true_row'].apply(lambda x: x.sum()/len(x))))

reduced_train=reduced_train.join(pd.DataFrame(train.groupby('game_session')['event_count_by_session'].max()))

reduced_train=reduced_train.join(pd.DataFrame(train.groupby('game_session')['accuracy_group'].mean()))

reduced_train=reduced_train.join(pd.DataFrame(train.groupby('game_session')['world'].mean()))

reduced_train=reduced_train.join(pd.DataFrame(train.groupby('game_session')[dummies_columns].sum()))
reduced_train.fillna(0,inplace=True)

reduced_train=reduced_train.reset_index(drop=True)
del train,train_labels,specs

gc.collect()
test=test[test['type']=='Assessment']
test=test.join(pd.get_dummies(test.title,prefix=('title_')))

test=test.join(pd.get_dummies(test.title_event_code,prefix=('title_event_')))

test=test.join(pd.get_dummies(test.event_code,prefix=('event_')))

test_first_column=pd.get_dummies(test.title,prefix=('title_')).columns[0]

test_last_column=test.columns[-1]

test_dummies_columns=list(test.loc[:,test_first_column:test_last_column].columns)
reduced_test=pd.DataFrame(test.groupby('game_session')['hourofday'].apply(lambda x: x.mode()[0]))

reduced_test=reduced_test.join(pd.DataFrame(test.groupby('game_session')['installation_id'].apply(lambda x: x.unique()[0])))

reduced_test=reduced_test.join(pd.DataFrame(test.groupby('game_session')['session_duration'].max()))

reduced_test=reduced_test.join(pd.DataFrame(test.groupby('game_session')['false_row'].apply(lambda x: x.sum()/len(x))))

reduced_test=reduced_test.join(pd.DataFrame(test.groupby('game_session')['true_row'].apply(lambda x: x.sum()/len(x))))

reduced_test=reduced_test.join(pd.DataFrame(test.groupby('game_session')['event_count_by_session'].max()))

reduced_test=reduced_test.join(pd.DataFrame(test.groupby('game_session')['world'].mean()))

reduced_test=reduced_test.join(pd.DataFrame(test.groupby('game_session')[test_dummies_columns].sum()))
test_session_installation=reduced_test.loc[:,['installation_id']]
reduced_test.fillna(0,inplace=True)

reduced_test.drop(columns='installation_id',inplace=True)

reduced_test=reduced_test.reset_index(drop=True)
del test

gc.collect()
test_session_installation.reset_index(inplace=True)
in_test_not_in_train=list(set(reduced_test.columns).difference(set(reduced_train.columns)))

in_train_not_in_test=list(set(reduced_train.columns).difference(set(reduced_test.columns)))

in_train_not_in_test.remove('accuracy_group')
reduced_train=reduced_train.join(pd.DataFrame(columns=in_test_not_in_train))

reduced_test=reduced_test.join(pd.DataFrame(columns=in_train_not_in_test))
reduced_train.fillna(0,inplace=True)

reduced_test.fillna(0,inplace=True)
pca=PCA(n_components=38)
train_last_column=reduced_train.columns[-1]

train_to_pca=reduced_train.loc[:,train_first_column:train_last_column]

reduced_train.drop(list(train_to_pca.columns),axis=1,inplace=True)

train_to_pca=train_to_pca.reindex(sorted(train_to_pca.columns),axis=1)

test_last_column=reduced_test.columns[-1]

test_to_pca=reduced_test.loc[:,test_first_column:test_last_column]

reduced_test.drop(list(train_to_pca.columns),axis=1,inplace=True)

test_to_pca=test_to_pca.reindex(sorted(test_to_pca.columns),axis=1)
to_pca=pd.concat([train_to_pca,test_to_pca])
to_pca=to_pca.reset_index(drop=True)

pca_ed=pca.fit_transform(to_pca)

pca_ed=pd.DataFrame(pca_ed)
train_pca_ed=pca_ed.iloc[:train_to_pca.shape[0],:]
test_pca_ed=pca_ed.iloc[-test_to_pca.shape[0]:,:].reset_index(drop=True)
reduced_train=pd.concat([reduced_train,train_pca_ed],axis=1)

reduced_test=pd.concat([reduced_test,test_pca_ed],axis=1).reset_index(drop=True)
reduced_test
y=reduced_train['accuracy_group']
reduced_train.drop(columns='accuracy_group',inplace=True)

X=reduced_train
params = {'n_estimators':2000,

            'gamma':0.01,

            'max_depth':8,

            'eta':0.01,

            'subsample': 0.5,

            'learning_rate': 0.04,

            'min_child_weight':1.5,

            'feature_fraction': 0.7,

            'colsample_bytree':0.5,

            'max_depth': 15,

            'reg_alpha': 1,  

            'reg_lambda': 1,

            'verbose':True

            }
skf=StratifiedKFold(n_splits=5)

counter=0

for train_index,val_index in skf.split(X,y):

    time1=time.time()

    X_train,X_valid=reduced_train.iloc[train_index,:],reduced_train.iloc[val_index,:]

    y_train,y_valid=y[train_index],y[val_index]

    reg=xgb.XGBClassifier(**params)

    reg.fit(X_train,y_train)

    y_predict=reg.predict(X_valid)

    score=make_scorer(cohen_kappa_score)(reg,X_valid,y_valid)

    time2=time.time()-time1

    counter+=1

    print('round{} time cost:{}'.format(counter,time2))

    print('round{} cohen_kappa_score:{:.3f}'.format(counter,score))
test_predict=pd.DataFrame(reg.predict(reduced_test),columns=['accuracy'])
test_session_installation=test_session_installation.join(test_predict)
submission=pd.DataFrame(test_session_installation.groupby('installation_id')['accuracy'].max())
submission=submission.reset_index()
sample_submission['accuracy_group']=submission['accuracy']
sample_submission.to_csv('submission.csv', index=False)