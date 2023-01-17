import pandas as pd
import numpy as np
mem_data = pd.read_csv('../input/mem_data.csv')
mem_tr = pd.read_csv('../input/transactions.csv')
song_info = pd.read_csv('../input/songs.csv')
Q1 = mem_data.age.quantile(0.25)
Q3 = mem_data.age.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
whisker = (mem_data.age >= (Q1 - 1.5 * IQR)) & (mem_data.age <= (Q3 + 1.5 * IQR))
mem_data.age = mem_data.age.where(whisker, other=0)
mem_tr = mem_tr.merge(song_info, how='left')
f = pd.to_datetime(mem_data.reg_date, format='%Y%m%d')
f = (pd.to_datetime('2017-12-31') - f).dt.days
mem_data['R_DAY'] = f
mem_data.R_DAY.describe()
f = pd.to_datetime(mem_data.ex_date, format='%Y%m%d')
f = (f.max() - f).dt.days
mem_data['E_DAY'] = f
mem_data.E_DAY.describe()
f = mem_tr.groupby('user_id')['listen'].agg({'total_listen':'sum'}).reindex().reset_index()
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
mem_data.head()
f = mem_tr[mem_tr.listen==1].groupby('user_id')['artist'].agg({'like_artists':'nunique'}).reindex().reset_index()
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0).astype('int')
f = mem_tr.groupby('user_id')['listen'].agg({'rec_ratio':'count'}).reindex().reset_index()
f = f.astype('int')
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0)
mem_data['rec_ratio'] = (mem_data['total_listen'] / mem_data['rec_ratio'].values).fillna(0).astype('float32')
f = mem_tr.merge(song_info, how='left')
f = f.groupby('user_id')['genre'].agg({'rec_genre':'nunique'}).reindex().reset_index()
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0)
d_col = ['reg_date','ex_date']
mem_data = mem_data.drop(d_col, axis=1)
mem_data.info()
mem_tr['country'] = mem_tr.isrc.str[:2]
mem_tr['song_code'] = mem_tr.isrc.str[3:5]
mem_tr['song_date'] = mem_tr.isrc.str[5:7]
mem_tr['song_date'] = np.where(mem_tr.song_date.astype(int) >30 ,'19'+mem_tr.song_date,'20'+mem_tr.song_date )
mem_tr['song_date'] = mem_tr.song_date.astype(int)
features = []
df = mem_tr.groupby(['user_id','genre'])['listen'].agg({'listen_genre_cnt':'count'}).reindex().reset_index()
df = df.pivot_table(values='listen_genre_cnt', index=df.user_id, columns='genre', aggfunc='first',fill_value=0).reset_index()
features.append(df);
df.shape
df = mem_tr.groupby(['user_id','rec_loc'])['listen'].agg({'rec_loc_count':'count'}).reindex().reset_index()
df = df.pivot_table(values='rec_loc_count', index=df.user_id, columns='rec_loc', aggfunc='first',fill_value=0).reset_index()
features.append(df)
df.shape
df = mem_tr.groupby(['user_id','rec_screen'])['listen'].agg({'rec_screen_count':'count'}).reindex().reset_index()
df = df.pivot_table(values='rec_screen_count', index=df.user_id, columns='rec_screen', aggfunc='first',fill_value=0).reset_index()
features.append(df)
df.shape
df = mem_tr.groupby(['user_id','entry'])['listen'].agg({'entry_count':'count'}).reindex().reset_index()
df = df.pivot_table(values='entry_count', index=df.user_id, columns='entry', aggfunc='first',fill_value=0).reset_index()
features.append(df)
df.shape
df = mem_tr.groupby(['user_id','artist'])['listen'].agg({'artist_count':'count'}).reindex().reset_index()
df = df.pivot_table(values='artist_count', index=df.user_id, columns='artist', aggfunc='first',fill_value=0).reset_index()
features.append(df);
df.shape
df = mem_tr.groupby(['user_id','composer'])['listen'].agg({'composer_count':'count'}).reindex().reset_index()
df = df.pivot_table(values='composer_count', index=df.user_id, columns='composer', aggfunc='first',fill_value=0).reset_index()
features.append(df);
df.shape
df = mem_tr.groupby(['user_id','lyricist'])['listen'].agg({'lyricist_count':'count'}).reindex().reset_index()
df = df.pivot_table(values='lyricist_count', index=df.user_id, columns='lyricist', aggfunc='first',fill_value=0).reset_index()
features.append(df);
df.shape
df = mem_tr.groupby(['user_id','language'])['listen'].agg({'language_count':'count'}).reindex().reset_index()
df = df.pivot_table(values='language_count', index=df.user_id, columns='language', aggfunc='first',fill_value=0).reset_index()
features.append(df);
df.shape
df = mem_tr.groupby(['user_id','country'])['listen'].agg({'country_count':'count'}).reindex().reset_index()
df = df.pivot_table(values='country_count', index=df.user_id, columns='country', aggfunc='first',fill_value=0).reset_index()
features.append(df);
df.shape
df = mem_tr.groupby(['user_id','listen'])['length'].agg({'country_count':'mean'}).reindex().reset_index()
nlisten = df[df.listen==0]
nlisten.drop(['listen'], axis=1,inplace=True)
listen = df[df.listen==1]
listen.drop(['listen'], axis=1,inplace=True)
features.append(nlisten);
features.append(listen);
df = mem_tr.groupby(['user_id','listen'])['song_date'].agg({'song_date_time':'mean'}).reindex().reset_index()
nlisten = df[df.listen==0]
nlisten.drop(['listen'], axis=1,inplace=True)
listen = df[df.listen==1]
listen.drop(['listen'], axis=1,inplace=True)
features.append(nlisten);
features.append(listen);

f = mem_data.groupby('user_id')['city'].agg([('city', lambda x: x.value_counts().index[0])]).reset_index()
f = pd.get_dummies(f, columns=['city'])
features.append(f);
f.shape
f = mem_data.groupby('user_id')['reg_method'].agg([('reg_method', lambda x: x.value_counts().index[0])]).reset_index()
f = pd.get_dummies(f, columns=['reg_method'])
features.append(f);
f.shape
for f in features :
    mem_data = pd.merge(mem_data, f, how='left',on='user_id')
display(mem_data.shape)
d_col = ['reg_method','city']
mem_data = mem_data.drop(d_col, axis=1)
mem_data.info()
mem_data.fillna(0,inplace=True)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import glob
input_data = 'pre_data.csv'
output_data = 'prediction_dd2.csv'
kfold = StratifiedKFold(n_splits=2)
n_it = 12
np.random.seed(724)
main = mem_data
train = main[main.gender!='unknown']
train.gender = (train.gender=='male').astype(int)
test = main[main.gender=='unknown']
test = test.sort_values('user_id')
t_final = test[['user_id', 'gender']]
test = test.drop(['gender','user_id'], axis=1)
target = train.gender.values
train = train.drop(['gender','user_id'], axis=1)
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split

train.columns = np.arange(0,len(train.columns))
test.columns = np.arange(0,len(test.columns))
from xgboost import XGBClassifier
parameters = {'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 1.0, 'learning_rate': 0.05, 
              'min_child_weight': 5, 'silent': True, 'n_estimators': 200}
XGB = XGBClassifier(**parameters, random_state=714,  n_jobs=-1)
#params = {'max_features':list(np.arange(1, train.shape[1])), 'bootstrap':[False], 'n_estimators': list(np.arange(50,100)), 'criterion':['gini','entropy']}
#model = RandomizedSearchCV(RandomForestClassifier(), param_distributions=params, n_iter=n_it, cv=kfold, scoring='roc_auc',n_jobs=-1, verbose=1)
print('MODELING.............................................................................')
XGB.fit(train, target)
#model = model.best_estimator_
score = cross_val_score(XGB, train, target, cv=5, scoring='roc_auc')
print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
print('COMPLETE')
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split

model = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=None, max_features=561, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=80, n_jobs=-1,
            oob_score=False, random_state=714, verbose=0,
            warm_start=False)
print('MODELING.............................................................................')
model.fit(train, target)
score = cross_val_score(model, train, target, cv=5, scoring='roc_auc')
print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
print('COMPLETE')
import lightgbm as lgb
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(train, target)
score = cross_val_score(model_lgb,train, target, cv=2, scoring='roc_auc')
print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
lgb = model_lgb.predict(test)
from sklearn.ensemble import GradientBoostingClassifier
GBoost_clf = GradientBoostingClassifier(learning_rate=0.05,
                                   max_depth=8, max_features=0.3,
                                   min_samples_leaf=100,
                                   loss="deviance",
                                   random_state =5)
GBoost_clf.fit(train, target)
score = cross_val_score(GBoost_clf, train, target, cv=2, scoring='roc_auc')
print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
from sklearn.ensemble import VotingClassifier
votingC = VotingClassifier(estimators=[('random', model), ('XGB',XGB), ('GBoost',GBoost_clf)], voting='soft', n_jobs=-1)
    
votingC = votingC.fit(train, target)
#score = cross_val_score(votingC, X_train, y_train, cv=5, scoring='roc_auc')
#print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
votingC = votingC.predict_proba(test)[:,1]
ensembled_prediction = (0.7*votingC)+(0.3*lgb)
t_final.gender = ensembled_prediction
t_final.to_csv(output_data, index=False)
print('COMPLETE')
