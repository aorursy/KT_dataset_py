import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option('max_columns', 10, 'max_rows', 10)
tr_train = pd.read_csv('../input/X_train.csv', encoding='cp949')
tr_test = pd.read_csv('../input/X_test.csv', encoding='cp949')
tr = pd.concat([tr_train, tr_test])
tr['real_amt']= tr.tot_amt / tr.inst_mon
tr['sdate'] = tr.sales_date.str[:10]
tr['dist_rate'] = (tr['dis_amt']/tr['tot_amt'])*100
tr.loc[456697,"sales_time"] = 1530
tr["time"] = tr['sdate'].astype(str).str.cat(tr["sales_time"].astype(str), sep =" ") 
tr["time"] = pd.to_datetime(tr.time, format='%Y-%m-%d %H%M')
features = []
#평균할인율
f = tr.groupby('custid')['dist_rate'].agg([('dis_rate', 'mean')]).reset_index()
features.append(f)
#display(f.isnull().sum().sum())
#display(f.shape)
#평균시간
f = tr.groupby(['custid'])['sales_time'].agg([('sales_time', 'mean')]).reset_index()
features.append(f)
#display(f.isnull().sum().sum())
#display(f.shape)
#남성파트
df = tr.groupby(['custid','part_nm'])['tot_amt'].agg([('tot_amt_part', 'sum')]).reset_index()
df['part_nm'] = np.where(df.part_nm.str.contains('남성'), '남성', '비남성')
df = df.pivot_table(values='tot_amt_part', index=df.custid, columns='part_nm', aggfunc='first',fill_value=0).reset_index()
df['남성part'] = (df['남성'] / (df['남성'] + df['비남성'])) * 100
df = df.fillna(0)
features.append(df)
#display(df.isnull().sum().sum())
#display(df.shape)
#화장품구매비율
df = tr.groupby(['custid','corner_nm'])['tot_amt'].agg([('tot_amt_corner', 'sum')]).reset_index()
df['corner_nm'] = np.where(df.corner_nm.str.contains('화장품'), '화장품', '비화장품')
df = df.pivot_table(values='tot_amt_corner', index=df.custid, columns='corner_nm', aggfunc='first',fill_value=0).reset_index()
df['화장품비율'] = (df['화장품'] / (df['화장품'] + df['비화장품'])) * 100
df = df.fillna(0)
features.append(df)
#display(df.isnull().sum().sum())
#display(df.shape)
#쇼핑시간
df = tr.groupby(['sdate','custid'])['time'].agg([('time', ['min','max'])]).reset_index()
df['shopping_time'] = (df['time']['max'] - df['time']['min']).dt.total_seconds()
df.drop(['sdate','time'], axis=1, inplace=True,level=0)
df = df.groupby(['custid'])['shopping_time'].agg([('shopping_time_mean','mean')]).reset_index()
features.append(df)
#display(df.isnull().sum().sum())
#display(df.shape)
#할부대비평균실구매
f = tr.groupby('custid')['real_amt'].agg([('real_amt', 'mean')]).reset_index()
features.append(f)
#display(f.isnull().sum().sum())
#display(f.shape)
#평균구매상품종류
df =tr.groupby(['custid','goodcd'])['tot_amt'].agg([('good_count', 'count')]).reset_index()
f = df.groupby(['custid'])['good_count'].agg([('good_count_mean', 'mean')]).reset_index()
features.append(f)
#display(f.isnull().sum().sum())
#display(f.shape)
#지역
df = tr.groupby(['custid','str_nm'])['tot_amt'].agg([('tot_amt_str', 'sum')]).reset_index()
df =df.pivot_table(values='tot_amt_str', index=df.custid, columns='str_nm', aggfunc='first',fill_value=0).reset_index()
features.append(df)
#display(df.isnull().sum().sum())
#display(df.shape)
#display(df.columns)
#display(df.index.name)
#팀별
df = tr.groupby(['custid','team_nm'])['tot_amt'].agg([('tot_amt_team', 'sum')]).reset_index()
df =df.pivot_table(values='tot_amt_team', index=df.custid, columns='team_nm', aggfunc='first',fill_value=0).reset_index()
features.append(df)
#display(df.isnull().sum().sum())
#display(df.shape)
#display(df.columns)
#display(df.index.name)
#총구매수입상품
df = tr.groupby(['custid'])['import_flg'].agg([('import_flg_sum', 'sum')]).reset_index()
features.append(df)
#display(df.isnull().sum().sum())
#display(df.shape)
#display(df.columns)
#display(df.index.name)
#월별총구매수입상품
df = tr.groupby(['custid'])['import_flg'].agg([('inst_mon_sum', 'sum')]).reset_index()
features.append(df)
#display(df.isnull().sum().sum())
#display(df.shape)
#display(df.columns)
#display(df.index.name)
#파트
df = tr.groupby(['custid','part_nm'])['tot_amt'].agg([('tot_amt_part', 'sum')]).reset_index()
df =df.pivot_table(values='tot_amt_part', index=df.custid, columns='part_nm', aggfunc='first',fill_value=0).reset_index()
features.append(df)
#display(df.isnull().sum().sum())
#display(df.shape)
#display(df.columns)
#display(df.index.name)
#코너
df = tr.groupby(['custid','corner_nm'])['tot_amt'].agg([('tot_amt_corner', 'sum')]).reset_index()
df =df.pivot_table(values='tot_amt_corner', index=df.custid, columns='corner_nm', aggfunc='first',fill_value=0).reset_index()
features.append(df)
#display(df.isnull().sum().sum())
#display(df.shape)
#display(df.columns)
#display(df.index.name)
#pc
df = tr.groupby(['custid','pc_nm'])['tot_amt'].agg([('tot_amt_pc', 'sum')]).reset_index()
df =df.pivot_table(values='tot_amt_pc', index=df.custid, columns='pc_nm', aggfunc='first',fill_value=0).reset_index()
features.append(df)
#display(df.isnull().sum().sum())
#display(df.shape)
#브랜드
df = tr.groupby(['custid','brd_nm'])['tot_amt'].agg([('tot_amt_brd', 'sum')]).reset_index()
df =df.pivot_table(values='tot_amt_brd', index=df.custid, columns='brd_nm', aggfunc='first',fill_value=0).reset_index()
features.append(df)
#display(df.isnull().sum().sum())
#display(df.shape)
#구매자
df = tr.groupby(['custid','buyer_nm'])['tot_amt'].agg([('tot_amt_buyer', 'sum')]).reset_index()
df =df.pivot_table(values='tot_amt_buyer', index=df.custid, columns='buyer_nm', aggfunc='first',fill_value=0).reset_index()
features.append(df)
#display(df.isnull().sum().sum())
#display(df.shape)
#구매시간
from datetime import timedelta as dt
test = tr.groupby(['custid'])['sales_date'].agg([('sales_date', 'max')]).reset_index()
test['days'] = (pd.to_datetime('2002-01-01') - pd.to_datetime(test.sales_date)).dt.days
test.drop(['sales_date'], axis=1, inplace=True)
features.append(test)
#display(test.isnull().sum().sum())
#display(test.shape)
#일평균구매액
test2 = tr.groupby(['sales_date','custid'])['tot_amt'].agg([('day_amt', 'sum')]).reset_index()
test2 = test2.groupby(['custid'])['day_amt'].agg([('일평균구매액', 'mean')]).reset_index()
features.append(test2)
#display(test2.isnull().sum().sum())
#display(test2.shape)
#일평균구매건
df = tr.groupby(['sales_date','custid'])['custid'].agg([('day_visit', 'count')]).reset_index()
f = df.groupby(['custid'])['day_visit'].agg([('일평균구매건', 'mean')]).reset_index()
features.append(f)
#display(f.isnull().sum().sum())
#display(f.shape)
#총구매액
f = tr.groupby('custid')['tot_amt'].agg([('총구매액', 'sum')]).reset_index()
features.append(f)
#display(f.isnull().sum().sum())
#display(f.shape)
f = tr.groupby('custid')['tot_amt'].agg([('구매건수', 'size')]).reset_index()
features.append(f)
display(f.isnull().sum().sum())
display(f.shape)
f = tr.groupby('custid')['tot_amt'].agg([('평균구매가격', 'mean')]).reset_index()
features.append(f)
display(f.isnull().sum().sum())
display(f.shape)
f = tr.groupby('custid')['inst_mon'].agg([('평균할부개월수', 'mean')]).reset_index()
f.iloc[:,1] = f.iloc[:,1].apply(round, args=(1,))
features.append(f)
display(f.isnull().sum().sum())
display(f.shape)
n = tr.corner_nm.nunique()
f = tr.groupby('custid')['brd_nm'].agg([('구매상품다양성', lambda x: len(x.unique()) / n)]).reset_index()
features.append(f)
display(f.isnull().sum().sum())
display(f.shape)
tr['sdate'] = tr.sales_date.str[:10]
f = tr.groupby(by = 'custid')['sdate'].agg([('내점일수','nunique')]).reset_index()
features.append(f)
display(f.isnull().sum().sum())
display(f.shape)
x = tr[tr['import_flg'] == 1].groupby('custid').size() / tr.groupby('custid').size()
f = x.reset_index().rename(columns={0: '수입상품_구매비율'}).fillna(0)
f.iloc[:,1] = (f.iloc[:,1]*100).apply(round, args=(1,))
features.append(f)
display(f.isnull().sum().sum())
display(f.shape)
#def weekday(x):
#    w = x.dayofweek 
#    if w < 4:
#        return 1 # 주중
#    else:
#        return 0 # 주말
#f = tr.groupby(by = 'custid')['sdate'].agg([('요일구매패턴', lambda x : pd.to_datetime(x).apply(weekday).value_counts().index[0])]).reset_index()
#features.append(f); f
def fw(x):
    k = x.dayofweek
    if k <= 4 :
        return('주중_방문')
    else :
        return('주말_방문')    
    
df = tr.copy()
df = df.drop_duplicates(['custid','sales_date'])

df['week'] = pd.to_datetime(df.sales_date).apply(fw)
df = pd.pivot_table(df, index='custid', columns='week', values='tot_amt', 
                   aggfunc=np.size, fill_value=0).reset_index()
df['주말방문비율'] = ((df.iloc[:,1] / (df.iloc[:,1]+df.iloc[:,2]))*100).apply(round, args=(1,))
f = df.copy().iloc[:,[0,-1]]
features.append(f)
display(f.isnull().sum().sum())
display(f.shape)
def f1(x):
    k = x.month
    if 3 <= k <= 5 :
        return('봄-구매건수')
    elif 6 <= k <= 8 :
        return('여름-구매건수')
    elif 9 <= k <= 11 :    
        return('가을-구매건수')
    else :
        return('겨울-구매건수')    
    
tr['season'] = pd.to_datetime(tr.sales_date).apply(f1)
f = pd.pivot_table(tr, index='custid', columns='season', values='tot_amt', 
                   aggfunc=np.size, fill_value=0).reset_index()
features.append(f)
display(f.isnull().sum().sum())
display(f.shape)
def f2(x):
    if 9 <= x <= 12 :
        return('아침_구매건수')
    elif 13 <= x <= 17 :
        return('점심_구매건수')
    else :
        return('저녁_구매건수')  # datatime 필드가 시간 형식에 맞지 않은 값을 갖는 경우 저녁시간으로 처리

tr['timeslot'] = tr.sales_date.str.split(' |:', expand=True).iloc[:,1].astype(int).apply(f2)
f = pd.pivot_table(tr, index='custid', columns='timeslot', values='tot_amt', 
                   aggfunc=np.size).reset_index()
features.append(f)
display(f.isnull().sum().sum())
display(f.shape)
f = tr.groupby('custid')['corner_nm'].agg([('주구매코너', lambda x: x.value_counts().index[0])]).reset_index()
f = pd.get_dummies(f, columns=['주구매코너'])  # This method performs One-hot-encoding
features.append(f)
display(f.isnull().sum().sum())
display(f.shape)
X_train = pd.DataFrame({'custid': tr_train.custid.unique()})
for f in features :
    X_train = pd.merge(X_train, f, how='left',on='custid')
display(X_train.shape)

X_test = pd.DataFrame({'custid': tr_test.custid.unique()})
for f in features :
    X_test = pd.merge(X_test, f, how='left',on='custid')
display(X_test.shape)

#y_train = pd.read_csv('../input/y_train.csv').gender
#X_train['gender'] = y_train
display(X_train.isnull().sum().sum())
display(X_test.isnull().sum().sum())
X_train['평균내점구매액'] = X_train['총구매액']/X_train['내점일수']
X_train['주중방문비율'] = (100 - X_train['주말방문비율'])
X_train['국내상품_구매비율'] = (100 - X_train['수입상품_구매비율'])
X_train['할부구매가격'] = X_train['평균구매가격'] / X_train['평균할부개월수']
X_train['구매상품다양성'] = X_train['총구매액'] / X_train['구매상품다양성']
X_train['주말방문수'] = (X_train['주말방문비율'] * X_train['내점일수']) / 100
X_train['주말방문수'] = X_train['주말방문수'].astype('int64')
X_train['주중방문수'] = X_train['내점일수'] - X_train['주말방문수']
X_train['주중방문수'] = X_train['주중방문수'].astype('int64')
X_train['내점당편균구매건수'] = X_train['구매건수']/X_train['내점일수']
X_train['주중구매액'] = X_train['총구매액']*(X_train['주중방문비율']/100)
X_train['주말구매액'] = X_train['총구매액'] - X_train['주중구매액']
X_test['평균내점구매액'] = X_test['총구매액']/X_test['내점일수']
X_test['주중방문비율'] = (100 - X_test['주말방문비율'])
X_test['국내상품_구매비율'] = (100 - X_test['수입상품_구매비율'])
X_test['할부구매가격'] = X_test['평균구매가격'] / X_test['평균할부개월수']
X_test['구매상품다양성'] = X_test['총구매액'] / X_test['구매상품다양성']
X_test['주말방문수'] = (X_test['주말방문비율'] * X_test['내점일수']) / 100
X_test['주말방문수'] = X_test['주말방문수'].astype('int64')
X_test['주중방문수'] = X_test['내점일수'] - X_test['주말방문수']
X_test['주중방문수'] = X_test['주중방문수'].astype('int64')
X_test['내점당편균구매건수'] = X_test['구매건수']/X_test['내점일수']
X_test['주중구매액'] = X_test['총구매액']*(X_test['주중방문비율']/100)
X_test['주말구매액'] = X_test['총구매액'] - X_test['주중구매액']
IDtest = X_test.custid;
X_train.drop(['custid'], axis=1, inplace=True)
X_test.drop(['custid'], axis=1, inplace=True)
y_train = pd.read_csv('../input/y_train.csv').gender
X_train.clip(lower=0,inplace=True)
X_train[X_train.총구매액<0]
X_train.columns = np.arange(0,len(X_train.columns))
X_test.columns = np.arange(0,len(X_train.columns))
max_features = X_train.shape[1]
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import GradientBoostingClassifier
kfold = StratifiedKFold(n_splits=2)
parameters = {'loss' : "deviance",
              'max_depth': 8,
              'min_samples_leaf': 100,
              'max_features': 0.3 
              } 
GBC = GradientBoostingClassifier(**parameters, random_state=123)
#score = cross_val_score(GBC, X_train, y_train, cv=5, scoring='roc_auc')
#print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
pred_gbc = GBC.fit(X_train, y_train).predict_proba(X_test)[:,1]
from xgboost import XGBClassifier
#parameters = {'max_depth': 7, 'n_estimators': 200}
#clf = RandomForestClassifier(**parameters, random_state=0)
#kfold = StratifiedKFold(n_splits=10)
#parameters = {'xgb__max_depth': 4, 'xgb__subsample': 0.7}
#clf = XGBClassifier(random_state=77, n_jobs=-1)
#clf = LogisticRegression()
#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
#clf = GridSearchCV(clf,param_grid = params, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
#clf.fit(X_train,y_train)
#clf = clf.best_estimator_
#LR.best_score_
#kfold = StratifiedKFold(n_splits=10)
#parameters = {'xgb__max_depth': 3, 'xgb__subsample': 0.7}
#clf = XGBClassifier(random_state=0, n_jobs=-1)
'''
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
#clf = LogisticRegression()
#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
clf = GridSearchCV(clf,param_grid = params, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
clf.fit(X_train,y_train)
clf = clf.best_estimator_
'''
parameters = {'xgb__max_depth': 4, 'xgb__subsample': 0.7,'gamma': 2}
clf = XGBClassifier(**parameters, random_state=123, n_jobs=-1)
'''
clf = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
'''
#clf = LogisticRegression()
#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
#clf = GridSearchCV(clf,param_grid = param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
clf.fit(X_train,y_train)
#clf = clf.best_estimator_
#LR.best_score_
#score = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')
#print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
from sklearn.ensemble import VotingClassifier
votingC = VotingClassifier(estimators=[('clf', clf), ('gbc', GBC)], voting='soft', n_jobs=-1)

votingC = votingC.fit(X_train, y_train)
#score = cross_val_score(votingC, X_train, y_train, cv=5, scoring='roc_auc')
#print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
pred = votingC.fit(X_train, y_train).predict_proba(X_test)[:,1]
df = tr.groupby(['custid'])['sales_time'].agg([('sales_time', 'mean')]).reset_index()
def f4(x):
    if x <= 978 :
        return('st_cat01')
    elif 979 <= x <= 1464 :
        return('st_cat02')
    elif 1465 <= x <= 1564 :
        return('st_cat03')
    elif 1565 <= x <= 1658 :
        return('st_cat04')
    else :
        return('st_cat05')  # datatime 필드가 시간 형식에 맞지 않은 값을 갖는 경우 저녁시간으로 처리
df['sale_ct'] = df.sales_time.apply(f4)
df.drop(['sales_time'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['sale_ct'])
train_1 = df.query('custid not in @IDtest').drop(columns=['custid'])
test_1 = df.query('custid in @IDtest').drop(columns=['custid'])
df = tr.groupby(['custid'])['dist_rate'].agg([('dist_amt', 'mean')]).reset_index()
def f3(x):
    if x <= 0 :
        return('cat01')
    elif 1 <= x <= 1.83 :
        return('cat02')
    elif 1.84 <= x <= 2.83 :
        return('cat03')
    elif 2.84 <= x <= 3.92 :
        return('cat04')
    else :
        return('cat05')  # datatime 필드가 시간 형식에 맞지 않은 값을 갖는 경우 저녁시간으로 처리
df['dist'] = df.dist_amt.apply(f3)
df.drop(['dist_amt'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['dist'])
train_2 = df.query('custid not in @IDtest').drop(columns=['custid'])
test_2 = df.query('custid in @IDtest').drop(columns=['custid'])
def f2(x):
    k = x.hour
    if 9 <= k <= 12 :
        return('아침_구매건수')
    elif 13 <= k <= 17 :
        return('점심_구매건수')
    else :
        return('저녁_구매건수')  # datatime 필드가 시간 형식에 맞지 않은 값을 갖는 경우 저녁시간으로 처리

tr['timeslot'] = tr.time.apply(f2)
def f1(x):
    k = x.month
    if 3 <= k <= 5 :
        return('봄-구매건수')
    elif 6 <= k <= 8 :
        return('여름-구매건수')
    elif 9 <= k <= 11 :    
        return('가을-구매건수')
    else :
        return('겨울-구매건수')    
    
tr['season'] = tr.time.apply(f1)
def fw(x):
    k = x.dayofweek
    if k <= 4 :
        return('주중_방문')
    else :
        return('주말_방문')    
    
tr['week'] = tr.time.apply(fw)
tr['sales_hour'] = tr['sales_time']//100;
tr['sales_wkday'] = pd.to_datetime(tr.sales_date).dt.weekday

def makeBOW(col):
    
    f = lambda x: np.where(len(x) >=1, 1, 0)

    train = pd.pivot_table(tr, index='custid', columns=col, values='tot_amt',
                             aggfunc=f, fill_value=0).reset_index(). \
                             query('custid not in @IDtest').drop(columns=['custid'])
    test = pd.pivot_table(tr, index='custid', columns=col, values='tot_amt',
                             aggfunc=f, fill_value=0).reset_index(). \
                             query('custid in @IDtest').drop(columns=['custid'])
    return train, test
train1, test1 = makeBOW('brd_nm')
train2, test2 = makeBOW('corner_nm')
train3, test3 = makeBOW('sales_hour')
train4, test4 = makeBOW('sales_wkday')
train5, test5 = makeBOW('timeslot')
train6, test6 = makeBOW('week')
train7, test7 = makeBOW('season')
#train6, test6 = makeBOW('dis_rate_ca')
X_train = pd.concat([train1, train2, train3,train4,train5,train6,train7,train_1,train_2], axis=1).values
X_test = pd.concat([test1, test2, test3,test4,test5,test6,test7,test_1,test_2], axis=1).values
max_features = X_train.shape[1]
from keras import models
from keras import layers
from keras.optimizers import RMSprop
from keras import regularizers
from keras.callbacks import EarlyStopping

model3 = models.Sequential()
model3.add(layers.Dense(1, input_shape=(max_features,), kernel_regularizer=regularizers.l2(0.01)))
model3.add(layers.Activation('sigmoid'))

model3.summary()

model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model3.fit(X_train, y_train, epochs=100, batch_size=64, 
                    validation_split=0.2, callbacks=[EarlyStopping(patience=5)])
pred_nn = model3.predict(X_test)[:,0]
ensembled_prediction = (0.5*pred)+(0.5*pred_nn)
fname = 'submissions.csv'
submissions = pd.concat([IDtest, pd.Series(ensembled_prediction, name="gender")] ,axis=1)
submissions.to_csv(fname, index=False)