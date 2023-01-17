!pip install scikit-misc
!pip install scikit-learn==0.21.3
!pip install fastai==0.7.0
import numpy as np 
import pandas as pd 
from fastai.imports import *
from fastai.structured import *
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv', low_memory=False)
test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv', low_memory=False)
bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv', low_memory=False)
bureau_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv', low_memory=False)
grouped = bureau_balance.groupby(['SK_ID_BUREAU'])

a = pd.DataFrame()
a['SK_ID_BUREAU'] = grouped.groups
a['MIN_MONTHS_BALANCE'] = grouped['MONTHS_BALANCE'].agg(np.min)
a['SUM_MONTHS_BALANCE'] = grouped['MONTHS_BALANCE'].agg(np.sum)
a['MEAN_MONTHS_BALANCE'] = grouped['MONTHS_BALANCE'].agg(np.mean)
a['SIZE_MONTHS_BALANCE'] = grouped['MONTHS_BALANCE'].agg(np.size)
a['NUM_STATUS_NUMBER'] = grouped['STATUS'].agg(np.size)
a
bureau = pd.merge(bureau, a, on = 'SK_ID_BUREAU', how='left')
bureau
train_cats(bureau)
testt, _, _ = proc_df(bureau, max_n_cat=60)
grouped = testt.groupby(['SK_ID_CURR'])
bu = pd.DataFrame()
bu['SK_ID_CURR'] = grouped.groups
col_name = grouped.get_group(100001).columns
for c in col_name:
    bu['MIN_{0}'.format(c)] = grouped[c].agg(np.min)
    bu['SUM_{0}'.format(c)] = grouped[c].agg(np.sum)
    bu['MEAN_{0}'.format(c)] = grouped[c].agg(np.mean)
    bu['SIZE_{0}'.format(c)] = grouped[c].agg(np.size)
bu
bu.reset_index(drop = True, inplace = True)
bu
bu.to_feather('burea_final')
train.shape, test.shape
trainn = pd.merge(train, bu, how='left', on='SK_ID_CURR')
testt = pd.merge(test, bu, how='left', on='SK_ID_CURR')
trainn.shape, testt.shape
train = trainn
test = testt
print('Training data shape: {}'.format(train.shape))
train.head()
print('Test data shape: {}'.format(test.shape))
test.head()
train['TARGET'].value_counts()
train['TARGET'].plot.hist()
train.dtypes.value_counts()
train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
train['NAME_EDUCATION_TYPE'].value_counts()
train['WEEKDAY_APPR_PROCESS_START'].value_counts()
train_cats(train)
train['NAME_EDUCATION_TYPE'].cat.categories
train['NAME_EDUCATION_TYPE'].cat.set_categories(['Lower secondary','Secondary / secondary special','Incomplete higher','Higher education','Academic degree'],
                                               ordered = True, inplace = True)
train['NAME_EDUCATION_TYPE'].cat.categories
train['WEEKDAY_APPR_PROCESS_START'].cat.set_categories(['MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY', 'SATURDAY', 'SUNDAY'],
                                               ordered = True, inplace = True)
train['WEEKDAY_APPR_PROCESS_START'].cat.categories
apply_cats(test, train)
train.shape
test.shape
X, y, nas = proc_df(train, 'TARGET', max_n_cat=60)
X_test,_,nas = proc_df(test, na_dict=nas,  max_n_cat=60)
X, y, nas = proc_df(train, 'TARGET', na_dict=nas,  max_n_cat=60)
X.shape, X_test.shape
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m, X_train, X_valid, y_train, y_valid):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)

def split_vals(a,n): return a[:n].copy(), a[n:].copy()
n_valid = 60000
n_trn = len(train)-n_valid
X_train, X_valid = split_vals(X, n_trn)
y_train, y_valid = split_vals(y, n_trn)
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
set_rf_samples(100000)
model = RandomForestClassifier(n_jobs=-1, n_estimators=40)
model.fit(X_train, y_train)
print_score(model, X_train, X_valid, y_train, y_valid)
prediction = model.predict_proba(X_test)[:,1]
submission = pd.DataFrame()
submission['SK_ID_CURR']=X_test.SK_ID_CURR
submission['TARGET']=prediction
submission.to_csv('submission.csv',index=False) # First Model
fi = rf_feat_importance(model, X_train)
fi[:30]
def plot_fi(fi): 
    return fi.plot('cols','imp','barh', figsize=(12,7), legend=False)
plot_fi(fi[fi.imp>0.0095])
important_col = fi[fi.imp>0.0095].cols
X_train_imp = X_train.copy()
X_valid_imp = X_valid.copy()
X_train_imp = X_train_imp[important_col]
X_valid_imp = X_valid_imp[important_col]
X_train_imp.shape, X_valid_imp.shape 
set_rf_samples(100000)
model = RandomForestClassifier(n_jobs=-1, n_estimators=40)
model.fit(X_train_imp, y_train)
print_score(model, X_train_imp, X_valid_imp, y_train, y_valid)
X_test_imp = X_test.copy()
X_test_imp = X_test_imp[important_col]
prediction = model.predict_proba(X_test_imp)[:,1]
submission = pd.DataFrame()
submission['SK_ID_CURR']=X_test_imp.SK_ID_CURR
submission['TARGET']=prediction
submission.to_csv('submission1.csv',index=False) # Second Model
fi = rf_feat_importance(model, X_train_imp)
fi
plot_fi(fi)
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(X_train_imp).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=X_train_imp.columns, 
      orientation='left', leaf_font_size=16)
plt.show()
def model_score(df_train, df_valid):
    set_rf_samples(100000)
    model = RandomForestClassifier(n_jobs=-1, n_estimators=40)
    model.fit(df_train, y_train)
    print_score(model, df_train, df_valid, y_train, y_valid)
for c in ['AMT_GOODS_PRICE', 'AMT_CREDIT']:
    print(c)
    temp_train = X_train_imp.drop(c, axis = 1)
    temp_valid = X_valid_imp.drop(c, axis = 1)
    model_score(temp_train, temp_valid)
X_train_imp = X_train_imp.drop(['AMT_GOODS_PRICE'], axis = 1)
X_valid_imp = X_valid_imp.drop(['AMT_GOODS_PRICE'], axis = 1)
X_test_imp = X_test_imp.drop(['AMT_GOODS_PRICE'], axis = 1)
X_train_imp.dtypes.value_counts()
X_train_imp.select_dtypes('int64').apply(pd.Series.nunique, axis = 0)
(X_train_imp['DAYS_BIRTH'] / 365).describe()
(X_train_imp['DAYS_EMPLOYED'] / 365).describe()
X_train_imp['DAYS_EMPLOYED'].plot.hist()
X_test_imp['DAYS_EMPLOYED'].plot.hist()
X_train_imp['DAYS_EMPLOYED'].replace({365243:np.nan}, inplace = True)
X_valid_imp['DAYS_EMPLOYED'].replace({365243:np.nan}, inplace = True)
X_test_imp['DAYS_EMPLOYED'].replace({365243:np.nan}, inplace = True)
X_train_imp['DAYS_EMPLOYED'].plot.hist()
X_valid_imp['DAYS_EMPLOYED'].plot.hist()
X_test_imp['DAYS_EMPLOYED'].plot.hist()
X_train_imp.shape, X_valid_imp.shape, X_test_imp.shape
X_train_imp = pd.concat([X_train_imp, X_valid_imp])
X_train_impp, _, nas = proc_df(X_train_imp)
X_test_impp,_ ,nas = proc_df(X_test_imp, na_dict=nas)
X_train_impp, _, nas = proc_df(X_train_imp, na_dict=nas)
n_valid = 60000
n_trn = len(train)-n_valid
X_train_impp, X_valid_impp = split_vals(X_train_impp, n_trn)
X_valid_impp['DAYS_EMPLOYED_na'].value_counts()
set_rf_samples(100000)
model = RandomForestClassifier(n_jobs=-1, n_estimators=40)
model.fit(X_train_impp, y_train)
print_score(model, X_train_impp, X_valid_impp, y_train, y_valid)
fi = rf_feat_importance(model, X_train_impp)
plot_fi(fi)
prediction = model.predict_proba(X_test_impp)[:,1]
submission = pd.DataFrame()
submission['SK_ID_CURR']=X_test_impp.SK_ID_CURR
submission['TARGET']=prediction
submission.to_csv('submission2.csv',index=False) # Third Model
fi = rf_feat_importance(model, X_train_impp)
plot_fi(fi)
X_train_istest = X_train_impp.copy()
X_valid_istest = X_valid_impp.copy()
X_test_istest = X_test_impp.copy()
X_train_istest['is_test'] = 0
X_valid_istest['is_test'] = 0
X_test_istest['is_test'] = 1
y_train_istest = X_train_istest['is_test']
y_valid_istest = X_valid_istest['is_test']
y_test_istest = X_test_istest['is_test']
X_train_istest = X_train_istest.drop('is_test', axis = 1)
X_valid_istest = X_valid_istest.drop('is_test', axis = 1)
X_test_istest = X_test_istest.drop('is_test', axis = 1)
y_train_istest.shape, y_test_istest.shape
XX = pd.concat([X_train_istest, X_valid_istest, X_test_istest])
yy = pd.concat([y_train_istest, y_valid_istest, y_test_istest])
set_rf_samples(100000)
model = RandomForestClassifier(n_jobs=-1, n_estimators=40, oob_score=True)
model.fit(XX, yy)
model.oob_score_
fi = rf_feat_importance(model, XX)
plot_fi(fi)
for c in ['MIN_SK_ID_BUREAU','AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_ID_PUBLISH', 'DAYS_BIRTH', 'DAYS_LAST_PHONE_CHANGE', 'SK_ID_CURR']:
    print(c)
    temp_train = X_train_impp.drop(c, axis = 1)
    temp_valid = X_valid_impp.drop(c, axis = 1)
    model_score(temp_train, temp_valid)
X_train_final = X_train_impp.drop('MIN_SK_ID_BUREAU', axis = 1)
X_valid_final = X_valid_impp.drop('MIN_SK_ID_BUREAU', axis = 1)
X_test_final = X_test_impp.drop('MIN_SK_ID_BUREAU', axis = 1)
set_rf_samples(100000)
model = RandomForestClassifier(n_jobs=-1, n_estimators=40)
model.fit(X_train_final, y_train)
print_score(model, X_train_final, X_valid_final, y_train, y_valid)
submission = pd.DataFrame()
submission['SK_ID_CURR']=X_test_impp.SK_ID_CURR
prediction = model.predict_proba(X_test_final)[:,1]
submission['TARGET']=prediction
submission.to_csv('submission3.csv',index=False) # Fourth Model
set_rf_samples(100000)
model = RandomForestClassifier(n_jobs=-1, n_estimators=40, max_features=0.5, min_samples_leaf=3)
model.fit(X_train_final, y_train)
print_score(model, X_train_final, X_valid_final, y_train, y_valid)
submission = pd.DataFrame()
submission['SK_ID_CURR']=X_test_impp.SK_ID_CURR
prediction = model.predict_proba(X_test_final)[:,1]
submission['TARGET']=prediction
submission.to_csv('submission4.csv',index=False) # Fifth Model
reset_rf_samples()
model = RandomForestClassifier(n_jobs=-1, n_estimators=200, max_features=0.5, min_samples_leaf=7)
model.fit(X_train_final, y_train)
print_score(model, X_train_final, X_valid_final, y_train, y_valid)
submission = pd.DataFrame()
submission['SK_ID_CURR']=X_test_impp.SK_ID_CURR
prediction = model.predict_proba(X_test_final)[:,1]
submission['TARGET']=prediction
submission.to_csv('submission5.csv',index=False) # Sixth Model