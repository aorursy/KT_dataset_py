import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
path = os.path.join('../input/train.csv')
data_train = pd.read_csv(path)
path = os.path.join('../input/test.csv')
data_test = pd.read_csv(path)
id = data_test['id']
data_test.info()
def drop_null(df):
    df = df.drop(['id','ps_car_03_cat', 'ps_car_05_cat', 'ps_reg_03'], axis=1)
    return df 

data_train = drop_null(data_train)
data_test = drop_null(data_test)
print(len(data_train.columns))
print(len(data_test.columns))
# data_train.info()
data_train.replace(-1, np.nan, inplace=True)
data_train.fillna(method='pad',inplace=True)
# data_train.fillna(0, inplace=True)
data_test.replace(-1, np.nan, inplace=True)
data_test.fillna(method='pad',inplace=True)
all_col = [i for i in data_train.columns if i not in ['id','target']]
bin_col = [i for i in all_col if 'bin' in i]
cat_col = [i for i in all_col if i.endswith('cat')]
else_col = [i for i in all_col if 'bin' not in i]
else_col = [i for i in else_col if 'cat' not in i]
print(bin_col, '\n', cat_col, '\n', else_col)
print(len(bin_col)+len(cat_col)+len(else_col))
#bin_
bin_values0 = []
bin_values1 = []
for i in bin_col:
    bin_values0.append(len(data_train[i])-sum(data_train[i]))
    bin_values1.append(sum(data_train[i]))

bin_values = pd.DataFrame({'bin_values0':bin_values0, 'bin_values1':bin_values1})
# bin_values.plot(kind='bar', stacked=True)
bin_col = bin_col[:4] + bin_col[8:]
print(bin_col)
#else
# data_train[else_col].head()
# plt.figure(figsize=(16,12))
df_else = pd.concat([data_train[else_col], data_train['target']], axis=1)
# sns.heatmap(df_else.corr(), annot=True)
corr = df_else.corr()
else_col = corr[corr['target']>0.001].index
else_col = else_col[:-1]
def transfrom_bin(train_test, bin_col=bin_col):
    df = train_test[bin_col]
    return df

def transfrom_cat(df, train_test):
    for i in cat_col:
        cat = pd.get_dummies(train_test[i])
        df = pd.concat([df, cat], axis=1)
    return df 

def transfrom_else(df, train_test):
    for i in else_col:
        els = train_test[i]
        df = pd.concat([df, els], axis=1)
    return df

def new_col(df, train_test):
    df['ps_car_13_ps_ind_03'] = train_test['ps_car_13']*train_test['ps_ind_03']
    return df
df_train = transfrom_bin(data_train)
df_train = transfrom_cat(df_train, data_train)
df_train = transfrom_else(df_train, data_train)
df_train = new_col(df_train, data_train)
print(df_train.shape)
df_test = transfrom_bin(data_test)
df_test = transfrom_cat(df_test, data_test)
df_test = transfrom_else(df_test, data_test)
df_test = new_col(df_test, data_test)
print(df_test.shape)
#Ensemble
#model1
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
rf_model = RandomForestClassifier(max_depth= 6, min_samples_leaf= 2, n_jobs= -1)
et_model = ExtraTreesClassifier(max_depth= 8, min_samples_leaf= 2, n_jobs= -1)
ad_model = AdaBoostClassifier(learning_rate= 0.75)
# sv_model = SVC(kernel= 'linear', C = 0.025)
lg_model = LogisticRegression(C=0.1, random_state=0)
gd_model = GradientBoostingClassifier(max_depth= 5,  min_samples_leaf=2)

all_model = [rf_model, et_model, ad_model, lg_model, gd_model]

from sklearn.model_selection import KFold
kf = KFold(n_splits=5,shuffle=False)
# for test_,train_ in kf.split(df_train.index):
#     print(len(test_), len(train_))
    
train_model1 = pd.DataFrame({'target':np.zeros(len(df_train))}, index=df_train.index)
test_model1 = pd.DataFrame({'target':np.zeros(len(df_test))}, index=df_test.index)

for i, (train_,val_) in enumerate(kf.split(df_train.index)):
    print('begin %d time' %i)
    model = all_model[i]
    train_kf = df_train.iloc[train_]
    validation_kf = df_train.iloc[train_]
    model.fit(train_kf, data_train['target'].iloc[train_])
    pre_val = model.predict_proba(validation_kf)
    train_model1['target'].iloc[train_] = pre_val[:,1]
    
    pre_test = model.predict_proba(df_test)
    test_model1['target'] += pre_test[:,1] 
    print('finish %d time' %i)
    
test_model1 = test_model1/5   
print(train_model1)
print(test_model1)
# test_model1 = test_model1/5
# print(test_model1)
'''
model2 = RandomForestClassifier(max_depth= 6, min_samples_leaf= 2, n_jobs= -1)
model2.fit(train_model1, data_train['target'])
pre_result = model2.predict_proba(test_model1)
print(pre_result[: 1])
'''
import xgboost as xgb
gbm = xgb.XGBClassifier(
 n_estimators= 200,
 max_depth= 4,
 min_child_weight= 2,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1)
gbm.fit(train_model1, data_train['target'])
gbm_result = gbm.predict_proba(test_model1)
print(gbm_result[: 1])

# target = mymodel.predict_proba(test_X)

# submission = pd.DataFrame({'target':gbm_result[: ,1]}, index=id)
# submission.to_csv('submission.csv')
submission = pd.DataFrame({'target':gbm_result[: ,1] , 'id':id})
submission.to_csv("Submission.csv", index=False)