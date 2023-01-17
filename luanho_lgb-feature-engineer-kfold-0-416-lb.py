import numpy as np 
import pandas as pd 
import os
import gc
from tqdm import tqdm 
from sklearn.preprocessing import LabelEncoder
print(os.listdir("../input"))
# Read data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Glimse data
print(train.shape)
train.head()
print(test.shape)
test.head()
df_combine = pd.concat([train,test], axis = 0)
id_name = 'Id'
target_name = 'Target'
cols = [f_ for f_ in df_combine.columns if df_combine[f_].dtype == 'object' and f_ != id_name ]
print(cols)

# mean encoding
for col in tqdm(cols):
    lb = LabelEncoder()
    lb.fit(df_combine[col])
    train[col] = lb.transform(train[col].astype(str))
    test[col] = lb.transform(test[col].astype(str))
    
del lb
gc.collect()
    
def extract_features(df):
    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']
    df['rent_to_rooms'] = df['v2a1']/df['rooms']
    df['tamhog_to_rooms'] = df['tamhog']/df['rooms']
    df['r4h1_percent_in_male'] = df['r4h1'] / df['r4h3']
    df['r4m1_percent_in_female'] = df['r4m1'] / df['r4m3']
    df['r4h1_percent_in_total'] = df['r4h1'] / df['hhsize']
    df['r4m1_percent_in_total'] = df['r4m1'] / df['hhsize']
    df['r4t1_percent_in_total'] = df['r4t1'] / df['hhsize']
   
    
extract_features(train)
extract_features(test)
cols_to_drop = [id_name ,target_name]
X_train = train.drop(cols_to_drop, axis = 1)
Y_train = train[target_name]
X_train.fillna(0)

import lightgbm as lgb
lgb = lgb.LGBMClassifier(class_weight='balanced',drop_rate=0.9, min_data_in_leaf=100, max_bin=255,
                                 n_estimators=500,min_sum_hessian_in_leaf=1,importance_type='gain',learning_rate=0.1,bagging_fraction = 0.85,
                                 colsample_bytree = 1.0,feature_fraction = 0.1,lambda_l1 = 5.0,lambda_l2 = 3.0,max_depth =  9,
                                 min_child_samples = 55,min_child_weight = 5.0,min_split_gain = 0.1,num_leaves = 45,subsample = 0.75)  



from sklearn.model_selection import KFold

k_folds = 5
kf = KFold(n_splits = k_folds, shuffle = True)

for train_index, test_index in kf.split(X_train, Y_train):
    x_train, x_val = X_train.iloc[train_index] , X_train.iloc[test_index]
    y_train, y_val = Y_train.iloc[train_index] , Y_train.iloc[test_index]
    lgb.fit(x_train, y_train , eval_set= [(x_val, y_val)], early_stopping_rounds = 400)
X_test = test.drop('Id', axis = 1)
X_test.fillna(0)

pres = lgb.predict(X_test)
id_test = test['Id']
subs = pd.DataFrame()
subs['Id'] = id_test
subs['Target'] = pres
subs.to_csv('mean_encoding_target.csv', index = False)
subs.head()
