import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#%matplotlib inline
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#from sklearn.linear_model import LogisticRegression
#from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#from sklearn.metrics import roc_curve, auc


df = pd.read_csv('../input/application_train.csv', index_col='SK_ID_CURR')
ts = df['TARGET']

df = df.drop(labels=['TARGET'], axis=1)

df_num = df.select_dtypes(include=[np.number])
df_cat = df.select_dtypes(exclude=[np.number])

criteria_s = df_num.nunique()==2
binary_cols = df_num.columns[criteria_s].tolist()

df_bin = df_num[binary_cols]
df_num = df_num.drop(labels=binary_cols, axis=1)

df_cat = pd.get_dummies(df_cat)
df_num = df_num.fillna(0)

k1 = df_num['AMT_CREDIT']/df_num['AMT_INCOME_TOTAL']
k2 = df_num['AMT_ANNUITY']/df_num['AMT_CREDIT']
k3 = df_num['AMT_ANNUITY']/df_num['AMT_INCOME_TOTAL']
k4 = df_num['AMT_GOODS_PRICE']/df_num['AMT_INCOME_TOTAL']
k5 = df_num['AMT_GOODS_PRICE']/df_num['AMT_CREDIT']
k6 = df_num['AMT_GOODS_PRICE']/(df_num['AMT_ANNUITY'].where(df_num['AMT_ANNUITY']!=0, 1))

k7 = df_num['AMT_CREDIT']/(df_num['APARTMENTS_MODE'].where(df_num['APARTMENTS_MODE']!=0, np.nan))
k7 = k7.fillna(0)

k8 = df_num['AMT_CREDIT']/(df_num['LIVINGAREA_MEDI'].where(df_num['LIVINGAREA_MEDI']!=0, np.nan))
k8 = k8.fillna(0)

k9 = df_num['AMT_INCOME_TOTAL']/(df_num['APARTMENTS_MODE'].where(df_num['APARTMENTS_MODE']!=0, np.nan))
k9 = k9.fillna(0)

k10 = df_num['AMT_INCOME_TOTAL']/(df_num['LIVINGAREA_MEDI'].where(df_num['LIVINGAREA_MEDI']!=0, np.nan))
k10 = k10.fillna(0)

s = df_num['DAYS_EMPLOYED'].copy()
s[s>0] = np.nan
df_num['DAYS_EMPLOYED'] = s
df_num = df_num.fillna(0)

k11 = df_num['AMT_INCOME_TOTAL']/(df_num['DAYS_EMPLOYED'].where(df_num['DAYS_EMPLOYED']!=0, np.nan))
k11 = k11.fillna(0).abs()

k12 = df_num['AMT_GOODS_PRICE'] * df_num['DAYS_EMPLOYED'] / df_num['AMT_INCOME_TOTAL']
k12 = k12.abs()

df_num['K1']= k1
df_num['K2']= k2
df_num['K3']= k3
df_num['K4']= k4
df_num['K5']= k5
df_num['K6']= k6
df_num['K7']= k7
df_num['K8']= k8
df_num['K9']= k9
df_num['K10']= k10
df_num['K11']= k11
df_num['K12']= k12

num_data_norm = StandardScaler().fit_transform(df_num.values)
df_num_norm = pd.DataFrame(data=num_data_norm, index=df.index)

big_df = pd.concat([df_num_norm, df_bin, df_cat], axis=1)

data = big_df.values
target = ts.values

tree = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=4, max_features=None, max_leaf_nodes=None, min_samples_leaf=50, min_weight_fraction_leaf=0.0, presort=False, random_state=120, splitter='best')
tree.fit(data, target)


leave_id = tree.apply(data)
leave_id = pd.Series(leave_id)

train_data_df = pd.DataFrame(data)
train_data_df['NODE'] = leave_id
train_data_df['TARGET'] = pd.Series(target)
grouped = train_data_df.groupby(['NODE'])

nodes_s = leave_id.value_counts()

rfs = {}



for node_num in nodes_s.keys().values:
    current_df = grouped.get_group(node_num)
    current_target_s = current_df['TARGET']
    if current_target_s.mean() == 0:
        rfs[node_num] = 0
        continue
    if current_target_s.mean() == 1:
        rfs[node_num] = 1
        continue
    current_data_df = current_df.drop(labels=['TARGET', 'NODE'], axis=1)
    rf = RandomForestClassifier( criterion='entropy', n_estimators=150, max_depth=5 )
    rf.fit(current_data_df.values, current_target_s.values)
    rfs[node_num] = rf


test_df = pd.read_csv('../input/application_test.csv', index_col='SK_ID_CURR')

df_num = test_df.select_dtypes(include=[np.number])
df_cat = test_df.select_dtypes(exclude=[np.number])

criteria_s = df_num.nunique()==2
criteria_s2 = df_num.nunique()==1
criteria_s3 = criteria_s | criteria_s2
binary_cols = df_num.columns[criteria_s3].tolist()

df_bin = df_num[binary_cols]
df_num = df_num.drop(labels=binary_cols, axis=1)

df_cat = pd.get_dummies(df_cat)
df_num = df_num.fillna(0)

k1 = df_num['AMT_CREDIT']/df_num['AMT_INCOME_TOTAL']
k2 = df_num['AMT_ANNUITY']/df_num['AMT_CREDIT']
k3 = df_num['AMT_ANNUITY']/df_num['AMT_INCOME_TOTAL']
k4 = df_num['AMT_GOODS_PRICE']/df_num['AMT_INCOME_TOTAL']
k5 = df_num['AMT_GOODS_PRICE']/df_num['AMT_CREDIT']
k6 = df_num['AMT_GOODS_PRICE']/(df_num['AMT_ANNUITY'].where(df_num['AMT_ANNUITY']!=0, 1))

k7 = df_num['AMT_CREDIT']/(df_num['APARTMENTS_MODE'].where(df_num['APARTMENTS_MODE']!=0, np.nan))
k7 = k7.fillna(0)

k8 = df_num['AMT_CREDIT']/(df_num['LIVINGAREA_MEDI'].where(df_num['LIVINGAREA_MEDI']!=0, np.nan))
k8 = k8.fillna(0)

k9 = df_num['AMT_INCOME_TOTAL']/(df_num['APARTMENTS_MODE'].where(df_num['APARTMENTS_MODE']!=0, np.nan))
k9 = k9.fillna(0)

k10 = df_num['AMT_INCOME_TOTAL']/(df_num['LIVINGAREA_MEDI'].where(df_num['LIVINGAREA_MEDI']!=0, np.nan))
k10 = k10.fillna(0)

s = df_num['DAYS_EMPLOYED'].copy()
s[s>0] = np.nan
df_num['DAYS_EMPLOYED'] = s
df_num = df_num.fillna(0)

k11 = df_num['AMT_INCOME_TOTAL']/(df_num['DAYS_EMPLOYED'].where(df_num['DAYS_EMPLOYED']!=0, np.nan))
k11 = k11.fillna(0).abs()

k12 = df_num['AMT_GOODS_PRICE'] * df_num['DAYS_EMPLOYED'] / df_num['AMT_INCOME_TOTAL']
k12 = k12.abs()

df_num['K1']= k1
df_num['K2']= k2
df_num['K3']= k3
df_num['K4']= k4
df_num['K5']= k5
df_num['K6']= k6
df_num['K7']= k7
df_num['K8']= k8
df_num['K9']= k9
df_num['K10']= k10
df_num['K11']= k11
df_num['K12']= k12

num_data_norm = StandardScaler().fit_transform(df_num.values)
df_num_norm = pd.DataFrame(data=num_data_norm, index=test_df.index)

big_test_df = pd.concat([df_num_norm, df_bin, df_cat], axis=1)

s = set(big_df.columns.tolist())
ss = set(big_test_df.columns.tolist())

dif = s - ss
a =list(dif)
b = {}
for i in range(len(a)):
    b[big_df.columns.get_loc(a[i])] = a[i]
for i in sorted(b):
    big_test_df.insert(i, b[i], 0)

labels = big_df.columns.tolist()
c={}
for i in range(len(labels)):
    c[big_df.columns.get_loc(labels[i])] = labels[i]
''' 
for i in sorted(c):
    if big_test_df.columns.get_loc(c[i]) != i:
        print(i, 'BAD')
    else:
        print(i, 'OK')
'''
test_data = big_test_df.values

leave_id = tree.apply(test_data)
leave_id = pd.Series(leave_id)

test_data_df = pd.DataFrame(test_data)
test_data_df['NODE'] = leave_id

grouped = test_data_df.groupby(['NODE'])

pred = pd.Series(index = test_data_df.index.tolist())


nodes_s = leave_id.value_counts()

for node_num in nodes_s.keys().values:
    current_df = grouped.get_group(node_num)
    current_data_df = current_df.drop(labels=['NODE'], axis=1)
    if current_data_df.empty:
        continue
    if rfs[node_num]==0:
        local_predictions = pd.Series(np.zeros(len(current_data_df.index)), index = current_data_df.index.tolist())
        pred.update(local_predictions)
        continue
    if rfs[node_num]==1:
        local_predictions = pd.Series( np.ones(len(current_data_df.index)), index = current_data_df.index.tolist())
        pred.update(local_predictions)
        continue  
    local_predictions = pd.Series( rfs[node_num].predict_proba(current_data_df.values)[:,1], index = current_data_df.index.tolist())
    pred.update(local_predictions)


print('Ok')
pred.shape
sub_df = pd.DataFrame()
sub_df['SK_ID_CURR'] = test_df.index
sub_df['TARGET'] = pred.values
sub_df = sub_df.set_index('SK_ID_CURR')
sub_df.to_csv('sub_1.csv')

print('Done')
