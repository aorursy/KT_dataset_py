import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import time
from IPython.display import display # Allows the use of display() for DataFrames
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.describe()
train['TARGET'].value_counts()
print (plt.style.available)
plt.style.use('ggplot')

# var15 is AGE
train['var15'].hist(bins=25)
plt.title('var15 = AGE');
# under 23 do not complain
print (len(train['TARGET'][train.var15<23]))
print (sum(train['TARGET'][train.var15<23]))
## use for spltting up combined train+test data, recording predictions
labels = train['TARGET'].values
df_train = train.drop(['TARGET'], axis=1)
df_test = test.copy()
id_test = test['ID']
piv_train = train.shape[0]
# Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

# Removing id
# we can also remove id later
df_all = df_all.drop(['ID'], axis=1)

print(df_all.isnull().sum() / df_all.shape[0])

df_all.head()
# count number of zeros
df_all['n0'] = (df_all==0).sum(axis=1)
print (df_all.shape)
# remove constant columns (std = 0)
remove = []
for col in df_all.columns:
    if df_all[col].std() == 0:
        remove.append(col)

df_all.drop(remove, axis=1, inplace=True)
print (df_all.shape)
# remove features that are constant for all rows
constants = []
for f in df_all.columns:
    print (len(np.unique(df_all[f])), '/', len(df_all[f]))
    if len(np.unique(df_all[f])) == 1:
        constants.append(f)
display(constants)

df_all = df_all.drop(constants, axis=1)
print (df_all.shape)
# keep track of columns to remove
remove = []
cols = df_all.columns

# loop thru cols to find equal values
for i in range(len(cols)-1):
    v = df_all[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,df_all[cols[j]].values):
            remove.append(cols[j])
print (len(remove))
print (remove)
print ((df_all.num_var13_medio).value_counts())
print ((df_all.saldo_medio_var13_medio_ult1).value_counts())
# extend the list of cols to remove
remove += ['num_var13_medio', 'saldo_medio_var13_medio_ult1']
len(remove)
# update df_all with dupes removed
df_all.drop(remove, axis=1, inplace=True)
print (df_all.shape)
# kaggle script: https://www.kaggle.com/zfturbo/santander-customer-satisfaction/to-the-top-v3/output 
# limit vars in test based on min and max vals of train
print('Setting min-max lims on test data...')

st = time.time()
for f in df_all.columns[:-1]:
    lim_min = np.min(df_train[f])
    df_all.loc[:, f][df_all[f] < lim_min] = lim_min

    lim_max = np.max(df_train[f])
    df_all.loc[:, f][df_all[f] > lim_max] = lim_max
    #print (f, ': min=', lim_min, ', max=', lim_max)

print (time.time() - st)
f = 'imp_ent_var16_ult1'
print (df_train[f].max())
print (df_all[f].max())
# look at log transform, train
plt.hist(np.log(train['var38']), bins=50)
plt.title('Training set')
plt.gcf().set_size_inches(8,3)
# look at log transform, test
plt.hist(np.log(test['var38']), bins=50)
plt.title('Test set')
plt.gcf().set_size_inches(8,3)
# look at distribution of number of zeros per row
plt.hist(df_all['n0'], bins=50)
plt.gcf().set_size_inches(8,3)
# log transform 'var38'
df_all['var38'] = np.log(df_all['var38'])

plt.hist(df_all['var38'], bins=50);
#Splitting train and test
vals = df_all.values
X = vals[:piv_train]
X_test_submit = vals[piv_train:]
y = labels

print (vals.shape)
print (X.shape)
print (X_test_submit.shape)
# normalize data
from sklearn.preprocessing import normalize
df_norm = normalize(df_all, axis=0)
#Splitting train and test
vals = df_norm
X = vals[:piv_train]
X_test_submit = vals[piv_train:]
y = labels
print (y.shape)
print (X.shape)
print (X_test_submit.shape)
from sklearn.model_selection import train_test_split

## split into train and validation
## use X,y for full training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.02, stratify=y, random_state=42)

print(X_train.shape)
print(y_train.shape)

print(X_valid.shape)
print(y_valid.shape)
#Classifier
# try using... objective: multi:softprob, rank:pairwise
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

ratio = float(np.sum(y == 1)) / np.sum(y==0)
est = XGBClassifier(max_depth=5, learning_rate=0.0202, n_estimators=556,
                    objective='binary:logistic', subsample=0.69, colsample_bytree=0.81, 
                    scale_pos_weight=ratio, seed=1776)                  

param = {
    'objective':'multi:softprob',                    
    'max_depth':6, 
    'learning_rate':0.25,
    'min_child_weight': 3,
    'n_estimators':43,                 
    'subsample':0.6, 
    'colsample_bytree':0.6,
    'num_class' :12
    }
# set the validation set for xgb training
eval_set = [(X_train,y_train), (X_valid,y_valid)]

# try using... eval_metric: mlogloss, merror, ndcg@n-, logloss, auc
est = XGBClassifier()
est.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=100, eval_metric='logloss')
# view the results, xgb
print(metrics.classification_report(y, est.predict(X)))
## xgb, predict probabilities
ypred = est.predict_proba(X)
print (metrics.roc_auc_score(y, ypred[:,1]))
# remember features that predict 0
var15 = test['var15']
saldo_medio_var5_hace2 = test['saldo_medio_var5_hace2']
saldo_var33 = test['saldo_var33']
var38 = test['var38']
var21 = test['var21']
num_var30 = test['num_var30']
num_var13_0 = test['num_var13_0']
num_var33_0 = test['num_var33_0']
imp_ent_var16_ult1 = test['imp_ent_var16_ult1']
imp_op_var39_comer_ult3 = test['imp_op_var39_comer_ult3']
saldo_medio_var5_ult3 = test['saldo_medio_var5_ult3']
## predict test, class probabilities
ypred_submit = est.predict_proba(X_test_submit)
print (X_test_submit.shape)
print (ypred_submit.shape)
print (ypred_submit[:7])
# remember features that predict 0
zero_feats = [('saldo_medio_var5_hace2', 160000), 
              ('saldo_var33', 0),
              ('var38', 3988596),
              ('var21', 7500),
              ('num_var30', 9),
              ('num_var13_0', 6),
              ('num_var33_0', 0),
              ('imp_ent_var16_ult1', 51003),
              ('imp_op_var39_comer_ult3', 13184),
              ('saldo_medio_var5_ult3', 108251),
             ]
for x,y in zero_feats:
    print (x, train[(train[x] > y) & (train.TARGET==1)].shape)
# Under 23 year olds are always happy
ypred_submit[list(var15.index[var15 < 23]), 1] = 0
ypred_submit[list(saldo_var33.index[saldo_var33 > 0]), 1] = 0
ypred_submit[list(var38.index[var38 > 3988596]), 1] = 0

#ypred_submit[list(saldo_medio_var5_hace2.index[saldo_medio_var5_hace2 > 160000]), 1] = 0
ypred_submit[list(var21.index[var21 > 7500]), 1] = 0
ypred_submit[list(num_var30.index[num_var30 > 9]), 1] = 0
ypred_submit[list(num_var13_0.index[num_var13_0 > 6]), 1] = 0
ypred_submit[list(num_var33_0.index[num_var33_0 > 0]), 1] = 0
ypred_submit[list(imp_ent_var16_ult1.index[imp_ent_var16_ult1 > 51003]), 1] = 0
ypred_submit[list(imp_op_var39_comer_ult3.index[imp_op_var39_comer_ult3 > 13184]), 1] = 0
ypred_submit[list(saldo_medio_var5_ult3.index[saldo_medio_var5_ult3 > 108251]), 1] = 0

ypred_submit[:7]
# Generate submission: stack ids and targets together into dataframe
sub = pd.concat([id_test, pd.Series(ypred_submit[:,1], name='TARGET')], axis=1)

# write dataframe to csv
#sub.to_csv('submission.csv',index=False)
sub.to_csv('submission.csv',index=False, float_format='%.16f')