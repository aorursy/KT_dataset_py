#-*- coding: UTF-8 -*-

# xgboost+lr,excited!!!!!!

import time

import re

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from sklearn.cross_validation import StratifiedKFold

from sklearn.cross_validation import cross_val_score

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.feature_selection import SelectFromModel, VarianceThreshold

import xgboost as xgb

from sklearn.cross_validation import KFold

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import normalize

from sklearn.decomposition import PCA

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print("start time: %s" % time.ctime())

check_time = time.time()

time_name = str(time.ctime())

file_time = "_".join(time_name.split()[:3])

GBDT_FILE = 'gbdt_imbalance_%s.txt' % file_time    #tree node file

original_seed = 1729

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

print('Load data...')

train = pd.read_csv("../input/train.csv")       #update

train_id = train['ID'].values

target = train['TARGET'].values

#train = train.drop(['ID','TARGET'],axis=1)



test = pd.read_csv("../input/test.csv")          #update

test_id = test['ID'].values

#test = test.drop(['ID'],axis=1)

print('load data complete, train records count: <<<<%s, test records count: <<<<%s, train columns count: <<<<%s, test columns count: <<<<%s, time: <<<<%s' % (train.shape[0], test.shape[0], train.shape[1], test.shape[1], round(((time.time() - check_time)/60),2)))

check_time = time.time()

#------------------------------------------data process-------------------------------------------------------------

#removing outliers, MANUALLY

train = train.replace(-999999,2)

test = test.replace(-999999,2)



#replace na

train = train.fillna(-1)

test = test.fillna(-1)



# remove constant columns (std = 0)

remove = []

for col in train.columns:

    if train[col].std() == 0:

        remove.append(col)



train.drop(remove, axis=1, inplace=True)

test.drop(remove, axis=1, inplace=True)

len1 = len(remove)

print(train.shape, test.shape)



# remove duplicated columns

remove = []

cols = train.columns

for i in range(len(cols)-1):

    v = train[cols[i]].values

    for j in range(i+1,len(cols)):

        if np.array_equal(v,train[cols[j]].values):

            remove.append(cols[j])



train.drop(remove, axis=1, inplace=True)

test.drop(remove, axis=1, inplace=True)

len2 = len(remove)



#label encoder

len3 = 0

cols = train.columns

origin_cols = train.columns[1:-1]        #original features

print("origin_cols: %s" % origin_cols)

for col in origin_cols:

     if train[col].dtype=='object':

            print(col)

            len3 += 1

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(train[col].values) + list(test[col].values))

            train[col] = lbl.transform(list(train[col].values))

            test[col] = lbl.transform(list(test[col].values))



print("data process ended, row droped:<<<<%s, row encoded:<<<<%s, time spend: <<<<%s" %(len1+len2, len3, round(((time.time() - check_time)/60),2)))

check_time = time.time()

#useless if test.txt already exists

#-----------------------------------------------------GBDT model------------------------------------------------------------------------

folds = 10

skf = StratifiedKFold(target,

                          n_folds=folds,

                          shuffle=False,

                          random_state=1580)  

a, sample_index = list(skf)[0]

train_sample = train.loc[sample_index,:]

target_sample = target[sample_index]

xgbc = xgb.XGBClassifier(n_estimators=500, seed=1580)

xgbc.fit(train_sample[origin_cols],target_sample)

xgbc.booster().dump_model(GBDT_FILE)

print('gbdt trained on sample, time spend:<<<<%s' % round(((time.time() - check_time)/60),2))

check_time = time.time()

#-----------------------------------------------------READ FEATURE----------------------------------------------------------------------

f = open(GBDT_FILE,'r')

feature_dict = {}

for line in f.readlines():

    if '<' in line:           #feature line

           line = line.split(':')[1].strip()

           feature_re = re.match('\[(.*)?\]', line)

           info = feature_re.group(0)              #should be only one group

           info = re.sub('\[|\]','',info)

           feature = info.split('<')[0].strip()

           value = float(info.split('<')[1].strip())

           value_set = feature_dict[feature] if feature in feature_dict else set()

           value_set.add(value)

           feature_dict[feature] = value_set



#feature encoder

for feature,value_set in feature_dict.items():

    #create two inf of the value_list

    value_list = sorted(list(value_set))

    min1 = value_list[0]

    max1 = value_list[-1]

    min0 = train[feature].min()

    max0 = train[feature].max()

    value_list.insert(0,min0)

    value_list.insert(len(value_list),max0)

    for i, value in enumerate(value_list):

         #no need for the last

         if len(value_list)==i+1:

             break

         #rule: right area of the value

         low_bound = value

         high_bound = value_list[i+1]

         col = "%s_gt_%s_lt_%s" % (feature, low_bound, high_bound)    #name the col

         train[col] = train[feature].apply(lambda x: 1 if x>=low_bound and x<high_bound else 0)

         test[col] = test[feature].apply(lambda x: 1 if x>=low_bound and x<high_bound else 0)



#remove original feature

train = train.drop(origin_cols, axis=1)

test = test.drop(origin_cols, axis=1)

print('feature generated base on gbdt sub-model, <<<<%s feature generated, time spend:<<<<%s' % (test.shape[0]-1, round(((time.time() - check_time)/60),2)))

check_time = time.time()

#--------------------------------------------------------UNDERSAMPLE-------------------------------------------------------------------

train1 = train[train['TARGET']==1]               #positive train samples

train2 = train[train['TARGET']==0]               #negative train samples

#train2 suppose to be the majority type, if not change it

if train2.shape[0]<train1.shape[0]:

    train1 = train[train['TARGET']==0]               #positive train samples

    train2 = train[train['TARGET']==1]               #negative train samples

train1 = train1.reset_index(drop=True)

train2 = train2.reset_index(drop=True)

fold = train2.shape[0] / train1.shape[0]

folds = int(fold)

skf1 = StratifiedKFold(train1.TARGET.values,

                          n_folds=folds,

                          shuffle=False,

                          random_state=1580)

skf2 = StratifiedKFold(train2.TARGET.values,

                          n_folds=folds,

                          shuffle=False,

                          random_state=1580)    

#------------------------------------------------------LVL1 Logistic Regression--------------------------------------------------------

clf = LogisticRegression(penalty='l1', random_state=1580, n_jobs=-1)

features = list(train.columns)

features.remove('ID')

features.remove('TARGET')

df_train_pred = []

df_pred = []

for i, (a, neg_index) in enumerate(skf2):

        fold_tag = "fold_%s" % i

        pos_index = list(skf1)[i][0]

        train_pos = train1.loc[pos_index, :]

        trainner = pd.concat((train_pos, train2.loc[neg_index,:]),axis=0, ignore_index=True)

        y = trainner.TARGET.values

        X = trainner[features]

        clf.fit(X,y)

        train_pred = clf.predict_proba(train[features])[:,1]

        y_pred = clf.predict_proba(test[features])[:,1]

        df_train_pred.append(train_pred)

        df_pred.append(y_pred)



#average results

train_preds = np.average(np.array(df_train_pred), axis=0)

test_preds = np.average(np.array(df_pred), axis=0)

print('LVL1 Logistic Model trained, Average AUC: <<<<%s, time spend:<<<<%s' % (roc_auc_score(train.TARGET.values, train_preds), round(((time.time() - check_time)/60),2)))

check_time = time.time()

#-----------------------------------------------------OUTPUT----------------------------------------------------------------------------

pd.DataFrame({"ID": test_id, "TARGET": test_preds}).to_csv('../output/xgb_lr_imbalance.csv',index=False)

#pd.DataFrame({"ID": test_id, "TARGET": test_preds}).to_csv('../output/gbdt_lr_sub_test.csv',index=False)         #test

print("end time: %s" % time.ctime())