import pandas as pd
import gc
import lightgbm as lgb
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold



import xgboost as xg

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.head(3)
train["DG3"].value_counts()
train["DG3"].isnull().value_counts()
train["DG6"].value_counts()
train.groupby("DG6")["is_female"].mean()
for x in train,test:
    train.loc[train["DG6"]!=2,"DG6"]==0
    test.loc[test["DG6"]!=2,"DG6"]==0
train.groupby("DL1")["is_female"].mean()
for x in train,test:
    train.loc[train["DL1"]!=7,"DL1"]==0
    test.loc[test["DL1"]!=7,"DL1"]==0
train.groupby("DG3")["is_female"].mean()
train.groupby(["DL0"])["is_female"].mean()
len(train.columns)
train["DG3A_OTHERS"].value_counts()
train.select_dtypes(include=[object]).isnull().sum()/len(train)
a=[x for x in train.columns if "OTHERS"  not in x]

train=train[a]

#alt=[x for x in a if x not in ["train_id","is_female"]]
#for x in alt:
#    train.loc[train[x]==96,x]=np.nan
#    test.loc[test[x]==96,x]=np.nan
categorical=["AA3","AA5","AA6","AA8","DG3","DG3A","DG14","DL2","DL5","DL27","DL28","MT1A","MT5","MT6","MT6A",
"MT6B","MT7A","MT9","MT11","FF13","MM10B","MM12","MM13","MM14","MM18","MM19","MM20","MM21","MM28","MM30",
"MM34","MM41","IFI5_1","IFI5_2","IFI5_3","IFI24","FL4","FL9A","FL9B","FL9C","FL10","FB2","FB19","FB20",
"FB21","FB24","FB25"]
#TODOS LOS G2P2,TODOS LOS MT13,TODOS LOS MT14 PERO NO LOS MT14A,TODOS LOS MM11,TODOS LOS FB2

a=[x for x in train.columns if "G2P2" in x]
b=[x for x in train.columns if "MT13" in x]
c=[x for x in train.columns if "MT14_" in x]
d=[x for x in train.columns if "MM11" in x]
e=[x for x in train.columns if "FB2" in x]
s=a+b+c+d+e

categorical.extend(s)
alt=[x for x in a if x not in ["train_id","is_female"] and x not in categorical]
for x in alt:
    train.loc[train[x]==96,x]=np.nan
    test.loc[test[x]==96,x]=np.nan
    train.loc[train[x]==99,x]=np.nan
    test.loc[test[x]==99,x]=np.nan
el=[]
for x in train.columns:
    if len(train.loc[train[x].isnull()])/len(train)<0.99 :
        el.append(x)
        
len(el)
len(train.columns)
train=train[el]
elt=[x for x in el if x in test]
test=test[elt]

#len(el)
len(categorical)
categorical=[x for x in categorical if x in train.columns]
for x in categorical:
    print (len(train[x].unique()))
    print ("")
categorical_2=train.select_dtypes(include=[object]).columns.tolist()

categorical.extend(categorical_2)


no_usar=["train_id","is_female"]

features=[x for x in train.columns if x not in no_usar ]

cat_ind=[features.index(x) for x in categorical]

for l in categorical:
    le = preprocessing.LabelEncoder()
    le.fit(list(train[l].dropna())+list(test[l].dropna()))

    train.loc[~train[l].isnull(),l]=le.transform(train.loc[~train[l].isnull(),l])
    test.loc[~test[l].isnull(),l]=le.transform(test.loc[~test[l].isnull(),l])
lgb_train = lgb.Dataset(train.loc[0:len(train)*9/10,features].values, train.loc[0:len(train)*9/10,["is_female"]].values.ravel())
lgb_eval = lgb.Dataset(train.loc[(len(train)*9/10)+1:,features].values, train.loc[(len(train)*9/10)+1:,["is_female"]].values.ravel(), reference=lgb_train)



params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': { 'auc'},
        #'num_leaves':128,
        'learning_rate': 0.01,
         "max_depth" : 6,
        'feature_fraction': 0.7,
         "bagging_freq": 1,
        'bagging_fraction': 0.85,
     "is_unbalance" : False,
        'verbose': 1
}


#lgbm3 = lgb.train(params,lgb_train,num_boost_round=800,valid_sets=lgb_eval,early_stopping_rounds=50,verbose_eval=5,categorical_feature=cat_ind)

lgbm3 = lgb.train(params,lgb_train,num_boost_round=4000,valid_sets=lgb_eval,early_stopping_rounds=50,verbose_eval=5,categorical_feature=cat_ind)
importancia=lgbm3.feature_importance(importance_type="gain")

def impxgb(valores,variables):
    dictimp={variables[a]:valores[a] for a in range(0,len(variables)) }
    xgimp=sorted(list(dictimp.items()), key=lambda x: x[1],reverse=True)

    return xgimp

ixg_l=impxgb(importancia,features)

ixg_l 
kf_previo=KFold(n_splits=5,random_state=256,shuffle=True)

i=1

r=[]

for train_index,test_index in kf_previo.split(train):

    lgb_train = lgb.Dataset(train.loc[train_index,features].values,train.loc[train_index,"is_female"].values.ravel())
    lgb_eval = lgb.Dataset(train.loc[test_index,features].values, train.loc[test_index,"is_female"].values.ravel(), reference=lgb_train)

    params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': { 'auc'},
            #"max_bin":512,
            'learning_rate': 0.01,
             "max_depth" : 9,
            'feature_fraction': 0.7,
             "bagging_freq": 1,
            'bagging_fraction': 0.85,
         "is_unbalance" : False,
            'verbose': 1
    }




    lgbm3 = lgb.train(params,lgb_train,num_boost_round=13100,valid_sets=lgb_eval,early_stopping_rounds=50,verbose_eval=0,categorical_feature=cat_ind)
    test["IS_FEMALE_FOLD_"+str(i)]=lgbm3.predict(test[features].values, num_iteration=lgbm3.best_iteration)
    
    print ("Fold_"+str(i))
    a= roc_auc_score(train.loc[test_index,"is_female"],lgbm3.predict(train.loc[test_index,features].values, num_iteration=lgbm3.best_iteration))
    r.append(a)
    print (a)
    print ("")
    
    i=i+1

print ("mean: "+str(np.mean(np.array(r))))
print ("std: "+str(np.std(np.array(r))))      




a=[x for x in test.columns if "FOLD" in x]

test["is_female"]=test[a].mean(axis=1)
test["test_id"]=range(0,len(test))
test[["test_id","is_female"]].to_csv("submission_24_reducing_catgeories.csv",index=False)
