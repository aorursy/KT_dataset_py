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
len(train.columns)
train["DG3A_OTHERS"].value_counts()
train.select_dtypes(include=[object]).isnull().sum()/len(train)
a=[x for x in train.columns if "OTHERS"  not in x]

train=train[a]

alt=[x for x in a if x not in ["train_id","is_female"]]
for x in alt:
    train.loc[train[x]==96,x]=np.nan
    test.loc[test[x]==96,x]=np.nan
categorical=["AA3","AA5","AA6","AA8","DG3","DG3A","DG6","DG14","DL1","DL2","DL5","DL27","DL28","MT1A","MT5","MT6","MT6A",
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
el=[]
for x in train.columns:
    if len(train.loc[train[x].isnull()])/len(train)<0.95 :
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

features=[x for x in train.columns if x not in no_usar and x not in categorical_2]


kf_previo=KFold(n_splits=5,random_state=256,shuffle=True)

i=1

r=[]


for train_index,test_index in kf_previo.split(train):

    
    params = {
        "objective":"binary:logistic",
        "tree_method":"hist", 
        "grow_policy":"depthwise",

        'eta': 0.01,
        'colsample_bytree': 0.4,
        'max_depth': 7,
        'subsample': 0.9,
        'silent': 1,
        'verbose_eval': True,
        "eval_metric":"auc"
    }


    xgtrain = xg.DMatrix(train.loc[train_index,features], label=train.loc[train_index,"is_female"])
    xgtest = xg.DMatrix(train.loc[test_index,features],label=train.loc[test_index,"is_female"])

    model = xg.train(params=params, dtrain=xgtrain,evals=[(xgtest,"test")], num_boost_round=10000,early_stopping_rounds=50,verbose_eval=False)

    test["IS_FEMALE_FOLD_"+str(i)]=model.predict(xg.DMatrix(test[features]), ntree_limit=model.best_ntree_limit)
   
    print ("Fold_"+str(i))
    
    a=roc_auc_score(train.loc[test_index,"is_female"],model.predict(xg.DMatrix(train.loc[test_index,features]), ntree_limit=model.best_iteration))
    
    r.append(a)
    
    print (roc_auc_score(train.loc[test_index,"is_female"],model.predict(xg.DMatrix(train.loc[test_index,features]), ntree_limit=model.best_iteration)))
    print ("")
    
    i=i+1
    
print ("mean: "+str(np.mean(np.array(r))))
print ("std: "+str(np.std(np.array(r))))  


    

a=[x for x in test.columns if "FOLD" in x]

test["is_female"]=test[a].mean(axis=1)
test["test_id"]=range(0,len(test))
test[["test_id","is_female"]].to_csv("submission_30_xgboost1_cleaned_data_2.csv",index=False)
