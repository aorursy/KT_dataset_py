#Imports
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import collections

#Chargement fichiers
t=time.time()
"""for file in os.listdir("../input"):
    var=pd.read_csv("../input/" + file)
    print("../input/" + file, var.shape)
    print("time for loading " + file + " : " + str(time.time()-t) + " s")
    t=time.time() """
train=pd.read_csv("../input/application_train.csv")
test=pd.read_csv("../input/application_test.csv")

#Distinction variables discrètes/continues
type_groups={}
for group in ["float64", "int64", "object"]:
    type_groups[group]=train.dtypes[train.dtypes==group].index.tolist()
train[type_groups["float64"]]

nunique_groups={}
nunique_groups[">=40"]=[]
nunique_groups["<40"]=[]
for x in list(train):
    if train[x].value_counts().shape[0]>=40:
        nunique_groups[">=40"].append(x)
    else:
        nunique_groups["<40"].append(x)

type_nunique_groups={}
for i in list(nunique_groups):
    for j in list(type_groups):
        type_nunique_groups[j + " " + i]=list(filter(lambda x : x in nunique_groups[i],type_groups[j]))
        
continuous = type_nunique_groups["int64 >=40"] + type_nunique_groups["float64 >=40"] + type_nunique_groups["float64 <40"] + ["CNT_CHILDREN"]
discrete=type_nunique_groups["object >=40"] + type_nunique_groups["object <40"] + list(filter(lambda x : x not in ["CNT_CHILDREN","TARGET"], type_nunique_groups["int64 <40"]))

#Gestion des valeurs manquantes
def fill_values(dataf,mask=None):
    t=time.time()
    #Variables discrètes
    var=pd.get_dummies(dataf[discrete[0]].fillna('missing'))
    var.columns=[discrete[0] + '_'+x for x in var.columns]
    mat=var

    for i in discrete[1:]:
        var=pd.get_dummies(dataf[i].fillna('missing'))
        var.columns=[str(i)+ '_'+str(x) for x in var.columns]
        mat=pd.concat([mat,var],axis=1)

    #Variables continues avec peu de valeurs manquantes
    missing=(dataf[continuous].count()/dataf.shape[0]).sort_values(ascending=False)
    to_fill=list(missing[missing>=0.8].index)

    mat2=dataf[to_fill[0]].fillna(dataf[to_fill[0]].median())
    for i in to_fill[1:]:
        var=dataf[i].fillna(dataf[i].median())
        mat2=pd.concat([mat2,var],axis=1)

    #Variables continues avec beaucoup de valeurs manquantes
    left_to_fill=list(missing[missing<0.8].index)
    name=left_to_fill[0]
    var=pd.cut(dataf[name],3).astype('category').cat.add_categories(['missing'])
    var=pd.get_dummies(var.fillna('missing'))
    var.columns=[name + "_" + str(1),name + "_" + str(2),name + "_" + str(3),name + "_" +'missing']
    mat3=var

    for i in left_to_fill[1:]:
        var=pd.cut(dataf[i],3).astype('category').cat.add_categories(['missing'])
        var=pd.get_dummies(var.fillna('missing'))
        var.columns=[i + "_" + str(1),i + "_" + str(2),i + "_" + str(3),i + "_" +'missing']
        mat3=pd.concat([mat3,var],axis=1)
    print("Temps pour nettoyage des données : ",time.time()-t,' s')
    fusion=pd.concat([mat,mat2,mat3],axis=1)
    #Ajout de colonnes le cas échéant
    if mask!=None:
        diff=list(filter(lambda x : x not in list(fusion),mask ))
        for i in diff : 
            fusion[i]=0
        diff=list(filter(lambda x : x not in mask,list(fusion) ))
        for i in diff:
            del fusion[i]
    return fusion[sorted(list(fusion))]
train_filled=fill_values(train)
test_filled=fill_values(test,list(train_filled))
"""#Modélisation
from sklearn.decomposition import LatentDirichletAllocation
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
X_train, X_val, y_train, y_val = train_test_split(train_filled, train["TARGET"], test_size=0.2, random_state=42)
clf=LatentDirichletAllocation()
#clf=LGBMClassifier(learning_rate =0.075, num_boost_round=1500,  nthread=8, seed=27,colsample_bytree=1, max_depth=3,
#                     min_child_weight=87.5467,min_split_gain=0.0950,num_leaves=22,reg_alpha=0.0019,reg_lambda=0.0406,subsample=0.8709)
#clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric= 'auc', verbose= 100, early_stopping_rounds= 50)
clf.fit(X_train,y_train)
#score=np.round(roc_auc_score(y_val, clf.predict_proba(X_val)[:,1]),4) 
score=np.round(roc_auc_score(y_val, clf.predict(X_val)),4) 
score  """
#Modélisation
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression

#Modèles
"""reg=RandomForestRegressor()
reg.fit(train_filled,train["TARGET"])
"""
clf=LGBMClassifier(learning_rate =0.075, num_boost_round=1500,  nthread=8, seed=27,colsample_bytree=1, max_depth=3,
                     min_child_weight=87.5467,min_split_gain=0.0950,num_leaves=22,reg_alpha=0.0019,reg_lambda=0.0406,subsample=0.8709)
clf.fit(train_filled, train["TARGET"], eval_metric= 'auc', verbose= 100)

sub=pd.Series(clf.predict_proba(test_filled)[:,1],name="TARGET")
#sub=pd.Series(reg.predict(test_filled),name="TARGET")
sub.loc[sub<0]=0
sub.loc[sub>1]=1
sub.index=test.index
submission=pd.concat([test["SK_ID_CURR"],sub],axis=1)
submission
submission.to_csv('submission.csv', index=False)
"""#Target encoding
def fill_values(dataf,target,continuous, discrete,nint):
    a=0
    #Variables discrètes
    dataf_tsf=pd.DataFrame(dataf[target])
    dataf_tsf.index=dataf.index
    if len(discrete) >0:
        #dataf[discrete]=dataf[discrete].fillna('missing')
        for i in discrete:
            print(a,i)
            a+=1
            dataf_tsf[i]=0
            groups=dataf.groupby(i)["TARGET"].mean()
            dicogroups={x:groups[x] for x in groups.index}
            meannull=dataf["TARGET"][pd.isnull(dataf[i])==True].mean() #Calcul de la valeur moyenne de la variable cible par modalité
            dicogroups[np.nan]=meannull"""