import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')

train_file = '../input/rg_train.csv'

test_file = '../input/rg_test.csv'
bd_train = pd.read_csv(train_file)

bd_test = pd.read_csv(test_file)
bd_test['Revenue.Grid'] = np.nan

bd_train['data'] = 'train'

bd_test['data'] = 'test'
bd_test=bd_test[bd_train.columns] #to ensure same order of columns
bd_all = pd.concat([bd_train, bd_test], axis = 0)
bd_all.head()
bd_all.dtypes
bd_all.nunique()
bd_all.drop(['REF_NO','post_code','post_area'],axis=1,inplace=True)
bd_all['children']=np.where(bd_all['children']=='Zero',0,bd_all['children'])

bd_all['children']=np.where(bd_all['children'].str[:1]=='4',4,bd_all['children'])

bd_all['children']=pd.to_numeric(bd_all['children'],errors='coerce')
bd_all['Revenue.Grid']=(bd_all['Revenue.Grid']==1).astype(int)
bd_all['family_income'].value_counts(dropna=False)
bd_all['family_income']=bd_all['family_income'].str.replace(',',"")

bd_all['family_income']=bd_all['family_income'].str.replace('<',"")

k=bd_all['family_income'].str.split('>=',expand=True)
for col in k.columns:

    k[col]=pd.to_numeric(k[col],errors='coerce')
bd_all['fi']=np.where(bd_all['family_income']=='Unknown',np.nan,

    np.where(k[0].isnull(),k[1],

    np.where(k[1].isnull(),k[0],0.5*(k[0]+k[1]))))
bd_all['age_band'].value_counts(dropna=False)
k=bd_all['age_band'].str.split('-',expand=True)

for col in k.columns:

    k[col]=pd.to_numeric(k[col],errors='coerce')
bd_all['ab']=np.where(bd_all['age_band'].str[:2]=='71',71,

             np.where(bd_all['age_band']=='Unknow',np.nan,0.5*(k[0]+k[1])))
del bd_all['age_band']

del bd_all['family_income']
cat_vars=bd_all.select_dtypes(['object']).columns

cat_vars=list(cat_vars)

cat_vars.remove('data')
for col in cat_vars:

    dummy=pd.get_dummies(bd_all[col],drop_first=True,prefix=col)

    bd_all=pd.concat([bd_all,dummy],axis=1)

    del bd_all[col]

    print(col)

del dummy
for col in bd_all.columns:

    if col=='data' or bd_all[col].isnull().sum()==0:

        continue

    bd_all.loc[bd_all[col].isnull(),col]=bd_all.loc[bd_all['data']=='train',col].mean()
bd_all.loc[bd_all[col].isnull(),col]=bd_all.loc[bd_all['data']=='train',col].mean()
train1=bd_all[bd_all['data']=='train']

del train1['data']

test1=bd_all[bd_all['data']=='test']

test1.drop(['Revenue.Grid','data'],axis=1,inplace=True)
from sklearn.linear_model import LogisticRegression
params={'class_weight':['balanced',None],

        'penalty':['l1','l2'],

# these are L1 and L2 written in lower case

# dont confuse them with numeric eleven and tweleve

        'C':np.linspace(0.0001,1000,10)}



# we can certainly try much higher ranges and number of values for theparameter 'C'

# grid search in this case , will be trying out 2*2*10=40 possiblecombination

# and will give us cross validated performance for all
model=LogisticRegression(fit_intercept=True)
from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(model,param_grid=params,cv=10,scoring="roc_auc")

# note that scoring is now roc_auc as we are solving a classification problem
x_train=train1.drop('Revenue.Grid',axis=1)

y_train=train1['Revenue.Grid']
grid_search.fit(x_train,y_train)
# predict_proba for predciting probabilities

# just predict, predicts hard classes considering 0.5 as score cutoff

# which is not always a great idea, we'll see in a moment

test_prediction = grid_search.predict_proba(test1)
test_prediction
# this will tell you which probability belongs to which class

grid_search.classes_
train_score=grid_search.predict_proba(x_train)[:,1]

real = y_train
cutoffs = np.linspace(.001,0.999, 999)
KS=[]
for cutoff in cutoffs:

    predicted=(train_score>cutoff).astype(int)

    TP=((real==1)&(predicted==1)).sum()

    FP=((real==0)&(predicted==1)).sum()

    TN=((real==0)&(predicted==0)).sum()

    FN=((real==1)&(predicted==0)).sum()

    ks=(TP/(TP+FN))-(FP/(TN+FP))

    KS.append(ks)
temp=pd.DataFrame({'cutoffs':cutoffs,'KS':KS})
import seaborn as sns

%matplotlib inline
sns.lmplot(x='cutoffs',y='KS',data=temp,fit_reg=False)
cutoffs[KS==max(KS)][0]
test_hard_classes=(test_prediction>cutoffs[KS==max(KS)][0]).astype(int)
test_hard_classes[:,0]
output = pd.DataFrame({'REF_NO' : bd_test.REF_NO, 'Revenue.Grid':test_hard_classes[:,0]})

output.to_csv('submission.csv', index=False)