# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/machinehackweekendinsurancechurnprediction/Insurance_Churn_ParticipantsData/Train.csv')

test = pd.read_csv('/kaggle/input/machinehackweekendinsurancechurnprediction/Insurance_Churn_ParticipantsData/Test.csv')

s = pd.read_excel('/kaggle/input/machinehackweekendinsurancechurnprediction/Insurance_Churn_ParticipantsData/sample_submission.xlsx')
train.head()
for k in train.columns:

    print(k,train[k].nunique())
df=train.append(test,ignore_index=True)

df.head()

# df['feat2_14']=df['feature_2']*df['feature_14']
# for i in ['feature_0', 'feature_1', 'feature_2', 'feature_3','feature_4', 'feature_5', 'feature_6']:



#     df[i]=np.expm1(df[i])
df.head(10)
# df['ft14_7']=(df['feature_7'].astype(str)+df['feature_14'].astype(str)).astype('category')
col=['feature_0', 'feature_1', 'feature_10', 'feature_11', 'feature_12',

       'feature_13', 'feature_14', 'feature_15', 'feature_2', 'feature_3',

       'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8',

       'feature_9']
cumcol=['feature_0', 'feature_1', 'feature_10', 'feature_11', 'feature_12',

       'feature_13', 'feature_15', 'feature_2', 'feature_3', 'feature_4',

       'feature_5', 'feature_6', 'feature_8', 'feature_9']

len(cumcol)
df[col].sort_values(['feature_2']).groupby(['feature_2']).cumsum().sort_index().columns

# len([x+'cumsum' for x in ['feature_0', 'feature_1', 'feature_10', 'feature_11', 'feature_12',

#        'feature_13', 'feature_14', 'feature_15', 'feature_3', 'feature_4',

#        'feature_5', 'feature_6', 'feature_8', 'feature_9']])



df[col].sort_values(['feature_2','feature_5']).groupby(['feature_2','feature_5']).cumsum().sort_index().columns
# for k in ['feature_0', 'feature_1', 'feature_3','feature_4', 'feature_5', 'feature_6']:

#     df[k+'by_2']=df[k]/(df['feature_2']+1)

# df[['feature_0', 'feature_1', 'feature_2', 'feature_3','feature_4', 'feature_5', 'feature_6']]/df['feature_2']



df[[x+'cumsum72' for x in cumcol]]=df[col].sort_values(['feature_2','feature_7']).groupby(['feature_7','feature_2']).cumsum().sort_index()



# df[[x+'cumsum142' for x in cumcol]]=df[col].sort_values(['feature_2','feature_14']).groupby(['feature_7','feature_14']).cumsum().sort_index()
# for k in col:

#     df[k+'count']=df.groupby(k).cumcount()
# train = train[['feature_0', 'feature_1', 'feature_2', 'feature_3','feature_4', 'feature_5', 'feature_6','labels']]



# sns.pairplot(train, kind="scatter")

# plt.show()
t=df[col].groupby(['feature_14','feature_7']).agg(['min','mean','sum','max'])

t.columns=['F1_' + '_'.join(col).strip() for col in t.columns.values]

t.reset_index(inplace=True)

t.head()



t1=df[col].groupby(['feature_5','feature_6']).agg(['min','mean','sum','max'])

t1.columns=['F2_' + '_'.join(col).strip() for col in t1.columns.values]

t1.reset_index(inplace=True)

t1.head()



t2=df[col].groupby(['feature_1']).agg(['min','mean','sum','max'])

t2.columns=['F3_' + '_'.join(col).strip() for col in t2.columns.values]

t2.reset_index(inplace=True)

t2.head()



t3=df[col].groupby(['feature_2']).agg(['min','mean','sum','max'])

t3.columns=['F4_' + '_'.join(col).strip() for col in t3.columns.values]

t3.reset_index(inplace=True)

t3.head()



# t4=df[col+['feature_101112','feature_915','feature_813']].groupby(['feature_101112','feature_915','feature_813']).agg(['min','mean','sum','max'])

# t4.columns=['F5_' + '_'.join(col).strip() for col in t4.columns.values]

# t4.reset_index(inplace=True)

# t4.head()



df=df.merge(t,on=['feature_14','feature_7'],how='left')

df=df.merge(t1,on=['feature_5','feature_6'],how='left')

df=df.merge(t2,on=['feature_1'],how='left')

df=df.merge(t3,on=['feature_2'],how='left')

# df=df.merge(t4,on=['feature_101112','feature_915','feature_813'],how='left')

df.head()
# df['feature_1_bin']=pd.qcut(df['feature_1'],5,labels=False).astype('category')

# df['feature_0_bin']=pd.qcut(df['feature_0'],5,labels=False).astype('category')

# df['feature_3_bin']=pd.qcut(df['feature_3'],5,labels=False).astype('category')

# df['feature_5_bin']=pd.qcut(df['feature_5'],,labels=False).astype('category')
# df[['feature_0', 'feature_1', 'feature_2', 'feature_3','feature_4', 'feature_5', 'feature_6']]

# for k in ['feature_0', 'feature_1', 'feature_2', 'feature_3','feature_4', 'feature_5', 'feature_6']:

#     df[k+'_na']=df[k].apply(lambda x: 1 if x>=0 else 0)

    

    

# df['sin_time_f2'] = np.sin(2*np.pi*df.feature_2/31)

# df['cos_time_f2'] = np.cos(2*np.pi*df.feature_2/31)



# df['sin_time_f7'] = np.sin(2*np.pi*df.feature_7/12)

# df['cos_time_f7'] = np.cos(2*np.pi*df.feature_7/12)



# df['sin_time_f14'] = np.sin(2*np.pi*df.feature_14/12)

# df['cos_time_f14'] = np.cos(2*np.pi*df.feature_14/12)



# for k in ['feature_0', 'feature_1', 'feature_2', 'feature_3','feature_4', 'feature_5', 'feature_6']:

#     for j in ['feature_0', 'feature_1', 'feature_2', 'feature_3','feature_4', 'feature_5', 'feature_6']:

#         if k!=j:

#             df[k+'+'+j]=df[k]/df[j]

#             df[k+'*'+j]=df[k]*df[j]
train.describe().T
for k in train.columns:

#     print(k,df[k].nunique())

    if df[k].nunique()<=31 and k!='labels':

        df[k]=df[k].astype('category')

df.info()
# df=pd.get_dummies(df,drop_first=True)
from catboost import CatBoostClassifier,Pool, cv

from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold,train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,f1_score

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt



# sns.heatmap(df_train.corr())


df_train=df[df['labels'].isnull()==False].copy()

df_test=df[df['labels'].isnull()==True].copy()



print(df_train.shape,df_test.shape)
from imblearn.over_sampling import SMOTE

# sm = SMOTE(random_state=1994)





from imblearn.combine import SMOTETomek

sm = SMOTETomek(random_state=1994) #best



X,y=df_train.drop(['labels'],axis=1),df_train['labels']



X_res,y_res=sm.fit_resample(X,y)



Xtest=df_test.drop(['labels'],axis=1)
print(X.shape,y.shape,Xtest.shape)

print(X_res.shape,y_res.shape,Xtest.shape)

# X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.20,random_state = 1994,stratify=y)

X_train,X_val,y_train,y_val = train_test_split(X_res,y_res,test_size=0.20,random_state = 1994,stratify=y_res)
# df_train=df_train.astype(np.float64)
# from sklearn.linear_model import LogisticRegression

# # df_train

# X,y=df_train.drop(['labels'],axis=1),df_train['labels']

# X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.20,random_state = 1994,stratify=y)



# lr=LogisticRegression(random_state=1994,C=2)

# lr.fit(X_train,y_train)

# y_pred=lr.predict(X_val)

# print(f1_score(y_val,y_pred))

train.labels.value_counts()/train.shape[0]
from sklearn.metrics import f1_score

def evaluate_F1_lgb(truth, predictions):  

    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483

    pred_labels = predictions.round()

    f1 = f1_score(truth, pred_labels)

    return ('F1', f1, True) 


# m=LGBMClassifier(n_estimators=2500,random_state=1994,learning_rate=0.03,reg_alpha=5,class_weight={1:8.8,0:1.1},num_leaves=200,min_child_samples=10,subsample=0.9,objective='binary')

m=LGBMClassifier(n_estimators=2500,random_state=1994,learning_rate=0.03,reg_alpha=5,class_weight={1:8.8,0:1.1},num_leaves=300,min_child_samples=10,subsample=0.9,objective='binary')

# m=RidgeCV(cv=4)

m.fit(X_train,y_train,eval_set=[(X_val, y_val)],eval_metric=evaluate_F1_lgb, early_stopping_rounds=100,verbose=200)

p=m.predict(X_val)



print(f1_score(y_val,p))
import matplotlib.pyplot as plt

import warnings

import seaborn as sns

sns.set_style('darkgrid')

warnings.filterwarnings('ignore')



%matplotlib inline

feature_imp = pd.DataFrame(sorted(zip(m.feature_importances_, X.columns), reverse=True)[:200], 

                           columns=['Value','Feature'])

plt.figure(figsize=(12,8))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()
err=[]

y_pred_tot=[]



# feature_importance_df = pd.DataFrame()

from sklearn.model_selection import KFold,StratifiedKFold

fold=StratifiedKFold(n_splits=5,shuffle=True,random_state=1994)

X_res,y_res=sm.fit_resample(X,y)

i=1

for train_index, test_index in fold.split(X_res,y_res):

    

    X_train, X_test = X_res.iloc[train_index], X_res.iloc[test_index]

    y_train, y_test = y_res[train_index], y_res[test_index]

    

    

    m=LGBMClassifier(n_estimators=5000,random_state=1994,reg_alpha=5,class_weight={1:8.8,0:1.1},num_leaves=300,min_child_samples=10,subsample=0.9)

    m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)],eval_metric=evaluate_F1_lgb, early_stopping_rounds=200,verbose=200)

    

    preds=m.predict_proba(X_test,num_iteration=m.best_iteration_)[:,-1]

    print("err: ",f1_score(y_test,preds.round()))

    err.append(f1_score(y_test,preds.round()))

    p = m.predict_proba(Xtest)[:,-1]

    i=i+1

    y_pred_tot.append(p)
# err=[]

# y_pred_tot=[]



# # feature_importance_df = pd.DataFrame()



# from sklearn.model_selection import KFold,StratifiedKFold

# fold=StratifiedKFold(n_splits=5,shuffle=True,random_state=1994)

# i=1

# for train_index, test_index in fold.split(X,y):

    

#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]

#     y_train, y_test = y[train_index], y[test_index]

#     m=LGBMClassifier(n_estimators=5000,random_state=1994,reg_alpha=5,class_weight={1:8.8,0:1.1})

#     m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)],eval_metric='auc', early_stopping_rounds=200,verbose=200)

    

#     preds=m.predict_proba(X_test,num_iteration=m.best_iteration_)[:,-1]

#     print("err: ",f1_score(y_test,preds.round()))

#     err.append(f1_score(y_test,preds.round()))

#     p = m.predict_proba(Xtest)[:,-1]

#     i=i+1

#     y_pred_tot.append(p)
np.mean(err)
# #catboost



# print(X.shape,Xtest.shape)





# # for i in categorical_features_indices:

# #     X.iloc[:,i]=pd.factorize(X.iloc[:,i])[1]

    

# X['feature_2']=X['feature_2'].astype(np.object)

# X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.25,random_state = 1994,stratify=y)

# categorical_features_indices = np.where(X_train.dtypes =='category')[0]

# categorical_features_indices


# m=CatBoostClassifier(n_estimators=2500,random_state=1994,learning_rate=0.1,eval_metric='AUC',l2_leaf_reg=2,bagging_temperature=1,scale_pos_weight=1)

# # m=RidgeCV(cv=4)



# m.fit(X_train,y_train,eval_set=[(X_val, y_val.values)], early_stopping_rounds=300,verbose=200,cat_features=categorical_features_indices)

# p=m.predict(X_val)

# print(f1_score(y_val,p))

# y_cat = m.predict_proba(Xtest,ntree_end=m.best_iteration_)[:,-1]

np.mean(y_pred_tot,0)
s['labels']=np.mean(y_pred_tot,0).round()

s
s['labels'].value_counts()

# s

s.to_excel('MH_weekend_churn_kv14.xlsx',index=False)