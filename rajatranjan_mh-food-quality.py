import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



plt.style.use('fivethirtyeight')

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss



import lightgbm as lgb

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm_notebook
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_excel("/kaggle/input/food-quality/Food_QUalityA_ParticipantsData/Data_Train.xlsx")

print(train.shape)

test = pd.read_excel("/kaggle/input/food-quality/Food_QUalityA_ParticipantsData/Data_Test.xlsx")

sample_sub = pd.read_excel('/kaggle/input/food-quality/Food_QUalityA_ParticipantsData/Sample_Submission.xlsx')
len(np.intersect1d(train.LicenseNo,test.LicenseNo))
test.LicenseNo.nunique(),train.LicenseNo.nunique()
train[train.LicenseNo==2136]
train[train.LocationID==81886.0]
# df.shape
ID_COL, TARGET_COL = 'ID', 'Inspection_Results'

df=train.append(test,ignore_index=True)

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True,errors='coerce',format='%d-%m-%Y')



df['dayofweek'] = df['Date'].dt.dayofweek

df['quarter'] = df['Date'].dt.quarter

df['month'] = df['Date'].dt.month

df['year'] = df['Date'].dt.year

df['dayofyear'] = df['Date'].dt.dayofyear

df['dayofmonth'] = df['Date'].dt.day

df['week'] = df['Date'].dt.week



df[(df.LicenseNo==17661) & (df.Inspection_Results.isnull()==False) ].sort_values('Date')[['Date','Inspection_Results','SectionViolations']].head(30)
df.head()
# df['log_entry'] = 1

# pivot_df = pd.pivot_table(df

#                           , values="log_entry"

#                           , index="Date"

#                           , columns="LocationID"

#                           , aggfunc="count"

#                           , fill_value=0).reset_index()



# pivot_df.columns = ["cat_two_"+ str(col) for col in pivot_df.columns]

# pivot_df = pivot_df.rename(columns={"cat_two_Date": "Date"})

# df=df.merge(pivot_df,on='Date',how='left')
# for k in [10,30,60,90]:

#     df=df.merge(ds.groupby(['Date']).mean()['SectionViolations'].rolling(k).mean().reset_index().rename(columns={'SectionViolations':'Date_rolling'+str(k)}),on='Date',how='left')



# train.groupby(['Date','LocationID','FacilityID']).transform('mean')

# xx=df.groupby(['FacilityID']).transform(lambda x: x.fillna(x.mean()))
# df['SectionViolations']=xx['SectionViolations']

df.head()
# df.Date_rolling60
df.isnull().sum()
df['Type'] = df['Type'].apply(lambda x: " ".join(str(x).split("/")))

import re

df['Reason'] = df['Reason'].apply(lambda x: re.sub("[^A-Za-z]"," ",x))



typetf=df['Type']

reasontf=df['Reason']





df=pd.get_dummies(df,columns=['City','State','RiskLevel'],drop_first=True)
# datetime.datetime.now()
import datetime

# df['days_since']=(datetime.datetime.now() - df['Date']).dt.days

df['days_since']=(datetime.datetime(2020,2,16) - df['Date']).dt.days
temp=df.sort_values(['Date','LicenseNo','LocationID'])
# df.groupby(['Date','FacilityID'])['SectionViolations'].apply(lambda x: x.shift().ewm(alpha=0.77).mean())

temp.groupby(['Date','LicenseNo']).cumcount()


temp['LL_cumcount']=temp.groupby(['LicenseNo','LocationID']).cumcount()

temp['LL_cumsum']=temp.groupby(['LicenseNo','LocationID']).cumsum()['SectionViolations']

temp['Date_cumsum']=temp.groupby(['Date']).cumsum()['SectionViolations']

temp['Date_lc_ld_cumsum']=temp.groupby(['Date','LicenseNo','LocationID']).cumsum()['SectionViolations']

temp['Date_cumcount']=temp.groupby(['Date','LicenseNo','LocationID']).cumcount()

# temp['F1']=temp.groupby(['Date','FacilityID']).cumsum()['SectionViolations']

# temp['F2']=temp.groupby(['Date','FacilityID']).cumcount()

v=[]

for k in [ 'FacilityID', 'FacilityName', 'Geo_Loc', 'LicenseNo',  'Street','LocationID']:

    temp[k+'_cumcount']=temp.groupby(k).cumcount()

    temp[k+'_cumsum']=temp.groupby(k).cumsum()['SectionViolations']

    v.append(k+'_cumcount')

    v.append(k+'_cumsum')

    

for k in [ 'FacilityID', 'FacilityName', 'Geo_Loc', 'LicenseNo',  'Street','LocationID']:

    for j in [ 'FacilityID', 'FacilityName', 'Geo_Loc', 'LicenseNo',  'Street','LocationID']:

        if k!=j:

            temp[k+'_cumcount'+j]=temp.groupby([k,j]).cumcount()

            v.append(k+'_cumcount'+j)

        

temp

df=df.merge(temp[['ID','LL_cumcount','LL_cumsum','Date_cumsum','Date_lc_ld_cumsum','Date_cumcount']+v],on='ID')
df['F3']=df['days_since']/df['SectionViolations']
# df.sort_values(['FacilityID','Date'])['SectionViolations'].diff(30)
# temp.groupby(['Date','LicenseNo','LocationID']).cumcount()
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

v_1 = TfidfVectorizer(ngram_range=(1,2),stop_words="english", analyzer='word')

typ_tr =v_1.fit_transform(typetf[:train.shape[0]])

typ_ts =v_1.transform(typetf[train.shape[0]:])



v_2 = TfidfVectorizer(ngram_range=(1,2),stop_words="english", analyzer='word')

res_tr =v_2.fit_transform(reasontf[:train.shape[0]])

res_ts =v_2.transform(reasontf[train.shape[0]:])



df.sample(10)
cat_cols = [ 'FacilityID', 'FacilityName', 'Geo_Loc', 'LicenseNo',  'Street','LocationID','Reason','Type','Date']

for k in cat_cols:

    df[k]=pd.factorize(df[k])[0]
for k in df.columns:

    print(k,df[k].nunique())
n=['mean','sum','min','max','median']

c=['count','nunique']

t1=df.groupby(['FacilityID']).agg({'FacilityName':c, 'Geo_Loc':c,

       'LicenseNo':c, 'LocationID':c, 'Reason':c, 'SectionViolations':n, 'Street':c,'Date':c,'days_since':n,

       'Type':c})

t1.columns=['FI_' + '_'.join(col).strip() for col in t1.columns.values]

t1.reset_index(inplace=True)



t2=df.groupby(['Geo_Loc']).agg({'FacilityName':c, 'FacilityID':c,

       'LicenseNo':c, 'LocationID':c, 'Reason':c, 'SectionViolations':n, 'Street':c,'Date':c,'days_since':n,

       'Type':c})

t2.columns=['G_' + '_'.join(col).strip() for col in t2.columns.values]

t2.reset_index(inplace=True)



t3=df.groupby(['LicenseNo']).agg({'FacilityName':c, 'FacilityID':c,

       'LocationID':c, 'Geo_Loc':c, 'Reason':c, 'SectionViolations':n, 'Street':c,'Date':c,'days_since':n,

       'Type':c})

t3.columns=['L_' + '_'.join(col).strip() for col in t3.columns.values]

t3.reset_index(inplace=True)





t4=df.groupby(['Date']).agg({'FacilityName':c, 'Geo_Loc':c,

       'LicenseNo':c, 'LocationID':c, 'Reason':c, 'SectionViolations':n, 'Street':c,'FacilityID':c,'days_since':n,

       'Type':c})

t4.columns=['D_' + '_'.join(col).strip() for col in t4.columns.values]

t4.reset_index(inplace=True)



t5=df.groupby(['FacilityName']).agg({'Geo_Loc':c, 'FacilityID':c,

       'LicenseNo':c, 'LocationID':c, 'Reason':c, 'SectionViolations':n, 'Street':c,'Date':c,'days_since':n,

       'Type':c})

t5.columns=['FN_' + '_'.join(col).strip() for col in t5.columns.values]

t5.reset_index(inplace=True)



# t6=df.groupby(['Street']).agg({'FacilityName':c, 'FacilityID':c,

#        'LocationID':c, 'Geo_Loc':c, 'Reason':c, 'SectionViolations':n, 'LicenseNo':c,'Date':c,

#        'Type':c})

# t6.columns=['S_' + '_'.join(col).strip() for col in t6.columns.values]

# t6.reset_index(inplace=True)



# t7=df.groupby(['dayofyear','month']).agg({'FacilityName':c, 'FacilityID':c,

#        'LocationID':c, 'Geo_Loc':c, 'Reason':c, 'SectionViolations':n, 'LicenseNo':c,'Date':c,

#        'Type':c})

# t7.columns=['DM_' + '_'.join(col).strip() for col in t7.columns.values]

# t7.reset_index(inplace=True)



t8=df.groupby(['Type','Reason']).agg({'FacilityName':c, 'FacilityID':c,

       'LocationID':c, 'Geo_Loc':c, 'SectionViolations':n, 'LicenseNo':c,'Date':c,'days_since':n

       })

t8.columns=['TR_' + '_'.join(col).strip() for col in t8.columns.values]

t8.reset_index(inplace=True)





t9=df.groupby(['LocationID']).agg({'FacilityName':c, 'FacilityID':c,

       'Street':c, 'Geo_Loc':c, 'Reason':c, 'SectionViolations':n, 'LicenseNo':c,'Date':c,'days_since':n,

       'Type':c})

t9.columns=['LI_' + '_'.join(col).strip() for col in t9.columns.values]

t9.reset_index(inplace=True)



t10=df.groupby(['LocationID','LicenseNo']).agg({'FacilityName':c, 'FacilityID':c,'days_since':n,

       'Street':c, 'Geo_Loc':c, 'Reason':c, 'SectionViolations':n, 'Date':c,

       'Type':c})

t10.columns=['LL_' + '_'.join(col).strip() for col in t10.columns.values]

t10.reset_index(inplace=True)



t11=df.groupby(['Date','FacilityID']).agg({'FacilityName':c, 'LicenseNo':c,'LocationID':c,'days_since':n,

       'Street':c, 'Geo_Loc':c, 'Reason':c, 'SectionViolations':n,

       'Type':c})

t11.columns=['DFI_' + '_'.join(col).strip() for col in t11.columns.values]

t11.reset_index(inplace=True)



t12=df.groupby(['Date','Geo_Loc']).agg({'FacilityName':c, 'LicenseNo':c,'LocationID':c,'days_since':n,

       'Street':c, 'FacilityID':c, 'Reason':c, 'SectionViolations':n,

       'Type':c})

t12.columns=['DG_' + '_'.join(col).strip() for col in t12.columns.values]

t12.reset_index(inplace=True)



t13=df.groupby(['Date','LicenseNo']).agg({'FacilityName':c, 'Street':c,'Geo_Loc':c,'days_since':n,

       'LocationID':c, 'FacilityID':c, 'Reason':c, 'SectionViolations':n,

       'Type':c})

t13.columns=['DLI_' + '_'.join(col).strip() for col in t13.columns.values]

t13.reset_index(inplace=True)



# t14=df.groupby(['Geo_Loc','FacilityID']).agg({'FacilityName':c, 'Street':c,'days_since':n,

#        'LicenseNo':c, 'Reason':c, 'SectionViolations':n,'LocationID':c,'Date':c,

#        'Type':c})

# t14.columns=['DYM_' + '_'.join(col).strip() for col in t14.columns.values]

# t14.reset_index(inplace=True)



df=df.merge(t1,on='FacilityID',how='left')

df=df.merge(t2,on='Geo_Loc',how='left')

df=df.merge(t3,on='LicenseNo',how='left')

df=df.merge(t4,on='Date',how='left')

df=df.merge(t5,on='FacilityName',how='left')

# df=df.merge(t6,on='Street',how='left')

# df=df.merge(t7,on=['dayofyear','month'],how='left')

df=df.merge(t8,on=['Type','Reason'],how='left')

df=df.merge(t9,on=['LocationID'],how='left')

df=df.merge(t10,on=['LocationID','LicenseNo'],how='left')

df=df.merge(t11,on=['Date','FacilityID'],how='left')

df=df.merge(t12,on=['Date','Geo_Loc'],how='left')

df=df.merge(t13,on=['Date','LicenseNo'],how='left')

# df=df.merge(t14,on=['Geo_Loc','FacilityID'],how='left')
df['SectionViolations_bin']=pd.factorize(pd.cut(df['SectionViolations'],7,labels=["ss"+str(x) for x in range(7)]))[0]



df.shape
ID_COL, TARGET_COL = 'ID', 'Inspection_Results'

features = [c for c in df.columns if c not in [ID_COL, TARGET_COL,'Type','Reason','log_entry']]

traindf, testdf = df.iloc[:train.shape[0]], df.iloc[train.shape[0]:]

# traindf=traindf[traindf.dayofyear.isnull()==False]

testdf.reset_index(drop=True, inplace=True)

target = traindf[TARGET_COL]
# df['SectionViolations_bin'].astype(int)
traindf[features].shape,testdf[features].shape
from scipy.sparse import csr_matrix

from scipy import sparse

final_features = sparse.hstack((traindf[features],typ_tr,res_tr)).tocsr()

final_featurest = sparse.hstack((testdf[features],typ_ts,res_ts )).tocsr()
X_trn, X_val, y_trn, y_val = train_test_split(final_features, target, test_size=0.3, stratify=target, random_state=1994)

X_test = final_featurest
final_features
import gc

# param = {'objective':"multiclass",

#          "boosting": "gbdt",

#          "metric": 'multi_logloss',

#          "num_class" : 7,

# #          "feature_fraction": 0.6,

# #          'learning_rate':0.03,

# #          "lambda_l1": 1,

# #          "lambda_l2": 1,

#          "verbosity": -1,

#         'n_estimators':2500,

#         'random_state':1994}



param = {'objective':"multiclass",

         "boosting": "gbdt",

         "metric": 'multi_logloss',

         "num_class" : 7,

         "feature_fraction": 0.7,

         'learning_rate':0.05,

         "lambda_l1": 1,

         "lambda_l2": 1,

         "verbosity": -1,

        'random_state':1994,

         'n_estimators': 2500,'colsample_bytree':0.3,

        'subsample_for_bin':20000,'subsample':0.9}

gc.collect()
from lightgbm import LGBMClassifier

# n_estimators=1500,random_state=1994,learning_rate=0.03,objective='binary',subsample_for_bin=20000,subsample=0.9

m=LGBMClassifier(**param)

m.fit(X_trn,y_trn,eval_set=[(X_trn,y_trn),(X_val, y_val)], early_stopping_rounds=100,verbose=200)

p=m.predict_proba(X_val)

log_loss(y_val,p)



# trn_data = lgb.Dataset(X_trn, y_trn)

# val_data = lgb.Dataset(X_val, y_val)

# num_round = 500000
# # m=LGBMClassifier(**param)

# # m.fit(X_trn,y_trn,eval_set=[(X_trn,y_trn),(X_val, y_val)], early_stopping_rounds=100,verbose=200)

# # p=m.predict_proba(X_val)



# # log_loss(y_val,p)



# import matplotlib.pyplot as plt

# import warnings

# import seaborn as sns

# sns.set_style('darkgrid')

# warnings.filterwarnings('ignore')



# %matplotlib inline

# feature_imp = pd.DataFrame(sorted(zip(m.feature_importances_, X_trn.columns), reverse=True)[:250], 

#                            columns=['Value','Feature'])

# plt.figure(figsize=(12,8))

# sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

# plt.title('LightGBM Features')

# plt.tight_layout()

# plt.show()

# main_cols=feature_imp.Feature[:250].values
# from scipy.sparse import csr_matrix

# from scipy import sparse

# final_features = sparse.hstack((traindf[main_cols],typ_tr,res_tr)).tocsr()

# final_featurest = sparse.hstack((testdf[main_cols],typ_ts,res_ts )).tocsr()



# X_trn, X_val, y_trn, y_val = train_test_split(final_features, target, test_size=0.3, stratify=target, random_state=1994)

# X_test = final_featurest
# m=LGBMClassifier(**param)

# m.fit(X_trn,y_trn,eval_set=[(X_trn,y_trn),(X_val, y_val)], early_stopping_rounds=100,verbose=200)

# p=m.predict_proba(X_val)



# log_loss(y_val,p)
# X_trn.shape,testdf[features].shape

# m.feature_importances_
X=final_features

y=target
y_pred_tot=[]

err=[]

feature_importance_df = pd.DataFrame()



from sklearn.model_selection import KFold,StratifiedKFold

fold=StratifiedKFold(n_splits=10,shuffle=True,random_state=1994)

i=1

for train_index, test_index in fold.split(X,y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    m=LGBMClassifier(**param)

    m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)], early_stopping_rounds=200,verbose=200)

    preds=m.predict_proba(X_test,num_iteration=m.best_iteration_)

    print("err: ",log_loss(y_test,preds))

    err.append(log_loss(y_test,preds))

    p = m.predict_proba(final_featurest)

    i=i+1

    y_pred_tot.append(p)
np.mean(err)
target_mapper = {0:'FACILITY CHANGED',

1:'FAIL',

2:'FURTHER INSPECTION REQUIRED',

3:'INSPECTION OVERRULED',

4:'PASS',

5:'PASS(CONDITIONAL)',

6:'SHUT-DOWN'}
sub_df = pd.DataFrame(np.mean(y_pred_tot,0))

sub_df.columns = [target_mapper[c] for c in sub_df.columns]

sub_df.to_excel("MH_fq_lgbmfold_s2k4.xlsx", index=False)
sub_df.head(10)