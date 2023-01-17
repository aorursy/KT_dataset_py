# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
policy=pd.read_csv('/kaggle/input/zindi-insuranceassurance/policy_data.csv')

payment=pd.read_csv('/kaggle/input/zindi-insuranceassurance/payment_history.csv')

client=pd.read_csv('/kaggle/input/zindi-insuranceassurance/client_data.csv')

sample_sub=pd.read_csv('/kaggle/input/zindi-insuranceassurance/sample_sub.csv')

df=pd.read_csv('/kaggle/input/zindi-insuranceassurance/train.csv')
df['Lapse Year'].value_counts()
test = df[(df['Lapse']=='?') & (df['Lapse Year']=='?')]

train = df[(df['Lapse']!='?') & (df['Lapse Year']!='?')]
test['Lapse'].replace({'?':np.nan},inplace=True)

test.drop(['Lapse Year'],axis=1,inplace=True)

test.head()
# df
df['Lapse'].replace({'?':0},inplace=True)

df['Lapse Year'].replace({'?':'2019','2017':'2019','2018':'2019'},inplace=True)

df.head()
df['Lapse']=df['Lapse'].astype(np.int)
df.groupby(['Lapse Year'])['Lapse'].mean()
df['Lapse Year'].value_counts()
df[df['Lapse']==1]
policy[policy['Policy ID']=='PID_C7EJ59O']
payment[payment['Policy ID']=='PID_C7EJ59O']
# client[client['Policy ID']=='PID_C7EJ59O']
# train['Policy ID X Year']=train['Policy ID'].astype(str)+" X "+train['Lapse Year'].astype(str)

# train



# year=pd.DataFrame({'Year':['2017','2018','2019']})

# year



# train1 = (

#     train.assign(key=1)

#     .merge(year.assign(key=1), on="key")

#     .drop("key", axis=1)

# )

# train1



# train1['Policy ID X Year']=train1['Policy ID'].astype(str)+" X "+train1['Year'].astype(str)

# train1



# train1['Lapse']=0



# train1.loc[train1['Policy ID X Year'].isin(train['Policy ID X Year']), 'Lapse'] = 1

# train1



# train1.drop(['Policy ID X Year','Lapse Year'],axis=1,inplace=True)



# train1
policy['NP2_EFFECTDATE']=pd.to_datetime(policy['NP2_EFFECTDATE'])

policy
policy[['NPR_PREMIUM_cumsum','NPR_SUMASSURED_cumsum','NLO_AMOUNT_cumsum']]=policy.sort_values('NP2_EFFECTDATE').groupby('Policy ID').cumsum().sort_index()[['NPR_PREMIUM','NPR_SUMASSURED','NLO_AMOUNT']]

policy['cumcount']=policy.sort_values('NP2_EFFECTDATE').groupby('Policy ID').cumcount()
policy.nunique()
# policy.groupby(['NPH_LASTNAME']).mean()
# policy.groupby(['Policy ID','PPR_PRODCD']).mean()
policy['NP2_EFFECTDATE_month']=policy['NP2_EFFECTDATE'].dt.month

policy['NP2_EFFECTDATE_day']=policy['NP2_EFFECTDATE'].dt.day

policy['NP2_EFFECTDATE_year']=policy['NP2_EFFECTDATE'].dt.year

policy['NP2_EFFECTDATE']=policy['NP2_EFFECTDATE'].astype('category')



policy.head()
# policy['NPR_SUMASSURED']/(policy['NPR_PREMIUM']+policy['NLO_AMOUNT'])
# policy['F1'] = policy['NPR_SUMASSURED']/policy['NPR_PREMIUM']



# policy['F2'] = policy['NPR_SUMASSURED']/(policy['NPR_PREMIUM']+policy['NLO_AMOUNT'])
# policy[policy['Policy ID']=='PID_EPZDSP8']
# policy.groupby(['NPH_LASTNAME']).mean()





c=['count','nunique']

n=['min','max','sum','mean']

d={  'PPR_PRODCD':c, 'NPR_PREMIUM':n,'NP2_EFFECTDATE':c,

       'Policy ID':c, 'CLF_LIFECD':c, 'NSP_SUBPROPOSAL':c, 'NPR_SUMASSURED':n,

       'NLO_TYPE':c, 'NLO_AMOUNT':n, 'AAG_AGCODE':c, 'PCL_LOCATCODE':c, 'OCCUPATION':c,

       'CATEGORY':c}

ls = policy.groupby('NPH_LASTNAME').agg(d)



ls.columns=['FL_' + '_'.join(col).strip() for col in ls.columns.values]

ls.reset_index(inplace=True)

ls
policy=policy.merge(ls,on='NPH_LASTNAME')


policy.columns

c=['count','nunique']

n=['min','max','sum','mean']

d={'NP2_EFFECTDATE':c, 'NP2_EFFECTDATE_year':['unique'],'NP2_EFFECTDATE_day':['unique'], 'PPR_PRODCD':c, 'NPR_PREMIUM':n,

   'NPR_PREMIUM_cumsum':n,'NPR_SUMASSURED_cumsum':n,'NLO_AMOUNT_cumsum':n,'cumcount':n,

       'NPH_LASTNAME':c, 'CLF_LIFECD':c, 'NSP_SUBPROPOSAL':c, 'NPR_SUMASSURED':n,

       'NLO_TYPE':c, 'NLO_AMOUNT':n, 'AAG_AGCODE':c, 'PCL_LOCATCODE':c, 'OCCUPATION':c,

       'CATEGORY':c}



for k in ['FL_PPR_PRODCD_count',

       'FL_PPR_PRODCD_nunique', 'FL_NPR_PREMIUM_min', 'FL_NPR_PREMIUM_max',

       'FL_NPR_PREMIUM_sum', 'FL_NPR_PREMIUM_mean', 'FL_NP2_EFFECTDATE_count',

       'FL_NP2_EFFECTDATE_nunique', 'FL_Policy ID_count',

       'FL_Policy ID_nunique', 'FL_CLF_LIFECD_count', 'FL_CLF_LIFECD_nunique',

       'FL_NSP_SUBPROPOSAL_count', 'FL_NSP_SUBPROPOSAL_nunique',

       'FL_NPR_SUMASSURED_min', 'FL_NPR_SUMASSURED_max',

       'FL_NPR_SUMASSURED_sum', 'FL_NPR_SUMASSURED_mean', 'FL_NLO_TYPE_count',

       'FL_NLO_TYPE_nunique', 'FL_NLO_AMOUNT_min', 'FL_NLO_AMOUNT_max',

       'FL_NLO_AMOUNT_sum', 'FL_NLO_AMOUNT_mean', 'FL_AAG_AGCODE_count',

       'FL_AAG_AGCODE_nunique', 'FL_PCL_LOCATCODE_count',

       'FL_PCL_LOCATCODE_nunique', 'FL_OCCUPATION_count',

       'FL_OCCUPATION_nunique', 'FL_CATEGORY_count', 'FL_CATEGORY_nunique']:

    d[k]=n

pol = policy.groupby('Policy ID').agg(d)
pol.columns=['FI_' + '_'.join(col).strip() for col in pol.columns.values]

pol.reset_index(inplace=True)

pol
# pol['FI_NP2_EFFECTDATE_unique'].value_counts()

# pol['FI_NP2_EFFECTDATE_unique'].apply(lambda x: " ".join([str(c) for c in x]))
pol['FI_NP2_EFFECTDATE_year_unique']=pol['FI_NP2_EFFECTDATE_year_unique'].apply(lambda x: " ".join([str(c) for c in x]))

pol['FI_NP2_EFFECTDATE_day_unique']=pol['FI_NP2_EFFECTDATE_day_unique'].apply(lambda x: " ".join([str(c) for c in x]))
# pol.drop('FI_NP2_EFFECTDATE_month_unique',axis=1,inplace=True)
# ls.columns

df
df = df.merge(pol,on='Policy ID')
# df['FI_NP2_EFFECTDATE_year_unique'].value_counts()
test=test.merge(pol,on='Policy ID')

test
# client.groupby(['Policy ID']).mean()
# # # len(np.intersect1d(df['Policy ID'],cl['Policy ID']))

# payment['DATEPAID']=pd.to_datetime(payment['DATEPAID'])



# payment['POSTDATE']=pd.to_datetime(payment['POSTDATE'])

# payment['PREMIUMDUEDATE']=pd.to_datetime(payment['PREMIUMDUEDATE'])

# payment['Diff1']=(payment['POSTDATE']-payment['DATEPAID'])/np.timedelta64(1,'D')

# payment['Diff2']=(payment['DATEPAID']-payment['PREMIUMDUEDATE'])/np.timedelta64(1,'D')

# payment['Diff3']=(payment['POSTDATE']-payment['PREMIUMDUEDATE'])/np.timedelta64(1,'D')



# # payment.drop(['DATEPAID','POSTDATE','PREMIUMDUEDATE'],axis=1,inplace=True)



# payment['DATEPAID_yr']=payment['DATEPAID'].dt.year

# payment['DATEPAID_dy']=payment['DATEPAID'].dt.month

# payment['POSTDATE_yr']=payment['POSTDATE'].dt.year

# payment['POSTDATE_dy']=payment['POSTDATE'].dt.month

# payment['PREMIUMDUEDATE_yr']=payment['PREMIUMDUEDATE'].dt.year

# payment['PREMIUMDUEDATE_dy']=payment['PREMIUMDUEDATE'].dt.month





# payment
# payment[payment['Policy ID']=='PID_NCXO0DU']
# payment.columns
# d={'AMOUNTPAID':n, 'DATEPAID':c, 'POSTDATE':c, 'PREMIUMDUEDATE':c,

#        'Diff1':n, 'Diff2':n, 'Diff3':n, 'DATEPAID_yr':['unique'], 'DATEPAID_dy':['unique'], 'POSTDATE_yr':['unique'],

#        'POSTDATE_dy':['unique'], 'PREMIUMDUEDATE_yr':['unique']}

# py=payment.groupby('Policy ID').agg(d)

# py.columns=['FP_' + '_'.join(col).strip() for col in py.columns.values]

# py.reset_index(inplace=True)

# py
# py['FP_DATEPAID_yr_unique']=py['FP_DATEPAID_yr_unique'].apply(lambda x: " ".join([str(c) for c in x]))

# py['FP_DATEPAID_dy_unique']=py['FP_DATEPAID_dy_unique'].apply(lambda x: " ".join([str(c) for c in x]))

# py['FP_POSTDATE_yr_unique']=py['FP_POSTDATE_yr_unique'].apply(lambda x: " ".join([str(c) for c in x]))

# py['FP_POSTDATE_dy_unique']=py['FP_POSTDATE_dy_unique'].apply(lambda x: " ".join([str(c) for c in x]))

# py['FP_PREMIUMDUEDATE_yr_unique']=py['FP_PREMIUMDUEDATE_yr_unique'].apply(lambda x: " ".join([str(c) for c in x]))
# py[[ 'FP_DATEPAID_yr_unique', 'FP_DATEPAID_dy_unique',

#        'FP_POSTDATE_yr_unique', 'FP_POSTDATE_dy_unique',

#        'FP_PREMIUMDUEDATE_yr_unique']].fillna('NA',inplace=True)
# py[[ 'FP_DATEPAID_yr_unique', 'FP_DATEPAID_dy_unique',

#        'FP_POSTDATE_yr_unique', 'FP_POSTDATE_dy_unique',

#        'FP_PREMIUMDUEDATE_yr_unique']]
# df=df.merge(py,on='Policy ID',how='left')

# test=test.merge(py,on='Policy ID',how='left')
# df.isnull().sum()
# test.isnull().sum()
# dftest = dftest.merge(client.drop('NPH_LASTNAME',axis=1),on=['Policy ID'],how='left')

# dftest['Policy ID'].nunique()



# dftrain = dftrain.merge(client.drop('NPH_LASTNAME',axis=1),on=['Policy ID'],how='left')

# dftrain['Policy ID'].nunique()
# payment['DATEPAID']=pd.to_datetime(payment['DATEPAID'])



# payment['POSTDATE']=pd.to_datetime(payment['POSTDATE'])

# payment['PREMIUMDUEDATE']=pd.to_datetime(payment['PREMIUMDUEDATE'])
# (payment['POSTDATE']-payment['PREMIUMDUEDATE'])/np.timedelta64(1,'D')
# payment['Diff1']=(payment['POSTDATE']-payment['DATEPAID'])/np.timedelta64(1,'D')

# payment['Diff2']=(payment['DATEPAID']-payment['PREMIUMDUEDATE'])/np.timedelta64(1,'D')

# payment['Diff3']=(payment['POSTDATE']-payment['PREMIUMDUEDATE'])/np.timedelta64(1,'D')



# payment.drop(['DATEPAID','POSTDATE','PREMIUMDUEDATE'],axis=1,inplace=True)
# dftrain = dftrain.merge(payment,on=['Policy ID'],how='left')



# dftest = dftest.merge(payment,on=['Policy ID'],how='left')
# dftrain
# dfmain=dftrain.append(dftest,ignore_index=True)

# dfmain.head()
# dfmain['year_passed'] = dfmain['Year'].astype(np.int) - dfmain['NP2_EFFECTDATE'].dt.year
# dfmain.info()
# df.info()
df.tail()
dftest=test.drop(['Lapse','Policy ID'],axis=1)

dftest.shape
# for k in [ 'FP_DATEPAID_yr_unique', 'FP_DATEPAID_dy_unique',

#        'FP_POSTDATE_yr_unique', 'FP_POSTDATE_dy_unique',

#        'FP_PREMIUMDUEDATE_yr_unique']:

#     df[k]=df[k].fillna('0')

#     dftest[k]=dftest[k].fillna('0')
# df[[ 'FP_DATEPAID_yr_unique', 'FP_DATEPAID_dy_unique',

#        'FP_POSTDATE_yr_unique', 'FP_POSTDATE_dy_unique',

#        'FP_PREMIUMDUEDATE_yr_unique']].replace({np.nan:'0'},inplace=True)

# dftest[[ 'FP_DATEPAID_yr_unique', 'FP_DATEPAID_dy_unique',

#        'FP_POSTDATE_yr_unique', 'FP_POSTDATE_dy_unique',

#        'FP_PREMIUMDUEDATE_yr_unique']]
# df[[ 'FP_DATEPAID_yr_unique', 'FP_DATEPAID_dy_unique',

#        'FP_POSTDATE_yr_unique', 'FP_POSTDATE_dy_unique',

#        'FP_PREMIUMDUEDATE_yr_unique']].fillna('',inplace=True)

# df
# df[[ 'FP_DATEPAID_yr_unique', 'FP_DATEPAID_dy_unique',

#        'FP_POSTDATE_yr_unique', 'FP_POSTDATE_dy_unique',

#        'FP_PREMIUMDUEDATE_yr_unique']]
for k in ['Policy ID','Lapse Year','FI_NP2_EFFECTDATE_year_unique','FI_NP2_EFFECTDATE_day_unique']:

    df[k]=df[k].astype('category')
df.isnull().sum()
df.isnull().sum()
# dfmain['Lapse'].replace({'?':np.nan},inplace=True)
# dfmain.drop(['NP2_EFFECTDATE'],axis=1,inplace=True)
# dfmain.columns
# dfmain[dfmain['Policy ID']=='PID_MFAAYNJ']
# dfmain.groupby(['Policy ID','Year']).mean().reset_index()
from catboost import CatBoostClassifier,Pool, cv

from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold,train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,f1_score,log_loss

from sklearn.naive_bayes import MultinomialNB

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt





# df_train=df[dfmain['Lapse'].isnull()==False].copy()

# df_test=df[dfmain['Lapse'].isnull()==True].reset_index(drop=True)



# print(df_train.shape,df_test.shape)
df['Lapse']=df['Lapse'].astype(np.int)
# X,y=df[df['Lapse Year']=='2019'].drop(['Lapse','Policy ID','Lapse Year'],axis=1),df[df['Lapse Year']=='2019']['Lapse']



X,y=df.drop(['Lapse','Lapse Year','Policy ID'],axis=1),df['Lapse']
# X,y=df[df['Lapse Year']=='2019'].drop(['Lapse','Lapse Year','Policy ID'],axis=1),df[df['Lapse Year']=='2019']['Lapse']



# Xtest=df.drop(['Lapse','Policy ID'],axis=1)

# X.columns
X.shape
df[df['Lapse Year']=='2019']['Lapse']
# df[df['Lapse Year']!='2019']['Lapse'].value_counts()


# X_val,y_val = df[df['Lapse Year']=='2019'].drop(['Lapse','Lapse Year','Policy ID'],axis=1),df[df['Lapse Year']=='2019']['Lapse']



# X_train,y_train = df[df['Lapse Year']!='2019'].drop(['Lapse','Lapse Year','Policy ID'],axis=1),df[df['Lapse Year']!='2019']['Lapse']
# X.info()
# X_train
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.20,random_state = 1994,stratify=y)
# m=LGBMClassifier(boosting_type='gbdt', class_weight='balanced',

#                 importance_type='split', learning_rate=0.03,col_sample_by_tree=0.9,

#                max_depth=-1, min_child_samples=20, min_child_weight=0.001,

#                min_split_gain=0.0, n_estimators=7000, n_jobs=-1, num_leaves=50,

#                objective='binary', random_state=1994,

#                reg_lambda=0.0, silent=True, subsample=1.0,

#                subsample_for_bin=200000, subsample_freq=0)

# # m=RidgeCV(cv=4)

# m.fit(X_train,y_train,eval_set=[(X_val, y_val)],eval_metric='loss', early_stopping_rounds=100,verbose=200)

# # p=m.predict_proba(X_val)[:-1]
cat_indx = np.where(X_train.dtypes=='category')[0]
cat_indx
m2  = CatBoostClassifier(n_estimators=3000,eval_metric='Logloss',learning_rate=0.03, random_seed= 1234,cat_features=cat_indx)

m2.fit(X_train,y_train,eval_set=[(X_train,y_train),((X_val, y_val))], early_stopping_rounds=100,verbose=200,)#erly100
p=m2.predict_proba(X_val)[:,-1]
log_loss(y_val,p)
import matplotlib.pyplot as plt

import warnings

import seaborn as sns

sns.set_style('darkgrid')

warnings.filterwarnings('ignore')



%matplotlib inline

feature_imp = pd.DataFrame(sorted(zip(m2.feature_importances_, X.columns), reverse=True)[:200], 

                           columns=['Value','Feature'])

plt.figure(figsize=(12,8))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()
# cols=feature_imp[:70]['Feature'].values
# X,y=df.drop(['Lapse','Lapse Year','Policy ID'],axis=1),df['Lapse']

# X=X[cols]

# X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.20,random_state = 1994,stratify=y)

# m2  = CatBoostClassifier(n_estimators=4000,eval_metric='Logloss',learning_rate=0.03, random_seed= 1234, use_best_model=True,cat_features=cat_indx)

# m2.fit(X_train,y_train,eval_set=[(X_train,y_train),((X_val, y_val))], early_stopping_rounds=100,verbose=200,)#erly100
# # dftest['Policy ID']=dftest['Policy ID'].astype('category')

# dftest=test.drop(['Lapse','Policy ID'],axis=1)

# dftest=dftest[cols]



for k in ['FI_NP2_EFFECTDATE_year_unique','FI_NP2_EFFECTDATE_day_unique']:

    dftest[k]=dftest[k].astype('category')
cat_indx
np.where(dftest.dtypes=='category')[0]
from catboost import CatBoostClassifier

errcb2=[]

y_pred_totcb2=[]

from sklearn.model_selection import KFold,StratifiedKFold, TimeSeriesSplit

from sklearn.metrics import mean_squared_error

fold=StratifiedKFold(n_splits=7)#15#5#10

i=1

for train_index, test_index in fold.split(X,y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    m2  = CatBoostClassifier(   n_estimators=4000,eval_metric='Logloss',learning_rate=0.03, random_seed= 1234, use_best_model=True ,cat_features=cat_indx)

    m2.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)], early_stopping_rounds=200,verbose=400)#erly100

    preds=m2.predict_proba(X_test)[:,-1]

    print("err: ",log_loss(y_test,preds))

    errcb2.append(log_loss(y_test,preds))

    p2 = m2.predict_proba(dftest)[:,-1]

    y_pred_totcb2.append(p2)

np.mean(errcb2)
# y_pred_tot1=[]

# err1=[]

# # feature_importance_df = pd.DataFrame()

# from sklearn.model_selection import KFold,StratifiedKFold

# fold=StratifiedKFold(n_splits=5,shuffle=True,random_state=1994)



# i=1

# for train_index, test_index in fold.split(X,y):

    

#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]

#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    

    

#     m=LGBMClassifier(boosting_type='gbdt', class_weight='balanced',

#                 importance_type='split', learning_rate=0.03,col_sample_by_tree=0.9,

#                max_depth=-1, min_child_samples=20, min_child_weight=0.001,

#                min_split_gain=0.0, n_estimators=7000, n_jobs=-1, num_leaves=50,

#                objective='binary', random_state=1994,

#                reg_lambda=0.0, silent=True, subsample=1.0,

#                subsample_for_bin=200000, subsample_freq=0)

#     m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)],early_stopping_rounds=100,verbose=200)

    

#     preds=m.predict_proba(X_test,ntree_end=m.best_iteration_)[:,-1]

#     print("err: ",log_loss(y_test,preds))

#     err1.append(log_loss(y_test,preds))

#     p = m.predict_proba(dftest)[:,-1]

#     i=i+1

#     y_pred_tot1.append(p)
np.mean(errcb2)
y_pred=np.mean(y_pred_totcb2,0)



# y_pred=m.predict_proba(dftest,n_iterations=m.best_iteration_)[:,-1]
y_pred
test['Lapse']=y_pred
s = test[['Policy ID','Lapse']]
s
sum(s['Lapse']>0.5)/s.shape[0]
sum(s['Lapse']>0.2)/s.shape[0]
s.to_csv('sNewCB_kv23.csv',index=False)