# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/final-participant-data/Final Participant Data Folder"))



# Any results you write to the current directory are saved as output.
train=pd.read_excel('../input/final-participant-data/Final Participant Data Folder/Final_Train.xlsx')

test=pd.read_excel('../input/final-participant-data/Final Participant Data Folder/Final_Test.xlsx')

train.head()
train.sample(20)
# df['main_place']=df['Place'].apply(lambda x : x.split(", ")[-1])

import re

def num_ex(val):

    try:

        t=int(re.sub("[^0-9]",'-',val).split('--')[1])

    except:

        t=0

    return t

def is_nan(x):

    return int(x is np.nan or x != x)

print(train.shape)

train=pd.DataFrame(data=train.loc[train['Place']!='e'],columns=train.columns).reset_index().drop('index',axis=1)

print(train.shape)

big_df = train.append(test)
big_df.head()
df=big_df.copy()

df['Experience']=df['Experience'].apply(lambda x:int( x.split(" ")[0]))

df['Rating'].fillna('0%',inplace=True)

df['Place'].fillna('Others',inplace=True)

df['isMisc']=df['Miscellaneous_Info'].apply(is_nan)

df['Miscellaneous_Info'].fillna('Others',inplace=True)

df['Rating']=df['Rating'].apply(lambda x:int(x[:-1]))

df['Exp_rate']=df['Experience']+df['Rating']

from sklearn.preprocessing import LabelEncoder

df['City']=df['Place'].apply(lambda x : x.split(", ")[-1])

df['Locality']=df['Place'].apply(lambda x : x.split(", ")[0])

df['City'].fillna("XXX",inplace = True)

df['Locality'].fillna("XXX",inplace = True)

df['isFeedback']=df['Miscellaneous_Info'].apply(lambda x :1 if 'Feedback' in str(x) else 0)

# df['misc_magic_n']=df['Miscellaneous_Info'].apply(num_ex)

df['Qual_len']=df['Qualification'].apply(lambda x : len(x.split(",")))

df['Qual_len2']=df['Qualification'].astype(str).apply(len)



# df['misc_len']=df['Miscellaneous_Info'].apply(lambda x : len(x.split(" ")))



df['Qual_1'] = df['Qualification'].str.split(',').str[0]

df['Qual_2'] = df['Qualification'].str.split(',').str[1]

df['Qual_2_len']=df['Qual_2'].astype(str).apply(len)

df['Qual_3'] = df['Qualification'].str.split(',').str[2]

df['Qual_1'].fillna("XXX",inplace = True)

df['Qual_2'].fillna("YYY",inplace = True)

df['Qual_2_len'].fillna(0,inplace = True)

df['Qual_3'].fillna("ZZZ",inplace = True)





df['Misc'] = df['Miscellaneous_Info'].str.split('%').str[0]

df['Misc_size'] = df['Miscellaneous_Info'].apply(len)

df['Misc_len'] = df['Misc'].str.len()

df.loc[df['Misc_len']>3, 'Misc'] = 0

df['Misc'].fillna(0,inplace = True)

df['Misc'] = df['Misc'].astype(int)

df['Misc_2'] = df['Miscellaneous_Info'].str.split('% ').str[1]

df['Misc_3'] = df['Misc_2'].str.split(' ').str[0]

df['Misc_3'].fillna(0,inplace = True)

df['Misc_3_len'] = df['Misc_3'].str.len()

df.loc[df['Misc_3_len']>3, 'Misc_3'] = 0

df.loc[df['Misc_3']==',', 'Misc_3'] = 0

df['Misc_3'] = df['Misc_3'].astype(int)

df['Misc_2'].fillna('XX',inplace=True)

df['Misc_3_len'].fillna(0,inplace=True)

df['Misc_4'] = df['Misc']*np.log((1+df['Misc_3']))

df['log_exp_rating']=np.log(1+df['Experience'])*np.log((1+df['Rating']))

df['var1']=df['Qual_2_len']/df['Qual_len']

# df['var2']=df['Misc_3']+df['Experience']

df['var3']=np.log(1+df['Rating'])*np.log(1+df['Misc_3'])

# df=pd.get_dummies(df,columns=['Profile','City'],drop_first=True)

df.head()
# np.log(1+df['Rating'])*np.log(1+df['Misc_3'])

# df['Misc_3']

# Miscellaneous_Info 3981

# Place 948

# Profile 6

# Qualification 1801

# City 9

# Locality 937

# Qual_1 181

# Qual_2 496

# Qual_3 482

# Misc_2 1897

dfcatboost=df.copy()

from sklearn.preprocessing import LabelEncoder

cat_cols=['Profile','City','Locality','Qual_1','Qual_2']

for i in cat_cols:

    l=LabelEncoder()

    df[i]=l.fit_transform(df[i])

df.head()
# big_df['Qual_1'] = big_df['Qualification'].str.split(',').str[0]

# big_df['Qual_2'] = big_df['Qualification'].str.split(',').str[1]

# big_df['Qual_3'] = big_df['Qualification'].str.split(',').str[2]

# big_df['Qual_1'].fillna("XXX",inplace = True)

# big_df['Qual_2'].fillna("XXX",inplace = True)

# big_df['Qual_3'].fillna("XXX",inplace = True)

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

plt.figure(figsize=(12,12))

sns.heatmap(df.corr(),annot=True)
for i in df.columns:

    if df[i].dtype!='object':

#         sns.jointplot(i,'Fees',df)

        pass

    else:

        print(i,df[i].nunique())
# big_df['years_exp'] = big_df['Experience'].str.slice(stop=2).astype(int)

# big_df['Rating'].fillna('0%',inplace = True)

# big_df['Rating'] = big_df['Rating'].str.slice(stop=-1).astype(int)

# big_df['City'] = big_df['Place'].str.split(',').str[1]

# big_df['Locality'] = big_df['Place'].str.split(',').str[0]

# big_df['City'].fillna("XXX",inplace = True)

# big_df['Locality'].fillna("XXX",inplace = True)





# big_df['Misc'] = big_df['Miscellaneous_Info'].str.split('%').str[0]

# big_df['Misc_len'] = big_df['Misc'].str.len()

# big_df.loc[big_df['Misc_len']>3, 'Misc'] = 0

# big_df['Misc'].fillna(0,inplace = True)

# big_df['Misc'] = big_df['Misc'].astype(int)

# big_df['Misc_2'] = big_df['Miscellaneous_Info'].str.split('% ').str[1]

# big_df['Misc_3'] = big_df['Misc_2'].str.split(' ').str[0]

# big_df['Misc_3'].fillna(0,inplace = True)

# big_df['Misc_3_len'] = big_df['Misc_3'].str.len()

# big_df.loc[big_df['Misc_3_len']>3, 'Misc_3'] = 0

# big_df.loc[big_df['Misc_3']==',', 'Misc_3'] = 0

# big_df['Misc_3'] = big_df['Misc_3'].astype(int)

df.head()
df.tail()
big_df = df.drop(['Miscellaneous_Info','Place','Qualification','Misc_2','Qual_3'], axis=1)

df_train = big_df[0:5960]

df_test = big_df[5960:]

df_test = df_test.drop(['Fees'], axis =1)







df_train.tail()
df_test.head()
# df_test_merge_1 = df_test[['Qual_1','Qual_1_code']].drop_duplicates()

# df_test_merge_2 = df_test[['Qual_2','Qual_2_code']].drop_duplicates()

# df_test_merge_3 = df_test[['Qual_3','Qual_3_code']].drop_duplicates()

# df_test_merge_4 = df_test[['Profile','Profile_code']].drop_duplicates()

# df_test_merge_5 = df_test[['City','City_code']].drop_duplicates()

# df_test_merge_6 = df_test[['Locality','Locality_code']].drop_duplicates()



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,f1_score,mean_squared_error,mean_squared_log_error

X=df_train.drop(['Fees'],axis=1)

y=np.log1p(df_train['Fees'])

# y=train['Fees']

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.25,random_state = 1994)



categorical_features_indices = np.where(X_train.dtypes =='object')[0]

categorical_features_indices
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression,Lasso,Ridge,RidgeCV,BayesianRidge

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



from sklearn.metrics import mean_squared_log_error

# def sigmoid(x, derivative=False):

#     sigm = 1. / (1. + np.exp(-x))

#     if derivative:

#         return sigm * (1. - sigm)

#     return sigm

import math



#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)

def rmsle(real, predicted):

    

    real=np.expm1(real)

    predicted=np.expm1(predicted)

#     print(real,predicted)

#     real=np.exp(real)

#     predicted=np.exp(predicted)

#     sum=0.0

#     for x in range(0,len(predicted)):

#         if predicted[x]<0 or real[x]<0: #check for negative values

#             continue

#         p = np.log(predicted[x]+1)

#         r = np.log(real[x]+1)

#         sum = sum + (p - r)**2

#     return (sum/len(predicted))**0.5

    return np.sqrt(mean_squared_log_error(real,predicted))

    



def rmsle_lgb(labels, preds):

    return 'rmsle', rmsle(preds,labels), False



# print(rmsle(np.expm1(y_val.values),np.expm1(p)))



# print(np.sqrt(mean_squared_log_error(y_val,p)))
from lightgbm import LGBMRegressor

m=LGBMRegressor(n_estimators=1000,verbose=100)

# m=RidgeCV(cv=4)

m.fit(X,y,eval_metric=rmsle_lgb, verbose=100,categorical_feature=cat_cols)

lgbmp=m.predict(X)

# m=Rid

# print(np.sqrt(mean_squared_log_error(np.expm1(y_val.values),np.expm1(p))))

# print(np.sqrt(mean_squared_log_error(y_val.values,p)))

# print(rmsle_lgb(y_val.values,lgbmp))
lgbmp
print(X_train.shape)

X_train.tail()
# from lightgbm import LGBMRegressor

# from sklearn.neural_network import MLPRegressor

# m=MLPRegressor(hidden_layer_sizes=(100,100,50))

# # m=RidgeCV(cv=4)

# m.fit(X_train,y_train)

# p=m.predict(X_val)

# # m=Rid

# print(np.sqrt(mean_squared_log_error(np.expm1(y_val.values),np.expm1(p))))

# # print(np.sqrt(mean_squared_log_error(y_val.values,p)))

# # print(rmsle_lgb(y_val.values,lgbmp))


errlgb=[]

y_pred_totlgb=[]

# low=99999

# jj=[]

# X=X.tocsr()

i=0

from sklearn.model_selection import KFold,StratifiedKFold

fold=KFold(n_splits=27,shuffle=True,random_state=1994)

for train_index, test_index in fold.split(X):

#     print(train_index, test_index)

    X_train, X_test = X.loc[train_index], X.loc[test_index]

#     print(X_train,X_test)

    y_train, y_test = y[train_index], y[test_index]

    lgbm_params = {'n_estimators': 2000, 

#                    'learning_rate': 0.05, 

#                    'nthread': 3, # Updated from nthread

#           'num_leaves': 64,

#                    'reg_alpha': 0.5,

#           'reg_lambda': 5,

#                    'boosting_type': 'gbdt',

#                    'objective':"regression",

#                    'max_depth': 5,

#                'num_leaves': 50, 

#                    'subsample': 0.6, 'min_child_weight': 5, 'max_depth': 5, 'learning_rate': 0.05, 'gamma': 5, 'colsample_bytree': 0.6,

#                    'subsample': 0.9,

#                    'colsample_bytree': 0.8,

#                'min_child_samples': 50,

                   'n_jobs': -1}

    rf=LGBMRegressor(**lgbm_params)

#     rf=CatBoostRegressor(n_estimators=2000,eval_metric='RMSE',learning_rate=0.05,max_depth=5)

#     rf=XGBRegressor(**lgbm_params)

    

    

    rf.fit(X_train,y_train,eval_set=[(X_test, y_test.values)],

         eval_metric=rmsle_lgb,

#            eval_metric='rmse',

           categorical_feature=cat_cols,

         verbose=200

         , early_stopping_rounds=100

          )

#     print('predict')

    pr=rf.predict(X_test)

#     print(1)

    print("errlgb: ",np.sqrt(mean_squared_log_error(np.expm1(y_test.values),np.expm1(pr))))

    

    errlgb.append(rmsle_lgb(y_test.values,pr)[1])

    p = rf.predict(df_test)

    print(p.shape)

#     s=pd.DataFrame({'Fees':np.expm1(p)})

#     s.to_excel('pred_docMH_5folds'+str(i)+'.xlsx',index=False)

    i+=1

#     p = rf.predict(pcft)

#     if low>np.sqrt(mean_squared_log_error(np.expm1(y_test.values),np.expm1(pr))):

#         low=np.sqrt(mean_squared_log_error(np.expm1(y_test.values),np.expm1(pr)))

#         jj=p

    y_pred_totlgb.append(p)
y_pred_totlgb[0].shape

print(np.mean(errlgb,0))
# from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor

# from xgboost import XGBRegressor

# errxgb=[]

# y_pred_totxgb=[]

# from sklearn.model_selection import KFold,StratifiedKFold

# fold=KFold(n_splits=25,shuffle=True,random_state=1994)

# for train_index, test_index in fold.split(X,y):

# #     print(1)

#     X_train, X_test = X.loc[train_index], X.loc[test_index]

#     y_train, y_test = y[train_index], y[test_index]

#     lgbm_params = {'n_estimators': 1000, 

# #                    'learning_rate': 0.1, 

# #                    'nthread': 3, # Updated from nthread

# #           'num_leaves': 64,

# #                    'reg_alpha': 0.5,

# #           'reg_lambda': 5,

# #                    'boosting_type': 'gbdt',

# #                    'objective':"regression",

#                    'max_depth': 5,

# #                'num_leaves': 50, 

# #                    'subsample': 0.6, 'min_child_weight': 5, 'max_depth': 5, 'learning_rate': 0.05, 'gamma': 5, 'colsample_bytree': 0.6,

# #                    'subsample': 0.9,

# #                    'colsample_bytree': 0.8,

# #                'min_child_samples': 50,

#                    'n_jobs': -1}

# #     rf=LGBMRegressor(**lgbm_params)

# #     rf=CatBoostRegressor(n_estimators=1000,eval_metric='RMSE')

#     rf=XGBRegressor(**lgbm_params)

# #     rf=XGBRegressor()

# #     rf=GradientBoostingRegressor(n_estimators=200)

# #     rf=BaggingRegressor(base_estimator=GradientBoostingRegressor(n_estimators=200))

#     rf.fit(X_train,y_train,eval_set=[(X_test, y_test.values)],

# #          eval_metric=rmsle_lgb,

#            eval_metric='rmse',

#          verbose=100, early_stopping_rounds=50)

#     prx=rf.predict(X_test)

#     print("err: ",rmsle(y_test.values,prx))

#     errxgb.append(rmsle(y_test.values,prx))

#     p = rf.predict(df_test)

#     y_pred_totxgb.append(p)
# np.mean(errxgb,0)
# m=XGBRegressor(n_estimators=1500)

# # m=RidgeCV(cv=4)

# m.fit(X,y,eval_set=[(X, y.values)],eval_metric='rmse', early_stopping_rounds=100,verbose=100)

# xgbp=m.predict(X)
big_dfcb = dfcatboost.drop(['Miscellaneous_Info','Place','Qualification','Misc_2','Qual_3'], axis=1)

df_train = big_dfcb[0:5960]

df_test = big_dfcb[5960:]

df_test = df_test.drop(['Fees'], axis =1)



X=df_train.drop(['Fees'],axis=1)

y=np.log1p(df_train['Fees'])

# y=train['Fees']

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.25,random_state = 1994)



categorical_features_indices = np.where(X_train.dtypes =='object')[0]

categorical_features_indices
from catboost import CatBoostRegressor

rf=CatBoostRegressor(n_estimators=2000,eval_metric='RMSE',learning_rate=0.05,max_depth=5)

#     rf=XGBRegressor(**lgbm_params)





rf.fit(X,y,

#          eval_metric=rmsle_lgb,

#            eval_metric='rmse',

       cat_features=categorical_features_indices,

     verbose=200

#          , early_stopping_rounds=100,use_best_model=True

      )

#     print('predict')

cbpr=rf.predict(X)

#     print(1)

# print("err: ",np.sqrt(mean_squared_log_error(np.expm1(y_val.values),np.expm1(cbpr))))



# err.append(rmsle_lgb(y_val.values,pr)[1])

# p = rf.predict(df_test)
# rf.fit(X,y,cat_features=categorical_features_indices,eval_set=(X, y),

#         plot=False,early_stopping_rounds=100,use_best_model=True,verbose=200)

# y_predd=rf.predict(df_test)



# pr

# print(rmsle_lgb(y_val.values,(p+pr)/2))
# y_predd


err=[]

y_pred_tot=[]

# low=99999

# jj=[]

# X=X.tocsr()

i=0

from sklearn.model_selection import KFold,StratifiedKFold

fold=KFold(n_splits=27,shuffle=True,random_state=1994)

for train_index, test_index in fold.split(X):

#     print(train_index, test_index)

    X_train, X_test = X.loc[train_index], X.loc[test_index]

#     print(X_train,X_test)

    y_train, y_test = y[train_index], y[test_index]

    lgbm_params = {'n_estimators': 1500, 

                   'learning_rate': 0.03, 

#                    'nthread': 3, # Updated from nthread

#           'num_leaves': 64,

#                    'reg_alpha': 0.5,

#           'reg_lambda': 5,

#                    'boosting_type': 'gbdt',

#                    'objective':"regression",

                   'max_depth': 9,

#                'num_leaves': 50, 

#                    'subsample': 0.6, 'min_child_weight': 5, 'max_depth': 5, 'learning_rate': 0.05, 'gamma': 5, 'colsample_bytree': 0.6,

#                    'subsample': 0.9,

#                    'colsample_bytree': 0.8,

#                'min_child_samples': 50,

                   'n_jobs': -1}

#     rf=LGBMRegressor(**lgbm_params)

    rf=CatBoostRegressor(n_estimators=2000,eval_metric='RMSE',learning_rate=0.05,max_depth=5)

#     rf=XGBRegressor(**lgbm_params)

    

    

    rf.fit(X_train,y_train,eval_set=[(X_test, y_test.values)],

#          eval_metric=rmsle_lgb,

#            eval_metric='rmse',

           cat_features=categorical_features_indices,

         verbose=200

#          , early_stopping_rounds=100,use_best_model=True

          )

#     print('predict')

    prc=rf.predict(X_test)

#     print(1)

    print("err: ",np.sqrt(mean_squared_log_error(np.expm1(y_test.values),np.expm1(prc))))

    

    err.append(rmsle_lgb(y_test.values,prc)[1])

    p = rf.predict(df_test)

#     s=pd.DataFrame({'Fees':np.expm1(p)})

#     s.to_excel('pred_docMH_5folds'+str(i)+'.xlsx',index=False)

    i+=1

#     p = rf.predict(pcft)

#     if low>np.sqrt(mean_squared_log_error(np.expm1(y_test.values),np.expm1(pr))):

#         low=np.sqrt(mean_squared_log_error(np.expm1(y_test.values),np.expm1(pr)))

#         jj=p

    y_pred_tot.append(p)
# np.where(X_train.dtype==np.object)

print(np.mean(err,0))

print(np.mean(errlgb,0))

# print(np.mean(errxgb,0))
# sorted(zip(rf.feature_importances_,X_train),reverse=True)

# X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state = 1994)

# from lightgbm import LGBMRegressor

# m=LGBMRegressor(n_estimators=1000)

# # m=RidgeCV(cv=4)

# m.fit(X_train,y_train,eval_set=[(X_val, y_val.values)],eval_metric=rmsle_lgb, early_stopping_rounds=100,categorical_feature="1,4,5,9,10,11")

# p=m.predict(X_val)

# # m=Rid

# # print(np.sqrt(mean_squared_log_error(np.expm1(y_val.values),np.expm1(p))))

# # print(np.sqrt(mean_squared_log_error(y_val.values,p)))

# print(rmsle_lgb(y_val.values,p))

np.mean(y_pred_totlgb,0)




s=pd.read_excel('../input/final-participant-data/Final Participant Data Folder/Sample_submission.xlsx')

s.head()

# xgbp

# main_df=pd.DataFrame({'lgb':lgbmp,'cb':cbpr,'p':y})

# main_df.head()

df_train['lgb']=np.expm1(lgbmp)

df_train['cb']=np.expm1(cbpr)

main_df=df_train.copy()

XX,yy=main_df.drop('Fees',axis=1),np.log1p(main_df.Fees)

print(XX.shape,yy.shape)

main_df.head()
# (np.mean(y_pred_tot,0)+np.mean(y_pred_totlgb,0)+np.mean(y_pred_totxgb,0))/3

df_test['lgb']=np.expm1(np.mean(y_pred_totlgb,0))

df_test['cb']=np.expm1(np.mean(y_pred_tot,0))

predict_df=df_test.copy()

# predict_df=pd.DataFrame({'lgb':np.mean(y_pred_totlgb,0),'cb':np.mean(y_pred_tot,0)})

predict_df.head()


errmain=[]

y_pred_main=[]

# low=99999

# jj=[]

# X=X.tocsr()

i=0

from sklearn.model_selection import KFold,StratifiedKFold

fold=KFold(n_splits=27,shuffle=True,random_state=1994)

for train_index, test_index in fold.split(XX):

#     print(train_index, test_index)

    X_train, X_test = XX.loc[train_index], XX.loc[test_index]

#     print(X_train,X_test)

    y_train, y_test = yy[train_index], yy[test_index]

    lgbm_params = {'n_estimators': 2000, 

                   'n_jobs': -1}

    rf=CatBoostRegressor(n_estimators=2000,eval_metric='RMSE',learning_rate=0.05,max_depth=5)

    rf.fit(X_train,y_train,eval_set=[(X_test, y_test.values)],

         verbose=200,cat_features=categorical_features_indices

#          , early_stopping_rounds=100

          )

    pr=rf.predict(X_test)

    print("errmain: ",np.sqrt(mean_squared_log_error(np.expm1(y_test.values),np.expm1(pr))))

    

    errmain.append(rmsle_lgb(y_test.values,pr)[1])

    pp = rf.predict(predict_df)

    print(pp.shape)

    y_pred_main.append(pp)
y_pred_main
s['Fees']=np.expm1((np.mean(y_pred_tot,0)+np.mean(y_pred_totlgb,0))/2)

s.head()


s.to_excel('pred_docMH_27folds_cb_lgbm3.xlsx',index=False)

s['Fees']=np.expm1(np.mean(y_pred_main,0))

s.head()
s.to_excel('pred_docMH_27folds_cb_lgbm_stacked1.xlsx',index=False)