import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectFromModel

import time

%matplotlib inline

# Classification

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier



import xgboost as xgb

import lightgbm as lgb

import catboost as cat

from catboost import CatBoostClassifier,Pool, cv



# Preprocessing

from sklearn.metrics import log_loss

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score



from sklearn.model_selection import StratifiedKFold,KFold

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score

from mlxtend.classifier import StackingClassifier

from datetime import datetime, timedelta

from sklearn.naive_bayes import MultinomialNB



import joblib

from sklearn.preprocessing import LabelEncoder
test = pd.read_csv('../input/janatahack-hr-analytics/test.csv')

train = pd.read_csv('../input/janatahack-hr-analytics/train.csv')

print(test.shape,train.shape)
target = 'target'

test_ids = test['enrollee_id'] 

org_feat = train.columns

df=train.append(test,ignore_index=True)

sns.heatmap(df.isnull())
df.info()
def logic_missing(df):

    df.loc[df['education_level'] == 'High School', 'major_discipline'] = 'not applicable'

    df.loc[df['education_level'] == 'Primary School', 'major_discipline'] = 'not applicable'

    df.loc[(df['experience']=='<1') & (df['company_type'].isna()) & (df['company_size'].isna()),'experience']=0

    df.loc[(df['experience']==0) & (df['company_type'].isna()) & (df['company_size'].isna()),'company_type']='Not Applicable'

    df.loc[(df['experience']==0) & (df['company_type']=='Not Applicable') & (df['company_size'].isna()),'company_size']='Not Applicable'

    return df
def impute_missing(df) : 

    missing_cols = list(df.columns[df.isnull().any()])

    print(f"Columns with missing values : {missing_cols}")

    df_org = df

    

    #### Impute Gender with unkniwn and add a _ismissing feature

    for col in missing_cols :

        if col!='target' :

            df[col+'_ismissing'] = 0

            df[col+'_ismissing'] = df[col].apply(lambda x : 1 if x!=x else 0) 

            df[col] = df[col].fillna('unknown')

        

    #df['gender'] = df['gender'].fillna('Unknown')

    return df
def cat_to_num(df) :

    df['company_size']=df['company_size'].apply(lambda x : 75 if x=='50-99' else x)

    df['company_size']=df['company_size'].apply(lambda x : 300 if x=='100-500' else x)

    df['company_size']=df['company_size'].apply(lambda x : 20000 if x=='10000+' else x)

    df['company_size']=df['company_size'].apply(lambda x : 30 if x=='10/49' else x)

    df['company_size']=df['company_size'].apply(lambda x : 3000 if x=='1000-4999' else x)

    df['company_size']=df['company_size'].apply(lambda x : 5 if x=='<10' else x)

    df['company_size']=df['company_size'].apply(lambda x : 750 if x=='500-999' else x)

    df['company_size']=df['company_size'].apply(lambda x : 7500 if x=='5000-9999' else x)

    df['company_size']=df['company_size'].apply(lambda x : -999 if x=='unknown' else x)

    df['company_size']=df['company_size'].apply(lambda x : -999 if x=='Not Applicable' else x)

    

    df['last_new_job'] = df['last_new_job'].apply(lambda x : 10 if x=='>4' else x)

    df['last_new_job'] = df['last_new_job'].apply(lambda x : 0 if x=='never' else x)

    df['last_new_job'] = df['last_new_job'].apply(lambda x : -999 if x=='unknown' else x)

    

    df['experience']=df['experience'].apply(lambda x : 0 if x=='<1' else x)

    df['experience']=df['experience'].apply(lambda x : 25 if x=='>20' else x)

    df['experience']=df['experience'].apply(lambda x : -999 if x=='unknown' else x)

    

    def ed_to_numeric(x):

        if x=='unknown' or x=='Primary School':

            return 0

        if x=='High School':

            return 1

        if x=='Graduate':

            return 2

        if x=='Masters':

            return 3

        if x=='Phd':

            return 4

    

    df['education_level'] = df['education_level'].apply(ed_to_numeric)

    

    df['last_new_job']=df['last_new_job'].astype(int)

    df['experience']=df['experience'].astype(int)

    #print(df['company_size'].value_counts())

    df['company_size']=df['company_size'].astype(int)



    

    return df
def feat_eng(df,cat) :

    df['experience_more_than20']=df['experience'].apply(lambda x : 1 if x>20 else 0)

    

    ##aggregate features

#     #cat_agg=['count','nunique']

#     num_agg=['min','mean','max','sum']

#     agg_col={

#         'experience':num_agg, 'company_size':num_agg, 'training_hours':num_agg,'last_new_job':num_agg}



#     agg_df=df.groupby('city').agg(agg_col)

#     agg_df.columns=['agg_' + '_'.join(col).strip() for col in agg_df.columns.values]

#     agg_df.reset_index(inplace=True)

    

#     df=df.merge(agg_df,on='city',how='left')

    

    ##ONE HOT ENCODING 

    if cat == 0:

        df=pd.get_dummies(df,columns=list(df.select_dtypes(include=['object']).columns),drop_first=True)

        cat_feat = 0

    else :

        cat_col = list(df.select_dtypes(include=['object']).columns)

        for c in cat_col :

            df[c] = df[c].astype('category')

        cat_feat = np.where(df.dtypes =='category')[0]



    return df,cat_feat
df_log = logic_missing(df)
df_imp = impute_missing(df_log)
df_num= cat_to_num(df_imp)
df_feat,cat_feat = feat_eng(df_num,0)
df_feat.info()
df = df_feat.drop(columns = ['enrollee_id'])

cat_feat = cat_feat-1
from imblearn.over_sampling import RandomOverSampler

os =RandomOverSampler(1)





df_train=df[df[target].isnull()==False].copy()

df_test=df[df[target].isnull()==True].copy()



df_test.drop(columns=[target],axis=1, inplace=True)



x = df_train.drop(target,axis=1)

y = df_train[target]

feat = df_test.columns



print(df_train.shape,df_test.shape)



x, y= os.fit_sample(x, y)
def make_sub(y_pred,name):

    df_sub = pd.DataFrame({'enrollee_id':test_ids,target:y_pred})

    import time

    times = time.strftime("%Y%m%d-%H%M%S")

    filename = 'submission-'+name+'_'+times+'.csv'

    df_sub.to_csv(filename,index=False)

    print(f"{filename} generated!")
def lgb_tune(x, y, target, plot=True):

    

    

    print("Parameter Tuning :")



    param_grid = {

    'num_leaves':[50],

    'max_depth':[-1],

    'colsample_bytree': [0.8],#,0.6,0.8],

    'min_child_samples': [10],#,10],

    'min_split_gain':[1],

    'subsample' : [0.7],

    'reg_alpha' : [0.7],#,0.6],

    'reg_lambda' : [0.6],#,0.7,0.8],

    'device': ['gpu']

    } 



    model = lgb.LGBMClassifier(

        objective='binary',

        boosting_type='gbdt', 

        learning_rate=0.003, 

        n_estimators=4000, 

        silent=False,

        #categorical_feature = cat_feat

    )

    

    skf = StratifiedKFold(n_splits=4, shuffle = True, random_state = 1001)

    

    lgb_grid = GridSearchCV(model, param_grid, cv=skf.split(x,y), scoring='roc_auc', verbose=1, n_jobs=4)

    lgb_grid.fit(x, y)



    print(lgb_grid.best_params_)

    print(lgb_grid.best_score_)



    #predictions = lgb_grid.predict(test[features]) 

    

    return lgb_grid.best_estimator_, #predictions, lgb_grid.best_params
def lgb_run (lgbM, cv) :

    err=[]

    y_pred_tot=[]

    from sklearn.model_selection import KFold,StratifiedKFold



    fold=StratifiedKFold(n_splits=cv,shuffle=True,random_state=1996)



    i=1



    for train_index, test_index in fold.split(x,y):

        print(f"\n\n-----------------FOLD {i}------------------------")

        x_train, x_test = x.iloc[train_index], x.iloc[test_index]

        y_train, y_test = y[train_index], y[test_index]

        m=lgbM

        m.fit(x_train,y_train,eval_set=[(x_test, y_test)],eval_metric='auc', early_stopping_rounds=200,verbose=100)#,categorical_feature = cat_feat)

        preds=m.predict_proba(x_test)[:,-1]

        print("err: ",roc_auc_score(y_test,preds.round()))

        err.append(roc_auc_score(y_test,preds.round()))

        p = m.predict_proba(df_test[feat])[:,-1]

        i=i+1

        y_pred_tot.append(p)

    print (f"Mean score : {np.mean(err,0)}")

    y_pred_lgb = np.mean(y_pred_tot, 0)

    return y_pred_lgb
lgbM = lgb_tune(x, y, target, True)
y_pred_lgb= lgb_run (lgbM[0], 10)
make_sub(y_pred_lgb,'lgb')
def xgb_tune(x, y,  target):

    

    print("Parameter Tuning :")



    param_grid = {

    'max_depth':[3,6], ##

    'subsample':[0.8],

    'colsample_bytree': [1],

    'min_child_weight': [0.4],

    'gamma': [0.5],

    'reg_lambda': [1],

    } 



    model = xgb.XGBClassifier(

        objective='binary:logistic',

        learning_rate=0.003, 

        n_estimators=4000, 

        tree_method = "gpu_hist",

        silent=False

    )



    skf = StratifiedKFold(n_splits=4, shuffle = True, random_state = 1001)



    xgb_grid = GridSearchCV(model, param_grid, cv=skf.split(x,y), scoring='roc_auc', verbose=1, n_jobs=4)

   

    xgb_grid.fit(x, y)



    print(xgb_grid.best_params_)

    print(xgb_grid.best_score_)





    return  xgb_grid.best_estimator_
def xgb_run (xgbM, cv) :

    err=[]

    y_pred_tot=[]

    from sklearn.model_selection import KFold,StratifiedKFold



    fold=StratifiedKFold(n_splits=cv,shuffle=True,random_state=1996)



    i=1



    for train_index, test_index in fold.split(x,y):



        x_train, x_test = x.iloc[train_index], x.iloc[test_index]

        y_train, y_test = y[train_index], y[test_index]

        m=xgbM

        m.fit(x_train,y_train,eval_set=[(x_test, y_test)],eval_metric='auc', early_stopping_rounds=200,verbose=100)

        preds=m.predict_proba(x_test)[:,-1]

        print("err: ",roc_auc_score(y_test,preds.round()))

        err.append(roc_auc_score(y_test,preds.round()))

        p = m.predict_proba(df_test[feat])[:,-1]

        i=i+1

        y_pred_tot.append(p)

    print (f"Mean score : {np.mean(err,0)}")

    y_pred_lgb = np.mean(y_pred_tot, 0)

    return y_pred_lgb
xgbM = xgb_tune(x, y, target)
y_pred_xg= xgb_run (xgbM, 10)
make_sub(y_pred_xg,'xgb')
make_sub(y_pred_xg*0.5+y_pred_lgb*0.5,'xgblgb')