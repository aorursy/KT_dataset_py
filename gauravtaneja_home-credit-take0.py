import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from functools import reduce
from matplotlib import pyplot
import os
import gc
from sklearn.ensemble import ExtraTreesClassifier
PATH = "../input"

# Preview of Categorical variables in a dataset

def data_preview_str(df, name):
    tot_feats = 0
    if name=='application_train':
        for col in [i for i in df.columns if (df[i].dtypes=='O') or (len(df[i].unique())<=10)]:
            tot_feats = tot_feats + len(df[col].unique())-1
            print(" %s "% col, '# unique: ',len(df[col].unique()))
            xdf = df.groupby(col).agg({'SK_ID_CURR':'count','TARGET':'sum'})
            xdf.columns = ['#Records','Response_Rate']
            xdf['Response_Rate'] = xdf['Response_Rate']/sum(xdf['Response_Rate'])
            xdf['%Records'] = xdf['#Records']*100/sum(xdf['#Records'])
            print(xdf.loc[:,['#Records','%Records','Response_Rate']])
            del xdf
            print("-"*50)        
    else:
        for col in [i for i in df.columns if (df[i].dtypes=='O') or (len(df[i].unique())<=15)]:
            tot_feats = tot_feats + len(df[col].unique())-1
            print(" %s "% col, '# unique: ',len(df[col].unique()))
            xdf = pd.DataFrame(df[col].value_counts())
            xdf['%Records'] = xdf[col]*100/sum(xdf[col])
            xdf.columns = ['#Records','%Records']
            print(xdf)
            del xdf
            print("-"*50)
    ll=[i for i in df.columns if (df[i].dtypes!='O') and (len(df[i].unique())>15)]
    tot_feats= tot_feats + len(ll)
    print("Creating dummies will lead to a total of %d features"% tot_feats)
    return 1

# MIssing data info
def missing_data(data, size):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    mdata = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    mdata = mdata.loc[mdata['Percent']>0,:].reset_index()
    if mdata.shape[0]==0:
        print("No missing data!")
        return 1
    else:
        sns.set_style("whitegrid")
        fig, ax =pyplot.subplots(figsize=size)
        ax = sns.barplot(x="Percent", y="index", data=mdata)
        return ax

# Create flags for Missing info
def missing_info_flag(df):
    mia_col_list = [i for i in df.columns if df[i].isnull().values.any().any()]
    counter = 0
    for col in mia_col_list:
        df[col+'_mia_'] = np.where(df[col].isnull(),1,0)
        counter +=1
    print("Created %d missing flags"% counter)
    
# Create dummy variables
def create_dummy_vars(df):
    dummy_varlist = [i for i in df.columns if df[i].dtypes=='O']
    dummy_df = pd.get_dummies(df, columns = dummy_varlist)
    return dummy_df
#     return pd.merge(df[[i for i in df.columns if df[i].dtypes!='O']],dummy_df, how='left', left_index=True, right_index = True)

# Pairplot for distribution of contnuous variables
def xdistributions(df, cols, kde):
    ncols = [i for i in cols if df[i].dtypes!='O']
    df.loc[:,'TARGET_char'] = np.where(df['TARGET']==0,'Clean','Default')
    if kde:
        ax = sns.pairplot(df.loc[:,ncols+['TARGET_char']], hue='TARGET_char', diag_kind='kde', diag_kws=dict(shade=True), plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    else:
        ax = sns.pairplot(df.loc[:,ncols+['TARGET_char']], hue='TARGET_char', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    df.drop('TARGET_char', axis=1, inplace=True)
    return ax

def prep_data(df):
    print("Before creating dummy variables #Columns: %d"% df.shape[1])
    p_df = create_dummy_vars(df)
    print("After creating dummy variables #Columns: %d"% p_df.shape[1])
    
    missing_info_flag(p_df)
    
    p_df = p_df.fillna(p_df.median())
    return p_df

# THis function calculates mean min max median for continuous columns and mena for dummy column sat SK_ID_CURR level
def agg_at_skid(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    flags = [ i for i in df.columns if (df[i].dtypes!='O') and (len(df[i].unique())<=2)]
    non_flags = [ i for i in df.columns if (df[i].dtypes!='O') and (len(df[i].unique())>2)]
    print("# unique SK_ID_CURR : ",len(df['SK_ID_CURR'].unique()))
    df_at_skid = pd.merge(df.loc[:,non_flags].groupby('SK_ID_CURR').agg(['min','max','mean','median']),
                              df.loc[:,flags+['SK_ID_CURR']].groupby('SK_ID_CURR').agg(['mean']),
                              how='inner', left_index=True, right_index=True)
    print("After merging shape : ",df_at_skid.shape)
    df_at_skid = pd.DataFrame(df_at_skid)
    df_at_skid.columns = ['_'.join(col) for col in df_at_skid.columns]
    df_at_skid.reset_index(inplace=True)
    df_at_skid.sort_values(by='SK_ID_CURR', ascending = [1])
    df_at_skid.reset_index(inplace=True, drop=True)
    return df_at_skid

def lgbm(data, test, y, **kwargs):
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import KFold
    from lightgbm import LGBMClassifier
    import gc
    # Get features
    excluded_feats = ['SK_ID_CURR','TARGET']
    features = [f_ for f_ in data.columns if f_ not in excluded_feats]

    # Run a 3 fold
    folds = KFold(n_splits=3, shuffle=True, random_state=123451)
    oof_preds = np.zeros(data.shape[0])
    sub_preds = np.zeros(test.shape[0])
    feature_importance_df = pd.DataFrame()
    print(kwargs)
    if kwargs !={}:
        for key, value in kwargs.items():
            print("The value of {} is {}".format(key, value))
    else:
        n_estimators=1500
        learning_rate = 0.01
        num_leaves = 50
        max_depth = 6 
        colsample_bytree = .8
        subsample =.9
        max_depth =6
        reg_alpha =.1
        reg_lambda =.1
        min_split_gain =.01
        min_child_weight =2
        silent =True
        verbose =-1
    
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data)):
        trn_x, trn_y = data[features].iloc[trn_idx], y.iloc[trn_idx]
        val_x, val_y = data[features].iloc[val_idx], y.iloc[val_idx]
        
        clf = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            max_depth=max_depth,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            silent=silent,
            verbose=verbose,
            early_stopping_rounds=10
        )

        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='auc', verbose=100, early_stopping_rounds=150
               )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))   
    return sub_preds, feature_importance_df
gc.collect()
application_train = pd.read_csv(PATH+"/application_train.csv")
print('Size of application_train '+'{:,}'.format(application_train.shape[0])+' rows and '+'{:,}'.format(application_train.shape[1])+' columns')
missing_data(application_train, size=(15,15))
app_tr_cont_cols = [i for i in application_train.columns if (application_train[i].dtypes!='O') and (len(application_train[i].unique())>10)]
application_train.loc[:,app_tr_cont_cols].describe()
data_preview_str(application_train,'application_train')
missing_info_flag(application_train)
application_train = prep_data(application_train)

# bureau
bureau = pd.read_csv(PATH+"/bureau.csv")
cols_ = [i for i in bureau.columns if (bureau[i].dtypes!='O') and (len(bureau[i].unique())>10)]
bureau.loc[:,cols_].describe()
missing_data(bureau,(5,5))
data_preview_str(bureau,'bureau_balance')
bureau = prep_data(bureau)
bureau_at_skid = agg_at_skid(bureau)
del bureau
import gc
gc.collect()
POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv")
print('POS_CASH_balance '+'{:,}'.format(POS_CASH_balance.shape[0])+' rows and '+'{:,}'.format(POS_CASH_balance.shape[1])+' columns')
cols_ = [i for i in POS_CASH_balance.columns if (POS_CASH_balance[i].dtypes!='O') and (len(POS_CASH_balance[i].unique())>10)]
POS_CASH_balance.loc[:,cols_].describe()
missing_data(POS_CASH_balance,(5,5))
data_preview_str(POS_CASH_balance,'POS_CASH_balance')
POS_CASH_balance = prep_data(POS_CASH_balance)
POS_CASH_balance_at_skid = agg_at_skid(POS_CASH_balance)
del POS_CASH_balance
gc.collect()
credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv")
print('credit_card_balance '+'{:,}'.format(credit_card_balance.shape[0])+' rows and '+'{:,}'.format(credit_card_balance.shape[1])+' columns')
cols_ = [i for i in credit_card_balance.columns if (credit_card_balance[i].dtypes!='O') and (len(credit_card_balance[i].unique())>10)]
credit_card_balance.loc[:,cols_].describe()
missing_data(credit_card_balance,(5,5))
data_preview_str(credit_card_balance,'credit_card_balance')
credit_card_balance = prep_data(credit_card_balance)
credit_card_balance_at_skid = agg_at_skid(credit_card_balance)
del credit_card_balance
gc.collect()
bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv")
print('bureau_balance '+'{:,}'.format(bureau_balance.shape[0])+' rows and '+'{:,}'.format(bureau_balance.shape[1])+' columns')
cols_ = [i for i in bureau_balance.columns if (bureau_balance[i].dtypes!='O') and (len(bureau_balance[i].unique())>10)]
bureau_balance.loc[:,cols_].describe()
missing_data(bureau_balance,(5,5))
data_preview_str(bureau_balance,'bureau_balance')
gc.collect()
# bureau_balance = prep_data(bureau_balance)
# bureau_balance_at_skid = agg_at_skid(bureau_balance)
del bureau_balance
gc.collect()
# previous_application
previous_application = pd.read_csv(PATH+"/previous_application.csv")
print('previous_application '+'{:,}'.format(previous_application.shape[0])+' rows and '+'{:,}'.format(previous_application.shape[1])+' columns')
cols_ = [i for i in previous_application.columns if (previous_application[i].dtypes!='O') and (len(previous_application[i].unique())>10)]
previous_application.loc[:,cols_].describe()
missing_data(previous_application,(5,5))
data_preview_str(previous_application,'previous_application')
# previous_application = prep_data(previous_application)
# previous_application_at_skid = agg_at_skid(previous_application)
del previous_application
gc.collect()
# installments_payments
installments_payments = pd.read_csv(PATH+"/installments_payments.csv")
print('installments_payments '+'{:,}'.format(installments_payments.shape[0])+' rows and '+'{:,}'.format(installments_payments.shape[1])+' columns')
cols_ = [i for i in installments_payments.columns if (installments_payments[i].dtypes!='O') and (len(installments_payments[i].unique())>10)]
installments_payments.loc[:,cols_].describe()
missing_data(installments_payments,(10,10))
data_preview_str(installments_payments,'installments_payments')
installments_payments = prep_data(installments_payments)
installments_payments_at_skid = agg_at_skid(installments_payments)
del installments_payments
gc.collect()
df_list = [application_train, bureau_at_skid, POS_CASH_balance_at_skid, installments_payments_at_skid, credit_card_balance_at_skid] #, bureau_balance_at_skid previous_application_at_skid
print("Before merging: ",application_train.shape)
X_train = reduce(lambda left,right: pd.merge(left,right,how='left',on='SK_ID_CURR'), df_list)
print("After merging: ",X_train.shape)
missing_data(X_train,(50,50))
del application_train
gc.collect()
application_test = pd.read_csv(PATH+"/application_test.csv")
missing_info_flag(application_test)
application_test = prep_data(application_test)
X_train.columns.values
X_train.drop([ 'SK_ID_PREV_min', 'SK_ID_PREV_max', 'SK_ID_PREV_mean','SK_ID_PREV_median'], axis=1, inplace=True)
df_list = [application_test, bureau_at_skid, POS_CASH_balance_at_skid, installments_payments_at_skid, credit_card_balance_at_skid]  #, bureau_balance_at_skid previous_application_at_skid
print("Before merging: ",application_test.shape)
X_test = reduce(lambda left,right: pd.merge(left,right,how='left',on='SK_ID_CURR'), df_list)
print("After merging: ",X_test.shape)
missing_data(X_test,(50,150))
X_test.drop(['SK_ID_PREV_min', 'SK_ID_PREV_max', 'SK_ID_PREV_mean','SK_ID_PREV_median'], axis=1, inplace=True)
del application_test, bureau_at_skid, POS_CASH_balance_at_skid, credit_card_balance_at_skid, installments_payments_at_skid #previous_application_at_skid
gc.collect()
# Dropping some ID aggregated variables
# dropvars = ['SK_ID_PREV_min_y', 'SK_ID_PREV_max_y', 'SK_ID_PREV_mean_y', 'SK_ID_PREV_median_y','SK_ID_BUREAU_min', 'SK_ID_BUREAU_max', 'SK_ID_BUREAU_mean', 'SK_ID_BUREAU_median']
# X_train.drop(dropvars, axis=1, inplace=True)
# X_test.drop(dropvars, axis=1, inplace=True)
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())
gc.collect()
# Remove variables where the value is same for more than 99% pf training sets
def variance_threshold_selector(data, threshold):
    from sklearn.feature_selection import VarianceThreshold
    print("# features before Variance Threshold: ", data.shape[1])
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

X_test_sel = variance_threshold_selector(data=X_test, threshold=0.99)
print("# features after Variance Threshold: ", X_test_sel.shape[1])
X_train_sel = variance_threshold_selector(data=X_train, threshold=0.99)
print("# features after Variance Threshold: ", X_train_sel.shape[1])
y = X_train['TARGET']
gc.collect()
del X_test, X_train
gc.collect()
preds, feature_importance_df = lgbm(data=X_train_sel.loc[:,[i for i in X_train_sel.columns if i in X_test_sel.columns]], test=X_test_sel, y=y)
X_train_sel.drop([i for i in X_train_sel.columns if i not in X_test_sel.columns], axis=1, inplace=True)
gc.collect()
X_test_sel['TARGET'] = preds

X_test_sel[['SK_ID_CURR', 'TARGET']].to_csv('take0_submission.csv', index=False, float_format='%.8f')
# Plot feature importances
import matplotlib.pyplot as plt
cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:50].index

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(8,10))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
# del X_test
# gc.collect()
