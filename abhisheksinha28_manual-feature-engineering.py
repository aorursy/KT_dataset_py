# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import sys

def return_size(df):
    """Return size of dataframe in gigabytes"""
    return round(sys.getsizeof(df) / 1e9, 2)

def convert_types(df):
    print(f'Original size of data: {return_size(df)} gb.')
    for c in df:
        if df[c].dtype == 'object':
            df[c] = df[c].astype('category')
    print(f'New size of data: {return_size(df)} gb.')
    return df
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
train = pd.read_csv('../input/application_train.csv').replace({365243: np.nan})
test = pd.read_csv('../input/application_test.csv').replace({365243: np.nan})
bureau = pd.read_csv('../input/bureau.csv').replace({365243: np.nan})
bureau_balance = pd.read_csv('../input/bureau_balance.csv').replace({365243: np.nan})

test['TARGET'] = np.nan
data = train.append(test, ignore_index=True, sort=True)

data = convert_types(data)
bureau = convert_types(bureau)
bureau_balance = convert_types(bureau_balance)

import gc
gc.enable()
del train, test
gc.collect()
def kde_target(var_name,df):
    corr = df['TARGET'].corr(df[var_name])
    sns.kdeplot(df.ix[df['TARGET']==0, var_name], label = 'target ==0')
    sns.kdeplot(df.ix[df['TARGET']==1, var_name], label= 'target ==1')
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend();
 
def agg_numeric(df, group_var, df_name):
    
    
    
    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    _, idx = np.unique(agg,axis=1,return_index=True)
    agg= agg.iloc[:,idx]
    
    return agg
def target_corrs(df):
    corrs = []
    for col in df :
        print(col)
        if col!= 'TARGET':
            corr = df['TARGET'].corr(df[col])
            corrs.append((col,corr))
    corrs = sorted(corrs, key = lambda x: abs(x[1]), reverse = True)
    return corrs
def agg_categorical(df,group_var,df_name) :
    categorical = pd.get_dummies(df.select_dtypes('category'))
    categorical[group_var] = df[group_var]
    categorical = categorical.groupby(group_var).agg(['sum','count','mean'])
    
    column_names = []
    
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['sum','count','mean']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    _,idx = np.unique(categorical, axis=1, return_index=True)
    categorical = categorical.iloc[:,idx]
    return categorical
import gc

def agg_child(df,parent_var,df_name):
    df_agg = agg_numeric(df,parent_var,df_name)
    df_cat = agg_categorical(df,parent_var,df_name)
    df_combined = df_agg.merge(df_cat,on=parent_var, how='outer')
    
    _,idx = np.unique(df_combined,axis=1,return_index=True)
    df_combined = df_combined.iloc[:,idx]
    
    gc.enable()
    del df_agg,df_cat
    gc.collect()
    
    return df_combined
def agg_grandchild(df, parent_df,parent_var, grandparent_var, df_name):
    parent_df = parent_df[[parent_var,grandparent_var]].copy().set_index(parent_var)
    df_agg = agg_numeric(df,parent_var,'%s_LOAN' % df_name)
    df_agg = df_agg.merge(parent_df, on = parent_var, how = 'left')
    df_agg_client = agg_numeric(df_agg,grandparent_var,'%s_CLIENT' %df_name)
    if any(df.dtypes == 'category'):
    
        # Aggregate the categorical variables at the parent level
        df_agg_cat = agg_categorical(df, parent_var, '%s_LOAN' % df_name)
        df_agg_cat = df_agg_cat.merge(parent_df,
                                      on = parent_var, how = 'left')

        # Aggregate the categorical variables at the grandparent level
        df_agg_cat_client = agg_numeric(df_agg_cat, grandparent_var, '%s_CLIENT' % df_name)
        df_info = df_agg_client.merge(df_agg_cat_client, on = grandparent_var, how = 'outer')
        
        gc.enable()
        del df_agg, df_agg_client, df_agg_cat, df_agg_cat_client
        gc.collect()
        
    else:
        df_info =df_agg_client.copy()
        gc.enable()
        del df_agg_client,df_agg
        gc.collect()
        
    _, idx = np.unique(df_info, axis=1, return_index =True)
    df_info = df_info.iloc[:,idx]
    return df_info
            
data['LOAN_RATE'] = data['AMT_ANNUITY'] / data['AMT_CREDIT'] 
data['CREDIT_INCOME_RATIO'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
data['EMPLOYED_BIRTH_RATIO'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
data['EXT_SOURCE_SUM'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis = 1)
data['EXT_SOURCE_MEAN'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
data['AMT_REQ_SUM'] = data[[x for x in data.columns if 'AMT_REQ_' in x]].sum(axis = 1)
bureau['LOAN_RATE'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM']
bureau_info = agg_child(bureau, 'SK_ID_CURR', 'BUREAU')
bureau_info.head()
bureau_info.shape
bureau_balance['PAST_DUE'] = bureau_balance['STATUS'].isin(['1', '2', '3', '4', '5'])
bureau_balance['ON_TIME'] = bureau_balance['STATUS'] == '0'
bureau_balance_info = agg_grandchild(bureau_balance, bureau, 'SK_ID_BUREAU', 'SK_ID_CURR', 'BB')
del bureau_balance, bureau
bureau_balance_info.head()
bureau_balance_info.shape
data = data.set_index('SK_ID_CURR')
data = data.merge(bureau_info, on = 'SK_ID_CURR', how = 'left')
del bureau_info
data.shape
data = data.merge(bureau_balance_info, on = 'SK_ID_CURR', how = 'left')
del bureau_balance_info
data.shape
previous = pd.read_csv('../input/previous_application.csv').replace({365243: np.nan})
previous = convert_types(previous)
previous['LOAN_RATE'] = previous['AMT_ANNUITY'] / previous['AMT_CREDIT']
previous["AMT_DIFFERENCE"] = previous['AMT_CREDIT'] - previous['AMT_APPLICATION']
previous_info = agg_child(previous, 'SK_ID_CURR', 'PREVIOUS')
previous_info.shape
data = data.merge(previous_info, on = 'SK_ID_CURR', how = 'left')
del previous_info
data.shape
installments = pd.read_csv('../input/installments_payments.csv').replace({365243: np.nan})
installments = convert_types(installments)
installments['LATE'] = installments['DAYS_ENTRY_PAYMENT'] > installments['DAYS_INSTALMENT']
installments['LOW_PAYMENT'] = installments['AMT_PAYMENT'] < installments['AMT_INSTALMENT']
installments_info = agg_grandchild(installments,previous,'SK_ID_PREV', 'SK_ID_CURR', 'IN')
del installments
installments_info.shape
data = data.merge(installments_info, on = 'SK_ID_CURR', how = 'left')
del installments_info
data.shape
cash = pd.read_csv('../input/POS_CASH_balance.csv').replace({365243: np.nan})
cash = convert_types(cash)
cash['LATE_PAYMENT'] = cash['SK_DPD'] > 0.0
cash['INSTALLMENTS_PAID'] = cash['CNT_INSTALMENT'] - cash['CNT_INSTALMENT_FUTURE']
cash_info = agg_grandchild(cash, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'CASH')
del cash
cash_info.shape

data = data.merge(cash_info, on='SK_ID_CURR', how='left')
del cash_info
data.shape
credit = pd.read_csv('../input/credit_card_balance.csv').replace({365243: np.nan})
credit = convert_types(credit)
credit['OVER_LIMIT'] = credit['AMT_BALANCE'] > credit['AMT_CREDIT_LIMIT_ACTUAL']
credit['BALANCE_CLEARED'] = credit['AMT_BALANCE'] == 0.0
credit['LOW_PAYMENT'] = credit['AMT_PAYMENT_CURRENT'] < credit['AMT_INST_MIN_REGULARITY']
credit['LATE'] = credit['SK_DPD'] > 0.0
credit_info = agg_grandchild(credit, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'CC')
del credit, previous
credit_info.shape
gc.collect()
gc.enable()

data = data.merge(credit_info, on = 'SK_ID_CURR', how = 'left')
del credit_info

data.shape
print('After manual feature engineering, there are {} features.'.format(data.shape[1] - 2))
gc.enable()
gc.collect()
print(f'Final size of data {return_size(data)}')
#data.to_csv('clean_manual_features.csv', chunksize = 100)

data.reset_index(inplace = True)
train, test = data[data['TARGET'].notnull()].copy(), data[data['TARGET'].isnull()].copy()
gc.enable()
del data
gc.collect()

import lightgbm as lgb

params = {'is_unbalance': True, 
              'n_estimators': 2673, 
              'num_leaves': 77, 
              'learning_rate': 0.00764, 
              'min_child_samples': 460, 
              'boosting_type': 'gbdt', 
              'subsample_for_bin': 240000, 
              'reg_lambda': 0.20, 
              'reg_alpha': 0.88, 
              'subsample': 0.95, 
              'colsample_bytree': 0.7}
train_labels = np.array(train.pop('TARGET')).reshape((-1, ))

test_ids = list(test.pop('SK_ID_CURR'))
test = test.drop(columns = ['TARGET'])
train = train.drop(columns = ['SK_ID_CURR'])

print('Training shape: ', train.shape)
print('Testing shape: ', test.shape)
model = lgb.LGBMClassifier(**params)

model.fit(train, train_labels)
preds = model.predict_proba(test)[:, 1]
submission = pd.DataFrame({'SK_ID_CURR': test_ids,
                           'TARGET': preds})

submission['SK_ID_CURR'] = submission['SK_ID_CURR'].astype(int)
submission['TARGET'] = submission['TARGET'].astype(float)
submission.to_csv('submission_manual.csv', index = False)
features = list(train.columns)
fi = pd.DataFrame({'feature': features,
                   'importance': model.feature_importances_})
def plot_feature_importances(df, n = 15, threshold = None):
    """
    Plots n most important features. Also plots the cumulative importance if
    threshold is specified and prints the number of features needed to reach threshold cumulative importance.
    Intended for use with any tree-based feature importances. 
    
    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be "feature" and "importance"
    
    n : int, default = 15
        Number of most important features to plot
    
    threshold : float, default = None
        Threshold for cumulative importance plot. If not provided, no plot is made
        
    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column
    
    Note
    --------
        * Normalization in this case means sums to 1. 
        * Cumulative importance is calculated by summing features from most to least important
    
    """
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    
    # Bar plot of n most important features
    df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = 'blue', edgecolor = 'k', figsize = (12, 8),
                            legend = False)

    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 
    plt.title(f'Top {n} Most Important Features', size = 18)
    plt.gca().invert_yaxis()
    
    if threshold:
        # Cumulative importance plot
        plt.figure(figsize = (8, 6))
        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 
        plt.title('Cumulative Feature Importance', size = 18);
        
        # Number of features needed for threshold cumulative importance
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        
        # Add vertical line to plot
        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.2, linestyles = '--', colors = 'red')
        plt.show();
        
        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 100 * threshold))
    
    return df
norm_fi = plot_feature_importances(fi, 25)
norm_fi.head(25)
threshold = 0.9

# Absolute value correlation matrix
corr_matrix = train.corr().abs()
corr_matrix.head()



























































































