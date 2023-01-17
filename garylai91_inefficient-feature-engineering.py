import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import gc
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
def bureau_balance_preprocess():
    # Import bureau and bureau_balance
    bureau = pd.read_csv("../input/bureau.csv")
    balance = pd.read_csv("../input/bureau_balance.csv")
    
    # Search for categorical variables
    cat_cols = []
    for i in balance.columns.values:
        if balance[i].dtype == 'object':
            cat_cols.append(i)
    
    # Search for numerical variables
    num_cols = [col for col in balance.columns if col not in cat_cols][1:]
    
    # One hot encoding of balance categorical variables
    balance = pd.get_dummies(balance, columns = ['STATUS'])
    
    cat_cols = [c for c in balance.columns.tolist() if c not in [cat_cols, num_cols]][2:]
    
    # Aggregate all variables
    cat_aggs = {}
    num_aggs = {}
    for i in cat_cols:
        cat_aggs[i] = ['mean']
    for j in num_cols:
        num_aggs[j] = ['size']
    
    # Groupby SK_ID_BUREAU and aggregate
    balance = balance.groupby('SK_ID_BUREAU').agg({**num_aggs, **cat_aggs})
    
    # Rename columns into 'column_name' + aggregation
    balance.columns = [i[0] + "_" + i[1]for i in balance.columns.tolist()]
    
    # Complete balance preprocessing
    balance.reset_index(inplace = True)
    
    # Search for categorical variables
    cat_cols = []
    for i in bureau.columns.values:
        if bureau[i].dtype == 'object':
            cat_cols.append(i)
    
    # Search for numerical variables
    num_cols = [col for col in bureau.columns if col not in cat_cols]
    
    # One hot encoding of bureau categorical variables
    bureau = pd.get_dummies(bureau, columns = cat_cols)
    
    # Merge balance and bureau dataframes
    df = bureau.merge(balance, how='left', on='SK_ID_BUREAU')
    
    # Drop SK_ID_BUREAU as it is redundant now
    df.drop(['SK_ID_BUREAU'], axis = 1, inplace = True)
    
    # Aggregate variables
    aggregations = {}
    for i in df.columns.tolist()[1:]:
        aggregations[i] = ['min', 'max', 'mean', 'median', 'var', 'size']
    
    # Groupby SK_ID_CURR and aggregate accordingly
    df = df.groupby('SK_ID_CURR').agg(aggregations)
    
    # Rename column names
    df.columns = [i[0] + "_" + i[1]for i in df.columns.tolist()]
    
    # bureau and balance preprocessing complete
    df = df.reset_index()
    return df
def prev_application():
    df = pd.read_csv("../input/previous_application.csv")
    df.loc[df['AMT_CREDIT'] > 6000000, 'AMT_CREDIT'] = np.nan
    df.loc[df['SELLERPLACE_AREA'] > 3500000, 'SELLERPLACE_AREA'] = np.nan
    df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    df['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    df['prev missing'] = df.isnull().sum(axis = 1).values
    df['prev AMT_APPLICATION / AMT_CREDIT'] = df['AMT_APPLICATION'] / df['AMT_CREDIT']
    df['prev AMT_APPLICATION - AMT_CREDIT'] = df['AMT_APPLICATION'] - df['AMT_CREDIT']
    df['prev AMT_APPLICATION - AMT_GOODS_PRICE'] = df['AMT_APPLICATION'] - df['AMT_GOODS_PRICE']
    df['prev AMT_GOODS_PRICE - AMT_CREDIT'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    df['prev DAYS_FIRST_DRAWING - DAYS_FIRST_DUE'] = df['DAYS_FIRST_DRAWING'] - df['DAYS_FIRST_DUE']
    df['prev DAYS_TERMINATION less -500'] = (df['DAYS_TERMINATION'] < -500).astype(int)
    df.rename(columns = lambda x : 'PREV_' + x if x  not in ['SK_ID_PREV', 'SK_ID_CURR'] else x, inplace = True)
    cat_cols = []
    for i in df.columns.values:
        if df[i].dtype == 'object':
            cat_cols.append(i)
    num_cols = [col for col in df.columns[2:] if col not in cat_cols]
    df = pd.get_dummies(df, columns = cat_cols)
    new_cat_cols = [c for c in df.columns.tolist() if c not in [cat_cols, num_cols]][2:]
    cat_aggs = {}
    num_aggs = {}
    for i in new_cat_cols:
        cat_aggs[i] = ['mean']
    for j in num_cols:
        num_aggs[j] = ['min', 'max', 'mean', 'median', 'var', 'size']
    df.drop(['SK_ID_PREV'], axis = 1, inplace = True)
    df = df.groupby('SK_ID_CURR').agg({**num_aggs, **cat_aggs})
    df.columns = [i[0] + "_" + i[1]for i in df.columns.tolist()]
    return df.reset_index()
def posh_cash_balance():
    df = pd.read_csv("../input/POS_CASH_balance.csv")
    df.loc[df['CNT_INSTALMENT_FUTURE'] > 60, 'CNT_INSTALMENT_FUTURE'] = np.nan
    df.drop(['SK_ID_PREV'], axis = 1, inplace = True)
    df['pos CNT_INSTALMENT more CNT_INSTALMENT_FUTURE'] = \
                    (df['CNT_INSTALMENT'] > df['CNT_INSTALMENT_FUTURE']).astype(int)
    cat_cols = []
    for i in df.columns.values:
        if df[i].dtype == 'object':
            cat_cols.append(i)
    num_cols = [c for c in df.columns.tolist() if c not in cat_cols][1:]
    df = pd.get_dummies(df, columns = ['NAME_CONTRACT_STATUS'])
    cat_cols = [c for c in df.columns.tolist() if c not in [cat_cols, num_cols]][1:]
    cat_aggs = {}
    num_aggs = {}
    for i in cat_cols:
        cat_aggs[i] = ['mean']
    for j in num_cols:
        num_aggs[j] = ['min', 'max', 'mean', 'median', 'var', 'size']
    df = df.groupby('SK_ID_CURR').agg({**num_aggs, **cat_aggs})
    df.columns = [i[0] + "_" + i[1]for i in df.columns.tolist()]
    df.reset_index(inplace = True)
    df['CNT_INSTALMENT_RATIO'] = df['CNT_INSTALMENT_mean'] / df['MONTHS_BALANCE_mean']
    df['CNT_INSTALMENT_FUTURE_RATIO'] = df['CNT_INSTALMENT_FUTURE_mean'] / df['MONTHS_BALANCE_mean']
    df['CNT_INSTALMENT_SUM_RATIO'] = (df['CNT_INSTALMENT_mean'] + df['CNT_INSTALMENT_FUTURE_mean']) / df['MONTHS_BALANCE_mean']
    return df
def cc_balance():
    df = pd.read_csv("../input/credit_card_balance.csv")
    df.loc[df['AMT_PAYMENT_CURRENT'] > 4000000, 'AMT_PAYMENT_CURRENT'] = np.nan
    df.loc[df['AMT_CREDIT_LIMIT_ACTUAL'] > 1000000, 'AMT_CREDIT_LIMIT_ACTUAL'] = np.nan
    df['card missing'] = df.isnull().sum(axis = 1).values
    df['card SK_DPD - MONTHS_BALANCE'] = df['SK_DPD'] - df['MONTHS_BALANCE']
    df['card SK_DPD_DEF - MONTHS_BALANCE'] = df['SK_DPD_DEF'] - df['MONTHS_BALANCE']
    df['card SK_DPD - SK_DPD_DEF'] = df['SK_DPD'] - df['SK_DPD_DEF']
    
    df['card AMT_TOTAL_RECEIVABLE - AMT_RECIVABLE'] = df['AMT_TOTAL_RECEIVABLE'] - df['AMT_RECIVABLE']
    df['card AMT_TOTAL_RECEIVABLE - AMT_RECEIVABLE_PRINCIPAL'] = df['AMT_TOTAL_RECEIVABLE'] - df['AMT_RECEIVABLE_PRINCIPAL']
    df['card AMT_RECIVABLE - AMT_RECEIVABLE_PRINCIPAL'] = df['AMT_RECIVABLE'] - df['AMT_RECEIVABLE_PRINCIPAL']

    df['card AMT_BALANCE - AMT_RECIVABLE'] = df['AMT_BALANCE'] - df['AMT_RECIVABLE']
    df['card AMT_BALANCE - AMT_RECEIVABLE_PRINCIPAL'] = df['AMT_BALANCE'] - df['AMT_RECEIVABLE_PRINCIPAL']
    df['card AMT_BALANCE - AMT_TOTAL_RECEIVABLE'] = df['AMT_BALANCE'] - df['AMT_TOTAL_RECEIVABLE']

    df['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_ATM_CURRENT'] = df['AMT_DRAWINGS_CURRENT'] - df['AMT_DRAWINGS_ATM_CURRENT']
    df['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_OTHER_CURRENT'] = df['AMT_DRAWINGS_CURRENT'] - df['AMT_DRAWINGS_OTHER_CURRENT']
    df['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_POS_CURRENT'] = df['AMT_DRAWINGS_CURRENT'] - df['AMT_DRAWINGS_POS_CURRENT']
    
    df.rename(columns = lambda x : 'PREV_' + x if x  not in ['SK_ID_PREV', 'SK_ID_CURR'] else x, inplace = True)
    cat_cols = []
    for i in df.columns.values:
        if df[i].dtype == 'object':
            cat_cols.append(i)
    num_cols = [col for col in df.columns[2:] if col not in cat_cols]
    df = pd.get_dummies(df, columns = cat_cols)
    new_cat_cols = [c for c in df.columns.tolist() if c not in [cat_cols, num_cols]][2:]
    cat_aggs = {}
    num_aggs = {}
    for i in new_cat_cols:
        cat_aggs[i] = ['mean']
    for j in num_cols:
        num_aggs[j] = ['min', 'max', 'mean', 'median', 'var', 'size']
    df.drop(['SK_ID_PREV'], axis = 1, inplace = True)
    df = df.groupby('SK_ID_CURR').agg({**num_aggs, **cat_aggs})
    df.columns = [i[0] + "_" + i[1]for i in df.columns.tolist()]
    return df.reset_index()
def installments():
    df = pd.read_csv("../input/installments_payments.csv")
    df.loc[df['NUM_INSTALMENT_VERSION'] > 70, 'NUM_INSTALMENT_VERSION'] = np.nan
    df.loc[df['DAYS_ENTRY_PAYMENT'] < -4000, 'DAYS_ENTRY_PAYMENT'] = np.nan
    df['ins DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
    df['ins NUM_INSTALMENT_NUMBER_100'] = (df['NUM_INSTALMENT_NUMBER'] == 100).astype(int)
    df['ins DAYS_INSTALMENT more NUM_INSTALMENT_NUMBER'] = (df['DAYS_INSTALMENT'] > df['NUM_INSTALMENT_NUMBER'] * 50 / 3 - 11500 / 3).astype(int)
    df['ins AMT_INSTALMENT - AMT_PAYMENT'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
    df['ins AMT_PAYMENT / AMT_INSTALMENT'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT']
    df.drop([])
    df.rename(columns = lambda x : 'PREV_' + x if x  not in ['SK_ID_PREV', 'SK_ID_CURR'] else x, inplace = True)
    cat_cols = []
    for i in df.columns.values:
        if df[i].dtype == 'object':
            cat_cols.append(i)
    num_cols = [col for col in df.columns[2:] if col not in cat_cols]
    df = pd.get_dummies(df, columns = cat_cols)
    new_cat_cols = [c for c in df.columns.tolist() if c not in [cat_cols, num_cols]][2:]
    cat_aggs = {}
    num_aggs = {}
    for i in new_cat_cols:
        cat_aggs[i] = ['mean']
    for j in num_cols:
        num_aggs[j] = ['min', 'max', 'mean', 'median', 'var', 'size']
    df.drop(['SK_ID_PREV'], axis = 1, inplace = True)
    df = df.groupby('SK_ID_CURR').agg({**num_aggs, **cat_aggs})
    df.columns = [i[0] + "_" + i[1]for i in df.columns.tolist()]
    return df.reset_index()
def get_beautiful_data():
    print('Getting train data...')
    train = pd.read_csv("../input/application_train.csv")
    
    print('Getting test data...')
    test = pd.read_csv("../input/application_test.csv")
    y = train.pop('TARGET')
    index_end = y.shape[0]
    test_index = test.pop('SK_ID_CURR')
    cat_cols = []
    for i in train.columns.values:
        if train[i].dtype == 'object':
            cat_cols.append(i)
    num_cols = [col for col in train.columns if col not in cat_cols]
    print('Merging train and tests data...')
    df = pd.concat([train, test], axis = 0, ignore_index = True)
    del train, test
    gc.collect()
    print('Merge complete.')
    
    print('Fetching bureau and balance data...')
    bureau = bureau_balance_preprocess()
    print('Merging to main data frame...')
    df = df.merge(bureau, how = 'left', on = 'SK_ID_CURR')
    del bureau
    gc.collect()
    print('Merge complete.')
    
    print('Fetching previous_application data...')
    prev = prev_application()
    print('Merging to main data frame...')
    df = df.merge(prev, how = 'left', on = 'SK_ID_CURR')
    del prev
    gc.collect()
    print('Merge complete.')
    
    print('Fetching POS_CASH_balance data...')
    pos = posh_cash_balance()
    print('Merging to main data frame...')
    df = df.merge(pos, how = 'left', on = 'SK_ID_CURR')
    del pos
    gc.collect()
    print('Merge complete.')
    
    print('Fetching credit_card_balance data...')
    cc = cc_balance()
    print('Merging to main data frame...')
    df = df.merge(cc, how = 'left', on = 'SK_ID_CURR')
    del cc
    gc.collect()
    print('Merge complete.')
    
    print('Fetching installments_payments data...')
    inst = installments()
    print('Merging to main data frame...')
    df = df.merge(inst, how = 'left', on = 'SK_ID_CURR')
    del inst
    gc.collect()
    print('Merge complete')
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    print('Adding more features...')
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']   
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    print('Done.')

    df = pd.get_dummies(df, columns = cat_cols, dummy_na= True)
        
    X = df.iloc[:index_end, 1:]
    X_test = df.iloc[index_end:, 1:]
    scale_pos_weight = y.value_counts()[0]/ y.value_counts()[1]
    
    X, X_test = X.align(X_test, join = 'inner', axis = 1)

    return X, X_test, y, scale_pos_weight, test_index
X, X_test, y, scale_pos_weight, test_index = get_beautiful_data()
# Remove Collinear Variables
# Threshold for removing correlated variables
threshold = 0.9

# Absolute value correlation matrix
corr_matrix = X.corr().abs()
corr_matrix.head()
# Upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head()
# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))
X = X.drop(columns = to_drop)
X_test = X_test.drop(columns = to_drop)

print('Training shape: ', X.shape)
print('Testing shape: ', X_test.shape)
# Train missing values (in percent)
train_missing = (X.isnull().sum() / len(X)).sort_values(ascending = False)
train_missing.head()
# Test missing values (in percent)
test_missing = (X_test.isnull().sum() / len(X_test)).sort_values(ascending = False)
test_missing.head()
# Identify missing values above threshold
train_missing = train_missing.index[train_missing > 0.75]
test_missing = test_missing.index[test_missing > 0.75]

all_missing = list(set(set(train_missing) | set(test_missing)))
print('There are %d columns with more than 75%% missing values' % len(all_missing))
X = X.drop(columns = all_missing)
X_test = X_test.drop(columns = all_missing)

print('Training shape: ', X.shape)
print('Testing shape: ', X_test.shape)
clf = xgb.XGBClassifier(learning_rate = 0.025, 
                        n_estimators = 10000, 
                        max_depth = 8, 
                        min_child_weight = 1, 
                        subsample = 0.8, 
                        colsample_bytree = 0.7, 
                        colsample_bylevel = 0.7,
                        objective = 'binary:logistic', 
                        n_jobs = -1,
                        scale_pos_weight = scale_pos_weight,
                        silent = True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)
clf.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_val, y_val)], eval_metric= 'auc', verbose= 100, early_stopping_rounds= 400)
result = clf.predict_proba(X_test)
submit = pd.DataFrame({'SK_ID_CURR': test_index, 'TARGET': result[:, 1]})
submit.to_csv('solution.csv', index = False)
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(clf, max_num_features=57, height=0.8, ax=ax)
ax.grid(False)
plt.title("XGBoost - Feature Importance", fontsize=15)
plt.show()
plt.savefig('feature_importance.png')
