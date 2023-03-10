import gc
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('../input/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('../input/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_AVG'] = df[docs].mean(axis=1)
    df['NEW_DOC_IND_STD'] = df[docs].std(axis=1)
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_LIVE_IND_STD'] = df[live].std(axis=1)
    df['NEW_LIVE_IND_KURT'] = df[live].kurtosis(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_EMPLOY_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('../input/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('../input/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    cols = active_agg.columns.tolist()
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    
    for e in cols:
        bureau_agg['NEW_RATIO_BURO_' + e[0] + "_" + e[1].upper()] = bureau_agg['ACTIVE_' + e[0] + "_" + e[1].upper()] / bureau_agg['CLOSED_' + e[0] + "_" + e[1].upper()]
    
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('../input/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    cols = approved_agg.columns.tolist()
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    
    for e in cols:
        prev_agg['NEW_RATIO_PREV_' + e[0] + "_" + e[1].upper()] = prev_agg['APPROVED_' + e[0] + "_" + e[1].upper()] / prev_agg['REFUSED_' + e[0] + "_" + e[1].upper()]
    
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('../input/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('../input/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('../input/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

debug = None
num_rows = 10000 if debug else None
df = application_train_test(num_rows)
with timer("Process bureau and bureau_balance"):
    bureau = bureau_and_balance(num_rows)
    print("Bureau df shape:", bureau.shape)
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    del bureau
    gc.collect()
with timer("Process previous_applications"):
    prev = previous_applications(num_rows)
    print("Previous applications df shape:", prev.shape)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    del prev
    gc.collect()
with timer("Process POS-CASH balance"):
    pos = pos_cash(num_rows)
    print("Pos-cash balance df shape:", pos.shape)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    del pos
    gc.collect()
with timer("Process installments payments"):
    ins = installments_payments(num_rows)
    print("Installments payments df shape:", ins.shape)
    df = df.join(ins, how='left', on='SK_ID_CURR')
    del ins
    gc.collect()
with timer("Process credit card balance"):
    cc = credit_card_balance(num_rows)
    print("Credit card balance df shape:", cc.shape)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    del cc
    gc.collect()
feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
for c in feats:
    ss = StandardScaler()
    df.loc[~np.isfinite(df[c]),c] = np.nan
    df.loc[~df[c].isnull(),c] = ss.fit_transform(df.loc[~df[c].isnull(),c].values.reshape(-1,1))
    df[c].fillna(-99999.,inplace=True)
train_df = df[df['TARGET'].notnull()]
test_df = df[df['TARGET'].isnull()]
train_df.columns = train_df.columns.str.replace('[^A-Za-z0-9_]', '_')
test_df.columns = test_df.columns.str.replace('[^A-Za-z0-9_]', '_')
feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
def Output(p):
    return 1./(1.+np.exp(-p))

def GP1(data):
    v = pd.DataFrame()
    v["i0"] = 0.010000*np.tanh((((((((-1.0*((((((((((data["CODE_GENDER"]) > (data["NEW_EXT_SOURCES_MEAN"]))*1.)) > ((-1.0*((data["CLOSED_AMT_CREDIT_SUM_SUM"])))))*1.)) + (((data["NEW_EXT_SOURCES_MEAN"]) * (3.141593)))))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i1"] = 0.034000*np.tanh((((((((((((-1.0*((data["EXT_SOURCE_2"])))) - (np.tanh((((data["EXT_SOURCE_3"]) * 2.0)))))) * 2.0)) * 2.0)) - (np.where(data["EXT_SOURCE_2"]<0, 1.570796, data["EXT_SOURCE_2"] )))) * 2.0)) 
    v["i2"] = 0.015000*np.tanh(((((((((np.where(data["NEW_SOURCES_PROD"]>0, data["NEW_EXT_SOURCES_MEAN"], data["NEW_CREDIT_TO_GOODS_RATIO"] )) - (((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)) - (np.tanh((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]))))))) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)) * 2.0)) 
    v["i3"] = 0.025000*np.tanh(((((np.maximum(((np.maximum(((-3.0)), ((data["NEW_SOURCES_PROD"]))))), ((data["EXT_SOURCE_3"])))) * (((-3.0) * 2.0)))) + (((data["EXT_SOURCE_2"]) * (np.minimum(((-3.0)), ((data["EXT_SOURCE_3"])))))))) 
    v["i4"] = 0.047000*np.tanh(((((((((data["NEW_EXT_SOURCES_MEAN"]) * (np.where(data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"]>0, -2.0, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"] )))) * 2.0)) * 2.0)) + (np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"]>0, (-1.0*((data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"]))), data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"] )))) 
    v["i5"] = 0.046000*np.tanh((((-1.0*((((data["NEW_EXT_SOURCES_MEAN"]) + (((data["NEW_EXT_SOURCES_MEAN"]) + (((((data["NEW_EXT_SOURCES_MEAN"]) + ((((data["EXT_SOURCE_3"]) > (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))*1.)))) + ((-1.0*((data["NEW_CREDIT_TO_GOODS_RATIO"]))))))))))))) * 2.0)) 
    v["i6"] = 0.010024*np.tanh(((((((((((((data["NEW_CREDIT_TO_GOODS_RATIO"]) - (np.tanh((data["APPROVED_APP_CREDIT_PERC_MAX"]))))) - (((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)))) - ((((data["NEW_EXT_SOURCES_MEAN"]) > (data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))*1.)))) * 2.0)) * 2.0)) * 2.0)) 
    v["i7"] = 0.029998*np.tanh((((((((((((((-1.0*((data["NEW_EXT_SOURCES_MEAN"])))) * 2.0)) - ((((((data["PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN"]) * 2.0)) < (data["INSTAL_AMT_PAYMENT_MEAN"]))*1.)))) * 2.0)) * 2.0)) - ((-1.0*((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"])))))) * 2.0)) 
    v["i8"] = 0.040000*np.tanh(((((((((((np.tanh((((((data["DAYS_EMPLOYED"]) + (data["NEW_CREDIT_TO_GOODS_RATIO"]))) - (data["BURO_CREDIT_ACTIVE_Closed_MEAN"]))))) + (np.tanh((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))))) - (data["EXT_SOURCE_2"]))) * 2.0)) * 2.0)) * 2.0)) 
    v["i9"] = 0.041000*np.tanh(((((((((((data["NEW_CREDIT_TO_GOODS_RATIO"]) + ((-1.0*((((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)) + (data["CODE_GENDER"])))))))) - (np.maximum(((data["EXT_SOURCE_3"])), ((data["NAME_EDUCATION_TYPE_Higher_education"])))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i10"] = 0.035006*np.tanh(((((((np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"])), (((((((-1.0*((((data["NEW_EXT_SOURCES_MEAN"]) + ((((data["NEW_EXT_SOURCES_MEAN"]) > (data["DAYS_EMPLOYED"]))*1.))))))) * 2.0)) + (data["NEW_CREDIT_TO_GOODS_RATIO"])))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i11"] = 0.043600*np.tanh(((((((((((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) + (np.where(data["NEW_EXT_SOURCES_MEAN"] > -1, data["DAYS_EMPLOYED"], 3.0 )))) - (((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)))) * 2.0)) - (data["CODE_GENDER"]))) * 2.0)) * 2.0)) 
    v["i12"] = 0.012760*np.tanh(((((((((((np.tanh((((data["DAYS_EMPLOYED"]) - (np.maximum(((data["PREV_APP_CREDIT_PERC_MEAN"])), ((((data["NEW_EXT_SOURCES_MEAN"]) - (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"])))))))))) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i13"] = 0.041000*np.tanh((((((-1.0*((((((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)) + ((((((data["INSTAL_AMT_PAYMENT_MIN"]) * 2.0)) > (data["DAYS_EMPLOYED"]))*1.)))) + ((((data["ORGANIZATION_TYPE_Bank"]) > (data["APPROVED_DAYS_DECISION_MIN"]))*1.))))))) * 2.0)) * 2.0)) 
    v["i14"] = 0.046700*np.tanh(((((data["NEW_CREDIT_TO_GOODS_RATIO"]) + (((((np.tanh((data["PREV_NAME_PRODUCT_TYPE_walk_in_MEAN"]))) + (((np.maximum((((-1.0*((data["NEW_EXT_SOURCES_MEAN"]))))), ((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])))) + (np.tanh((data["DAYS_EMPLOYED"]))))))) * 2.0)))) * 2.0)) 
    v["i15"] = 0.000015*np.tanh(((((np.where(data["EXT_SOURCE_1"]<0, ((data["NEW_CREDIT_TO_GOODS_RATIO"]) + (((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) + ((((-1.0*((data["EXT_SOURCE_3"])))) * 2.0))))), ((data["CC_AMT_TOTAL_RECEIVABLE_MEAN"]) - (data["EXT_SOURCE_1"])) )) * 2.0)) * 2.0)) 
    v["i16"] = 0.005000*np.tanh(((data["NEW_EXT_SOURCES_MEAN"]) * (((np.minimum(((np.minimum(((((data["NEW_EXT_SOURCES_MEAN"]) * (((data["NEW_RATIO_PREV_AMT_DOWN_PAYMENT_MAX"]) + (data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"])))))), ((data["EXT_SOURCE_1"]))))), ((data["EXT_SOURCE_3"])))) + (data["EXT_SOURCE_3"]))))) 
    v["i17"] = 0.049700*np.tanh(((((((((np.maximum(((data["CC_AMT_RECIVABLE_MAX"])), (((-1.0*(((((data["DAYS_EMPLOYED"]) < (((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) - (data["APPROVED_CNT_PAYMENT_MEAN"]))))*1.)))))))) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)) - (data["NAME_EDUCATION_TYPE_Higher_education"]))) * 2.0)) 
    v["i18"] = 0.045088*np.tanh(((((((np.tanh((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))) + (((((np.maximum(((data["DAYS_EMPLOYED"])), ((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"])))) - (data["NEW_EXT_SOURCES_MEAN"]))) + ((((data["DEF_30_CNT_SOCIAL_CIRCLE"]) + (data["NEW_CREDIT_TO_GOODS_RATIO"]))/2.0)))))) * 2.0)) * 2.0)) 
    v["i19"] = 0.047500*np.tanh(((((np.where(data["NEW_SOURCES_PROD"] > -1, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], ((((data["NEW_DOC_IND_KURT"]) - (((data["EXT_SOURCE_3"]) - (((data["DAYS_BIRTH"]) - (data["EXT_SOURCE_2"]))))))) * 2.0) )) - (data["EXT_SOURCE_3"]))) * 2.0)) 
    v["i20"] = 0.042704*np.tanh(((((data["NEW_CREDIT_TO_GOODS_RATIO"]) + (((((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]) + (((((((data["PREV_CNT_PAYMENT_MEAN"]) - (data["NEW_EXT_SOURCES_MEAN"]))) - (data["CODE_GENDER"]))) - (data["APPROVED_APP_CREDIT_PERC_MAX"]))))) - (data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"]))))) * 2.0)) 
    v["i21"] = 0.048960*np.tanh(((((((((((((((data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]) - (data["NEW_EXT_SOURCES_MEAN"]))) + (data["INSTAL_PAYMENT_DIFF_MAX"]))) + (data["INSTAL_PAYMENT_DIFF_MAX"]))) * 2.0)) + (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))) + (data["NEW_CREDIT_TO_GOODS_RATIO"]))) + (data["NAME_INCOME_TYPE_Working"]))) 
    v["i22"] = 0.049970*np.tanh(((((((((np.tanh((((np.where(data["EXT_SOURCE_1"] > -1, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], ((data["PREV_NAME_PRODUCT_TYPE_walk_in_MEAN"]) - (data["BURO_CREDIT_ACTIVE_Closed_MEAN"])) )) * 2.0)))) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)) * 2.0)) * 2.0)) 
    v["i23"] = 0.049950*np.tanh(((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) + (((data["REGION_RATING_CLIENT_W_CITY"]) + (((np.where(np.maximum(((data["PREV_AMT_DOWN_PAYMENT_MAX"])), ((data["EXT_SOURCE_3"])))>0, ((data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"]) * 2.0), (-1.0*((data["EXT_SOURCE_3"]))) )) * 2.0)))))) * 2.0)) 
    v["i24"] = 0.049985*np.tanh(((((np.where(data["EXT_SOURCE_1"] > -1, data["CODE_GENDER"], data["DAYS_BIRTH"] )) + (((data["REGION_RATING_CLIENT_W_CITY"]) + (data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))))) + (((data["PREV_NAME_YIELD_GROUP_high_MEAN"]) + (((data["PREV_CNT_PAYMENT_MEAN"]) - (data["CODE_GENDER"]))))))) 
    v["i25"] = 0.049800*np.tanh(((((((((((((((data["PREV_CNT_PAYMENT_MEAN"]) - (data["POS_MONTHS_BALANCE_SIZE"]))) - (np.maximum(((data["NEW_CAR_TO_BIRTH_RATIO"])), ((data["CODE_GENDER"])))))) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)) + (data["NEW_DOC_IND_KURT"]))) * 2.0)) * 2.0)) 
    v["i26"] = 0.049993*np.tanh(((((data["DAYS_EMPLOYED"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))) - (((data["EXT_SOURCE_2"]) - (((((np.tanh((data["REFUSED_DAYS_DECISION_MAX"]))) + (((((data["INSTAL_PAYMENT_DIFF_MEAN"]) * 2.0)) * 2.0)))) - (data["NAME_EDUCATION_TYPE_Higher_education"]))))))) 
    v["i27"] = 0.049841*np.tanh(((np.tanh(((-1.0*((data["BURO_CREDIT_ACTIVE_Closed_MEAN"])))))) - (((((((data["CODE_GENDER"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))) + (data["NEW_EXT_SOURCES_MEAN"]))) - (np.where(data["POS_MONTHS_BALANCE_SIZE"]<0, data["FLAG_DOCUMENT_3"], data["REFUSED_AMT_GOODS_PRICE_MAX"] )))))) 
    v["i28"] = 0.050000*np.tanh(((data["APPROVED_CNT_PAYMENT_MEAN"]) - (((data["NEW_EXT_SOURCES_MEAN"]) + (((data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]) + (((np.maximum(((data["NEW_CAR_TO_BIRTH_RATIO"])), ((((((data["INSTAL_AMT_PAYMENT_MIN"]) * 2.0)) + (data["INSTAL_DBD_SUM"])))))) * 2.0)))))))) 
    v["i29"] = 0.049911*np.tanh(((((data["DEF_30_CNT_SOCIAL_CIRCLE"]) + (((((((((data["INSTAL_PAYMENT_DIFF_MAX"]) * 2.0)) - (data["PREV_APP_CREDIT_PERC_MAX"]))) + ((-1.0*((data["APPROVED_AMT_ANNUITY_MEAN"])))))) * 2.0)))) - (((data["NEW_EXT_SOURCES_MEAN"]) - (data["AMT_ANNUITY"]))))) 
    v["i30"] = 0.046032*np.tanh(((((((data["DAYS_LAST_PHONE_CHANGE"]) + (((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) + (data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))))) - (data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))) + (np.where(data["EXT_SOURCE_1"]>0, data["REFUSED_DAYS_DECISION_MEAN"], ((data["NEW_CREDIT_TO_GOODS_RATIO"]) + (data["DAYS_EMPLOYED"])) )))) 
    v["i31"] = 0.049970*np.tanh(((data["NEW_CREDIT_TO_GOODS_RATIO"]) + (((((((data["PREV_NAME_CLIENT_TYPE_New_MEAN"]) - (data["INSTAL_AMT_PAYMENT_MIN"]))) + (((data["PREV_CNT_PAYMENT_MEAN"]) + (((((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]) - (data["INSTAL_AMT_PAYMENT_MIN"]))) - (data["POS_MONTHS_BALANCE_SIZE"]))))))) * 2.0)))) 
    v["i32"] = 0.049504*np.tanh(((((np.where(data["EXT_SOURCE_3"] > -1, data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"], np.maximum(((data["REFUSED_DAYS_DECISION_MAX"])), ((np.maximum(((data["BURO_DAYS_CREDIT_MAX"])), ((((((data["EXT_SOURCE_3"]) - (data["EXT_SOURCE_1"]))) - (data["CODE_GENDER"])))))))) )) * 2.0)) * 2.0)) 
    v["i33"] = 0.049920*np.tanh(np.where(data["NAME_EDUCATION_TYPE_Higher_education"]>0, data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"], ((((np.where(data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]>0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], np.where(data["PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN"]>0, data["CC_CNT_DRAWINGS_CURRENT_MAX"], ((data["NEW_CAR_TO_BIRTH_RATIO"]) * (data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"])) ) )) * 2.0)) * 2.0) )) 
    v["i34"] = 0.048501*np.tanh(np.where(data["INSTAL_DPD_MEAN"]<0, ((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((data["FLAG_DOCUMENT_3"]) + (data["FLAG_DOCUMENT_3"])))), data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"] )) * 2.0)) * 2.0), (-1.0*((data["CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN"]))) )) 
    v["i35"] = 0.048340*np.tanh(np.where(data["INSTAL_DPD_MEAN"]>0, 3.0, ((((((((((data["APPROVED_CNT_PAYMENT_MEAN"]) - (data["APPROVED_AMT_ANNUITY_MEAN"]))) + (data["REG_CITY_NOT_LIVE_CITY"]))) * 2.0)) + (((data["REGION_RATING_CLIENT_W_CITY"]) - (data["CODE_GENDER"]))))) * 2.0) )) 
    v["i36"] = 0.048800*np.tanh(((((((((((np.where(((data["AMT_ANNUITY"]) - (data["APPROVED_AMT_ANNUITY_MEAN"]))<0, data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"], (-1.0*((np.tanh((data["NEW_CAR_TO_EMPLOY_RATIO"]))))) )) * 2.0)) - (data["NAME_INCOME_TYPE_State_servant"]))) * 2.0)) * 2.0)) * 2.0)) 
    v["i37"] = 0.049972*np.tanh(((np.where(data["POS_SK_DPD_DEF_MAX"]>0, 3.141593, ((data["DEF_30_CNT_SOCIAL_CIRCLE"]) + (np.where(data["NEW_SOURCES_PROD"] > -1, data["DEF_30_CNT_SOCIAL_CIRCLE"], ((((((data["PREV_CNT_PAYMENT_SUM"]) - (data["POS_MONTHS_BALANCE_SIZE"]))) * 2.0)) * 2.0) ))) )) * 2.0)) 
    v["i38"] = 0.046442*np.tanh(((((((data["INSTAL_PAYMENT_DIFF_MAX"]) - (data["APPROVED_AMT_ANNUITY_MEAN"]))) + (((np.maximum(((data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"])), ((np.maximum(((data["APPROVED_CNT_PAYMENT_MEAN"])), ((data["DEF_60_CNT_SOCIAL_CIRCLE"]))))))) - (np.maximum(((data["APPROVED_APP_CREDIT_PERC_VAR"])), ((data["NAME_FAMILY_STATUS_Married"])))))))) * 2.0)) 
    v["i39"] = 0.049950*np.tanh(((((((np.where(np.maximum(((data["BURO_CREDIT_ACTIVE_Closed_MEAN"])), ((data["APPROVED_HOUR_APPR_PROCESS_START_MAX"])))<0, ((data["DAYS_ID_PUBLISH"]) - (data["FLOORSMAX_AVG"])), np.where(data["AMT_ANNUITY"]<0, data["CC_CNT_DRAWINGS_CURRENT_MEAN"], data["NEW_CREDIT_TO_GOODS_RATIO"] ) )) * 2.0)) * 2.0)) * 2.0)) 
    v["i40"] = 0.049998*np.tanh(((((data["INSTAL_AMT_INSTALMENT_MAX"]) + (((((data["INSTAL_DPD_MEAN"]) - (((data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]) + (data["PREV_NAME_YIELD_GROUP_low_normal_MEAN"]))))) - (np.maximum(((data["NEW_SOURCES_PROD"])), ((((data["CODE_GENDER"]) + (data["APPROVED_AMT_ANNUITY_MEAN"])))))))))) * 2.0)) 
    v["i41"] = 0.047701*np.tanh(((((((((((((data["INSTAL_PAYMENT_DIFF_MEAN"]) + (np.minimum(((data["AMT_ANNUITY"])), ((data["REGION_RATING_CLIENT_W_CITY"])))))) - (((data["POS_MONTHS_BALANCE_SIZE"]) * 2.0)))) + (data["APPROVED_CNT_PAYMENT_SUM"]))) * 2.0)) + (data["APPROVED_CNT_PAYMENT_SUM"]))) * 2.0)) 
    v["i42"] = 0.049932*np.tanh(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (((((np.where(data["CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN"] > -1, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], np.tanh((data["DAYS_EMPLOYED"])) )) + (((np.maximum(((data["INSTAL_PAYMENT_DIFF_MAX"])), ((data["ACTIVE_DAYS_CREDIT_MAX"])))) * 2.0)))) - (data["OCCUPATION_TYPE_Core_staff"]))))) 
    v["i43"] = 0.049520*np.tanh(((((np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"]>0, data["CC_AMT_INST_MIN_REGULARITY_SUM"], ((np.where(data["NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MIN"]>0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], ((data["FLAG_WORK_PHONE"]) + (((data["DAYS_LAST_PHONE_CHANGE"]) - (data["INSTAL_AMT_PAYMENT_MIN"])))) )) * 2.0) )) * 2.0)) * 2.0)) 
    v["i44"] = 0.049798*np.tanh(((data["PREV_CNT_PAYMENT_MEAN"]) + (((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, ((data["PREV_CNT_PAYMENT_MEAN"]) - (data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"])), data["PREV_DAYS_DECISION_MIN"] )) - (((data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"]) + (((data["APPROVED_AMT_APPLICATION_MIN"]) + (data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]))))))))) 
    v["i45"] = 0.049566*np.tanh(np.where(data["INSTAL_DPD_MEAN"]<0, ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) - (((data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]) * (data["PREV_DAYS_DECISION_MEAN"])))), np.where(data["NEW_RATIO_BURO_DAYS_CREDIT_MAX"]>0, data["DAYS_LAST_PHONE_CHANGE"], (((-1.0*((data["CC_AMT_CREDIT_LIMIT_ACTUAL_SUM"])))) - (data["CC_AMT_CREDIT_LIMIT_ACTUAL_SUM"])) ) )) 
    v["i46"] = 0.049300*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]<0, np.where(((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) - (data["DEF_30_CNT_SOCIAL_CIRCLE"]))<0, ((data["DEF_30_CNT_SOCIAL_CIRCLE"]) - (np.maximum(((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"])), ((((data["NAME_FAMILY_STATUS_Married"]) * 2.0)))))), data["ACTIVE_DAYS_CREDIT_MAX"] ), data["ACTIVE_DAYS_CREDIT_MAX"] )) 
    v["i47"] = 0.049902*np.tanh((-1.0*((np.where(data["PREV_APP_CREDIT_PERC_MEAN"]<0, np.where(data["CC_AMT_CREDIT_LIMIT_ACTUAL_SUM"]<0, data["LIVINGAREA_AVG"], data["CC_AMT_CREDIT_LIMIT_ACTUAL_SUM"] ), (((data["PREV_NAME_TYPE_SUITE_nan_MEAN"]) < ((((data["CC_AMT_BALANCE_MEAN"]) < (data["PREV_APP_CREDIT_PERC_MEAN"]))*1.)))*1.) ))))) 
    v["i48"] = 0.047562*np.tanh(((data["REGION_RATING_CLIENT_W_CITY"]) + (np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"]<0, np.where(data["AMT_GOODS_PRICE"] > -1, np.where(data["OCCUPATION_TYPE_Core_staff"]>0, -3.0, ((data["PREV_NAME_TYPE_SUITE_nan_MEAN"]) - (data["NEW_CAR_TO_EMPLOY_RATIO"])) ), data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"] ), -3.0 )))) 
    v["i49"] = 0.049943*np.tanh(np.where(np.maximum(((data["ORGANIZATION_TYPE_Construction"])), ((np.maximum(((data["CC_CNT_DRAWINGS_CURRENT_VAR"])), ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))))))>0, 3.141593, ((((((data["ORGANIZATION_TYPE_Self_employed"]) * 2.0)) * 2.0)) + (np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((data["PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN"]))))) )) 
    v["i50"] = 0.049280*np.tanh(((((((np.where(((data["PREV_AMT_ANNUITY_MEAN"]) + (data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"]))>0, ((data["APPROVED_CNT_PAYMENT_MEAN"]) - (data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"])), ((data["AMT_ANNUITY"]) + (((data["NEW_DOC_IND_KURT"]) * 2.0))) )) * 2.0)) * 2.0)) * 2.0)) 
    v["i51"] = 0.044800*np.tanh(((((((np.maximum(((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])), ((np.where(data["NEW_SOURCES_PROD"]>0, data["REG_CITY_NOT_LIVE_CITY"], ((np.where(data["NEW_EXT_SOURCES_MEAN"] > -1, data["NEW_EXT_SOURCES_MEAN"], data["BURO_AMT_CREDIT_SUM_DEBT_SUM"] )) - (data["APPROVED_AMT_DOWN_PAYMENT_MEAN"])) ))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i52"] = 0.049551*np.tanh(((data["FLAG_WORK_PHONE"]) + (((data["INSTAL_PAYMENT_DIFF_MEAN"]) + (((((data["DAYS_ID_PUBLISH"]) - (data["CODE_GENDER"]))) + (((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"] )) - (data["INSTAL_AMT_PAYMENT_SUM"]))))))))) 
    v["i53"] = 0.050000*np.tanh(np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"]>0, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_LIMIT_MEAN"], np.where(data["NEW_RATIO_BURO_DAYS_CREDIT_MAX"]>0, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_LIMIT_MEAN"], ((((data["EXT_SOURCE_3"]) * 2.0)) - (np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]<0, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_LIMIT_MEAN"], (-1.0*((data["INSTAL_PAYMENT_DIFF_MEAN"]))) ))) ) )) 
    v["i54"] = 0.049550*np.tanh(((((((np.where(data["OWN_CAR_AGE"] > -1, data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__high_MEAN"], np.where(data["CLOSED_DAYS_CREDIT_VAR"] > -1, data["ACTIVE_AMT_CREDIT_SUM_SUM"], ((data["NAME_INCOME_TYPE_Working"]) - ((((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) + (data["NEW_EMPLOY_TO_BIRTH_RATIO"]))/2.0))) ) )) * 2.0)) * 2.0)) * 2.0)) 
    v["i55"] = 0.049100*np.tanh(((np.where(data["CC_CNT_INSTALMENT_MATURE_CUM_SUM"]>0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], ((data["NEW_DOC_IND_KURT"]) + (((np.where(data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"] > -1, data["CC_AMT_RECIVABLE_MEAN"], ((data["DAYS_REGISTRATION"]) - (data["NEW_EMPLOY_TO_BIRTH_RATIO"])) )) * 2.0))) )) * 2.0)) 
    v["i56"] = 0.048334*np.tanh(((((((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) * 2.0)) + (data["REG_CITY_NOT_LIVE_CITY"]))) + (np.maximum(((data["APPROVED_CNT_PAYMENT_SUM"])), ((np.maximum(((data["OCCUPATION_TYPE_Drivers"])), ((np.where(data["POS_SK_DPD_DEF_MEAN"]<0, data["BURO_CREDIT_TYPE_Microloan_MEAN"], (-1.0*((data["NEW_RATIO_PREV_AMT_APPLICATION_MIN"]))) )))))))))) 
    v["i57"] = 0.050000*np.tanh(((((((data["INSTAL_PAYMENT_DIFF_SUM"]) - (((np.where(data["NEW_DOC_IND_STD"] > -1, np.where(data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"]>0, data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"], data["PREV_NAME_PAYMENT_TYPE_Cash_through_the_bank_MEAN"] ), data["NEW_DOC_IND_STD"] )) * 2.0)))) - (data["CODE_GENDER"]))) - (data["NAME_FAMILY_STATUS_Married"]))) 
    v["i58"] = 0.049904*np.tanh(((np.where(data["AMT_GOODS_PRICE"] > -1, ((((((((((((data["NEW_DOC_IND_KURT"]) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (data["REGION_RATING_CLIENT_W_CITY"]))) * 2.0)) * 2.0)) * 2.0), data["NEW_CREDIT_TO_ANNUITY_RATIO"] )) * 2.0)) 
    v["i59"] = 0.048700*np.tanh(np.where(data["NEW_DOC_IND_AVG"]>0, ((np.where(data["NEW_CREDIT_TO_GOODS_RATIO"]>0, ((data["AMT_CREDIT"]) + ((((((data["AMT_CREDIT"]) > (data["INSTAL_DBD_SUM"]))*1.)) * 2.0))), ((data["BURO_AMT_CREDIT_SUM_DEBT_SUM"]) * 2.0) )) * 2.0), data["INSTAL_PAYMENT_DIFF_MEAN"] )) 
    v["i60"] = 0.045160*np.tanh(((((((np.where((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) > (data["BURO_AMT_CREDIT_SUM_DEBT_SUM"]))*1.)>0, data["CC_CNT_DRAWINGS_POS_CURRENT_MIN"], data["BURO_CREDIT_TYPE_Microloan_MEAN"] )) - (np.maximum(((data["NEW_CAR_TO_BIRTH_RATIO"])), ((data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"])))))) - (data["BURO_CREDIT_TYPE_Mortgage_MEAN"]))) - (data["NAME_INCOME_TYPE_State_servant"]))) 
    v["i61"] = 0.049550*np.tanh(np.where(data["NEW_EXT_SOURCES_MEAN"]<0, np.where(data["ACTIVE_DAYS_CREDIT_MEAN"]<0, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"], (-1.0*((data["NEW_EXT_SOURCES_MEAN"]))) ), np.where(data["NAME_EDUCATION_TYPE_Higher_education"]>0, (-1.0*((data["NEW_EXT_SOURCES_MEAN"]))), (-1.0*((data["NEW_SOURCES_PROD"]))) ) )) 
    v["i62"] = 0.047401*np.tanh(((((((data["POS_SK_DPD_DEF_MAX"]) - (data["APPROVED_AMT_APPLICATION_MIN"]))) * 2.0)) - (np.maximum(((np.maximum(((data["BURO_CREDIT_TYPE_Car_loan_MEAN"])), ((np.maximum(((((data["NAME_INCOME_TYPE_Commercial_associate"]) + (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])))), ((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])))))))), ((data["FLAG_DOCUMENT_8"])))))) 
    v["i63"] = 0.048088*np.tanh(np.where(data["INSTAL_AMT_INSTALMENT_MAX"]>0, data["APPROVED_CNT_PAYMENT_MEAN"], ((((((((np.maximum(((data["DEF_60_CNT_SOCIAL_CIRCLE"])), ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])))) * 2.0)) + (data["NEW_SCORES_STD"]))) + (np.maximum(((data["PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN"])), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"])))))) * 2.0) )) 
    v["i64"] = 0.049282*np.tanh(((data["POS_SK_DPD_DEF_MAX"]) - (np.where((((data["NEW_EXT_SOURCES_MEAN"]) + (data["INSTAL_DBD_SUM"]))/2.0) > -1, ((data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"]) - (((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)) * 2.0))), ((data["NEW_EXT_SOURCES_MEAN"]) + (data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"])) )))) 
    v["i65"] = 0.049900*np.tanh(np.where(((data["ORGANIZATION_TYPE_Military"]) - (data["DAYS_BIRTH"]))>0, ((((data["EXT_SOURCE_1"]) / 2.0)) - (data["DAYS_BIRTH"])), (-1.0*((((data["DAYS_BIRTH"]) + (((data["DAYS_BIRTH"]) + (data["EXT_SOURCE_1"]))))))) )) 
    v["i66"] = 0.049906*np.tanh(np.where(data["LANDAREA_AVG"]>0, data["CC_MONTHS_BALANCE_VAR"], ((np.where(data["NEW_RATIO_BURO_DAYS_CREDIT_MAX"]<0, data["NEW_RATIO_BURO_DAYS_CREDIT_MAX"], data["CC_AMT_CREDIT_LIMIT_ACTUAL_SUM"] )) * (np.where((((data["INSTAL_PAYMENT_DIFF_MEAN"]) + (data["INSTAL_DPD_MEAN"]))/2.0)<0, data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"], data["LANDAREA_AVG"] ))) )) 
    v["i67"] = 0.050000*np.tanh(((data["PREV_NAME_CLIENT_TYPE_New_MEAN"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (np.maximum(((((((data["DAYS_ID_PUBLISH"]) - ((-1.0*((data["REGION_RATING_CLIENT_W_CITY"])))))) + (((data["PREV_NAME_PORTFOLIO_XNA_MEAN"]) + (data["APPROVED_CNT_PAYMENT_MEAN"])))))), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"])))))))) 
    v["i68"] = 0.048988*np.tanh(((((data["WALLSMATERIAL_MODE_Stone__brick"]) + (((((np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((np.where(data["APPROVED_AMT_ANNUITY_MEAN"] > -1, data["POS_SK_DPD_DEF_MAX"], ((((-1.0*((data["NEW_PHONE_TO_BIRTH_RATIO"])))) > (data["NEW_EXT_SOURCES_MEAN"]))*1.) ))))) * 2.0)) * 2.0)))) * 2.0)) 
    v["i69"] = 0.050000*np.tanh((-1.0*((np.maximum(((np.where((((data["INSTAL_DAYS_ENTRY_PAYMENT_MEAN"]) + (data["INSTAL_AMT_INSTALMENT_MAX"]))/2.0)>0, data["YEARS_BUILD_MEDI"], data["INSTAL_COUNT"] ))), ((((np.maximum(((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])), ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])))) + (data["BURO_CREDIT_TYPE_Car_loan_MEAN"]))))))))) 
    v["i70"] = 0.044000*np.tanh(((((((np.where(((data["APPROVED_AMT_GOODS_PRICE_MAX"]) - (np.tanh((data["POS_MONTHS_BALANCE_SIZE"]))))>0, data["PREV_DAYS_DECISION_MAX"], (((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]) + (data["AMT_ANNUITY"]))/2.0) )) * 2.0)) * 2.0)) - (data["NEW_DOC_IND_AVG"]))) 
    v["i71"] = 0.037587*np.tanh(((np.where(data["BURO_CREDIT_ACTIVE_Closed_MEAN"]>0, np.minimum(((data["REGION_POPULATION_RELATIVE"])), (((-1.0*((data["PREV_AMT_ANNUITY_MEAN"])))))), np.maximum(((data["FLAG_WORK_PHONE"])), ((np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((((data["BASEMENTAREA_AVG"]) * (data["PREV_CODE_REJECT_REASON_XAP_MEAN"])))))))) )) * 2.0)) 
    v["i72"] = 0.026552*np.tanh(((((np.where(data["ACTIVE_AMT_CREDIT_SUM_SUM"]<0, ((data["APPROVED_CNT_PAYMENT_SUM"]) - (((np.tanh((np.maximum(((np.maximum(((data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"])), ((data["POS_COUNT"]))))), ((data["INSTAL_AMT_PAYMENT_SUM"])))))) * 2.0))), data["BURO_CREDIT_TYPE_Credit_card_MEAN"] )) * 2.0)) * 2.0)) 
    v["i73"] = 0.047540*np.tanh(((np.where(data["BURO_CREDIT_TYPE_Microloan_MEAN"]>0, data["BURO_CREDIT_TYPE_Microloan_MEAN"], np.where(data["PREV_AMT_DOWN_PAYMENT_MIN"]>0, data["BURO_CREDIT_TYPE_Microloan_MEAN"], np.where(data["NAME_INCOME_TYPE_Commercial_associate"]>0, data["INSTAL_AMT_PAYMENT_MIN"], ((data["INSTAL_AMT_PAYMENT_MIN"]) * (data["NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MAX"])) ) ) )) - (data["PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN"]))) 
    v["i74"] = 0.049300*np.tanh(((((np.maximum(((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])), ((data["NAME_FAMILY_STATUS_Separated"])))) + (data["NAME_HOUSING_TYPE_Municipal_apartment"]))) + (((np.where(data["APPROVED_APP_CREDIT_PERC_MIN"]>0, data["INSTAL_DPD_MEAN"], (((data["CC_AMT_PAYMENT_CURRENT_SUM"]) < (data["POS_SK_DPD_DEF_MAX"]))*1.) )) * 2.0)))) 
    v["i75"] = 0.042080*np.tanh(((np.tanh((data["NEW_EXT_SOURCES_MEAN"]))) - (((((data["NEW_EXT_SOURCES_MEAN"]) + ((((data["NEW_EXT_SOURCES_MEAN"]) < (data["OCCUPATION_TYPE_Accountants"]))*1.)))) + ((((data["FLAG_PHONE"]) + (np.maximum(((data["NEW_EMPLOY_TO_BIRTH_RATIO"])), ((data["BURO_STATUS_0_MEAN_MEAN"])))))/2.0)))))) 
    v["i76"] = 0.039200*np.tanh(((np.maximum(((data["BURO_STATUS_1_MEAN_MEAN"])), ((np.maximum(((data["ORGANIZATION_TYPE_Self_employed"])), ((((data["BURO_CREDIT_TYPE_Microloan_MEAN"]) + (data["BURO_AMT_CREDIT_SUM_DEBT_SUM"]))))))))) + (np.where(data["PREV_CODE_REJECT_REASON_XAP_MEAN"] > -1, data["BURO_AMT_CREDIT_SUM_DEBT_SUM"], ((data["BURO_DAYS_CREDIT_MEAN"]) * (data["BURO_AMT_CREDIT_SUM_DEBT_SUM"])) )))) 
    v["i77"] = 0.045016*np.tanh(((np.where(((data["WEEKDAY_APPR_PROCESS_START_MONDAY"]) * (data["FLAG_DOCUMENT_18"]))<0, data["NEW_RATIO_BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"], np.where(((data["INSTAL_AMT_INSTALMENT_SUM"]) * (data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"]))<0, data["OCCUPATION_TYPE_Laborers"], ((data["ORGANIZATION_TYPE_Industry__type_9"]) * (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) ) )) * 2.0)) 
    v["i78"] = 0.044600*np.tanh(np.where(data["FLOORSMIN_MODE"]<0, ((np.where(data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]<0, ((((data["APPROVED_AMT_CREDIT_MAX"]) - (data["INSTAL_AMT_PAYMENT_SUM"]))) * 2.0), ((data["PREV_NAME_CONTRACT_STATUS_Canceled_MEAN"]) * (data["FLOORSMIN_MODE"])) )) * 2.0), data["NEW_DOC_IND_KURT"] )) 
    v["i79"] = 0.047706*np.tanh(((((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > -1, ((data["FLOORSMAX_AVG"]) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"])))), data["NEW_CREDIT_TO_ANNUITY_RATIO"] )) - (data["NAME_INCOME_TYPE_State_servant"]))) - (data["NAME_INCOME_TYPE_State_servant"]))) - (data["WEEKDAY_APPR_PROCESS_START_SATURDAY"]))) 
    v["i80"] = 0.019998*np.tanh(np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]<0, np.minimum(((data["NEW_DOC_IND_KURT"])), ((np.where(data["CC_AMT_CREDIT_LIMIT_ACTUAL_MIN"]>0, ((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) * 2.0), ((data["AMT_CREDIT"]) + (data["NEW_DOC_IND_KURT"])) )))), ((data["APPROVED_CNT_PAYMENT_SUM"]) - (data["REFUSED_CNT_PAYMENT_SUM"])) )) 
    v["i81"] = 0.049499*np.tanh(((((data["FLAG_WORK_PHONE"]) + (np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), (((((np.maximum(((data["APPROVED_CNT_PAYMENT_SUM"])), ((data["ORGANIZATION_TYPE_Construction"])))) + (np.maximum(((np.maximum(((data["NAME_HOUSING_TYPE_Rented_apartment"])), ((data["NEW_SCORES_STD"]))))), ((data["CC_CNT_DRAWINGS_CURRENT_MAX"])))))/2.0))))))) * 2.0)) 
    v["i82"] = 0.047499*np.tanh((((((data["INSTAL_DBD_MAX"]) + ((-1.0*((np.maximum(((data["OCCUPATION_TYPE_Accountants"])), ((data["ORGANIZATION_TYPE_School"]))))))))/2.0)) - ((((((((data["EXT_SOURCE_2"]) + (data["NEW_PHONE_TO_EMPLOY_RATIO"]))) + (data["APPROVED_HOUR_APPR_PROCESS_START_MAX"]))/2.0)) + (data["NEW_LIVE_IND_SUM"]))))) 
    v["i83"] = 0.048937*np.tanh(((np.where(data["BURO_STATUS_1_MEAN_MEAN"] > -1, data["AMT_ANNUITY"], np.where(data["AMT_ANNUITY"]<0, data["DAYS_ID_PUBLISH"], data["PREV_CODE_REJECT_REASON_LIMIT_MEAN"] ) )) - (((((data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"]) - (data["OCCUPATION_TYPE_Drivers"]))) - (data["POS_SK_DPD_MEAN"]))))) 
    v["i84"] = 0.047002*np.tanh(np.where(data["BURO_CREDIT_ACTIVE_Active_MEAN"] > -1, np.minimum(((data["NEW_DOC_IND_KURT"])), ((data["REGION_POPULATION_RELATIVE"]))), np.where(data["CLOSED_DAYS_CREDIT_MAX"]>0, data["CC_CNT_DRAWINGS_CURRENT_MEAN"], np.where(data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]>0, data["BURO_CREDIT_ACTIVE_Active_MEAN"], ((data["EXT_SOURCE_3"]) - (data["BURO_CREDIT_ACTIVE_Active_MEAN"])) ) ) )) 
    v["i85"] = 0.049968*np.tanh(np.where((((data["INSTAL_DPD_MEAN"]) > (data["NAME_EDUCATION_TYPE_Lower_secondary"]))*1.)>0, data["INSTAL_DAYS_ENTRY_PAYMENT_MEAN"], np.where(data["NEW_PHONE_TO_BIRTH_RATIO"] > -1, data["REFUSED_AMT_CREDIT_MAX"], ((data["INSTAL_DAYS_ENTRY_PAYMENT_MEAN"]) * (data["REFUSED_AMT_CREDIT_MAX"])) ) )) 
    v["i86"] = 0.048000*np.tanh(np.where((((data["PREV_NAME_CONTRACT_STATUS_Canceled_MEAN"]) > (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))*1.)>0, np.where(data["PREV_NAME_CONTRACT_STATUS_Canceled_MEAN"]>0, data["PREV_NAME_YIELD_GROUP_high_MEAN"], data["NEW_CREDIT_TO_ANNUITY_RATIO"] ), ((((data["REFUSED_AMT_CREDIT_MAX"]) * (data["WEEKDAY_APPR_PROCESS_START_SUNDAY"]))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) )) 
    v["i87"] = 0.047521*np.tanh(((data["NAME_FAMILY_STATUS_Married"]) * (np.where((((data["NAME_FAMILY_STATUS_Married"]) < (np.minimum(((data["PREV_AMT_GOODS_PRICE_MEAN"])), (((-1.0*((data["DAYS_BIRTH"]))))))))*1.)>0, data["REFUSED_AMT_CREDIT_MIN"], ((((data["DAYS_BIRTH"]) * 2.0)) * 2.0) )))) 
    v["i88"] = 0.035104*np.tanh(((data["INSTAL_AMT_PAYMENT_MIN"]) - ((-1.0*((((np.where(data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"]<0, data["PREV_NAME_PORTFOLIO_XNA_MEAN"], ((((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]) + (data["PREV_CHANNEL_TYPE_Contact_center_MEAN"]))) + (data["NAME_HOUSING_TYPE_Rented_apartment"])) )) - (data["BURO_AMT_CREDIT_SUM_MEAN"])))))))) 
    v["i89"] = 0.049998*np.tanh(np.where(((data["NEW_DOC_IND_KURT"]) - (data["DAYS_BIRTH"])) > -1, np.where(data["NEW_INC_PER_CHLD"]<0, data["CLOSED_DAYS_CREDIT_MIN"], ((((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) * (data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"]))) * 2.0) ), (-1.0*((data["DAYS_BIRTH"]))) )) 
    v["i90"] = 0.049998*np.tanh(np.where(data["CLOSED_AMT_CREDIT_SUM_SUM"]>0, ((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) + (np.where(data["ACTIVE_AMT_CREDIT_SUM_SUM"]<0, data["NEW_RATIO_PREV_APP_CREDIT_PERC_MEAN"], data["REGION_RATING_CLIENT_W_CITY"] ))), ((((((((data["APPROVED_CNT_PAYMENT_SUM"]) - (data["POS_COUNT"]))) * 2.0)) * 2.0)) * 2.0) )) 
    v["i91"] = 0.049698*np.tanh(np.where(data["APPROVED_AMT_GOODS_PRICE_MEAN"]>0, data["CC_AMT_BALANCE_MIN"], np.where(data["POS_MONTHS_BALANCE_MEAN"]>0, ((((((data["INSTAL_DPD_MEAN"]) * 2.0)) * 2.0)) * 2.0), ((data["BURO_STATUS_X_MEAN_MEAN"]) - (((data["NEW_CAR_TO_BIRTH_RATIO"]) * 2.0))) ) )) 
    v["i92"] = 0.049598*np.tanh(np.where(data["NEW_RATIO_BURO_DAYS_CREDIT_MEAN"]<0, (((((data["EXT_SOURCE_3"]) > (-3.0))*1.)) - ((((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]) > (data["NEW_EXT_SOURCES_MEAN"]))*1.))), np.where(data["EXT_SOURCE_3"] > -1, data["EXT_SOURCE_3"], data["PREV_NAME_SELLER_INDUSTRY_XNA_MEAN"] ) )) 
    v["i93"] = 0.049600*np.tanh(((((((np.minimum(((data["PREV_NAME_GOODS_CATEGORY_Consumer_Electronics_MEAN"])), ((data["NAME_EDUCATION_TYPE_Lower_secondary"])))) + ((((np.minimum(((data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__middle_MEAN"])), ((data["DAYS_ID_PUBLISH"])))) > (data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"]))*1.)))) + (np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"])))))) * 2.0)) 
    v["i94"] = 0.049793*np.tanh(((np.where((-1.0*((data["EXT_SOURCE_2"]))) > -1, np.maximum(((data["CC_AMT_BALANCE_MIN"])), ((((np.minimum(((data["NAME_CONTRACT_TYPE_Cash_loans"])), ((data["OBS_30_CNT_SOCIAL_CIRCLE"])))) * (data["PREV_AMT_GOODS_PRICE_MEAN"]))))), (((data["NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MIN"]) + (data["REGION_RATING_CLIENT_W_CITY"]))/2.0) )) * 2.0)) 
    v["i95"] = 0.047992*np.tanh(np.where(data["EXT_SOURCE_3"] > -1, np.where(data["CODE_GENDER"] > -1, data["PREV_PRODUCT_COMBINATION_POS_household_without_interest_MEAN"], (-1.0*((data["CLOSED_DAYS_CREDIT_MAX"]))) ), np.maximum(((data["CC_AMT_RECIVABLE_MEAN"])), ((np.maximum(((data["REFUSED_CNT_PAYMENT_SUM"])), ((np.maximum(((data["REFUSED_DAYS_DECISION_MEAN"])), ((data["ACTIVE_DAYS_CREDIT_MEAN"]))))))))) )) 
    v["i96"] = 0.048061*np.tanh(np.where(data["PREV_NAME_GOODS_CATEGORY_Photo___Cinema_Equipment_MEAN"]>0, data["REFUSED_AMT_DOWN_PAYMENT_MIN"], np.where(data["REFUSED_AMT_DOWN_PAYMENT_MIN"] > -1, data["PREV_AMT_GOODS_PRICE_MAX"], np.where(data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]>0, data["REFUSED_AMT_DOWN_PAYMENT_MIN"], (((data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"]) > (((data["PREV_NAME_GOODS_CATEGORY_Photo___Cinema_Equipment_MEAN"]) * (data["BURO_DAYS_CREDIT_MAX"]))))*1.) ) ) )) 
    v["i97"] = 0.022584*np.tanh(((np.where(np.where(data["BURO_AMT_CREDIT_SUM_DEBT_SUM"] > -1, data["CODE_GENDER"], data["BURO_AMT_CREDIT_SUM_MEAN"] ) > -1, data["BURO_AMT_CREDIT_SUM_DEBT_SUM"], ((data["EXT_SOURCE_2"]) * (np.where(data["BURO_AMT_CREDIT_SUM_DEBT_SUM"] > -1, data["BURO_MONTHS_BALANCE_SIZE_SUM"], data["BURO_AMT_CREDIT_SUM_MEAN"] ))) )) - (data["BURO_AMT_CREDIT_SUM_MEAN"]))) 
    v["i98"] = 0.046920*np.tanh((-1.0*((np.where(data["DAYS_BIRTH"]>0, np.where(data["DAYS_EMPLOYED"]>0, np.maximum(((data["BURO_DAYS_CREDIT_MIN"])), ((data["CC_AMT_CREDIT_LIMIT_ACTUAL_MAX"]))), np.where(data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"] > -1, data["CC_AMT_PAYMENT_TOTAL_CURRENT_MAX"], (-1.0*((data["PREV_NAME_PRODUCT_TYPE_x_sell_MEAN"]))) ) ), data["CODE_GENDER"] ))))) 
    v["i99"] = 0.049880*np.tanh((((((((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) > (np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_MEAN"])), ((data["DEF_30_CNT_SOCIAL_CIRCLE"])))))*1.)) + (np.tanh((np.tanh((np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"] > -1, data["NEW_RATIO_BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"], data["POS_SK_DPD_DEF_MAX"] )))))))) * 2.0)) * 2.0)) 
    v["i100"] = 0.049848*np.tanh(np.where(np.where((-1.0*((data["APPROVED_DAYS_DECISION_MAX"]))) > -1, data["NEW_EXT_SOURCES_MEAN"], data["BURO_AMT_CREDIT_SUM_MEAN"] )>0, (-1.0*((data["EXT_SOURCE_1"]))), ((data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"]) * (np.where(data["NEW_RATIO_PREV_APP_CREDIT_PERC_MEAN"]>0, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_MAX"], data["APPROVED_DAYS_DECISION_MAX"] ))) )) 
    v["i101"] = 0.047541*np.tanh(np.where(data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"]>0, data["EXT_SOURCE_2"], (-1.0*((((data["CC_CNT_DRAWINGS_POS_CURRENT_MIN"]) * ((-1.0*((np.where(data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]>0, data["INSTAL_PAYMENT_DIFF_MAX"], ((data["DEF_60_CNT_SOCIAL_CIRCLE"]) * (data["PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN"])) ))))))))) )) 
    v["i102"] = 0.047686*np.tanh(np.where(data["CC_AMT_RECIVABLE_VAR"] > -1, ((((data["CC_AMT_RECIVABLE_VAR"]) + (data["CC_CNT_DRAWINGS_CURRENT_VAR"]))) - (data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"])), np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"]>0, data["CC_CNT_DRAWINGS_CURRENT_VAR"], (((data["NEW_SOURCES_PROD"]) > (data["CC_CNT_DRAWINGS_CURRENT_VAR"]))*1.) ) )) 
    v["i103"] = 0.046496*np.tanh(((data["POS_NAME_CONTRACT_STATUS_Completed_MEAN"]) * (np.maximum(((((np.where(data["POS_NAME_CONTRACT_STATUS_Completed_MEAN"] > -1, ((data["APPROVED_AMT_APPLICATION_MIN"]) * 2.0), data["INSTAL_COUNT"] )) * 2.0))), (((((((((data["AMT_CREDIT"]) + (data["WEEKDAY_APPR_PROCESS_START_SATURDAY"]))/2.0)) * 2.0)) * 2.0))))))) 
    v["i104"] = 0.049850*np.tanh(np.where(data["BURO_DAYS_CREDIT_MAX"] > -1, np.where(data["BURO_CREDIT_TYPE_Car_loan_MEAN"]<0, np.where(data["PREV_CHANNEL_TYPE_Contact_center_MEAN"]>0, data["PREV_CHANNEL_TYPE_Contact_center_MEAN"], data["INSTAL_AMT_PAYMENT_MIN"] ), data["CC_AMT_CREDIT_LIMIT_ACTUAL_SUM"] ), np.maximum(((data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"])), (((-1.0*((data["PREV_NAME_SELLER_INDUSTRY_XNA_MEAN"])))))) )) 
    v["i105"] = 0.049100*np.tanh(((np.where(data["BURO_STATUS_0_MEAN_MEAN"] > -1, data["PREV_CNT_PAYMENT_SUM"], np.maximum(((data["PREV_NAME_CONTRACT_TYPE_Revolving_loans_MEAN"])), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"]))) )) - (np.maximum(((data["BURO_STATUS_0_MEAN_MEAN"])), ((np.maximum(((np.maximum(((data["NAME_FAMILY_STATUS_Married"])), ((data["EXT_SOURCE_1"]))))), ((data["EXT_SOURCE_1"]))))))))) 
    v["i106"] = 0.049003*np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, np.where((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (((data["PREV_CODE_REJECT_REASON_SCO_MEAN"]) + (((data["PREV_NAME_CLIENT_TYPE_Refreshed_MEAN"]) * 2.0)))))*1.)>0, data["BURO_STATUS_0_MEAN_MEAN"], (((data["PREV_CNT_PAYMENT_SUM"]) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.) ), data["NAME_EDUCATION_TYPE_Higher_education"] )) 
    v["i107"] = 0.041200*np.tanh(((np.where(data["BURO_CREDIT_TYPE_Microloan_MEAN"] > -1, data["REGION_RATING_CLIENT_W_CITY"], data["AMT_ANNUITY"] )) + (((data["PREV_CHANNEL_TYPE_Credit_and_cash_offices_MEAN"]) * (np.where(data["PREV_NAME_PRODUCT_TYPE_XNA_MEAN"]<0, data["REGION_RATING_CLIENT_W_CITY"], np.where(data["REGION_RATING_CLIENT_W_CITY"]<0, data["PREV_WEEKDAY_APPR_PROCESS_START_SATURDAY_MEAN"], data["BURO_CREDIT_TYPE_Mortgage_MEAN"] ) )))))) 
    v["i108"] = 0.048457*np.tanh(((((((((((((data["DAYS_BIRTH"]) * (((data["PREV_NAME_PORTFOLIO_POS_MEAN"]) * 2.0)))) - (data["ORGANIZATION_TYPE_Military"]))) - (data["ORGANIZATION_TYPE_Industry__type_9"]))) - (data["OCCUPATION_TYPE_High_skill_tech_staff"]))) - (data["ORGANIZATION_TYPE_Bank"]))) - (data["OCCUPATION_TYPE_Medicine_staff"]))) 
    v["i109"] = 0.049920*np.tanh(np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"]>0, data["NEW_RATIO_BURO_AMT_ANNUITY_MEAN"], np.where(data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]>0, data["BURO_DAYS_CREDIT_MEAN"], np.where(data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"] > -1, data["ACTIVE_AMT_CREDIT_SUM_SUM"], np.where(data["BURO_DAYS_CREDIT_VAR"] > -1, data["ACTIVE_AMT_CREDIT_SUM_SUM"], (-1.0*((data["INSTAL_DBD_SUM"]))) ) ) ) )) 
    v["i110"] = 0.048617*np.tanh(((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) * ((((((data["INSTAL_AMT_PAYMENT_MEAN"]) + (data["INSTAL_AMT_INSTALMENT_SUM"]))/2.0)) - (np.where(data["REGION_RATING_CLIENT"] > -1, data["EXT_SOURCE_3"], (((((((data["APPROVED_CNT_PAYMENT_MEAN"]) > (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)) * 2.0)) * 2.0) )))))) 
    v["i111"] = 0.031776*np.tanh((((((((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MAX"]<0, np.where(data["INSTAL_PAYMENT_DIFF_MAX"]<0, data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"], data["INSTAL_DBD_MAX"] ), (-1.0*((data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]))) )) + (((data["BURO_DAYS_CREDIT_ENDDATE_MAX"]) * (data["OBS_60_CNT_SOCIAL_CIRCLE"]))))/2.0)) * 2.0)) * 2.0)) 
    v["i112"] = 0.049079*np.tanh(np.where(data["NEW_CREDIT_TO_GOODS_RATIO"] > -1, ((((((np.minimum(((np.minimum((((-1.0*((data["REFUSED_AMT_ANNUITY_MIN"]))))), ((data["NEW_CREDIT_TO_GOODS_RATIO"]))))), ((data["AMT_ANNUITY"])))) * 2.0)) * 2.0)) * 2.0), ((data["INSTAL_DBD_MAX"]) - (data["REFUSED_AMT_ANNUITY_MIN"])) )) 
    v["i113"] = 0.048979*np.tanh(np.where(np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])), ((data["NEW_EXT_SOURCES_MEAN"]))) > -1, data["BURO_CREDIT_TYPE_Microloan_MEAN"], np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"]<0, np.maximum(((data["NEW_CREDIT_TO_GOODS_RATIO"])), ((np.where(data["ORGANIZATION_TYPE_Business_Entity_Type_3"]<0, data["INSTAL_PAYMENT_DIFF_MEAN"], data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"] )))), data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"] ) )) 
    v["i114"] = 0.047000*np.tanh(np.where(data["CC_AMT_RECEIVABLE_PRINCIPAL_MAX"] > -1, data["REGION_POPULATION_RELATIVE"], ((data["REFUSED_DAYS_DECISION_MEAN"]) * (((((((data["APPROVED_AMT_CREDIT_MIN"]) + (data["NEW_PHONE_TO_EMPLOY_RATIO"]))/2.0)) + ((((data["INSTAL_PAYMENT_DIFF_VAR"]) < ((((data["APPROVED_AMT_CREDIT_MIN"]) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))/2.0)))*1.)))/2.0))) )) 
    v["i115"] = 0.048568*np.tanh(((np.where(data["ACTIVE_AMT_CREDIT_SUM_SUM"]<0, ((np.where(data["CLOSED_AMT_CREDIT_SUM_SUM"]>0, data["PREV_PRODUCT_COMBINATION_POS_other_with_interest_MEAN"], (((-2.0) > (np.where(data["INSTAL_PAYMENT_DIFF_VAR"]>0, data["LIVINGAPARTMENTS_AVG"], data["EXT_SOURCE_2"] )))*1.) )) * 2.0), data["ACTIVE_DAYS_CREDIT_MEAN"] )) * 2.0)) 
    v["i116"] = 0.049750*np.tanh(np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]>0, data["INSTAL_PAYMENT_DIFF_SUM"], np.where(np.where(data["APPROVED_APP_CREDIT_PERC_MEAN"]<0, data["DAYS_BIRTH"], data["NAME_EDUCATION_TYPE_Higher_education"] )<0, data["APPROVED_AMT_APPLICATION_MAX"], np.where(data["DAYS_BIRTH"]<0, data["NAME_EDUCATION_TYPE_Higher_education"], (-1.0*((data["NAME_EDUCATION_TYPE_Higher_education"]))) ) ) )) 
    v["i117"] = 0.046000*np.tanh(((np.where(data["BASEMENTAREA_MEDI"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (-1.0*((np.where(data["CC_AMT_DRAWINGS_ATM_CURRENT_MAX"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], ((data["DEF_60_CNT_SOCIAL_CIRCLE"]) * (data["NEW_ANNUITY_TO_INCOME_RATIO"])) ) )))) )) * 2.0)) 
    v["i118"] = 0.049042*np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_MEAN"]>0, data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"], ((np.where(data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]>0, data["CC_AMT_PAYMENT_CURRENT_MIN"], ((data["PREV_WEEKDAY_APPR_PROCESS_START_SUNDAY_MEAN"]) * (np.minimum(((data["INSTAL_DBD_MEAN"])), ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["APPROVED_DAYS_DECISION_MAX"]))))))) )) * 2.0) )) 
    v["i119"] = 0.046682*np.tanh(np.where(data["CC_AMT_PAYMENT_CURRENT_MEAN"]>0, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"], np.where(data["PREV_NAME_YIELD_GROUP_middle_MEAN"]>0, data["PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN"], np.maximum(((data["CC_CNT_DRAWINGS_CURRENT_MAX"])), ((np.maximum(((np.where(data["PREV_DAYS_DECISION_MIN"] > -1, data["ORGANIZATION_TYPE_Construction"], data["BURO_AMT_CREDIT_SUM_DEBT_SUM"] ))), ((data["PREV_PRODUCT_COMBINATION_Cash_Street__middle_MEAN"])))))) ) )) 
    v["i120"] = 0.037398*np.tanh(((((np.where(data["CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN"]>0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], np.maximum(((np.maximum((((((data["NAME_EDUCATION_TYPE_Lower_secondary"]) + ((((data["NEW_RATIO_PREV_DAYS_DECISION_MAX"]) > (data["NAME_EDUCATION_TYPE_Lower_secondary"]))*1.)))/2.0))), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"]))))), ((data["CLOSED_AMT_CREDIT_SUM_DEBT_MAX"]))) )) * 2.0)) * 2.0)) 
    v["i121"] = 0.044000*np.tanh(np.where(data["INSTAL_DPD_MEAN"]<0, ((np.maximum((((((data["INSTAL_DPD_MEAN"]) > (((((data["POS_MONTHS_BALANCE_MEAN"]) / 2.0)) / 2.0)))*1.))), (((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"]) > (data["ORGANIZATION_TYPE_Construction"]))*1.))))) * 2.0), data["POS_MONTHS_BALANCE_MEAN"] )) 
    v["i122"] = 0.005400*np.tanh(np.where(data["DAYS_ID_PUBLISH"] > -1, np.where(data["POS_MONTHS_BALANCE_MEAN"]<0, np.where(data["DAYS_ID_PUBLISH"]<0, data["PREV_NAME_TYPE_SUITE_Family_MEAN"], -3.0 ), ((((np.maximum(((data["CC_AMT_RECIVABLE_VAR"])), ((data["INSTAL_DPD_MEAN"])))) * 2.0)) * 2.0) ), data["FLOORSMIN_MEDI"] )) 
    v["i123"] = 0.040397*np.tanh(np.where(data["EXT_SOURCE_2"]>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], np.where(data["ACTIVE_MONTHS_BALANCE_MAX_MAX"]<0, ((np.where(data["DAYS_ID_PUBLISH"] > -1, ((data["NEW_DOC_IND_KURT"]) + (data["APPROVED_APP_CREDIT_PERC_MEAN"])), data["REFUSED_AMT_DOWN_PAYMENT_MIN"] )) - (data["NEW_ANNUITY_TO_INCOME_RATIO"])), data["EXT_SOURCE_1"] ) )) 
    v["i124"] = 0.049016*np.tanh(((np.where(data["NEW_EXT_SOURCES_MEAN"] > -1, data["NAME_HOUSING_TYPE_Municipal_apartment"], data["NEW_CREDIT_TO_GOODS_RATIO"] )) + (((np.maximum(((data["BURO_CREDIT_ACTIVE_Sold_MEAN"])), ((np.maximum(((np.where(data["NEW_SCORES_STD"] > -1, data["BURO_CREDIT_TYPE_Microloan_MEAN"], data["NEW_CREDIT_TO_GOODS_RATIO"] ))), ((data["ORGANIZATION_TYPE_Construction"]))))))) * 2.0)))) 
    v["i125"] = 0.023880*np.tanh(np.maximum(((data["ORGANIZATION_TYPE_Transport__type_3"])), ((((data["NEW_RATIO_BURO_DAYS_CREDIT_UPDATE_MEAN"]) + (((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) * (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * ((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) < (np.tanh((data["NEW_ANNUITY_TO_INCOME_RATIO"]))))*1.))))))))))) 
    v["i126"] = 0.031000*np.tanh(np.where(data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"] > -1, np.where(data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]>0, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"], (((np.where(data["NEW_DOC_IND_STD"] > -1, data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"], data["INSTAL_DAYS_ENTRY_PAYMENT_MEAN"] )) > (data["BURO_DAYS_CREDIT_MAX"]))*1.) ), data["NEW_DOC_IND_STD"] )) 
    v["i127"] = 0.015002*np.tanh((((-1.0*((((data["LIVE_CITY_NOT_WORK_CITY"]) * (np.where(data["BURO_DAYS_CREDIT_MIN"]>0, data["PREV_PRODUCT_COMBINATION_POS_other_with_interest_MEAN"], (-1.0*((((((data["PREV_PRODUCT_COMBINATION_POS_other_with_interest_MEAN"]) * 2.0)) * 2.0)))) ))))))) - (np.maximum(((data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"])), ((data["ORGANIZATION_TYPE_Military"])))))) 
    v["i128"] = 0.048040*np.tanh(((((np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]>0, data["ORGANIZATION_TYPE_Business_Entity_Type_3"], np.maximum(((np.maximum(((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_MAX"])), ((data["BURO_STATUS_1_MEAN_MEAN"]))))), ((data["PREV_PRODUCT_COMBINATION_POS_other_with_interest_MEAN"]))))), ((np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), ((data["ACTIVE_AMT_CREDIT_SUM_SUM"])))))) )) * 2.0)) * 2.0)) 
    v["i129"] = 0.031766*np.tanh(np.where(data["CC_AMT_CREDIT_LIMIT_ACTUAL_MAX"] > -1, data["PREV_CNT_PAYMENT_MEAN"], np.maximum(((((((data["NEW_SOURCES_PROD"]) - (data["BURO_STATUS_X_MEAN_MEAN"]))) * 2.0))), (((((data["BURO_AMT_CREDIT_SUM_SUM"]) < (np.where(data["NAME_INCOME_TYPE_Commercial_associate"]>0, data["BURO_AMT_CREDIT_SUM_SUM"], data["PREV_PRODUCT_COMBINATION_Card_Street_MEAN"] )))*1.)))) )) 
    v["i130"] = 0.042002*np.tanh(((data["NEW_RATIO_BURO_DAYS_CREDIT_MAX"]) * (np.where(data["PREV_CHANNEL_TYPE_Regional___Local_MEAN"]<0, np.where(data["FLOORSMIN_MODE"]<0, data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"], ((data["PREV_CHANNEL_TYPE_Regional___Local_MEAN"]) + (data["NEW_RATIO_BURO_DAYS_CREDIT_MEAN"])) ), ((data["POS_MONTHS_BALANCE_MEAN"]) - (data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"])) )))) 
    v["i131"] = 0.010397*np.tanh(np.where(data["NEW_RATIO_PREV_AMT_ANNUITY_MAX"]<0, np.where(data["EXT_SOURCE_3"] > -1, ((np.where(data["LANDAREA_MODE"]<0, (-1.0*((((data["CLOSED_DAYS_CREDIT_ENDDATE_MIN"]) * (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]))))), data["PREV_NAME_CONTRACT_TYPE_Revolving_loans_MEAN"] )) * 2.0), data["FLAG_WORK_PHONE"] ), data["NEW_RATIO_BURO_DAYS_CREDIT_VAR"] )) 
    v["i132"] = 0.049340*np.tanh(np.where(data["BURO_DAYS_CREDIT_MAX"]<0, data["INSTAL_PAYMENT_DIFF_MEAN"], (-1.0*(((((data["BURO_DAYS_CREDIT_MAX"]) < (np.tanh((((((((data["BURO_DAYS_CREDIT_MAX"]) > ((((data["BURO_DAYS_CREDIT_MAX"]) > (data["BURO_AMT_CREDIT_SUM_MEAN"]))*1.)))*1.)) > (data["BURO_AMT_CREDIT_SUM_MAX"]))*1.)))))*1.)))) )) 
    v["i133"] = 0.049700*np.tanh(np.where(data["AMT_INCOME_TOTAL"]>0, np.where(data["NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MIN"]>0, data["NAME_CONTRACT_TYPE_Cash_loans"], data["PREV_NAME_CLIENT_TYPE_Repeater_MEAN"] ), ((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) - (np.where(data["NEW_RATIO_PREV_AMT_CREDIT_MIN"]<0, data["PREV_NAME_TYPE_SUITE_Children_MEAN"], 2.0 )))) - (data["INSTAL_AMT_INSTALMENT_SUM"])) )) 
    v["i134"] = 0.049820*np.tanh(np.where(data["FLAG_DOCUMENT_8"]<0, (((np.where(((data["NAME_INCOME_TYPE_State_servant"]) + (data["PREV_NAME_YIELD_GROUP_low_normal_MEAN"]))<0, data["PREV_NAME_TYPE_SUITE_Spouse__partner_MEAN"], ((data["INSTAL_DPD_MAX"]) * 2.0) )) < (((data["INSTAL_DPD_MAX"]) * 2.0)))*1.), ((data["PREV_NAME_YIELD_GROUP_low_normal_MEAN"]) * 2.0) )) 
    v["i135"] = 0.034598*np.tanh(((data["PREV_AMT_DOWN_PAYMENT_MEAN"]) * (np.tanh((((data["EXT_SOURCE_3"]) - (((((data["NAME_FAMILY_STATUS_Separated"]) - (((((((((-1.0*((data["NAME_CONTRACT_TYPE_Cash_loans"])))) < (data["ACTIVE_DAYS_CREDIT_MIN"]))*1.)) * 2.0)) * 2.0)))) / 2.0)))))))) 
    v["i136"] = 0.049281*np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] > -1, np.where(data["BURO_STATUS_0_MEAN_MEAN"] > -1, data["CLOSED_AMT_ANNUITY_MEAN"], np.where(data["NEW_INC_BY_ORG"] > -1, data["INSTAL_AMT_PAYMENT_MIN"], data["NEW_CREDIT_TO_ANNUITY_RATIO"] ) ), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (((data["REFUSED_AMT_APPLICATION_MIN"]) + (data["BURO_STATUS_0_MEAN_MEAN"])))) )) 
    v["i137"] = 0.018362*np.tanh(np.where(data["POS_MONTHS_BALANCE_SIZE"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (-1.0*((np.maximum(((data["NAME_CONTRACT_TYPE_Cash_loans"])), ((np.where(np.where(data["APPROVED_AMT_APPLICATION_MIN"]<0, data["NEW_DOC_IND_AVG"], data["COMMONAREA_MEDI"] )<0, ((data["FLOORSMAX_AVG"]) * 2.0), data["PREV_NAME_YIELD_GROUP_high_MEAN"] ))))))) )) 
    v["i138"] = 0.000340*np.tanh(np.where(data["YEARS_BUILD_AVG"] > -1, data["NAME_HOUSING_TYPE_Municipal_apartment"], np.where(data["ORGANIZATION_TYPE_Police"]<0, ((data["WEEKDAY_APPR_PROCESS_START_SATURDAY"]) * (np.where(data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"]>0, (((data["REFUSED_APP_CREDIT_PERC_MAX"]) + (data["PREV_NAME_CONTRACT_STATUS_Canceled_MEAN"]))/2.0), data["NAME_HOUSING_TYPE_Municipal_apartment"] ))), data["YEARS_BUILD_AVG"] ) )) 
    v["i139"] = 0.047401*np.tanh((-1.0*(((((((data["ORGANIZATION_TYPE_Industry__type_9"]) + (((data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]) + (data["ORGANIZATION_TYPE_Military"]))))) + (np.maximum(((data["NEW_DOC_IND_AVG"])), ((((data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"]) + (((data["PREV_NAME_GOODS_CATEGORY_Furniture_MEAN"]) + (data["AMT_REQ_CREDIT_BUREAU_YEAR"])))))))))/2.0))))) 
    v["i140"] = 0.045253*np.tanh(((np.maximum(((np.where(data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"]>0, data["POS_COUNT"], ((data["APPROVED_AMT_CREDIT_MAX"]) - (data["INSTAL_AMT_INSTALMENT_SUM"])) ))), ((np.where(data["PREV_NAME_PRODUCT_TYPE_x_sell_MEAN"]<0, data["PREV_PRODUCT_COMBINATION_Cash_Street__middle_MEAN"], ((data["OCCUPATION_TYPE_Drivers"]) - (data["INSTAL_AMT_INSTALMENT_SUM"])) ))))) * 2.0)) 
    v["i141"] = 0.047007*np.tanh(((np.where(data["NEW_EXT_SOURCES_MEAN"]<0, ((np.where(data["NEW_EXT_SOURCES_MEAN"] > -1, data["NEW_EXT_SOURCES_MEAN"], data["CC_AMT_BALANCE_MAX"] )) - (((data["NEW_EXT_SOURCES_MEAN"]) + ((((data["WEEKDAY_APPR_PROCESS_START_MONDAY"]) > (data["NEW_EXT_SOURCES_MEAN"]))*1.))))), data["REGION_RATING_CLIENT_W_CITY"] )) * 2.0)) 
    v["i142"] = 0.048014*np.tanh(np.where(data["CC_AMT_PAYMENT_CURRENT_MEAN"]>0, ((data["ORGANIZATION_TYPE_Transport__type_3"]) + (data["NEW_RATIO_BURO_AMT_CREDIT_SUM_LIMIT_MEAN"])), (((data["ORGANIZATION_TYPE_Transport__type_3"]) < ((((((data["ACTIVE_DAYS_CREDIT_MAX"]) + (np.tanh((((data["NAME_HOUSING_TYPE_Rented_apartment"]) + (-3.0))))))/2.0)) / 2.0)))*1.) )) 
    v["i143"] = 0.010000*np.tanh((((((data["INSTAL_AMT_PAYMENT_MIN"]) > (np.where(data["INSTAL_DBD_SUM"]<0, (((data["NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MAX"]) < (np.where(data["CC_AMT_PAYMENT_TOTAL_CURRENT_SUM"] > -1, data["PREV_APP_CREDIT_PERC_MIN"], data["BURO_CREDIT_TYPE_Car_loan_MEAN"] )))*1.), ((data["APPROVED_AMT_ANNUITY_MIN"]) - (data["REG_CITY_NOT_LIVE_CITY"])) )))*1.)) * 2.0)) 
    v["i144"] = 0.048499*np.tanh(((((((data["CODE_GENDER"]) * (((data["DAYS_BIRTH"]) + (np.where(data["NAME_EDUCATION_TYPE_Secondary___secondary_special"] > -1, np.where(data["CODE_GENDER"] > -1, data["DAYS_BIRTH"], data["EXT_SOURCE_2"] ), data["CODE_GENDER"] )))))) * 2.0)) * 2.0)) 
    v["i145"] = 0.020720*np.tanh(np.where(data["PREV_NAME_GOODS_CATEGORY_Sport_and_Leisure_MEAN"]<0, np.where(data["PREV_NAME_PORTFOLIO_Cards_MEAN"]<0, (-1.0*((((data["NEW_PHONE_TO_EMPLOY_RATIO"]) + ((((data["PREV_CODE_REJECT_REASON_LIMIT_MEAN"]) < (data["APPROVED_CNT_PAYMENT_MEAN"]))*1.)))))), (((data["PREV_NAME_TYPE_SUITE_Other_B_MEAN"]) < (data["PREV_PRODUCT_COMBINATION_Cash_MEAN"]))*1.) ), data["NEW_PHONE_TO_EMPLOY_RATIO"] )) 
    v["i146"] = 0.049683*np.tanh((((np.where(data["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"] > -1, ((data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"]) * 2.0), np.where(data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"]<0, data["ORGANIZATION_TYPE_Transport__type_3"], ((data["REFUSED_AMT_CREDIT_MAX"]) * (data["NEW_INC_BY_ORG"])) ) )) + (((data["CC_AMT_TOTAL_RECEIVABLE_MEAN"]) - (data["CC_AMT_PAYMENT_TOTAL_CURRENT_SUM"]))))/2.0)) 
    v["i147"] = 0.048699*np.tanh(((data["PREV_WEEKDAY_APPR_PROCESS_START_THURSDAY_MEAN"]) * (np.where((((data["PREV_WEEKDAY_APPR_PROCESS_START_THURSDAY_MEAN"]) + (data["BURO_CREDIT_TYPE_Credit_card_MEAN"]))/2.0) > -1, data["PREV_NAME_SELLER_INDUSTRY_XNA_MEAN"], ((data["PREV_AMT_APPLICATION_MAX"]) * (np.where(data["PREV_WEEKDAY_APPR_PROCESS_START_THURSDAY_MEAN"]>0, data["PREV_AMT_APPLICATION_MAX"], data["REGION_POPULATION_RELATIVE"] ))) )))) 
    v["i148"] = 0.041840*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]>0, ((((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) - (data["ACTIVE_AMT_CREDIT_SUM_MAX"]))) - (data["ACTIVE_AMT_CREDIT_SUM_MAX"]))) * 2.0), np.maximum(((np.maximum(((data["ORGANIZATION_TYPE_Transport__type_3"])), ((data["ACTIVE_AMT_CREDIT_SUM_MEAN"]))))), ((data["CC_AMT_RECEIVABLE_PRINCIPAL_VAR"]))) )) 
    v["i149"] = 0.049911*np.tanh(np.minimum((((((((data["NEW_EXT_SOURCES_MEAN"]) > (data["PREV_CODE_REJECT_REASON_HC_MEAN"]))*1.)) - (data["YEARS_BUILD_AVG"])))), (((((((-1.0*(((((data["NEW_EXT_SOURCES_MEAN"]) > (((1.0) - (data["ORGANIZATION_TYPE_Industry__type_9"]))))*1.))))) * 2.0)) * 2.0))))) 
    return Output(v.sum(axis=1))

def GP2(data):
    v = pd.DataFrame()
    v["i0"] = 0.010000*np.tanh(((((((((np.tanh((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))) - (((np.where((((data["NEW_EXT_SOURCES_MEAN"]) < (data["NEW_DOC_IND_AVG"]))*1.)>0, ((data["NEW_EXT_SOURCES_MEAN"]) * 2.0), data["NEW_EXT_SOURCES_MEAN"] )) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) 
    v["i1"] = 0.034000*np.tanh((-1.0*((((((((((((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)) - (np.where(np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"])), ((data["NEW_EXT_SOURCES_MEAN"])))>0, data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"], data["NAME_INCOME_TYPE_Working"] )))) * 2.0)) * 2.0)) * 2.0)) * 2.0))))) 
    v["i2"] = 0.048500*np.tanh(((data["OCCUPATION_TYPE_Drivers"]) + (((data["FLAG_DOCUMENT_3"]) + (((((data["NEW_EXT_SOURCES_MEAN"]) + (((np.tanh((data["CC_AMT_TOTAL_RECEIVABLE_MEAN"]))) + ((((((-1.0*((data["NEW_EXT_SOURCES_MEAN"])))) * 2.0)) * 2.0)))))) * 2.0)))))) 
    v["i3"] = 0.030000*np.tanh(((((((((((data["NEW_CREDIT_TO_GOODS_RATIO"]) - ((-1.0*((np.tanh((data["CC_AMT_DRAWINGS_ATM_CURRENT_VAR"])))))))) - (((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)) * 2.0)))) - (data["CODE_GENDER"]))) * 2.0)) * 2.0)) 
    v["i4"] = 0.049000*np.tanh(((((((((((np.tanh((((np.tanh((data["DAYS_EMPLOYED"]))) - (data["EXT_SOURCE_3"]))))) - ((((data["EXT_SOURCE_2"]) < (data["NAME_EDUCATION_TYPE_Higher_education"]))*1.)))) - (data["EXT_SOURCE_2"]))) * 2.0)) * 2.0)) * 2.0)) 
    v["i5"] = 0.017800*np.tanh((((((((-1.0*((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0))))) + (((data["NEW_CREDIT_TO_GOODS_RATIO"]) + ((((-1.0*((np.maximum(((data["CODE_GENDER"])), ((data["APPROVED_APP_CREDIT_PERC_MAX"]))))))) - (data["NEW_EXT_SOURCES_MEAN"]))))))) * 2.0)) * 2.0)) 
    v["i6"] = 0.029394*np.tanh(((((((((((np.tanh((data["NEW_CREDIT_TO_GOODS_RATIO"]))) - (data["NEW_EXT_SOURCES_MEAN"]))) - (((((((data["NEW_EXT_SOURCES_MEAN"]) > (data["DAYS_EMPLOYED"]))*1.)) > (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))*1.)))) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)) * 2.0)) 
    v["i7"] = 0.035000*np.tanh(((((((((np.tanh(((((data["DAYS_EMPLOYED"]) + (((np.tanh((((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]) + (data["PREV_NAME_PRODUCT_TYPE_walk_in_MEAN"]))))) + (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))))/2.0)))) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)) * 2.0)) * 2.0)) 
    v["i8"] = 0.034200*np.tanh(((((np.minimum((((((((((-1.0*((data["NEW_EXT_SOURCES_MEAN"])))) * 2.0)) + (data["NEW_CREDIT_TO_GOODS_RATIO"]))) * 2.0))), (((((((-1.0*((data["NEW_EXT_SOURCES_MEAN"])))) - (data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))) * 2.0))))) * 2.0)) * 2.0)) 
    v["i9"] = 0.030000*np.tanh(((((((np.maximum(((np.minimum(((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"])), ((np.tanh((data["DAYS_EMPLOYED"]))))))), ((((np.tanh((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))) * 2.0))))) + (data["NEW_CREDIT_TO_GOODS_RATIO"]))) - (((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)))) * 2.0)) 
    v["i10"] = 0.041000*np.tanh(((((((((((np.tanh((((data["NEW_CREDIT_TO_GOODS_RATIO"]) + (data["DAYS_EMPLOYED"]))))) - (data["EXT_SOURCE_2"]))) - (np.tanh((data["EXT_SOURCE_3"]))))) * 2.0)) + (np.tanh((data["REFUSED_DAYS_DECISION_MAX"]))))) * 2.0)) 
    v["i11"] = 0.049000*np.tanh(((((((((np.tanh((data["CC_AMT_INST_MIN_REGULARITY_MEAN"]))) - (((data["APPROVED_APP_CREDIT_PERC_MAX"]) - (data["PREV_NAME_CLIENT_TYPE_New_MEAN"]))))) - (((data["APPROVED_AMT_CREDIT_MIN"]) - (data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))))) - (data["NEW_EXT_SOURCES_MEAN"]))) - (data["NEW_EXT_SOURCES_MEAN"]))) 
    v["i12"] = 0.049901*np.tanh(((((((((np.maximum(((data["NEW_CREDIT_TO_GOODS_RATIO"])), ((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"])))) + (data["DEF_30_CNT_SOCIAL_CIRCLE"]))) + (np.tanh((data["CC_CNT_DRAWINGS_CURRENT_MAX"]))))) - (((data["CODE_GENDER"]) + (((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)))))) * 2.0)) 
    v["i13"] = 0.049040*np.tanh(((((((np.maximum(((np.tanh((data["PREV_CNT_PAYMENT_MEAN"])))), ((data["CC_CNT_DRAWINGS_CURRENT_MAX"])))) + (((np.minimum(((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"])), ((np.tanh((data["DAYS_EMPLOYED"])))))) - (((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)))))) * 2.0)) * 2.0)) 
    v["i14"] = 0.044680*np.tanh(((((data["NEW_DOC_IND_KURT"]) + (data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))) + (((((((((((data["DAYS_EMPLOYED"]) + (data["INSTAL_PAYMENT_DIFF_MAX"]))) - (data["EXT_SOURCE_2"]))) * 2.0)) - (data["PREV_RATE_DOWN_PAYMENT_MAX"]))) - (data["PREV_AMT_ANNUITY_MEAN"]))))) 
    v["i15"] = 0.047043*np.tanh(((((((((np.tanh((np.where(data["PREV_APP_CREDIT_PERC_MEAN"]>0, data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"], ((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) + ((-1.0*((data["EXT_SOURCE_3"]))))) )))) + ((-1.0*((data["NEW_EXT_SOURCES_MEAN"])))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i16"] = 0.049100*np.tanh((((((((-1.0*((((data["NEW_EXT_SOURCES_MEAN"]) + (np.maximum(((data["NEW_CAR_TO_BIRTH_RATIO"])), ((((data["CODE_GENDER"]) - (np.maximum(((data["PREV_NAME_PRODUCT_TYPE_walk_in_MEAN"])), ((data["NEW_CREDIT_TO_GOODS_RATIO"])))))))))))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i17"] = 0.048520*np.tanh((((((data["NEW_CREDIT_TO_GOODS_RATIO"]) + (((((np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), (((((-1.0*((((((1.28758943080902100)) + (data["NEW_EXT_SOURCES_MEAN"]))/2.0))))) - (np.tanh((data["EXT_SOURCE_3"])))))))) * 2.0)) * 2.0)))/2.0)) * 2.0)) 
    v["i18"] = 0.049320*np.tanh(((data["DEF_60_CNT_SOCIAL_CIRCLE"]) + (((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) + (((((((((np.tanh((data["NEW_ANNUITY_TO_INCOME_RATIO"]))) - (data["NEW_EXT_SOURCES_MEAN"]))) - (np.tanh((np.tanh((data["BURO_CREDIT_ACTIVE_Closed_MEAN"]))))))) * 2.0)) * 2.0)))))) 
    v["i19"] = 0.049995*np.tanh(((((((data["PREV_CNT_PAYMENT_MEAN"]) - (((data["NEW_EXT_SOURCES_MEAN"]) - (((data["NEW_CREDIT_TO_GOODS_RATIO"]) + (((data["PREV_CNT_PAYMENT_SUM"]) + (((((data["INSTAL_PAYMENT_DIFF_MAX"]) - (data["POS_MONTHS_BALANCE_SIZE"]))) * 2.0)))))))))) * 2.0)) * 2.0)) 
    v["i20"] = 0.049870*np.tanh(((((((np.maximum(((data["DAYS_EMPLOYED"])), ((data["NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MIN"])))) - (data["NEW_EXT_SOURCES_MEAN"]))) - (data["NAME_EDUCATION_TYPE_Higher_education"]))) + (((((data["INSTAL_PAYMENT_DIFF_MAX"]) + (((data["INSTAL_PAYMENT_DIFF_MAX"]) - (data["POS_MONTHS_BALANCE_SIZE"]))))) * 2.0)))) 
    v["i21"] = 0.049513*np.tanh(((((((np.where(data["APPROVED_AMT_DOWN_PAYMENT_MAX"]>0, -2.0, (((-1.0*((data["EXT_SOURCE_3"])))) - (np.where(data["CODE_GENDER"]>0, data["CODE_GENDER"], data["NEW_SOURCES_PROD"] ))) )) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)) * 2.0)) 
    v["i22"] = 0.049435*np.tanh(((((((data["PREV_DAYS_DECISION_MIN"]) + (data["PREV_CNT_PAYMENT_MEAN"]))) + (((data["NEW_DOC_IND_KURT"]) + (((((data["PREV_NAME_YIELD_GROUP_high_MEAN"]) - (data["CODE_GENDER"]))) + (data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))))))) - (data["NEW_EXT_SOURCES_MEAN"]))) 
    v["i23"] = 0.049304*np.tanh(((((np.where(data["NEW_SOURCES_PROD"] > -1, np.minimum(((data["REGION_RATING_CLIENT_W_CITY"])), ((data["CC_AMT_TOTAL_RECEIVABLE_MEAN"]))), ((data["DAYS_BIRTH"]) + (np.maximum(((data["PREV_NAME_PRODUCT_TYPE_walk_in_MEAN"])), ((np.maximum(((data["CC_AMT_TOTAL_RECEIVABLE_MEAN"])), ((data["NEW_CREDIT_TO_GOODS_RATIO"])))))))) )) * 2.0)) * 2.0)) 
    v["i24"] = 0.049200*np.tanh(((((((data["NEW_DOC_IND_KURT"]) + (((np.where(data["INSTAL_AMT_PAYMENT_MIN"]>0, data["REFUSED_CNT_PAYMENT_SUM"], np.maximum(((data["REFUSED_CNT_PAYMENT_SUM"])), ((data["PREV_DAYS_DECISION_MIN"]))) )) - (((data["APPROVED_HOUR_APPR_PROCESS_START_MAX"]) + (data["NEW_EXT_SOURCES_MEAN"]))))))) * 2.0)) * 2.0)) 
    v["i25"] = 0.049792*np.tanh(((data["REGION_RATING_CLIENT_W_CITY"]) + (((np.where(data["INSTAL_DPD_MEAN"]<0, np.where(data["EXT_SOURCE_1"] > -1, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], np.where(data["NEW_CAR_TO_BIRTH_RATIO"] > -1, data["REFUSED_DAYS_DECISION_MAX"], data["DAYS_EMPLOYED"] ) ), (-1.0*((data["NEW_CAR_TO_BIRTH_RATIO"]))) )) * 2.0)))) 
    v["i26"] = 0.049621*np.tanh(((((((np.where(data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]<0, ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (data["CODE_GENDER"]))) - (data["NEW_EXT_SOURCES_MEAN"])), data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"] )) - (data["INSTAL_DBD_SUM"]))) - (data["APPROVED_AMT_ANNUITY_MEAN"]))) + (data["PREV_CNT_PAYMENT_MEAN"]))) 
    v["i27"] = 0.049640*np.tanh(((((((((((data["APPROVED_CNT_PAYMENT_MEAN"]) + (data["PREV_NAME_CLIENT_TYPE_New_MEAN"]))) + (data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))) + (np.where(data["CC_CNT_INSTALMENT_MATURE_CUM_MEAN"] > -1, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], np.tanh((data["DAYS_EMPLOYED"])) )))) - (data["NAME_EDUCATION_TYPE_Higher_education"]))) * 2.0)) 
    v["i28"] = 0.049849*np.tanh(((np.where(data["POS_SK_DPD_DEF_MAX"]>0, (5.0), ((((((np.where(data["AMT_CREDIT"] > -1, data["PREV_NAME_YIELD_GROUP_high_MEAN"], data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"] )) + (data["DEF_30_CNT_SOCIAL_CIRCLE"]))) + (data["NEW_CREDIT_TO_GOODS_RATIO"]))) - (data["POS_MONTHS_BALANCE_SIZE"])) )) * 2.0)) 
    v["i29"] = 0.049352*np.tanh(((((((np.where(data["NEW_CAR_TO_EMPLOY_RATIO"]>0, data["CC_AMT_PAYMENT_CURRENT_MIN"], ((data["INSTAL_PAYMENT_DIFF_MAX"]) - (data["APPROVED_AMT_ANNUITY_MEAN"])) )) - (np.where(data["BURO_CREDIT_ACTIVE_Closed_MEAN"]<0, data["YEARS_BUILD_AVG"], data["CODE_GENDER"] )))) * 2.0)) + (data["NEW_DOC_IND_KURT"]))) 
    v["i30"] = 0.049816*np.tanh((((((((np.where(data["INSTAL_AMT_PAYMENT_MIN"]>0, data["REFUSED_DAYS_DECISION_MAX"], data["APPROVED_DAYS_DECISION_MIN"] )) + (data["INSTAL_PAYMENT_DIFF_MEAN"]))/2.0)) + (((data["APPROVED_CNT_PAYMENT_MEAN"]) + (np.where(data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"] > -1, data["BURO_DAYS_CREDIT_MEAN"], data["DAYS_LAST_PHONE_CHANGE"] )))))) * 2.0)) 
    v["i31"] = 0.049908*np.tanh(((np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]>0, 2.0, np.where(np.minimum(((((data["INSTAL_AMT_INSTALMENT_MAX"]) * 2.0))), ((((data["NEW_EXT_SOURCES_MEAN"]) / 2.0)))) > -1, data["NEW_RATIO_BURO_AMT_ANNUITY_MEAN"], ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (data["EXT_SOURCE_3"])) ) )) * 2.0)) 
    v["i32"] = 0.049859*np.tanh(((data["DAYS_ID_PUBLISH"]) + (((((((data["PREV_CNT_PAYMENT_MEAN"]) - (data["POS_MONTHS_BALANCE_SIZE"]))) + (((((((data["INSTAL_PAYMENT_DIFF_MEAN"]) - (data["INSTAL_AMT_PAYMENT_MIN"]))) - (data["INSTAL_AMT_PAYMENT_MIN"]))) - (data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"]))))) * 2.0)))) 
    v["i33"] = 0.049192*np.tanh(((((np.where(data["APPROVED_AMT_ANNUITY_MEAN"]<0, ((data["DAYS_LAST_PHONE_CHANGE"]) + (((data["AMT_ANNUITY"]) + (data["NAME_INCOME_TYPE_Working"])))), data["PREV_CNT_PAYMENT_MEAN"] )) + (data["REGION_RATING_CLIENT_W_CITY"]))) + (((data["REG_CITY_NOT_LIVE_CITY"]) - (data["CODE_GENDER"]))))) 
    v["i34"] = 0.049730*np.tanh(((((np.where(data["NEW_SOURCES_PROD"] > -1, data["BURO_STATUS_1_MEAN_MEAN"], np.where(data["DAYS_BIRTH"] > -1, ((np.where(data["AMT_GOODS_PRICE"] > -1, (-1.0*((data["NEW_CAR_TO_BIRTH_RATIO"]))), data["CC_CNT_DRAWINGS_CURRENT_MEAN"] )) * 2.0), data["BURO_STATUS_1_MEAN_MEAN"] ) )) * 2.0)) * 2.0)) 
    v["i35"] = 0.050000*np.tanh(((((np.maximum(((data["APPROVED_CNT_PAYMENT_MEAN"])), ((data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"])))) + (((((np.maximum(((data["ACTIVE_DAYS_CREDIT_MAX"])), ((data["DEF_60_CNT_SOCIAL_CIRCLE"])))) + (((data["INSTAL_DPD_MEAN"]) * ((10.86021709442138672)))))) - (data["OCCUPATION_TYPE_Core_staff"]))))) * 2.0)) 
    v["i36"] = 0.048505*np.tanh((((7.0)) * (np.where(data["INSTAL_DPD_MEAN"]>0, (1.50285637378692627), ((np.where(data["NAME_FAMILY_STATUS_Married"]>0, np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]>0, data["NAME_EDUCATION_TYPE_Secondary___secondary_special"], data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"] ), (-1.0*((data["FLOORSMAX_MEDI"]))) )) * 2.0) )))) 
    v["i37"] = 0.049500*np.tanh(((((np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"]>0, data["CC_AMT_RECIVABLE_MIN"], np.where(data["CC_AMT_INST_MIN_REGULARITY_VAR"] > -1, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], np.where(data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]>0, data["CC_AMT_INST_MIN_REGULARITY_VAR"], ((data["AMT_ANNUITY"]) - (data["APPROVED_AMT_ANNUITY_MEAN"])) ) ) )) * 2.0)) * 2.0)) 
    v["i38"] = 0.048958*np.tanh(((((data["REG_CITY_NOT_LIVE_CITY"]) + (data["PREV_CNT_PAYMENT_MEAN"]))) + (((((((data["INSTAL_AMT_INSTALMENT_MAX"]) - (data["PREV_NAME_PAYMENT_TYPE_Cash_through_the_bank_MEAN"]))) - (data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"]))) - (np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"] > -1, data["PREV_AMT_ANNUITY_MEAN"], data["NEW_CAR_TO_BIRTH_RATIO"] )))))) 
    v["i39"] = 0.049650*np.tanh(((((((np.maximum(((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])), ((np.maximum(((np.where(data["POS_SK_DPD_DEF_MAX"]<0, data["CC_AMT_BALANCE_MIN"], data["INSTAL_PAYMENT_DIFF_MEAN"] ))), ((((((data["BURO_CREDIT_TYPE_Microloan_MEAN"]) * 2.0)) * 2.0)))))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i40"] = 0.049700*np.tanh(((data["REGION_RATING_CLIENT_W_CITY"]) + (((((np.where(((data["POS_MONTHS_BALANCE_SIZE"]) * 2.0) > -1, data["PREV_CNT_PAYMENT_MEAN"], np.where(data["NEW_RATIO_BURO_DAYS_CREDIT_MAX"]>0, -3.0, ((data["CLOSED_DAYS_CREDIT_VAR"]) * (data["INSTAL_COUNT"])) ) )) * 2.0)) * 2.0)))) 
    v["i41"] = 0.049099*np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, ((((((data["AMT_ANNUITY"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (np.maximum(((data["FLAG_WORK_PHONE"])), (((((data["REGION_RATING_CLIENT_W_CITY"]) > (data["FLAG_WORK_PHONE"]))*1.))))))))) * 2.0)) * 2.0), data["REFUSED_AMT_DOWN_PAYMENT_MIN"] )) 
    v["i42"] = 0.049706*np.tanh(((((np.where(data["NEW_EXT_SOURCES_MEAN"] > -1, np.where(data["NEW_EXT_SOURCES_MEAN"]<0, data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"], ((data["REGION_RATING_CLIENT_W_CITY"]) - (data["EXT_SOURCE_1"])) ), np.maximum(((data["ACTIVE_DAYS_CREDIT_MEAN"])), (((-1.0*((data["CODE_GENDER"])))))) )) * 2.0)) * 2.0)) 
    v["i43"] = 0.048092*np.tanh(((((((data["INSTAL_PAYMENT_DIFF_MEAN"]) - (((data["APPROVED_AMT_ANNUITY_MAX"]) + (((data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]) - (data["INSTAL_AMT_PAYMENT_MAX"]))))))) * 2.0)) - (np.maximum(((np.maximum(((data["PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN"])), ((data["NEW_EMPLOY_TO_BIRTH_RATIO"]))))), ((data["PREV_NAME_YIELD_GROUP_low_normal_MEAN"])))))) 
    v["i44"] = 0.049950*np.tanh(((((((((((data["AMT_ANNUITY"]) * 2.0)) - (data["PREV_AMT_ANNUITY_MEAN"]))) - (((((data["PREV_NAME_CONTRACT_TYPE_Consumer_loans_MEAN"]) + (data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"]))) - (data["DEF_60_CNT_SOCIAL_CIRCLE"]))))) - (data["INSTAL_DBD_SUM"]))) * 2.0)) 
    v["i45"] = 0.049975*np.tanh(((((((((data["PREV_CODE_REJECT_REASON_SCOFR_MEAN"]) - (data["INSTAL_DBD_SUM"]))) * 2.0)) + (((data["APPROVED_CNT_PAYMENT_SUM"]) - (np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]<0, data["NAME_FAMILY_STATUS_Married"], ((data["BURO_MONTHS_BALANCE_SIZE_MEAN"]) * 2.0) )))))) * 2.0)) 
    v["i46"] = 0.047492*np.tanh(((np.where(data["AMT_ANNUITY"]<0, np.where(data["DAYS_BIRTH"]>0, (-1.0*((((data["DAYS_BIRTH"]) + (data["EXT_SOURCE_1"]))))), data["CC_CNT_DRAWINGS_CURRENT_MAX"] ), ((((((data["NEW_CREDIT_TO_GOODS_RATIO"]) * 2.0)) * 2.0)) * 2.0) )) * 2.0)) 
    v["i47"] = 0.049920*np.tanh(((((((((((data["REGION_RATING_CLIENT_W_CITY"]) - (data["NEW_EMPLOY_TO_BIRTH_RATIO"]))) - (np.maximum(((np.maximum(((data["PREV_RATE_DOWN_PAYMENT_MEAN"])), ((np.maximum(((data["NAME_INCOME_TYPE_State_servant"])), ((data["FLOORSMAX_MEDI"])))))))), ((data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"])))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i48"] = 0.048000*np.tanh(((((((np.where(data["ACTIVE_DAYS_CREDIT_MAX"]>0, data["BURO_AMT_CREDIT_SUM_DEBT_SUM"], data["INSTAL_PAYMENT_DIFF_MAX"] )) * 2.0)) * 2.0)) + (((np.where(data["BURO_DAYS_CREDIT_VAR"] > -1, data["ACTIVE_DAYS_CREDIT_MAX"], data["PREV_PRODUCT_COMBINATION_POS_mobile_with_interest_MEAN"] )) + (data["ORGANIZATION_TYPE_Self_employed"]))))) 
    v["i49"] = 0.048000*np.tanh(((((np.where(data["NEW_CAR_TO_BIRTH_RATIO"]>0, -3.0, (((((((((data["NEW_EXT_SOURCES_MEAN"]) > (data["PREV_NAME_GOODS_CATEGORY_Furniture_MEAN"]))*1.)) + (data["FLAG_WORK_PHONE"]))) * 2.0)) * 2.0) )) - (data["OCCUPATION_TYPE_Core_staff"]))) - (data["NEW_EXT_SOURCES_MEAN"]))) 
    v["i50"] = 0.049213*np.tanh(((np.maximum(((data["APPROVED_CNT_PAYMENT_MEAN"])), ((np.maximum(((data["NEW_SCORES_STD"])), ((np.maximum(((data["REG_CITY_NOT_LIVE_CITY"])), ((np.maximum(((data["PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN"])), ((data["CC_CNT_DRAWINGS_CURRENT_MAX"]))))))))))))) - (((np.maximum(((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])), ((data["CODE_GENDER"])))) * 2.0)))) 
    v["i51"] = 0.047502*np.tanh(((np.maximum(((data["INSTAL_PAYMENT_DIFF_MEAN"])), ((np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))))))) - (np.maximum(((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])), (((((-1.0*((((data["AMT_ANNUITY"]) - (data["NEW_DOC_IND_AVG"])))))) + (data["NAME_INCOME_TYPE_State_servant"])))))))) 
    v["i52"] = 0.042032*np.tanh(np.where(data["EXT_SOURCE_1"] > -1, (-1.0*((data["DAYS_BIRTH"]))), ((((data["DAYS_BIRTH"]) - (data["FLAG_DOCUMENT_8"]))) - (np.where((-1.0*((data["DAYS_BIRTH"])))<0, data["OCCUPATION_TYPE_High_skill_tech_staff"], data["CODE_GENDER"] ))) )) 
    v["i53"] = 0.049812*np.tanh(((np.where(((data["INSTAL_COUNT"]) * 2.0) > -1, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], ((data["LIVINGAREA_MEDI"]) * (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"])) )) - (np.where(data["BURO_CREDIT_TYPE_Microloan_MEAN"]>0, data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"], ((data["PREV_AMT_ANNUITY_MEAN"]) - (data["PREV_AMT_APPLICATION_MAX"])) )))) 
    v["i54"] = 0.049694*np.tanh(((np.where(data["PREV_AMT_ANNUITY_MAX"]<0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (((data["NONLIVINGAPARTMENTS_MODE"]) - (data["PREV_NAME_TYPE_SUITE_nan_MEAN"])))), np.where(data["PREV_NAME_TYPE_SUITE_nan_MEAN"] > -1, ((data["APPROVED_CNT_PAYMENT_MEAN"]) + (data["OCCUPATION_TYPE_Drivers"])), data["PREV_NAME_TYPE_SUITE_nan_MEAN"] ) )) * 2.0)) 
    v["i55"] = 0.049701*np.tanh(((((np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] > -1, ((((((((data["BURO_AMT_CREDIT_SUM_DEBT_SUM"]) - (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))) + (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))) * 2.0)) - (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), data["INSTAL_PAYMENT_DIFF_MEAN"] )) * 2.0)) * 2.0)) 
    v["i56"] = 0.049080*np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_MEAN"] > -1, ((((np.where(data["BURO_CREDIT_ACTIVE_Active_MEAN"]>0, data["ACTIVE_DAYS_CREDIT_MAX"], np.where(data["ACTIVE_AMT_CREDIT_SUM_MEAN"]>0, data["ACTIVE_DAYS_CREDIT_MAX"], data["BURO_CREDIT_ACTIVE_Active_MEAN"] ) )) * 2.0)) * 2.0), np.maximum(((data["EXT_SOURCE_3"])), ((data["REFUSED_DAYS_DECISION_MAX"]))) )) 
    v["i57"] = 0.049955*np.tanh(((np.where(data["PREV_AMT_APPLICATION_MAX"]<0, np.minimum(((data["AMT_ANNUITY"])), ((data["NEW_DOC_IND_KURT"]))), data["APPROVED_CNT_PAYMENT_SUM"] )) + (((((np.minimum(((data["REGION_POPULATION_RELATIVE"])), ((data["AMT_ANNUITY"])))) - (data["APPROVED_AMT_ANNUITY_MEAN"]))) - (data["APPROVED_AMT_ANNUITY_MEAN"]))))) 
    v["i58"] = 0.048588*np.tanh(((np.where(((data["NEW_DOC_IND_KURT"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) > -1, (((((-1.0*((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0))))) - (data["WEEKDAY_APPR_PROCESS_START_SATURDAY"]))) * 2.0), np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((data["INSTAL_PAYMENT_DIFF_SUM"]))) )) * 2.0)) 
    v["i59"] = 0.039981*np.tanh(((np.where(data["NEW_CAR_TO_EMPLOY_RATIO"] > -1, data["REFUSED_DAYS_DECISION_MAX"], ((np.maximum(((np.maximum(((data["DEF_30_CNT_SOCIAL_CIRCLE"])), ((((data["APPROVED_CNT_PAYMENT_MEAN"]) + (data["BURO_CREDIT_TYPE_Microloan_MEAN"]))))))), ((data["DAYS_ID_PUBLISH"])))) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])) )) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))) 
    v["i60"] = 0.035776*np.tanh(((((np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"]<0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], -2.0 )) + (np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((data["INSTAL_DBD_MAX"])))))) - (np.where(data["NEW_RATIO_BURO_DAYS_CREDIT_MAX"]<0, data["PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN"], (-1.0*((-2.0))) )))) 
    v["i61"] = 0.049702*np.tanh(((((((data["ORGANIZATION_TYPE_Construction"]) + (np.maximum(((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])), ((np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((np.where(np.maximum(((data["INSTAL_DPD_MEAN"])), ((data["CC_AMT_BALANCE_MAX"])))<0, data["PREV_CNT_PAYMENT_SUM"], data["INSTAL_DAYS_ENTRY_PAYMENT_MEAN"] )))))))))) * 2.0)) * 2.0)) 
    v["i62"] = 0.049299*np.tanh(((((((np.where((-1.0*((data["NEW_EXT_SOURCES_MEAN"]))) > -1, ((np.where(data["CC_AMT_TOTAL_RECEIVABLE_MEAN"] > -1, data["NEW_EXT_SOURCES_MEAN"], (((data["NEW_EXT_SOURCES_MEAN"]) > (data["PREV_AMT_DOWN_PAYMENT_MAX"]))*1.) )) * 2.0), data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"] )) * 2.0)) * 2.0)) * 2.0)) 
    v["i63"] = 0.033400*np.tanh(((data["WALLSMATERIAL_MODE_Stone__brick"]) + (((((data["POS_SK_DPD_DEF_MAX"]) - (data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"]))) + ((((((data["APPROVED_APP_CREDIT_PERC_MIN"]) < (data["INSTAL_PAYMENT_DIFF_MEAN"]))*1.)) - (np.maximum(((data["INSTAL_AMT_PAYMENT_SUM"])), ((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])))))))))) 
    v["i64"] = 0.049672*np.tanh(((((((np.maximum(((data["INSTAL_DPD_MEAN"])), ((np.maximum(((data["ORGANIZATION_TYPE_Construction"])), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"]))))))) * 2.0)) * 2.0)) - (((((data["NEW_PHONE_TO_BIRTH_RATIO"]) - (np.tanh((data["PREV_CHANNEL_TYPE_Contact_center_MEAN"]))))) - (data["NAME_FAMILY_STATUS_Separated"]))))) 
    v["i65"] = 0.048401*np.tanh(((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) - (((((data["POS_MONTHS_BALANCE_SIZE"]) - (np.maximum(((data["APPROVED_CNT_PAYMENT_SUM"])), ((data["POS_SK_DPD_MAX"])))))) - (np.maximum(((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"])), ((np.maximum(((data["INSTAL_PAYMENT_DIFF_MEAN"])), ((data["NAME_HOUSING_TYPE_Municipal_apartment"]))))))))))) 
    v["i66"] = 0.036240*np.tanh((((((data["REGION_RATING_CLIENT_W_CITY"]) + (((np.where(data["NEW_EXT_SOURCES_MEAN"]>0, data["NAME_EDUCATION_TYPE_Secondary___secondary_special"], ((((((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_SUM"])), ((data["BURO_AMT_CREDIT_SUM_DEBT_SUM"])))) * 2.0)) - (data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"]))) * 2.0) )) * 2.0)))/2.0)) * 2.0)) 
    v["i67"] = 0.049448*np.tanh(np.where(data["NEW_CREDIT_TO_GOODS_RATIO"] > -1, np.minimum(((data["NEW_CREDIT_TO_GOODS_RATIO"])), ((((data["ACTIVE_AMT_CREDIT_SUM_MAX"]) * (np.where(data["PREV_AMT_ANNUITY_MIN"]>0, data["CODE_GENDER"], data["INSTAL_AMT_INSTALMENT_SUM"] )))))), np.where(data["PREV_AMT_ANNUITY_MIN"]>0, data["PREV_NAME_CONTRACT_STATUS_Approved_MEAN"], data["CODE_GENDER"] ) )) 
    v["i68"] = 0.045202*np.tanh(((((data["ORGANIZATION_TYPE_Self_employed"]) + (((data["INSTAL_DBD_MAX"]) - (((data["APPROVED_AMT_CREDIT_MIN"]) + (np.maximum(((data["NEW_PHONE_TO_EMPLOY_RATIO"])), ((data["BURO_STATUS_0_MEAN_MEAN"])))))))))) - (np.maximum(((data["APPROVED_AMT_CREDIT_MIN"])), ((data["BURO_AMT_CREDIT_SUM_MEAN"])))))) 
    v["i69"] = 0.050000*np.tanh(np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"]>0, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"], np.where(data["BURO_CREDIT_TYPE_Credit_card_MEAN"] > -1, np.where(data["PREV_NAME_CONTRACT_TYPE_Consumer_loans_MEAN"] > -1, ((data["EXT_SOURCE_3"]) - (data["NEW_EXT_SOURCES_MEAN"])), data["BURO_AMT_CREDIT_SUM_DEBT_SUM"] ), ((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)) * 2.0) ) )) 
    v["i70"] = 0.047500*np.tanh((-1.0*((((((np.maximum(((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])), ((((np.maximum(((((data["FLAG_DOCUMENT_18"]) * 2.0))), ((np.where(data["AMT_GOODS_PRICE"] > -1, data["CC_AMT_PAYMENT_CURRENT_SUM"], (-1.0*((data["CC_AMT_BALANCE_MEAN"]))) ))))) * 2.0))))) * 2.0)) * 2.0))))) 
    v["i71"] = 0.049392*np.tanh(((((((((((((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_SUM"])), ((data["BURO_AMT_CREDIT_SUM_DEBT_SUM"])))) - (np.where(data["DAYS_ID_PUBLISH"] > -1, data["BURO_AMT_CREDIT_SUM_MEAN"], 0.318310 )))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i72"] = 0.049200*np.tanh(np.where(data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"] > -1, np.where(data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]<0, data["CC_AMT_BALANCE_MEAN"], (-1.0*((data["INSTAL_PAYMENT_DIFF_MAX"]))) ), np.maximum(((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"])), ((((np.maximum(((data["FLAG_WORK_PHONE"])), ((data["INSTAL_PAYMENT_DIFF_MAX"])))) + (data["DAYS_REGISTRATION"]))))) )) 
    v["i73"] = 0.050000*np.tanh(((data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__high_MEAN"]) + (np.where(data["CLOSED_DAYS_CREDIT_MAX"]>0, data["BURO_CREDIT_TYPE_Microloan_MEAN"], ((data["REFUSED_AMT_ANNUITY_MAX"]) * ((((((data["PREV_NAME_GOODS_CATEGORY_Computers_MEAN"]) + (data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"]))/2.0)) - (data["EXT_SOURCE_3"])))) )))) 
    v["i74"] = 0.049580*np.tanh(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((((((data["REGION_RATING_CLIENT_W_CITY"]) + (np.where(data["CODE_GENDER"] > -1, data["REGION_POPULATION_RELATIVE"], (((-1.0*((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["BURO_CREDIT_TYPE_Mortgage_MEAN"])))))) * 2.0) )))) * 2.0)) * 2.0)))) 
    v["i75"] = 0.047500*np.tanh(np.where(data["NEW_DOC_IND_STD"]<0, data["PREV_NAME_TYPE_SUITE_nan_MEAN"], ((((data["CC_AMT_RECIVABLE_MAX"]) - (data["CC_AMT_PAYMENT_CURRENT_SUM"]))) + (np.where(data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"] > -1, data["EXT_SOURCE_2"], ((data["INSTAL_COUNT"]) * (data["EXT_SOURCE_2"])) ))) )) 
    v["i76"] = 0.049748*np.tanh(np.where(np.tanh((data["PREV_APP_CREDIT_PERC_MIN"])) > -1, np.where(data["POS_NAME_CONTRACT_STATUS_Completed_MEAN"] > -1, ((data["NAME_EDUCATION_TYPE_Lower_secondary"]) - (((((data["PREV_APP_CREDIT_PERC_MIN"]) * 2.0)) * 2.0))), ((data["CC_CNT_INSTALMENT_MATURE_CUM_VAR"]) * (data["BURO_AMT_CREDIT_SUM_MAX"])) ), data["NEW_DOC_IND_STD"] )) 
    v["i77"] = 0.048004*np.tanh(np.where(data["CC_AMT_BALANCE_VAR"]>0, data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"], ((np.where(data["CC_AMT_CREDIT_LIMIT_ACTUAL_SUM"] > -1, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_MEAN"], ((data["DAYS_BIRTH"]) * (data["NAME_FAMILY_STATUS_Married"])) )) - (np.where(data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"]<0, data["NAME_FAMILY_STATUS_Married"], data["REFUSED_AMT_CREDIT_MIN"] ))) )) 
    v["i78"] = 0.049471*np.tanh(np.where(data["INSTAL_COUNT"]>0, np.where(data["EXT_SOURCE_2"]<0, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_MAX"], data["APPROVED_CNT_PAYMENT_SUM"] ), ((-1.0) - (np.where(data["EXT_SOURCE_3"] > -1, data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"], ((data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_MAX"]) - (data["EXT_SOURCE_3"])) ))) )) 
    v["i79"] = 0.047294*np.tanh((-1.0*((((data["NEW_RATIO_PREV_AMT_CREDIT_MIN"]) * (np.where(((((data["FLAG_DOCUMENT_8"]) + (data["WEEKDAY_APPR_PROCESS_START_MONDAY"]))) + (data["NEW_DOC_IND_AVG"]))>0, data["BASEMENTAREA_MODE"], ((data["POS_SK_DPD_DEF_MEAN"]) - (data["ORGANIZATION_TYPE_Military"])) ))))))) 
    v["i80"] = 0.049797*np.tanh(((data["AMT_ANNUITY"]) - (np.where(data["POS_SK_DPD_DEF_MAX"]<0, np.maximum(((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"])), ((((np.where(data["APPROVED_APP_CREDIT_PERC_MAX"]<0, data["INSTAL_AMT_PAYMENT_SUM"], ((np.tanh((data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"]))) * 2.0) )) * 2.0)))), data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] )))) 
    v["i81"] = 0.049970*np.tanh(np.where(data["NEW_PHONE_TO_BIRTH_RATIO"] > -1, ((data["PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN"]) + (np.where(data["BURO_DAYS_CREDIT_ENDDATE_MAX"]>0, data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"], data["APPROVED_AMT_CREDIT_MAX"] ))), ((data["WALLSMATERIAL_MODE_Stone__brick"]) - (np.where(data["PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN"]<0, data["POS_MONTHS_BALANCE_MEAN"], (9.0) ))) )) 
    v["i82"] = 0.047933*np.tanh(np.where(data["CLOSED_DAYS_CREDIT_MAX"]<0, np.maximum(((data["CC_CNT_DRAWINGS_CURRENT_MAX"])), ((np.where(data["CC_AMT_BALANCE_VAR"] > -1, data["CC_CNT_DRAWINGS_CURRENT_MAX"], np.maximum(((np.maximum(((((data["PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN"]) * 2.0))), ((data["DEF_60_CNT_SOCIAL_CIRCLE"]))))), ((data["PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN"]))) )))), data["ACTIVE_AMT_CREDIT_SUM_SUM"] )) 
    v["i83"] = 0.049578*np.tanh(np.where(data["LANDAREA_AVG"]>0, data["PREV_NAME_YIELD_GROUP_XNA_MEAN"], np.where(data["BURO_AMT_CREDIT_SUM_MEAN"]>0, data["NAME_INCOME_TYPE_Commercial_associate"], (-1.0*((np.where(((data["NEW_PHONE_TO_EMPLOY_RATIO"]) + (data["BURO_DAYS_CREDIT_MAX"])) > -1, data["NAME_INCOME_TYPE_Commercial_associate"], data["CC_AMT_CREDIT_LIMIT_ACTUAL_MAX"] )))) ) )) 
    v["i84"] = 0.049998*np.tanh(((np.where(((data["AMT_GOODS_PRICE"]) * (data["CC_AMT_RECIVABLE_VAR"]))>0, ((((data["APPROVED_CNT_PAYMENT_SUM"]) + (data["AMT_ANNUITY"]))) - (np.where(data["NEW_DOC_IND_KURT"] > -1, data["INSTAL_AMT_INSTALMENT_SUM"], data["NEW_DOC_IND_AVG"] ))), data["PREV_NAME_CONTRACT_TYPE_Revolving_loans_MEAN"] )) * 2.0)) 
    v["i85"] = 0.000199*np.tanh(np.minimum(((np.minimum(((data["NEW_DOC_IND_KURT"])), (((-1.0*(((((data["WEEKDAY_APPR_PROCESS_START_SUNDAY"]) + (((data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]) + (np.where(data["POS_SK_DPD_DEF_MAX"]<0, data["OCCUPATION_TYPE_Medicine_staff"], data["PREV_NAME_PORTFOLIO_Cash_MEAN"] )))))/2.0))))))))), (((-1.0*((data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"]))))))) 
    v["i86"] = 0.044998*np.tanh(((np.where((((data["POS_COUNT"]) < (np.tanh((data["APPROVED_CNT_PAYMENT_SUM"]))))*1.)>0, np.where(data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]>0, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"], (((data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]) < (data["POS_COUNT"]))*1.) ), ((data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"]) * 2.0) )) * 2.0)) 
    v["i87"] = 0.049818*np.tanh(((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) - (((np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]<0, data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"], data["NEW_CAR_TO_BIRTH_RATIO"] )) - (np.where(data["PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN"]>0, data["PREV_PRODUCT_COMBINATION_POS_household_without_interest_MEAN"], (((data["INSTAL_PAYMENT_DIFF_MAX"]) > (data["PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN"]))*1.) )))))) * 2.0)) 
    v["i88"] = 0.048104*np.tanh(((data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"]) - (((data["CC_AMT_PAYMENT_CURRENT_MEAN"]) - (np.where(data["INSTAL_DPD_MEAN"]<0, ((((data["APPROVED_CNT_PAYMENT_MEAN"]) * (data["APPROVED_DAYS_DECISION_MAX"]))) - (data["INSTAL_DAYS_ENTRY_PAYMENT_MEAN"])), ((((data["INSTAL_DAYS_ENTRY_PAYMENT_MEAN"]) * 2.0)) * 2.0) )))))) 
    v["i89"] = 0.043416*np.tanh(((((np.where(data["CC_AMT_PAYMENT_CURRENT_SUM"]<0, np.where(data["ORGANIZATION_TYPE_School"]<0, np.where(data["NAME_INCOME_TYPE_State_servant"]<0, data["NEW_DOC_IND_KURT"], data["REFUSED_CNT_PAYMENT_MEAN"] ), ((data["REFUSED_CNT_PAYMENT_SUM"]) * 2.0) ), data["WALLSMATERIAL_MODE_Stone__brick"] )) * 2.0)) * 2.0)) 
    v["i90"] = 0.049199*np.tanh(((((data["POS_SK_DPD_DEF_MAX"]) - (np.where(data["NAME_CONTRACT_TYPE_Cash_loans"] > -1, data["NAME_CONTRACT_TYPE_Cash_loans"], data["LIVINGAPARTMENTS_MEDI"] )))) + (np.maximum(((np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((np.maximum(((data["OCCUPATION_TYPE_Drivers"])), ((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"])))))))), ((data["NAME_HOUSING_TYPE_Municipal_apartment"])))))) 
    v["i91"] = 0.041880*np.tanh(np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"] > -1, ((data["REFUSED_RATE_DOWN_PAYMENT_MEAN"]) * (((data["APPROVED_AMT_DOWN_PAYMENT_MIN"]) * 2.0))), ((data["PREV_NAME_YIELD_GROUP_high_MEAN"]) * (np.where(data["FLOORSMIN_MEDI"]<0, data["AMT_GOODS_PRICE"], ((data["REFUSED_RATE_DOWN_PAYMENT_MEAN"]) * (data["APPROVED_AMT_DOWN_PAYMENT_MIN"])) ))) )) 
    v["i92"] = 0.049800*np.tanh(np.where(data["APPROVED_HOUR_APPR_PROCESS_START_MAX"]>0, (((data["EXT_SOURCE_3"]) + (data["APPROVED_CNT_PAYMENT_SUM"]))/2.0), np.where(data["EXT_SOURCE_3"] > -1, data["OCCUPATION_TYPE_Laborers"], np.maximum(((data["NEW_EXT_SOURCES_MEAN"])), ((((np.maximum(((data["ACTIVE_DAYS_CREDIT_MAX"])), ((data["DEF_30_CNT_SOCIAL_CIRCLE"])))) * 2.0)))) ) )) 
    v["i93"] = 0.049748*np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MAX"]<0, np.where(data["AMT_GOODS_PRICE"]<0, np.where(((data["AMT_GOODS_PRICE"]) + (data["NEW_DOC_IND_AVG"]))>0, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_LIMIT_MEAN"], ((data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"]) * (data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"])) ), data["APPROVED_AMT_DOWN_PAYMENT_MAX"] ), data["AMT_GOODS_PRICE"] )) 
    v["i94"] = 0.049770*np.tanh(((np.where(data["INSTAL_PAYMENT_DIFF_VAR"] > -1, np.where(data["AMT_INCOME_TOTAL"]<0, data["AMT_ANNUITY"], np.minimum(((data["NEW_DOC_IND_KURT"])), ((np.where(data["REFUSED_AMT_GOODS_PRICE_MEAN"]>0, (-1.0*((data["DAYS_BIRTH"]))), data["NEW_CREDIT_TO_ANNUITY_RATIO"] )))) ), data["NEW_DOC_IND_STD"] )) * 2.0)) 
    v["i95"] = 0.048000*np.tanh(np.where(data["BURO_DAYS_CREDIT_UPDATE_MEAN"] > -1, np.maximum(((data["NAME_EDUCATION_TYPE_Lower_secondary"])), ((np.where(data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]<0, data["BURO_CREDIT_TYPE_Microloan_MEAN"], (-1.0*((data["ACTIVE_MONTHS_BALANCE_MAX_MAX"]))) )))), ((data["YEARS_BUILD_AVG"]) * (((data["NAME_EDUCATION_TYPE_Lower_secondary"]) - (data["REFUSED_APP_CREDIT_PERC_VAR"])))) )) 
    v["i96"] = 0.049784*np.tanh(((((((np.where(data["CODE_GENDER"]>0, np.where(data["PREV_CNT_PAYMENT_SUM"] > -1, data["DAYS_BIRTH"], data["BURO_DAYS_CREDIT_VAR"] ), (((((data["AMT_CREDIT"]) < (data["DAYS_BIRTH"]))*1.)) - (data["DAYS_BIRTH"])) )) * 2.0)) * 2.0)) * 2.0)) 
    v["i97"] = 0.035002*np.tanh(np.where(data["NEW_RATIO_BURO_DAYS_CREDIT_MIN"]<0, np.where(data["NAME_HOUSING_TYPE_Rented_apartment"]<0, np.where(data["LIVINGAPARTMENTS_AVG"]<0, data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"], ((data["INSTAL_AMT_PAYMENT_MIN"]) - (data["NEW_RATIO_PREV_AMT_ANNUITY_MAX"])) ), data["NAME_HOUSING_TYPE_Rented_apartment"] ), ((data["INSTAL_AMT_PAYMENT_MIN"]) - (data["NEW_LIVE_IND_SUM"])) )) 
    v["i98"] = 0.035000*np.tanh(((((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]) * ((((data["APPROVED_CNT_PAYMENT_SUM"]) < (np.where(data["EXT_SOURCE_3"]<0, data["POS_MONTHS_BALANCE_SIZE"], data["BURO_AMT_CREDIT_SUM_MAX"] )))*1.)))) - (np.where(data["DAYS_EMPLOYED"]<0, data["NEW_CAR_TO_BIRTH_RATIO"], data["APPROVED_CNT_PAYMENT_SUM"] )))) 
    v["i99"] = 0.049903*np.tanh(((np.where(data["ORGANIZATION_TYPE_Industry__type_9"]>0, data["CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN"], ((data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"]) * (data["INSTAL_PAYMENT_DIFF_MAX"])) )) - (np.where(data["OBS_30_CNT_SOCIAL_CIRCLE"]<0, data["INSTAL_AMT_PAYMENT_SUM"], ((data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"]) - (data["APPROVED_AMT_CREDIT_MEAN"])) )))) 
    v["i100"] = 0.048280*np.tanh(((((np.where(data["CLOSED_DAYS_CREDIT_MAX"]>0, data["ACTIVE_AMT_CREDIT_SUM_SUM"], np.where(data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]<0, np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"] > -1, data["INSTAL_PAYMENT_DIFF_MAX"], (-1.0*((data["ACTIVE_AMT_CREDIT_SUM_SUM"]))) ), data["CLOSED_MONTHS_BALANCE_MIN_MIN"] ) )) * 2.0)) + (data["POS_SK_DPD_DEF_MAX"]))) 
    v["i101"] = 0.048002*np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_MEAN"]>0, data["REGION_RATING_CLIENT_W_CITY"], np.where(data["REFUSED_APP_CREDIT_PERC_MAX"]>0, data["PREV_WEEKDAY_APPR_PROCESS_START_THURSDAY_MEAN"], np.maximum(((data["CC_AMT_BALANCE_VAR"])), ((np.maximum((((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"]) > (((data["OCCUPATION_TYPE_Core_staff"]) / 2.0)))*1.))), ((data["ORGANIZATION_TYPE_Transport__type_3"])))))) ) )) 
    v["i102"] = 0.046002*np.tanh(np.where((((data["NEW_EXT_SOURCES_MEAN"]) + (data["OCCUPATION_TYPE_Core_staff"]))/2.0) > -1, ((data["NEW_EXT_SOURCES_MEAN"]) - (((data["NEW_EXT_SOURCES_MEAN"]) * (data["NEW_EXT_SOURCES_MEAN"])))), ((data["NEW_EXT_SOURCES_MEAN"]) * (data["NEW_EXT_SOURCES_MEAN"])) )) 
    v["i103"] = 0.049800*np.tanh(((np.maximum(((data["NEW_RATIO_PREV_APP_CREDIT_PERC_MEAN"])), ((np.where(data["NEW_CREDIT_TO_GOODS_RATIO"]>0, data["PREV_DAYS_DECISION_MAX"], np.where(data["NEW_CREDIT_TO_GOODS_RATIO"] > -1, data["ORGANIZATION_TYPE_Business_Entity_Type_3"], (((data["INSTAL_DBD_MAX"]) > (data["PREV_NAME_CONTRACT_STATUS_Canceled_MEAN"]))*1.) ) ))))) * 2.0)) 
    v["i104"] = 0.049408*np.tanh(np.where(data["DAYS_ID_PUBLISH"] > -1, ((data["DAYS_BIRTH"]) * (((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) + (np.where(data["DAYS_BIRTH"] > -1, np.where(data["PREV_NAME_PRODUCT_TYPE_walk_in_MEAN"]<0, data["NAME_EDUCATION_TYPE_Secondary___secondary_special"], data["CLOSED_AMT_ANNUITY_MEAN"] ), data["NAME_INCOME_TYPE_Working"] ))))), data["CLOSED_AMT_ANNUITY_MEAN"] )) 
    v["i105"] = 0.046500*np.tanh(np.where(data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"] > -1, data["AMT_GOODS_PRICE"], (-1.0*((np.where(data["PREV_NAME_PORTFOLIO_Cards_MEAN"]>0, data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"], np.where(data["NEW_SCORES_STD"]>0, (-1.0*((data["NEW_SCORES_STD"]))), (((data["PREV_NAME_PORTFOLIO_Cards_MEAN"]) > (data["NEW_EXT_SOURCES_MEAN"]))*1.) ) )))) )) 
    v["i106"] = 0.047502*np.tanh(np.where(data["CODE_GENDER"]<0, data["NAME_FAMILY_STATUS_Separated"], np.where(data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]<0, np.where(data["DAYS_BIRTH"]<0, data["NEW_DOC_IND_AVG"], data["PREV_NAME_GOODS_CATEGORY_Consumer_Electronics_MEAN"] ), np.where(data["AMT_REQ_CREDIT_BUREAU_MON"] > -1, data["CLOSED_DAYS_CREDIT_MEAN"], data["NEW_INC_PER_CHLD"] ) ) )) 
    v["i107"] = 0.048562*np.tanh(((np.where(data["NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MIN"]>0, data["BURO_CREDIT_TYPE_Credit_card_MEAN"], ((data["POS_SK_DPD_DEF_MAX"]) - (data["APPROVED_AMT_GOODS_PRICE_MIN"])) )) - (np.maximum(((data["BURO_CREDIT_TYPE_Car_loan_MEAN"])), ((np.maximum(((data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"])), ((np.maximum(((data["WEEKDAY_APPR_PROCESS_START_SUNDAY"])), ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"])))))))))))) 
    v["i108"] = 0.049957*np.tanh(np.where(data["NEW_RATIO_BURO_DAYS_CREDIT_MEAN"]>0, ((data["INSTAL_AMT_PAYMENT_MIN"]) - (np.tanh((2.0)))), ((data["CC_AMT_INST_MIN_REGULARITY_MAX"]) - (((data["CC_AMT_CREDIT_LIMIT_ACTUAL_MIN"]) - ((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MAX"]) < (((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) * 2.0)))*1.))))) )) 
    v["i109"] = 0.037995*np.tanh(np.where(data["POS_NAME_CONTRACT_STATUS_Completed_MEAN"]>0, data["AMT_ANNUITY"], np.where(data["REFUSED_HOUR_APPR_PROCESS_START_MIN"]>0, data["PREV_RATE_DOWN_PAYMENT_MAX"], np.maximum(((np.maximum(((np.where(data["ACTIVE_DAYS_CREDIT_MIN"]>0, data["CLOSED_MONTHS_BALANCE_SIZE_SUM"], data["NAME_EDUCATION_TYPE_Higher_education"] ))), ((data["FLAG_WORK_PHONE"]))))), ((data["FLAG_WORK_PHONE"]))) ) )) 
    v["i110"] = 0.045640*np.tanh(((np.where(data["CC_AMT_CREDIT_LIMIT_ACTUAL_MIN"]<0, np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"]>0, data["CC_AMT_CREDIT_LIMIT_ACTUAL_MIN"], (-1.0*((((data["PREV_DAYS_DECISION_MIN"]) * (np.where(data["REFUSED_AMT_APPLICATION_MAX"] > -1, data["INSTAL_DBD_MAX"], data["NEW_DOC_IND_AVG"] )))))) ), data["CC_AMT_INST_MIN_REGULARITY_VAR"] )) * 2.0)) 
    v["i111"] = 0.050000*np.tanh(((data["CC_NAME_CONTRACT_STATUS_Active_MEAN"]) * ((((data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"]) < (np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MAX"]>0, data["PREV_NAME_CONTRACT_STATUS_Canceled_MEAN"], ((np.where(data["INSTAL_DPD_MEAN"]>0, np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MAX"])), ((data["PREV_CNT_PAYMENT_MEAN"]))), data["LANDAREA_MEDI"] )) / 2.0) )))*1.)))) 
    v["i112"] = 0.046700*np.tanh(((data["WALLSMATERIAL_MODE_Stone__brick"]) + ((((data["ORGANIZATION_TYPE_Construction"]) + ((-1.0*((np.where(data["DEF_30_CNT_SOCIAL_CIRCLE"]>0, data["CC_AMT_INST_MIN_REGULARITY_SUM"], np.where(data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"]>0, (((data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"]) + (data["REFUSED_AMT_CREDIT_MIN"]))/2.0), data["NEW_PHONE_TO_EMPLOY_RATIO"] ) ))))))/2.0)))) 
    v["i113"] = 0.049002*np.tanh(np.where(data["NEW_EMPLOY_TO_BIRTH_RATIO"]>0, (((((data["NEW_CREDIT_TO_INCOME_RATIO"]) * (data["BURO_DAYS_CREDIT_MAX"]))) + (data["BURO_DAYS_CREDIT_MEAN"]))/2.0), np.where(data["BURO_DAYS_CREDIT_MAX"]>0, data["ACTIVE_AMT_CREDIT_SUM_SUM"], (((data["ORGANIZATION_TYPE_Transport__type_3"]) > (data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"]))*1.) ) )) 
    v["i114"] = 0.049400*np.tanh((-1.0*((((data["OCCUPATION_TYPE_Accountants"]) + (((data["ORGANIZATION_TYPE_Military"]) + (np.where(data["PREV_NAME_GOODS_CATEGORY_Computers_MEAN"]>0, data["EXT_SOURCE_2"], ((data["OCCUPATION_TYPE_Core_staff"]) + ((((((data["ENTRANCES_AVG"]) > (data["PREV_NAME_GOODS_CATEGORY_Computers_MEAN"]))*1.)) * 2.0))) ))))))))) 
    v["i115"] = 0.045008*np.tanh(np.where(data["FLOORSMIN_AVG"] > -1, data["PREV_CHANNEL_TYPE_Country_wide_MEAN"], np.where(data["PREV_NAME_CLIENT_TYPE_New_MEAN"]>0, data["DAYS_REGISTRATION"], np.maximum((((((data["ORGANIZATION_TYPE_Construction"]) + (data["PREV_NAME_CLIENT_TYPE_New_MEAN"]))/2.0))), ((np.where(data["NEW_EMPLOY_TO_BIRTH_RATIO"]>0, data["BURO_STATUS_1_MEAN_MEAN"], data["AMT_INCOME_TOTAL"] )))) ) )) 
    v["i116"] = 0.049852*np.tanh(((data["POS_COUNT"]) * (np.where(data["CLOSED_DAYS_CREDIT_MEAN"]<0, np.where(data["PREV_NAME_SELLER_INDUSTRY_Consumer_electronics_MEAN"]>0, data["PREV_RATE_DOWN_PAYMENT_MAX"], data["INSTAL_AMT_PAYMENT_MAX"] ), np.maximum(((np.maximum(((data["PREV_WEEKDAY_APPR_PROCESS_START_SUNDAY_MEAN"])), ((data["INSTAL_DBD_MEAN"]))))), ((data["DAYS_REGISTRATION"]))) )))) 
    v["i117"] = 0.048914*np.tanh(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]) * (np.where(data["AMT_INCOME_TOTAL"]>0, data["PREV_WEEKDAY_APPR_PROCESS_START_SUNDAY_MEAN"], (-1.0*((np.maximum(((data["PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN"])), ((np.where(data["LIVINGAREA_AVG"]>0, (-1.0*((data["EXT_SOURCE_1"]))), data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"] ))))))) )))) 
    v["i118"] = 0.049651*np.tanh(((((((data["NEW_CREDIT_TO_GOODS_RATIO"]) * (((((data["AMT_ANNUITY"]) - (data["INSTAL_DBD_SUM"]))) - (((data["OCCUPATION_TYPE_Core_staff"]) * (np.where(data["INSTAL_DBD_SUM"]>0, data["OCCUPATION_TYPE_Core_staff"], data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"] )))))))) * 2.0)) * 2.0)) 
    v["i119"] = 0.037400*np.tanh(np.where(data["CC_NAME_CONTRACT_STATUS_Active_VAR"] > -1, data["PREV_PRODUCT_COMBINATION_Cash_MEAN"], (((((-1.0*((np.where(data["NEW_RATIO_PREV_AMT_GOODS_PRICE_MEAN"] > -1, data["BURO_CREDIT_ACTIVE_Closed_MEAN"], np.where(data["PREV_PRODUCT_COMBINATION_Cash_MEAN"]>0, (-1.0*((data["NEW_RATIO_BURO_AMT_CREDIT_SUM_SUM"]))), data["ORGANIZATION_TYPE_Industry__type_9"] ) ))))) * 2.0)) * 2.0) )) 
    v["i120"] = 0.049499*np.tanh((((((((data["POS_SK_DPD_MEAN"]) - (data["PREV_WEEKDAY_APPR_PROCESS_START_SATURDAY_MEAN"]))) + (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, data["FLAG_DOCUMENT_3"], np.tanh((data["APARTMENTS_MEDI"])) )))/2.0)) - ((((data["NEW_RATIO_PREV_AMT_APPLICATION_MEAN"]) > (data["PREV_NAME_PAYMENT_TYPE_XNA_MEAN"]))*1.)))) 
    v["i121"] = 0.049600*np.tanh(((data["REGION_RATING_CLIENT_W_CITY"]) * (((data["NEW_EXT_SOURCES_MEAN"]) + (np.where(data["TOTALAREA_MODE"]<0, np.where(data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]<0, data["FLAG_DOCUMENT_8"], data["NEW_EXT_SOURCES_MEAN"] ), np.where(data["REGION_RATING_CLIENT_W_CITY"]<0, data["ACTIVE_MONTHS_BALANCE_MIN_MIN"], data["FLAG_DOCUMENT_8"] ) )))))) 
    v["i122"] = 0.050000*np.tanh(np.where(data["POS_COUNT"]<0, np.where(data["NAME_EDUCATION_TYPE_Lower_secondary"]<0, ((data["FLAG_WORK_PHONE"]) * (data["EXT_SOURCE_2"])), data["NAME_EDUCATION_TYPE_Lower_secondary"] ), ((((((data["PREV_NAME_CONTRACT_TYPE_Consumer_loans_MEAN"]) - (data["FLAG_PHONE"]))) * 2.0)) - (data["FLAG_WORK_PHONE"])) )) 
    v["i123"] = 0.049800*np.tanh(np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"]>0, data["EXT_SOURCE_2"], np.where(np.where(data["ACTIVE_DAYS_CREDIT_MIN"] > -1, data["INSTAL_DBD_SUM"], data["CLOSED_MONTHS_BALANCE_MIN_MIN"] )<0, data["ORGANIZATION_TYPE_Transport__type_3"], (-1.0*((((((((data["CC_AMT_BALANCE_MEAN"]) * 2.0)) * 2.0)) * 2.0)))) ) )) 
    v["i124"] = 0.049001*np.tanh(np.where(data["BURO_CREDIT_TYPE_Microloan_MEAN"]>0, data["BURO_CREDIT_TYPE_Microloan_MEAN"], np.minimum((((((data["NEW_SOURCES_PROD"]) > (data["NEW_RATIO_BURO_DAYS_CREDIT_VAR"]))*1.))), (((((((data["DAYS_BIRTH"]) + (data["NEW_SOURCES_PROD"]))/2.0)) * (((data["NEW_RATIO_BURO_DAYS_CREDIT_VAR"]) - (data["DAYS_BIRTH"]))))))) )) 
    v["i125"] = 0.048804*np.tanh(((data["NEW_RATIO_BURO_AMT_CREDIT_SUM_MAX"]) - (np.where(data["DAYS_LAST_PHONE_CHANGE"] > -1, np.where(data["PREV_DAYS_DECISION_MIN"] > -1, data["ACTIVE_AMT_CREDIT_SUM_DEBT_MAX"], data["BURO_MONTHS_BALANCE_MIN_MIN"] ), np.where(data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]>0, ((data["BURO_MONTHS_BALANCE_MIN_MIN"]) - (data["DAYS_LAST_PHONE_CHANGE"])), data["CC_AMT_DRAWINGS_CURRENT_VAR"] ) )))) 
    v["i126"] = 0.047502*np.tanh((((-1.0*((np.where(data["EXT_SOURCE_3"] > -1, data["BURO_DAYS_CREDIT_MAX"], (-1.0*((np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"] > -1, data["BURO_DAYS_CREDIT_MAX"], (-1.0*((np.where(data["NEW_RATIO_PREV_AMT_ANNUITY_MIN"] > -1, data["PREV_NAME_CLIENT_TYPE_Refreshed_MEAN"], data["PREV_RATE_DOWN_PAYMENT_MIN"] )))) )))) ))))) * 2.0)) 
    v["i127"] = 0.049420*np.tanh((((((data["INSTAL_DBD_SUM"]) < (data["APPROVED_AMT_CREDIT_MAX"]))*1.)) - (((data["OCCUPATION_TYPE_Medicine_staff"]) - (np.where(data["NEW_CAR_TO_BIRTH_RATIO"] > -1, data["APPROVED_AMT_ANNUITY_MEAN"], np.maximum(((((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]) * 2.0))), ((data["NAME_HOUSING_TYPE_Rented_apartment"]))) )))))) 
    v["i128"] = 0.030880*np.tanh(np.where(((data["BURO_AMT_CREDIT_SUM_DEBT_SUM"]) * ((-1.0*((data["PREV_AMT_DOWN_PAYMENT_MAX"])))))<0, ((data["FLAG_WORK_PHONE"]) * 2.0), np.where(data["INSTAL_PAYMENT_PERC_SUM"]>0, ((data["BURO_AMT_CREDIT_SUM_DEBT_SUM"]) * (data["BURO_AMT_CREDIT_SUM_DEBT_SUM"])), ((data["INSTAL_AMT_PAYMENT_MIN"]) * 2.0) ) )) 
    v["i129"] = 0.047000*np.tanh(np.where(data["PREV_AMT_DOWN_PAYMENT_MIN"]>0, data["REGION_RATING_CLIENT_W_CITY"], ((np.maximum(((data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"])), ((np.where(data["REGION_POPULATION_RELATIVE"] > -1, np.where(data["INSTAL_PAYMENT_DIFF_MEAN"] > -1, data["NEW_EXT_SOURCES_MEAN"], (-1.0*((data["NEW_EXT_SOURCES_MEAN"]))) ), data["REGION_RATING_CLIENT_W_CITY"] ))))) * 2.0) )) 
    v["i130"] = 0.048680*np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_DEBT_SUM"] > -1, ((data["BURO_AMT_CREDIT_SUM_DEBT_SUM"]) - (((data["ACTIVE_AMT_CREDIT_SUM_MEAN"]) * (((((((-1.0*((data["BURO_AMT_CREDIT_SUM_DEBT_SUM"])))) < ((((data["BURO_AMT_CREDIT_SUM_MAX"]) < (data["BURO_AMT_CREDIT_SUM_DEBT_SUM"]))*1.)))*1.)) * 2.0))))), data["POS_SK_DPD_DEF_MAX"] )) 
    v["i131"] = 0.049972*np.tanh(np.where(data["OBS_60_CNT_SOCIAL_CIRCLE"]<0, np.where(data["FLAG_DOCUMENT_3"]>0, data["REGION_POPULATION_RELATIVE"], np.where(data["REGION_POPULATION_RELATIVE"]>0, data["FLAG_DOCUMENT_3"], np.where(data["AMT_INCOME_TOTAL"]>0, data["FLAG_DOCUMENT_3"], (-1.0*((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))) ) ) ), data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] )) 
    v["i132"] = 0.048698*np.tanh(np.where(data["APARTMENTS_MODE"]>0, data["CC_NAME_CONTRACT_STATUS_Active_SUM"], np.where(data["BURO_AMT_CREDIT_SUM_SUM"]>0, data["PREV_CODE_REJECT_REASON_LIMIT_MEAN"], (-1.0*((((data["INSTAL_AMT_PAYMENT_SUM"]) + (((data["NEW_DOC_IND_AVG"]) - ((((data["INSTAL_AMT_PAYMENT_SUM"]) < (data["REFUSED_AMT_GOODS_PRICE_MAX"]))*1.)))))))) ) )) 
    v["i133"] = 0.049969*np.tanh(np.where(data["ORGANIZATION_TYPE_School"]>0, data["BURO_STATUS_1_MEAN_MEAN"], np.where(((data["EXT_SOURCE_1"]) / 2.0) > -1, data["BURO_AMT_CREDIT_SUM_DEBT_SUM"], (-1.0*((((data["BURO_AMT_CREDIT_SUM_MEAN"]) - (np.where(data["CLOSED_AMT_CREDIT_SUM_MAX"]>0, data["ORGANIZATION_TYPE_Self_employed"], data["NEW_LIVE_IND_STD"] )))))) ) )) 
    v["i134"] = 0.049501*np.tanh(np.where(data["NEW_RATIO_BURO_DAYS_CREDIT_MAX"]>0, data["PREV_NAME_YIELD_GROUP_XNA_MEAN"], ((((np.where(((data["NEW_RATIO_PREV_AMT_CREDIT_MIN"]) * 2.0)>0, ((data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__high_MEAN"]) * 2.0), (((data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"]) > (((data["BURO_AMT_CREDIT_SUM_LIMIT_SUM"]) / 2.0)))*1.) )) * 2.0)) * 2.0) )) 
    v["i135"] = 0.046770*np.tanh(np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] > -1, np.where(data["COMMONAREA_MODE"] > -1, data["OCCUPATION_TYPE_Drivers"], ((((((((data["PREV_CODE_REJECT_REASON_HC_MEAN"]) + (data["PREV_PRODUCT_COMBINATION_POS_other_with_interest_MEAN"]))/2.0)) + (data["NAME_HOUSING_TYPE_Rented_apartment"]))/2.0)) * 2.0) ), data["APPROVED_AMT_APPLICATION_MAX"] )) 
    v["i136"] = 0.047902*np.tanh((-1.0*(((((np.maximum(((np.maximum(((np.maximum(((data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"])), ((data["PREV_NAME_GOODS_CATEGORY_Sport_and_Leisure_MEAN"]))))), ((data["PREV_NAME_GOODS_CATEGORY_Photo___Cinema_Equipment_MEAN"]))))), ((data["OCCUPATION_TYPE_High_skill_tech_staff"])))) + (np.where(data["NEW_PHONE_TO_BIRTH_RATIO"] > -1, data["EXT_SOURCE_2"], data["PREV_NAME_SELLER_INDUSTRY_XNA_MEAN"] )))/2.0))))) 
    v["i137"] = 0.048544*np.tanh(np.where(data["ORGANIZATION_TYPE_Kindergarten"]>0, data["AMT_REQ_CREDIT_BUREAU_QRT"], np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"]>0, data["ACTIVE_CREDIT_DAY_OVERDUE_MAX"], ((((np.maximum(((np.where(data["APPROVED_DAYS_DECISION_MAX"]>0, data["APPROVED_AMT_APPLICATION_MAX"], data["PREV_NAME_CLIENT_TYPE_Repeater_MEAN"] ))), ((data["INSTAL_DPD_MEAN"])))) * 2.0)) * 2.0) ) )) 
    v["i138"] = 0.048922*np.tanh(np.where((((data["CC_AMT_DRAWINGS_CURRENT_VAR"]) + (data["NEW_SOURCES_PROD"]))/2.0)<0, np.where(data["PREV_CODE_REJECT_REASON_SCO_MEAN"]<0, ((data["NEW_SOURCES_PROD"]) - ((((data["CC_AMT_DRAWINGS_CURRENT_VAR"]) + (data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]))/2.0))), data["CC_AMT_DRAWINGS_CURRENT_VAR"] ), (-1.0*(((5.22825956344604492)))) )) 
    v["i139"] = 0.047001*np.tanh(((data["NEW_EXT_SOURCES_MEAN"]) - (((data["NEW_EXT_SOURCES_MEAN"]) * ((((((data["EXT_SOURCE_3"]) / 2.0)) + (np.maximum(((data["NEW_SCORES_STD"])), ((np.maximum(((data["PREV_AMT_ANNUITY_MIN"])), ((((data["NEW_EXT_SOURCES_MEAN"]) * (data["NEW_EXT_SOURCES_MEAN"]))))))))))/2.0)))))) 
    v["i140"] = 0.020806*np.tanh(((np.where(data["PREV_CNT_PAYMENT_MEAN"]<0, data["EXT_SOURCE_3"], ((data["INSTAL_AMT_INSTALMENT_SUM"]) * (data["PREV_CNT_PAYMENT_MEAN"])) )) - (((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MAX"]) + (((data["INSTAL_AMT_INSTALMENT_SUM"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["PREV_CNT_PAYMENT_MEAN"]))))))))) 
    v["i141"] = 0.005614*np.tanh(np.where(data["REFUSED_AMT_GOODS_PRICE_MEAN"]>0, data["PREV_NAME_SELLER_INDUSTRY_Consumer_electronics_MEAN"], np.where(data["CC_AMT_CREDIT_LIMIT_ACTUAL_SUM"]>0, data["PREV_AMT_GOODS_PRICE_MIN"], (((data["INSTAL_AMT_PAYMENT_SUM"]) < (((np.where(data["BURO_CREDIT_ACTIVE_Closed_MEAN"]>0, data["REGION_POPULATION_RELATIVE"], ((data["PREV_AMT_GOODS_PRICE_MAX"]) * 2.0) )) / 2.0)))*1.) ) )) 
    v["i142"] = 0.049999*np.tanh((((((((data["INSTAL_AMT_INSTALMENT_SUM"]) < (np.where(data["APPROVED_AMT_APPLICATION_MIN"]>0, data["POS_NAME_CONTRACT_STATUS_Completed_MEAN"], (-1.0*((np.where(data["PREV_PRODUCT_COMBINATION_POS_household_without_interest_MEAN"]>0, data["APPROVED_APP_CREDIT_PERC_MIN"], (((data["PREV_NAME_GOODS_CATEGORY_Audio_Video_MEAN"]) < (data["APPROVED_AMT_APPLICATION_MIN"]))*1.) )))) )))*1.)) * 2.0)) * 2.0)) 
    v["i143"] = 0.040040*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_SUM"]>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], ((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"] > -1, np.minimum(((data["HOUR_APPR_PROCESS_START"])), (((((-1.0*((data["DAYS_EMPLOYED"])))) / 2.0)))), np.maximum(((data["APPROVED_CNT_PAYMENT_SUM"])), ((data["PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN"]))) )) * 2.0) )) 
    v["i144"] = 0.049999*np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_OVERDUE_MEAN"]>0, 3.141593, ((np.maximum(((data["HOUR_APPR_PROCESS_START"])), ((((data["BASEMENTAREA_AVG"]) + (np.maximum(((data["NONLIVINGAREA_MEDI"])), ((((data["HOUR_APPR_PROCESS_START"]) * 2.0)))))))))) * (data["WEEKDAY_APPR_PROCESS_START_SATURDAY"])) )) 
    v["i145"] = 0.049975*np.tanh(((data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"]) * (np.where(data["CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN"] > -1, data["ACTIVE_MONTHS_BALANCE_MIN_MIN"], np.tanh((np.where(data["REGION_POPULATION_RELATIVE"]<0, data["CLOSED_AMT_CREDIT_SUM_SUM"], np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"]<0, (-1.0*((data["ACTIVE_AMT_ANNUITY_MAX"]))), data["CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN"] ) ))) )))) 
    v["i146"] = 0.042400*np.tanh(np.where(data["PREV_PRODUCT_COMBINATION_Cash_MEAN"]>0, data["BURO_CREDIT_TYPE_Credit_card_MEAN"], np.where((-1.0*(((((data["PREV_PRODUCT_COMBINATION_Cash_MEAN"]) < (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))*1.)))) > -1, data["NEW_RATIO_BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"], (((data["NAME_INCOME_TYPE_State_servant"]) > (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))*1.) ) )) 
    v["i147"] = 0.049749*np.tanh(np.where(np.maximum(((((data["APPROVED_DAYS_DECISION_MIN"]) * ((((data["INSTAL_DPD_MEAN"]) > (data["NAME_EDUCATION_TYPE_Lower_secondary"]))*1.))))), ((data["APARTMENTS_MEDI"])))>0, (((data["REFUSED_AMT_ANNUITY_MEAN"]) < (data["INSTAL_DPD_MEAN"]))*1.), ((((data["NAME_EDUCATION_TYPE_Lower_secondary"]) * 2.0)) * 2.0) )) 
    v["i148"] = 0.049970*np.tanh(np.where(data["INSTAL_DPD_SUM"]>0, data["PREV_APP_CREDIT_PERC_MIN"], np.where(data["CC_AMT_PAYMENT_TOTAL_CURRENT_MAX"]<0, (((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"]) > (np.maximum(((data["ORGANIZATION_TYPE_Construction"])), ((data["ORGANIZATION_TYPE_Construction"])))))*1.)) * 2.0), ((data["PREV_NAME_TYPE_SUITE_Family_MEAN"]) - (data["DAYS_LAST_PHONE_CHANGE"])) ) )) 
    v["i149"] = 0.048725*np.tanh(np.where(data["NEW_INC_PER_CHLD"]<0, np.minimum(((data["REGION_POPULATION_RELATIVE"])), ((((data["BURO_STATUS_X_MEAN_MEAN"]) * (((data["NAME_FAMILY_STATUS_Separated"]) * 2.0)))))), np.where(data["CLOSED_DAYS_CREDIT_VAR"]<0, data["NAME_FAMILY_STATUS_Separated"], ((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) * (data["BURO_STATUS_X_MEAN_MEAN"])) ) ))
    return Output(v.sum(axis=1))
roc_auc_score(train_df.TARGET,GP1(train_df))
roc_auc_score(train_df.TARGET,GP2(train_df))
x = test_df[['SK_ID_CURR']].copy()
x['TARGET'] = .5*GP1(test_df)+.5*GP2(test_df)
x.to_csv('pure_submission.csv', index = False)