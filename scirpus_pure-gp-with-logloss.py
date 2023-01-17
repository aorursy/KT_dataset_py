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
    v["i0"] = 0.005976*np.tanh(((((np.minimum(((((((((np.tanh((np.minimum(((data["DAYS_EMPLOYED"])), ((data["REGION_RATING_CLIENT_W_CITY"])))))) - (data["EXT_SOURCE_3"]))) * 2.0)) + (data["NEW_CREDIT_TO_GOODS_RATIO"])))), (((-1.0*((data["NEW_SOURCES_PROD"]))))))) * 2.0)) * 2.0)) 
    v["i1"] = 0.040171*np.tanh((((((((((((data["DAYS_EMPLOYED"]) > (((data["EXT_SOURCE_2"]) + (np.maximum(((data["EXT_SOURCE_1"])), ((data["EXT_SOURCE_3"])))))))*1.)) - (np.maximum(((data["EXT_SOURCE_3"])), ((data["EXT_SOURCE_2"])))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i2"] = 0.049975*np.tanh(((((((((((((((((np.tanh((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]))) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)) - (np.tanh((data["CC_AMT_PAYMENT_CURRENT_SUM"]))))) * 2.0)) * 2.0)) * 2.0)) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)) 
    v["i3"] = 0.049570*np.tanh(((((((((data["NAME_INCOME_TYPE_Working"]) + ((((((((-1.0*((data["NEW_EXT_SOURCES_MEAN"])))) * 2.0)) - ((((data["CC_AMT_PAYMENT_CURRENT_MIN"]) < (data["BURO_DAYS_CREDIT_ENDDATE_MAX"]))*1.)))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) 
    v["i4"] = 0.040171*np.tanh((((((-1.0*(((((((data["APPROVED_DAYS_DECISION_MIN"]) < ((((data["NEW_EXT_SOURCES_MEAN"]) > (((data["INSTAL_DPD_MEAN"]) * 2.0)))*1.)))*1.)) + (((data["NEW_EXT_SOURCES_MEAN"]) * 2.0))))))) * 2.0)) * 2.0)) 
    v["i5"] = 0.040171*np.tanh(((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) - (((data["NEW_EXT_SOURCES_MEAN"]) + (((np.tanh((data["PREV_RATE_DOWN_PAYMENT_MAX"]))) + ((((((((data["CODE_GENDER"]) > (data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))*1.)) + (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)))))))) * 2.0)) 
    v["i6"] = 0.049088*np.tanh((((((((((((((-1.0*((data["NEW_EXT_SOURCES_MEAN"])))) * 2.0)) + (np.tanh((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))))))) + (((data["DAYS_BIRTH"]) / 2.0)))) * 2.0)) * 2.0)) * 2.0)) 
    v["i7"] = 0.040171*np.tanh(((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)) * (np.minimum(((data["NEW_EXT_SOURCES_MEAN"])), ((((((((((data["NEW_EXT_SOURCES_MEAN"]) * (data["REFUSED_APP_CREDIT_PERC_MAX"]))) + (data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]))/2.0)) + ((-1.0*(((8.0))))))/2.0))))))) 
    v["i8"] = 0.047400*np.tanh((((((((((-1.0*((((data["NEW_EXT_SOURCES_MEAN"]) + (np.maximum(((np.maximum((((-1.0*((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))))), ((data["NEW_CAR_TO_BIRTH_RATIO"]))))), ((data["NEW_EMPLOY_TO_BIRTH_RATIO"]))))))))) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)) * 2.0)) * 2.0)) 
    v["i9"] = 0.040171*np.tanh((((((((((((data["DAYS_EMPLOYED"]) - (data["NAME_EDUCATION_TYPE_Higher_education"]))) + (data["NEW_CREDIT_TO_GOODS_RATIO"]))/2.0)) - (((data["EXT_SOURCE_2"]) - (np.tanh((((data["ACTIVE_DAYS_CREDIT_MAX"]) - (data["BURO_CREDIT_TYPE_Mortgage_MEAN"]))))))))) * 2.0)) * 2.0)) 
    v["i10"] = 0.049975*np.tanh((-1.0*((((((((data["NEW_EXT_SOURCES_MEAN"]) - ((-1.0*((np.tanh((((((data["CODE_GENDER"]) - (((data["DAYS_EMPLOYED"]) + (data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))))) - (data["PREV_CNT_PAYMENT_MEAN"])))))))))) * 2.0)) * 2.0))))) 
    v["i11"] = 0.049620*np.tanh(((((((((((((((data["NEW_CREDIT_TO_GOODS_RATIO"]) - ((((data["EXT_SOURCE_3"]) > (np.tanh((data["CC_AMT_DRAWINGS_CURRENT_MEAN"]))))*1.)))) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)) * 2.0)) * 2.0)) 
    v["i12"] = 0.050000*np.tanh(((((((((np.minimum(((data["NEW_CREDIT_TO_GOODS_RATIO"])), ((data["NEW_DOC_IND_KURT"])))) - (((data["NEW_EXT_SOURCES_MEAN"]) - (np.tanh((((data["DAYS_EMPLOYED"]) - (data["NEW_DOC_IND_KURT"]))))))))) * 2.0)) * 2.0)) - (data["NAME_EDUCATION_TYPE_Higher_education"]))) 
    v["i13"] = 0.050000*np.tanh(((((((((np.tanh((((data["DAYS_EMPLOYED"]) + (data["INSTAL_PAYMENT_DIFF_MAX"]))))) - (data["NEW_EXT_SOURCES_MEAN"]))) + (np.tanh((((np.tanh((data["REFUSED_DAYS_DECISION_MEAN"]))) + (data["NEW_CREDIT_TO_GOODS_RATIO"]))))))) * 2.0)) * 2.0)) 
    v["i14"] = 0.047100*np.tanh(((((((((((data["NEW_CREDIT_TO_GOODS_RATIO"]) - (data["NAME_EDUCATION_TYPE_Higher_education"]))) - (data["CODE_GENDER"]))) - (data["EXT_SOURCE_2"]))) + (((((np.tanh((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]))) - (data["EXT_SOURCE_3"]))) * 2.0)))) * 2.0)) 
    v["i15"] = 0.049950*np.tanh(((((((data["NEW_CREDIT_TO_GOODS_RATIO"]) - (data["NEW_EXT_SOURCES_MEAN"]))) - (((((data["INSTAL_DBD_SUM"]) - (np.tanh((data["CC_AMT_RECIVABLE_MEAN"]))))) - (np.maximum(((data["PREV_CNT_PAYMENT_MEAN"])), ((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"])))))))) * 2.0)) 
    v["i16"] = 0.046512*np.tanh(((((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) - (data["EXT_SOURCE_2"]))) + (np.where(data["EXT_SOURCE_3"] < -99998, data["DAYS_EMPLOYED"], ((((data["NAME_INCOME_TYPE_Working"]) - (data["CODE_GENDER"]))) - (((data["EXT_SOURCE_3"]) * 2.0))) )))) * 2.0)) 
    v["i17"] = 0.049820*np.tanh(((((np.where(data["APPROVED_AMT_DOWN_PAYMENT_MAX"]>0, data["REFUSED_DAYS_DECISION_MAX"], data["DAYS_LAST_PHONE_CHANGE"] )) + (((data["NEW_DOC_IND_KURT"]) + (((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) - (data["NEW_EXT_SOURCES_MEAN"]))) - (data["CODE_GENDER"]))))))) - (data["NEW_EXT_SOURCES_MEAN"]))) 
    v["i18"] = 0.049119*np.tanh(((((data["DEF_30_CNT_SOCIAL_CIRCLE"]) + (((((((((data["PREV_CNT_PAYMENT_MEAN"]) - (data["POS_MONTHS_BALANCE_SIZE"]))) - (data["NEW_EXT_SOURCES_MEAN"]))) + (((((((data["INSTAL_DPD_MEAN"]) * 2.0)) * 2.0)) * 2.0)))) * 2.0)))) * 2.0)) 
    v["i19"] = 0.048000*np.tanh(((data["PREV_NAME_YIELD_GROUP_high_MEAN"]) - (((((data["PREV_NAME_CONTRACT_STATUS_Approved_MEAN"]) + (data["NEW_EXT_SOURCES_MEAN"]))) + (((((np.maximum(((data["CODE_GENDER"])), ((data["APPROVED_RATE_DOWN_PAYMENT_MAX"])))) + (np.tanh((data["BURO_CREDIT_ACTIVE_Closed_MEAN"]))))) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))))) 
    v["i20"] = 0.049970*np.tanh(((((((data["REGION_RATING_CLIENT_W_CITY"]) + (np.where(np.maximum(((np.minimum(((data["DAYS_EMPLOYED"])), ((data["FLAG_DOCUMENT_3"]))))), ((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"])))>0, (-1.0*((data["EXT_SOURCE_3"]))), data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"] )))) * 2.0)) + (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))) 
    v["i21"] = 0.049799*np.tanh(((data["NEW_CREDIT_TO_GOODS_RATIO"]) + ((-1.0*((((data["NEW_EXT_SOURCES_MEAN"]) + (((data["APPROVED_AMT_GOODS_PRICE_MIN"]) - (((((((((data["INSTAL_PAYMENT_DIFF_MAX"]) * 2.0)) - (data["POS_MONTHS_BALANCE_SIZE"]))) * 2.0)) * 2.0))))))))))) 
    v["i22"] = 0.049870*np.tanh(((((np.where(data["NEW_CAR_TO_BIRTH_RATIO"]>0, data["REFUSED_DAYS_DECISION_MAX"], np.maximum(((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"])), ((data["AMT_ANNUITY"]))) )) - (((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) + (data["CODE_GENDER"]))))) - (((data["NEW_EXT_SOURCES_MEAN"]) - (data["PREV_NAME_YIELD_GROUP_high_MEAN"]))))) 
    v["i23"] = 0.047896*np.tanh(((((((np.where(data["CC_AMT_BALANCE_MAX"] < -99998, ((data["DAYS_EMPLOYED"]) - (((np.maximum(((data["APPROVED_AMT_DOWN_PAYMENT_MAX"])), ((data["FLOORSMAX_AVG"])))) * 2.0))), ((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) * 2.0) )) * 2.0)) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)) 
    v["i24"] = 0.046500*np.tanh(((((((((((((data["PREV_CNT_PAYMENT_MEAN"]) - (data["POS_MONTHS_BALANCE_SIZE"]))) * 2.0)) - (data["NEW_EXT_SOURCES_MEAN"]))) + (((data["PREV_NAME_YIELD_GROUP_high_MEAN"]) + (np.tanh((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))))))) + (data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))) * 2.0)) 
    v["i25"] = 0.048000*np.tanh(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (((data["NEW_EXT_SOURCES_MEAN"]) - (((data["FLAG_DOCUMENT_3"]) - (np.maximum(((((np.maximum(((data["NEW_CAR_TO_BIRTH_RATIO"])), ((((((data["INSTAL_AMT_PAYMENT_MIN"]) * 2.0)) * 2.0))))) * 2.0))), ((data["BURO_CREDIT_ACTIVE_Closed_MEAN"])))))))))) 
    v["i26"] = 0.047900*np.tanh(((np.where(data["NAME_EDUCATION_TYPE_Higher_education"]>0, data["CC_AMT_RECIVABLE_MAX"], np.where(data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]>0, data["CC_AMT_RECIVABLE_MEAN"], ((np.where(data["DAYS_EMPLOYED"]<0, (-1.0*((data["BURO_CREDIT_ACTIVE_Closed_MEAN"]))), (-1.0*((data["EXT_SOURCE_1"]))) )) * 2.0) ) )) * 2.0)) 
    v["i27"] = 0.049976*np.tanh(((((((((((data["PREV_NAME_PRODUCT_TYPE_walk_in_MEAN"]) - (((data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"]) - (data["APPROVED_DAYS_DECISION_MIN"]))))) - (data["APPROVED_AMT_ANNUITY_MEAN"]))) + (data["NEW_DOC_IND_KURT"]))) - (((data["INSTAL_AMT_PAYMENT_MIN"]) - (data["INSTAL_AMT_INSTALMENT_MAX"]))))) * 2.0)) 
    v["i28"] = 0.049718*np.tanh(((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], np.where(data["POS_MONTHS_BALANCE_SIZE"]>0, data["APPROVED_CNT_PAYMENT_MEAN"], np.where(data["INSTAL_AMT_PAYMENT_MIN"]>0, data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"], (-1.0*((data["NEW_SOURCES_PROD"]))) ) ) )) * 2.0)) * 2.0)) 
    v["i29"] = 0.045700*np.tanh(((((((((data["INSTAL_PAYMENT_DIFF_SUM"]) + (data["PREV_CNT_PAYMENT_MEAN"]))) + (((((data["PREV_NAME_CLIENT_TYPE_New_MEAN"]) + (((data["NEW_CREDIT_TO_GOODS_RATIO"]) + (data["DEF_30_CNT_SOCIAL_CIRCLE"]))))) + (data["PREV_NAME_YIELD_GROUP_XNA_MEAN"]))))) - (data["CODE_GENDER"]))) * 2.0)) 
    v["i30"] = 0.049198*np.tanh(((((((((np.tanh((data["AMT_ANNUITY"]))) + (((data["PREV_CNT_PAYMENT_SUM"]) - (np.maximum(((data["NEW_CAR_TO_EMPLOY_RATIO"])), ((data["POS_COUNT"])))))))) - (((data["INSTAL_AMT_PAYMENT_MIN"]) + (data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"]))))) * 2.0)) * 2.0)) 
    v["i31"] = 0.049397*np.tanh(((((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, ((((data["INSTAL_PAYMENT_DIFF_MAX"]) * 2.0)) - (data["APPROVED_AMT_ANNUITY_MEAN"])), ((data["APPROVED_CNT_PAYMENT_MEAN"]) + (((data["ACTIVE_DAYS_CREDIT_MEAN"]) - (data["BURO_CREDIT_ACTIVE_Closed_MEAN"])))) )) + (data["DEF_60_CNT_SOCIAL_CIRCLE"]))) * 2.0)) 
    v["i32"] = 0.048984*np.tanh(((((data["REGION_RATING_CLIENT_W_CITY"]) + (np.where(((data["APPROVED_AMT_ANNUITY_MAX"]) - (((data["INSTAL_DPD_MEAN"]) - (data["CODE_GENDER"]))))<0, ((data["REGION_RATING_CLIENT_W_CITY"]) - (data["NEW_CAR_TO_BIRTH_RATIO"])), data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"] )))) - (data["PREV_NAME_PORTFOLIO_POS_MEAN"]))) 
    v["i33"] = 0.049560*np.tanh(((((data["AMT_ANNUITY"]) - (((np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]>0, data["EXT_SOURCE_1"], ((np.maximum(((((data["NAME_FAMILY_STATUS_Married"]) + (data["EXT_SOURCE_3"])))), ((data["APPROVED_AMT_DOWN_PAYMENT_MAX"])))) * 2.0) )) - (data["REGION_RATING_CLIENT_W_CITY"]))))) * 2.0)) 
    v["i34"] = 0.049700*np.tanh(((((((np.maximum(((data["NEW_CREDIT_TO_GOODS_RATIO"])), ((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])))) + (((((((data["PREV_NAME_CLIENT_TYPE_New_MEAN"]) + (((data["APPROVED_CNT_PAYMENT_MEAN"]) - (data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]))))) - (data["PREV_CODE_REJECT_REASON_XAP_MEAN"]))) * 2.0)))) * 2.0)) * 2.0)) 
    v["i35"] = 0.049646*np.tanh(np.where(data["POS_SK_DPD_DEF_MAX"]<0, ((((((((data["FLAG_WORK_PHONE"]) + (((np.where(data["EXT_SOURCE_1"] < -99998, data["DAYS_EMPLOYED"], data["CC_CNT_DRAWINGS_CURRENT_MEAN"] )) * 2.0)))) * 2.0)) * 2.0)) * 2.0), 3.141593 )) 
    v["i36"] = 0.049390*np.tanh(((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) + (((data["BURO_CREDIT_ACTIVE_Active_MEAN"]) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (((((np.where(data["CLOSED_DAYS_CREDIT_VAR"] < -99998, ((data["INSTAL_AMT_PAYMENT_SUM"]) * (data["CLOSED_DAYS_CREDIT_VAR"])), data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"] )) * 2.0)) * 2.0)))))))) 
    v["i37"] = 0.049750*np.tanh(((((np.where(np.where(data["PREV_NAME_CLIENT_TYPE_Repeater_MEAN"]>0, data["PREV_APP_CREDIT_PERC_MEAN"], data["NEW_CAR_TO_BIRTH_RATIO"] )>0, ((data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"]) * 2.0), (-1.0*((np.where(data["OCCUPATION_TYPE_Core_staff"]<0, data["ENTRANCES_MEDI"], data["OCCUPATION_TYPE_Core_staff"] )))) )) * 2.0)) * 2.0)) 
    v["i38"] = 0.049300*np.tanh(((((((((np.maximum(((np.minimum(((data["REGION_RATING_CLIENT_W_CITY"])), ((data["AMT_ANNUITY"]))))), ((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) * 2.0))))) - (((data["APPROVED_AMT_ANNUITY_MEAN"]) - (data["DEF_30_CNT_SOCIAL_CIRCLE"]))))) + (data["INSTAL_PAYMENT_DIFF_MEAN"]))) * 2.0)) * 2.0)) 
    v["i39"] = 0.049060*np.tanh(((((data["ORGANIZATION_TYPE_Self_employed"]) + (((np.where(data["POS_SK_DPD_DEF_MEAN"]<0, data["REG_CITY_NOT_LIVE_CITY"], (8.0) )) - (np.where(data["BURO_AMT_CREDIT_SUM_DEBT_SUM"]<0, np.maximum(((data["NAME_FAMILY_STATUS_Married"])), ((data["POS_SK_DPD_DEF_MAX"]))), data["NEW_CAR_TO_EMPLOY_RATIO"] )))))) * 2.0)) 
    v["i40"] = 0.048602*np.tanh(((((np.where(np.maximum(((data["CC_AMT_CREDIT_LIMIT_ACTUAL_SUM"])), ((np.maximum(((np.where(data["INSTAL_DBD_MEAN"]<0, data["NEW_RATIO_BURO_DAYS_CREDIT_MAX"], data["NAME_FAMILY_STATUS_Married"] ))), ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))))))<0, (-1.0*((data["FLOORSMAX_AVG"]))), data["APPROVED_CNT_PAYMENT_MEAN"] )) * 2.0)) * 2.0)) 
    v["i41"] = 0.048139*np.tanh(np.where(np.maximum(((data["INSTAL_DPD_MEAN"])), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"])))<0, ((np.where(data["NEW_EXT_SOURCES_MEAN"]>0, data["DAYS_ID_PUBLISH"], data["ACTIVE_DAYS_CREDIT_MEAN"] )) - (np.where(data["NEW_CAR_TO_EMPLOY_RATIO"]>0, 3.141593, data["NAME_EDUCATION_TYPE_Higher_education"] ))), 3.141593 )) 
    v["i42"] = 0.048501*np.tanh(((((((np.maximum(((data["APPROVED_CNT_PAYMENT_MEAN"])), ((data["FLAG_WORK_PHONE"])))) - (data["CODE_GENDER"]))) + (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - (data["PREV_AMT_APPLICATION_MEAN"]))) * 2.0), data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_MAX"] )))) * 2.0)) 
    v["i43"] = 0.049898*np.tanh(((((np.where(data["APPROVED_AMT_CREDIT_MEAN"]>0, data["APPROVED_CNT_PAYMENT_SUM"], ((data["AMT_ANNUITY"]) - (data["PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN"])) )) - (np.maximum(((data["NEW_DOC_IND_AVG"])), ((data["PREV_NAME_YIELD_GROUP_low_action_MEAN"])))))) - (((data["INSTAL_DBD_SUM"]) + (data["NAME_INCOME_TYPE_State_servant"]))))) 
    v["i44"] = 0.049002*np.tanh(((((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])), ((((np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((data["NEW_CREDIT_TO_GOODS_RATIO"])))) - (np.maximum(((np.where(data["POS_SK_DPD_DEF_MAX"]<0, data["NAME_FAMILY_STATUS_Married"], data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"] ))), ((data["CLOSED_DAYS_CREDIT_MAX"]))))))))) * 2.0)) * 2.0)) 
    v["i45"] = 0.049600*np.tanh(((((((((((np.maximum(((np.where(data["CC_AMT_TOTAL_RECEIVABLE_MEAN"] < -99998, data["INSTAL_DPD_MEAN"], data["CC_AMT_RECEIVABLE_PRINCIPAL_MIN"] ))), ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])))) * 2.0)) * 2.0)) * 2.0)) + (np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((data["APPROVED_CNT_PAYMENT_SUM"])))))) * 2.0)) 
    v["i46"] = 0.049079*np.tanh((((((((((-1.0*((np.maximum(((data["EXT_SOURCE_1"])), ((data["POS_COUNT"]))))))) + (np.where(np.maximum(((data["APPROVED_CNT_PAYMENT_SUM"])), ((data["INSTAL_DBD_SUM"])))<0, data["OCCUPATION_TYPE_Laborers"], data["APPROVED_CNT_PAYMENT_SUM"] )))) * 2.0)) * 2.0)) * 2.0)) 
    v["i47"] = 0.049180*np.tanh(((((np.minimum(((((((((data["DAYS_REGISTRATION"]) - (data["BURO_CREDIT_TYPE_Mortgage_MEAN"]))) - (data["OCCUPATION_TYPE_Core_staff"]))) - (data["BURO_CREDIT_TYPE_Mortgage_MEAN"])))), ((((data["INSTAL_PAYMENT_DIFF_SUM"]) - (data["APPROVED_HOUR_APPR_PROCESS_START_MAX"])))))) - (data["NAME_EDUCATION_TYPE_Incomplete_higher"]))) * 2.0)) 
    v["i48"] = 0.049515*np.tanh(((((((np.where(data["CC_AMT_RECEIVABLE_PRINCIPAL_SUM"] < -99998, ((data["AMT_ANNUITY"]) - (data["APPROVED_AMT_ANNUITY_MEAN"])), data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"] )) + (np.maximum(((data["OCCUPATION_TYPE_Drivers"])), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"])))))) * 2.0)) * 2.0)) 
    v["i49"] = 0.049802*np.tanh(((data["INSTAL_PAYMENT_DIFF_MAX"]) + (np.where(data["BURO_CREDIT_ACTIVE_Active_MEAN"]<0, np.where(data["BURO_DAYS_CREDIT_MAX"]<0, data["DAYS_LAST_PHONE_CHANGE"], data["BURO_CREDIT_ACTIVE_Active_MEAN"] ), ((data["FLAG_WORK_PHONE"]) - (((data["PREV_HOUR_APPR_PROCESS_START_MEAN"]) - (((data["ACTIVE_DAYS_CREDIT_MAX"]) * 2.0))))) )))) 
    v["i50"] = 0.049043*np.tanh((-1.0*((np.where(data["EXT_SOURCE_1"]>0, data["NEW_EXT_SOURCES_MEAN"], ((((((((data["NEW_EXT_SOURCES_MEAN"]) - (((np.tanh((np.tanh((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)))))) * 2.0)))) * 2.0)) * 2.0)) * 2.0) ))))) 
    v["i51"] = 0.047443*np.tanh(((((data["ORGANIZATION_TYPE_Construction"]) + (((data["REG_CITY_NOT_LIVE_CITY"]) + (((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) + (np.maximum(((data["NEW_SCORES_STD"])), ((((np.maximum(((data["PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN"])), ((data["PREV_CODE_REJECT_REASON_SCOFR_MEAN"])))) + (data["PREV_CNT_PAYMENT_MEAN"])))))))))))) * 2.0)) 
    v["i52"] = 0.049973*np.tanh(((data["REGION_RATING_CLIENT_W_CITY"]) + (np.where(data["NEW_EXT_SOURCES_MEAN"]<0, ((((np.where(data["POS_SK_DPD_DEF_MAX"]<0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], data["POS_MONTHS_BALANCE_MEAN"] )) * 2.0)) * 2.0), ((((data["POS_MONTHS_BALANCE_MEAN"]) - (data["NEW_CAR_TO_BIRTH_RATIO"]))) * 2.0) )))) 
    v["i53"] = 0.047044*np.tanh(((((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"], data["NEW_CREDIT_TO_ANNUITY_RATIO"] )) - (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))) - (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))) + (np.maximum(((data["ACTIVE_DAYS_CREDIT_ENDDATE_MAX"])), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"])))))) 
    v["i54"] = 0.049700*np.tanh(((((((((((((data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__high_MEAN"]) + (np.maximum(((data["PREV_PRODUCT_COMBINATION_Cash_Street__high_MEAN"])), ((data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"])))))) - (data["INSTAL_AMT_PAYMENT_SUM"]))) * 2.0)) * 2.0)) - (np.maximum(((data["PREV_APP_CREDIT_PERC_MEAN"])), ((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])))))) * 2.0)) 
    v["i55"] = 0.049609*np.tanh(((((np.where(data["EXT_SOURCE_3"] < -99998, data["REFUSED_DAYS_DECISION_MAX"], ((data["EXT_SOURCE_3"]) * (data["EXT_SOURCE_3"])) )) - (data["BURO_DAYS_CREDIT_MAX"]))) + (np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, data["EXT_SOURCE_3"], -1.0 )))) 
    v["i56"] = 0.049496*np.tanh(((np.maximum(((data["DAYS_ID_PUBLISH"])), ((data["ACTIVE_DAYS_CREDIT_MAX"])))) - (np.maximum(((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])), ((((((((data["NEW_DOC_IND_AVG"]) + (data["NAME_INCOME_TYPE_State_servant"]))) + (data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"]))) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))) 
    v["i57"] = 0.048440*np.tanh(((np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"]<0, data["AMT_ANNUITY"], data["APPROVED_CNT_PAYMENT_MEAN"] )) + (((np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"])))) + (((data["ORGANIZATION_TYPE_Construction"]) + (((data["EXT_SOURCE_2"]) * (data["INSTAL_DBD_SUM"]))))))))) 
    v["i58"] = 0.044197*np.tanh(((((np.where(data["CODE_GENDER"]<0, np.maximum(((data["WALLSMATERIAL_MODE_Stone__brick"])), ((data["FLAG_DOCUMENT_3"]))), np.where(np.where(data["PREV_PRODUCT_COMBINATION_Cash_MEAN"]<0, data["FLAG_WORK_PHONE"], data["CC_CNT_DRAWINGS_CURRENT_VAR"] )<0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], data["FLAG_DOCUMENT_3"] ) )) * 2.0)) * 2.0)) 
    v["i59"] = 0.049650*np.tanh(((((data["WALLSMATERIAL_MODE_Stone__brick"]) - (np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]>0, data["NEW_RATIO_BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"], np.maximum(((np.maximum(((data["NEW_SOURCES_PROD"])), ((data["ACTIVE_AMT_CREDIT_SUM_MAX"]))))), ((np.maximum(((data["APPROVED_AMT_CREDIT_MIN"])), ((data["BURO_STATUS_0_MEAN_MEAN"])))))) )))) - (data["NAME_INCOME_TYPE_Commercial_associate"]))) 
    v["i60"] = 0.047884*np.tanh(np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"]>0, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_MAX"], np.where(((data["NEW_RATIO_BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]) * (data["NEW_EXT_SOURCES_MEAN"])) < -99998, data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"], ((((data["ORGANIZATION_TYPE_Self_employed"]) + ((((data["PREV_NAME_GOODS_CATEGORY_Furniture_MEAN"]) < (data["NEW_EXT_SOURCES_MEAN"]))*1.)))) * 2.0) ) )) 
    v["i61"] = 0.048710*np.tanh(np.where(data["EXT_SOURCE_3"] < -99998, data["REFUSED_CNT_PAYMENT_SUM"], ((((((((-1.0) - (np.minimum(((((data["EXT_SOURCE_2"]) / 2.0))), ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])))))) * 2.0)) - (data["EXT_SOURCE_3"]))) - (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) )) 
    v["i62"] = 0.049366*np.tanh(np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"]>0, data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"], ((np.where(data["NEW_RATIO_BURO_DAYS_CREDIT_MAX"]>0, data["CC_CNT_DRAWINGS_CURRENT_SUM"], ((((data["NEW_RATIO_PREV_AMT_CREDIT_MIN"]) * (data["OCCUPATION_TYPE_Accountants"]))) - (data["AMT_REQ_CREDIT_BUREAU_YEAR"])) )) - (data["NEW_PHONE_TO_EMPLOY_RATIO"])) )) 
    v["i63"] = 0.048000*np.tanh(((np.where(data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]>0, data["BURO_DAYS_CREDIT_MEAN"], np.where(data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"] < -99998, data["DEF_30_CNT_SOCIAL_CIRCLE"], ((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]) * 2.0)) * 2.0) ) )) - (np.where(data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]>0, data["ACTIVE_MONTHS_BALANCE_MIN_MIN"], data["NAME_FAMILY_STATUS_Married"] )))) 
    v["i64"] = 0.048498*np.tanh(((np.where(data["AMT_GOODS_PRICE"]<0, np.maximum(((data["AMT_ANNUITY"])), ((((((((((data["AMT_ANNUITY"]) - (data["PREV_AMT_ANNUITY_MIN"]))) * 2.0)) - (data["PREV_AMT_ANNUITY_MIN"]))) * 2.0)))), data["INSTAL_AMT_INSTALMENT_MAX"] )) - (data["NEW_DOC_IND_STD"]))) 
    v["i65"] = 0.045500*np.tanh(((data["APPROVED_CNT_PAYMENT_SUM"]) - (((data["INSTAL_COUNT"]) - (np.where(data["NEW_CREDIT_TO_GOODS_RATIO"]<0, ((data["INSTAL_DBD_MAX"]) - (data["PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN"])), (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (data["NEW_EXT_SOURCES_MEAN"]))) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))/2.0) )))))) 
    v["i66"] = 0.049995*np.tanh(((((((np.maximum(((data["EXT_SOURCE_3"])), ((data["NEW_EXT_SOURCES_MEAN"])))) + (((data["NEW_EXT_SOURCES_MEAN"]) * (((data["NEW_EXT_SOURCES_MEAN"]) + (data["INSTAL_PAYMENT_DIFF_MEAN"]))))))) + (((data["NEW_EXT_SOURCES_MEAN"]) + (data["INSTAL_PAYMENT_DIFF_MEAN"]))))) * 2.0)) 
    v["i67"] = 0.039797*np.tanh(((((np.maximum(((data["PREV_CHANNEL_TYPE_Contact_center_MEAN"])), ((np.where(data["PREV_DAYS_DECISION_MIN"]>0, ((np.maximum(((((np.maximum(((data["INSTAL_DPD_MEAN"])), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"])))) * 2.0))), ((data["CC_AMT_RECIVABLE_VAR"])))) * 2.0), data["NEW_CREDIT_TO_ANNUITY_RATIO"] ))))) * 2.0)) * 2.0)) 
    v["i68"] = 0.049954*np.tanh(((((((data["NAME_FAMILY_STATUS_Separated"]) + (np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((((np.where(data["INSTAL_COUNT"]>0, data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"], data["PREV_DAYS_DECISION_MEAN"] )) - ((((data["INSTAL_DPD_MAX"]) < (data["PREV_NAME_TYPE_SUITE_Other_B_MEAN"]))*1.))))))))) * 2.0)) * 2.0)) 
    v["i69"] = 0.045722*np.tanh(((((np.where(data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"]>0, -1.0, ((np.where(data["INSTAL_AMT_INSTALMENT_SUM"]>0, data["ACTIVE_AMT_CREDIT_SUM_SUM"], (-1.0*((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, data["PREV_NAME_YIELD_GROUP_low_action_MEAN"], data["PREV_NAME_PRODUCT_TYPE_XNA_MEAN"] )))) )) * 2.0) )) * 2.0)) * 2.0)) 
    v["i70"] = 0.048797*np.tanh(((np.where(np.maximum(((data["APPROVED_AMT_CREDIT_MAX"])), ((data["AMT_ANNUITY"])))<0, data["AMT_ANNUITY"], (-1.0*((data["PREV_AMT_ANNUITY_MEAN"]))) )) + (((data["POS_SK_DPD_DEF_MAX"]) + (np.maximum(((data["ACTIVE_DAYS_CREDIT_MAX"])), ((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"])))))))) 
    v["i71"] = 0.047984*np.tanh((-1.0*((np.where(data["REGION_RATING_CLIENT_W_CITY"]>0, data["BURO_CREDIT_TYPE_Mortgage_MEAN"], np.where(data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]>0, data["CC_AMT_PAYMENT_TOTAL_CURRENT_SUM"], ((np.maximum(((np.maximum(((data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"])), ((data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"]))))), ((data["CC_AMT_PAYMENT_TOTAL_CURRENT_SUM"])))) + (data["NEW_PHONE_TO_EMPLOY_RATIO"])) ) ))))) 
    v["i72"] = 0.049980*np.tanh(np.where(data["EXT_SOURCE_3"] < -99998, data["REFUSED_AMT_APPLICATION_MAX"], ((np.maximum(((np.maximum((((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"]) > (((data["OCCUPATION_TYPE_Core_staff"]) / 2.0)))*1.))), ((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"]) * (data["BURO_CREDIT_TYPE_Microloan_MEAN"]))))))), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"])))) * 2.0) )) 
    v["i73"] = 0.049350*np.tanh(((((((((((data["BURO_AMT_CREDIT_SUM_DEBT_SUM"]) - ((((data["BURO_AMT_CREDIT_SUM_MEAN"]) > (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))*1.)))) - (data["BURO_AMT_CREDIT_SUM_MEAN"]))) + (np.tanh((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))))) - (data["BURO_AMT_CREDIT_SUM_MEAN"]))) - (data["BURO_AMT_CREDIT_SUM_MEAN"]))) 
    v["i74"] = 0.049056*np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, np.where(data["PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN"]<0, data["DAYS_REGISTRATION"], data["NEW_CREDIT_TO_ANNUITY_RATIO"] ), np.where(data["OWN_CAR_AGE"] < -99998, ((((data["REGION_POPULATION_RELATIVE"]) - (data["PREV_NAME_CONTRACT_TYPE_Cash_loans_MEAN"]))) - (data["INSTAL_AMT_PAYMENT_SUM"])), -2.0 ) )) 
    v["i75"] = 0.041842*np.tanh(np.maximum(((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])), ((np.where(data["PREV_NAME_PAYMENT_TYPE_XNA_MEAN"]<0, data["APPROVED_AMT_GOODS_PRICE_MAX"], ((((((data["PREV_CNT_PAYMENT_MEAN"]) + (data["ORGANIZATION_TYPE_Construction"]))) + (data["NEW_CREDIT_TO_GOODS_RATIO"]))) + (((data["ORGANIZATION_TYPE_Transport__type_3"]) + (data["PREV_PRODUCT_COMBINATION_Cash_Street__middle_MEAN"])))) ))))) 
    v["i76"] = 0.049296*np.tanh((((((data["NEW_EXT_SOURCES_MEAN"]) > (data["NAME_INCOME_TYPE_State_servant"]))*1.)) + (((((((data["OCCUPATION_TYPE_Drivers"]) - (data["NEW_DOC_IND_AVG"]))) - (data["INSTAL_AMT_INSTALMENT_SUM"]))) + (np.where(data["PREV_AMT_CREDIT_MEAN"]<0, data["REGION_POPULATION_RELATIVE"], data["OBS_60_CNT_SOCIAL_CIRCLE"] )))))) 
    v["i77"] = 0.048658*np.tanh(((((data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"]) - (data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"]))) + (np.where(data["POS_SK_DPD_MEAN"] < -99998, data["APPROVED_CNT_PAYMENT_SUM"], ((((((data["APPROVED_AMT_APPLICATION_MAX"]) - (data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"]))) - (data["WEEKDAY_APPR_PROCESS_START_MONDAY"]))) - (data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"])) )))) 
    v["i78"] = 0.049800*np.tanh(np.where(data["EXT_SOURCE_1"] < -99998, ((((data["DAYS_BIRTH"]) * 2.0)) * 2.0), ((((((data["ORGANIZATION_TYPE_Medicine"]) - (data["EXT_SOURCE_1"]))) - (((((data["DAYS_BIRTH"]) * 2.0)) * 2.0)))) - (data["EXT_SOURCE_1"])) )) 
    v["i79"] = 0.048000*np.tanh(np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"])), ((np.where(data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"]<0, ((np.where(data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"] < -99998, ((data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]) - (data["CC_AMT_PAYMENT_CURRENT_SUM"])), data["CC_NAME_CONTRACT_STATUS_Active_SUM"] )) - (data["NEW_PHONE_TO_EMPLOY_RATIO"])), data["EXT_SOURCE_2"] ))))) 
    v["i80"] = 0.045128*np.tanh(np.where(data["APPROVED_AMT_APPLICATION_MAX"] < -99998, data["NEW_DOC_IND_AVG"], np.where(data["WEEKDAY_APPR_PROCESS_START_SATURDAY"]<0, np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, (((data["POS_COUNT"]) < (data["PREV_AMT_GOODS_PRICE_MAX"]))*1.), np.maximum(((data["APPROVED_AMT_APPLICATION_MAX"])), ((data["APARTMENTS_MEDI"]))) ), data["REFUSED_DAYS_DECISION_MAX"] ) )) 
    v["i81"] = 0.049278*np.tanh((((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < ((((data["NAME_HOUSING_TYPE_Rented_apartment"]) + (data["NEW_DOC_IND_KURT"]))/2.0)))*1.)) + (np.minimum(((np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((data["REG_CITY_NOT_LIVE_CITY"]))))), ((data["REGION_RATING_CLIENT_W_CITY"])))))) * 2.0)) * 2.0)) 
    v["i82"] = 0.049000*np.tanh(((data["BURO_CREDIT_TYPE_Credit_card_MEAN"]) - (np.where(data["PREV_NAME_CONTRACT_STATUS_Approved_MEAN"]>0, data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"], np.maximum(((data["BURO_CREDIT_TYPE_Car_loan_MEAN"])), ((((data["BURO_CREDIT_TYPE_Credit_card_MEAN"]) + (((np.maximum(((data["DAYS_BIRTH"])), ((data["ORGANIZATION_TYPE_School"])))) + (data["PREV_NAME_CONTRACT_STATUS_Approved_MEAN"]))))))) )))) 
    v["i83"] = 0.049400*np.tanh(np.where(data["POS_SK_DPD_DEF_MAX"]>0, data["INSTAL_DPD_MAX"], ((((np.where(data["AMT_ANNUITY"]<0, (-1.0*(((((data["INSTAL_AMT_INSTALMENT_SUM"]) > (np.maximum(((data["AMT_CREDIT"])), ((data["PREV_AMT_GOODS_PRICE_MAX"])))))*1.)))), data["NEW_CREDIT_TO_GOODS_RATIO"] )) * 2.0)) * 2.0) )) 
    v["i84"] = 0.046006*np.tanh(np.where(data["OWN_CAR_AGE"]>0, ((data["PREV_AMT_CREDIT_MEAN"]) - (data["DAYS_BIRTH"])), ((data["FLAG_WORK_PHONE"]) * (((data["ORGANIZATION_TYPE_Kindergarten"]) + (((data["WEEKDAY_APPR_PROCESS_START_SUNDAY"]) + (np.maximum(((data["DAYS_BIRTH"])), ((data["NEW_ANNUITY_TO_INCOME_RATIO"]))))))))) )) 
    v["i85"] = 0.009576*np.tanh(((((np.where(data["EXT_SOURCE_3"] < -99998, data["NEW_EXT_SOURCES_MEAN"], np.where(data["NEW_DOC_IND_KURT"]<0, data["ORGANIZATION_TYPE_Self_employed"], ((data["PREV_NAME_SELLER_INDUSTRY_XNA_MEAN"]) * (np.where(data["CC_CNT_DRAWINGS_CURRENT_VAR"]<0, data["CLOSED_DAYS_CREDIT_MEAN"], data["CC_AMT_BALANCE_MEAN"] ))) ) )) * 2.0)) * 2.0)) 
    v["i86"] = 0.042600*np.tanh((((((((data["APPROVED_AMT_APPLICATION_MIN"]) < (data["NEW_EXT_SOURCES_MEAN"]))*1.)) - (((data["NEW_DOC_IND_AVG"]) + (((data["NEW_EXT_SOURCES_MEAN"]) - (np.tanh((((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)) * 2.0)))))))))) - (data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"]))) 
    v["i87"] = 0.048500*np.tanh((((-1.0*((np.where(data["REFUSED_AMT_GOODS_PRICE_MAX"]>0, data["DAYS_LAST_PHONE_CHANGE"], (-1.0*((((((((data["CC_CNT_DRAWINGS_CURRENT_MAX"]) - (np.where(data["BURO_CREDIT_TYPE_Car_loan_MEAN"]<0, data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"], data["POS_MONTHS_BALANCE_MAX"] )))) * 2.0)) * 2.0)))) ))))) * 2.0)) 
    v["i88"] = 0.049749*np.tanh(np.where(((data["PREV_DAYS_DECISION_MEAN"]) + (np.tanh((((((data["PREV_NAME_PAYMENT_TYPE_XNA_MEAN"]) + (data["PREV_NAME_PAYMENT_TYPE_XNA_MEAN"]))) + (data["REFUSED_AMT_ANNUITY_MIN"]))))))<0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (-1.0*((((data["REFUSED_AMT_ANNUITY_MIN"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))) )) 
    v["i89"] = 0.039400*np.tanh(np.where(data["INSTAL_DPD_MEAN"]<0, np.where(data["BURO_AMT_CREDIT_SUM_DEBT_SUM"]<0, np.where(data["ACTIVE_CREDIT_DAY_OVERDUE_MAX"] < -99998, data["PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN"], np.where(data["PREV_AMT_DOWN_PAYMENT_MEAN"]>0, data["INSTAL_AMT_PAYMENT_MIN"], data["PREV_CNT_PAYMENT_SUM"] ) ), data["ORGANIZATION_TYPE_Business_Entity_Type_3"] ), data["INSTAL_DAYS_ENTRY_PAYMENT_MEAN"] )) 
    v["i90"] = 0.045801*np.tanh(((((np.where(data["ACTIVE_AMT_CREDIT_SUM_SUM"]>0, ((((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) - (data["ACTIVE_AMT_CREDIT_SUM_MAX"]))) * 2.0), np.where(data["DEF_60_CNT_SOCIAL_CIRCLE"]<0, data["POS_SK_DPD_DEF_MAX"], (-1.0*((data["ACTIVE_AMT_CREDIT_SUM_MAX"]))) ) )) + (data["NAME_EDUCATION_TYPE_Lower_secondary"]))) * 2.0)) 
    v["i91"] = 0.049800*np.tanh(((((data["PREV_CNT_PAYMENT_MEAN"]) - ((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]) < (np.tanh((data["PREV_AMT_CREDIT_MEAN"]))))*1.)))) * (((((data["PREV_DAYS_DECISION_MAX"]) + (data["PREV_NAME_YIELD_GROUP_high_MEAN"]))) + (((data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"]) + (data["OBS_60_CNT_SOCIAL_CIRCLE"]))))))) 
    v["i92"] = 0.049738*np.tanh(np.where(data["NEW_DOC_IND_STD"]>0, np.where(data["CLOSED_MONTHS_BALANCE_SIZE_SUM"] < -99998, data["INSTAL_PAYMENT_DIFF_MAX"], ((((np.where(data["BURO_STATUS_1_MEAN_MEAN"]<0, data["PREV_NAME_CONTRACT_TYPE_Revolving_loans_MEAN"], data["PREV_NAME_PRODUCT_TYPE_XNA_MEAN"] )) - (data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"]))) - (data["BURO_CREDIT_TYPE_Mortgage_MEAN"])) ), data["INSTAL_PAYMENT_DIFF_MAX"] )) 
    v["i93"] = 0.049512*np.tanh(((np.maximum(((data["ORGANIZATION_TYPE_Transport__type_3"])), ((data["INSTAL_AMT_PAYMENT_MIN"])))) + (((data["NAME_EDUCATION_TYPE_Lower_secondary"]) + (((data["INSTAL_AMT_PAYMENT_MIN"]) + (((data["ORGANIZATION_TYPE_Construction"]) - (np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"]>0, data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"], data["APPROVED_AMT_CREDIT_MIN"] )))))))))) 
    v["i94"] = 0.047402*np.tanh(np.where(data["NEW_DOC_IND_STD"]<0, data["APPROVED_DAYS_DECISION_MIN"], np.maximum(((data["CC_CNT_DRAWINGS_CURRENT_VAR"])), ((np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((np.maximum(((((data["POS_SK_DPD_MAX"]) / 2.0))), (((((data["PREV_NAME_CASH_LOAN_PURPOSE_Medicine_MEAN"]) < (((data["NEW_RATIO_PREV_DAYS_DECISION_MAX"]) / 2.0)))*1.)))))))))) )) 
    v["i95"] = 0.044002*np.tanh(np.where(data["CC_AMT_PAYMENT_TOTAL_CURRENT_SUM"]>0, data["NEW_RATIO_BURO_MONTHS_BALANCE_MIN_MIN"], ((data["NAME_HOUSING_TYPE_Municipal_apartment"]) + (np.maximum(((data["INSTAL_PAYMENT_DIFF_SUM"])), ((np.maximum(((data["ORGANIZATION_TYPE_Transport__type_3"])), ((np.where(data["REFUSED_AMT_CREDIT_MIN"] < -99998, data["NEW_SCORES_STD"], data["ACTIVE_AMT_CREDIT_SUM_SUM"] ))))))))) )) 
    v["i96"] = 0.050000*np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_MEAN"]<0, np.where(data["BURO_DAYS_CREDIT_MAX"]>0, data["CC_AMT_BALANCE_VAR"], np.maximum(((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"])), (((((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]) < (data["NEW_EXT_SOURCES_MEAN"]))*1.)))) ), np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]<0, data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"], data["BURO_DAYS_CREDIT_MAX"] ) )) 
    v["i97"] = 0.048906*np.tanh((((((np.maximum(((data["INSTAL_DBD_MAX"])), ((data["BURO_AMT_CREDIT_SUM_OVERDUE_MEAN"])))) + (np.where(data["OBS_60_CNT_SOCIAL_CIRCLE"]<0, data["ORGANIZATION_TYPE_Transport__type_3"], ((data["ACTIVE_DAYS_CREDIT_ENDDATE_MAX"]) + (data["PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN"])) )))/2.0)) - ((((data["NEW_RATIO_PREV_AMT_CREDIT_MIN"]) > (data["PREV_PRODUCT_COMBINATION_POS_household_without_interest_MEAN"]))*1.)))) 
    v["i98"] = 0.049860*np.tanh(((np.where(data["CODE_GENDER"]<0, data["NEW_ANNUITY_TO_INCOME_RATIO"], np.where(((data["DAYS_BIRTH"]) - (data["NEW_DOC_IND_AVG"]))<0, data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"], data["DAYS_ID_PUBLISH"] ) )) - ((((data["DAYS_BIRTH"]) + (((data["OCCUPATION_TYPE_Medicine_staff"]) * 2.0)))/2.0)))) 
    v["i99"] = 0.041003*np.tanh(np.maximum(((data["BURO_STATUS_1_MEAN_MEAN"])), ((((data["DAYS_REGISTRATION"]) * (((((data["ACTIVE_DAYS_CREDIT_MIN"]) - (data["INSTAL_AMT_INSTALMENT_MAX"]))) * (np.where(((data["BURO_CREDIT_ACTIVE_Closed_MEAN"]) + (data["DAYS_REGISTRATION"])) < -99998, data["APPROVED_AMT_CREDIT_MAX"], data["PREV_AMT_CREDIT_MIN"] ))))))))) 
    v["i100"] = 0.049800*np.tanh(np.where(data["EXT_SOURCE_3"] < -99998, np.where(data["PREV_AMT_APPLICATION_MEAN"] < -99998, data["PREV_NAME_PAYMENT_TYPE_XNA_MEAN"], ((data["NEW_EXT_SOURCES_MEAN"]) * 2.0) ), ((data["EXT_SOURCE_3"]) * (((((((data["INSTAL_PAYMENT_DIFF_MAX"]) + (data["NEW_EXT_SOURCES_MEAN"]))/2.0)) + (data["INSTAL_PAYMENT_DIFF_MEAN"]))/2.0))) )) 
    v["i101"] = 0.039360*np.tanh(((((data["PREV_NAME_CASH_LOAN_PURPOSE_XNA_MEAN"]) * (np.maximum((((((data["BURO_CREDIT_TYPE_Credit_card_MEAN"]) + (data["ORGANIZATION_TYPE_Military"]))/2.0))), ((np.where(((data["BURO_CREDIT_TYPE_Credit_card_MEAN"]) * (data["PREV_CNT_PAYMENT_MEAN"])) < -99998, data["PREV_CNT_PAYMENT_SUM"], data["REFUSED_RATE_DOWN_PAYMENT_MAX"] ))))))) * 2.0)) 
    v["i102"] = 0.047204*np.tanh((-1.0*((np.maximum(((((data["YEARS_BUILD_MEDI"]) - (data["NEW_DOC_IND_AVG"])))), ((((data["OCCUPATION_TYPE_High_skill_tech_staff"]) + (np.maximum(((data["CLOSED_AMT_CREDIT_SUM_MEAN"])), ((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), ((((data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]) + (data["OCCUPATION_TYPE_High_skill_tech_staff"]))))))))))))))))) 
    v["i103"] = 0.049790*np.tanh(np.where(data["PREV_NAME_GOODS_CATEGORY_Audio_Video_MEAN"]>0, data["INSTAL_PAYMENT_DIFF_SUM"], np.where(data["ACTIVE_MONTHS_BALANCE_MAX_MAX"]>0, data["NEW_RATIO_BURO_AMT_ANNUITY_MAX"], (-1.0*((((data["ORGANIZATION_TYPE_Medicine"]) + (np.maximum(((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])), (((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_MAX"]) > (data["BURO_AMT_CREDIT_SUM_DEBT_MAX"]))*1.))))))))) ) )) 
    v["i104"] = 0.049046*np.tanh((-1.0*(((((data["ORGANIZATION_TYPE_School"]) > (np.where((((data["AMT_GOODS_PRICE"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)>0, data["PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN"], ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) < (np.tanh((data["PREV_PRODUCT_COMBINATION_Cash_MEAN"]))))*1.))) )))*1.))))) 
    v["i105"] = 0.049750*np.tanh(np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"]>0, data["DAYS_LAST_PHONE_CHANGE"], ((np.where(data["INSTAL_DPD_MEAN"]>0, data["INSTAL_DAYS_ENTRY_PAYMENT_MEAN"], np.maximum(((((data["INSTAL_DAYS_ENTRY_PAYMENT_MEAN"]) * (data["CC_AMT_BALANCE_MEAN"])))), ((((data["PREV_DAYS_DECISION_MAX"]) - (data["APPROVED_APP_CREDIT_PERC_MIN"]))))) )) * 2.0) )) 
    v["i106"] = 0.045968*np.tanh(np.where(((data["DAYS_ID_PUBLISH"]) - (data["DAYS_BIRTH"]))<0, ((data["DAYS_ID_PUBLISH"]) * ((((((data["CNT_FAM_MEMBERS"]) + (((data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]) - (data["DAYS_BIRTH"]))))/2.0)) * 2.0))), data["DAYS_ID_PUBLISH"] )) 
    v["i107"] = 0.049699*np.tanh(((np.maximum(((data["APPROVED_CNT_PAYMENT_SUM"])), ((np.where(data["NEW_CAR_TO_EMPLOY_RATIO"] < -99998, data["DAYS_BIRTH"], data["INSTAL_AMT_INSTALMENT_MEAN"] ))))) - (((data["INSTAL_AMT_PAYMENT_SUM"]) + ((((np.minimum(((data["INSTAL_AMT_PAYMENT_SUM"])), ((data["PREV_AMT_CREDIT_MAX"])))) < (data["INSTAL_AMT_PAYMENT_SUM"]))*1.)))))) 
    v["i108"] = 0.007000*np.tanh(np.where(np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MAX"]<0, data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"], data["BURO_AMT_CREDIT_SUM_SUM"] )>0, (10.0), ((np.where(data["PREV_NAME_GOODS_CATEGORY_Photo___Cinema_Equipment_MEAN"]<0, np.where(data["NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MAX"]>0, data["WALLSMATERIAL_MODE_Stone__brick"], data["POS_SK_DPD_DEF_MAX"] ), data["NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MEAN"] )) * 2.0) )) 
    v["i109"] = 0.046106*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]<0, (-1.0*((((((data["NEW_LIVE_IND_SUM"]) - (((np.maximum(((data["PREV_AMT_APPLICATION_MIN"])), ((((data["BURO_CREDIT_ACTIVE_Sold_MEAN"]) + (data["BURO_CREDIT_TYPE_Microloan_MEAN"])))))) * 2.0)))) - (data["POS_SK_DPD_MAX"]))))), data["BURO_AMT_CREDIT_SUM_OVERDUE_MEAN"] )) 
    v["i110"] = 0.042843*np.tanh((-1.0*((np.where(data["AMT_INCOME_TOTAL"]>0, data["BURO_AMT_CREDIT_SUM_MEAN"], np.where(data["CC_AMT_RECEIVABLE_PRINCIPAL_VAR"]>0, data["ACTIVE_DAYS_CREDIT_ENDDATE_MAX"], np.where(data["FLAG_WORK_PHONE"]>0, data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"], np.where(data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"]>0, data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"], data["NEW_CREDIT_TO_GOODS_RATIO"] ) ) ) ))))) 
    v["i111"] = 0.049018*np.tanh(np.where(data["BURO_STATUS_X_MEAN_MEAN"] < -99998, ((np.where(data["PREV_NAME_YIELD_GROUP_high_MEAN"]>0, data["FLAG_WORK_PHONE"], np.tanh((data["LIVINGAREA_AVG"])) )) - (((data["EXT_SOURCE_2"]) + (data["NEW_INC_BY_ORG"])))), np.tanh((((data["EXT_SOURCE_2"]) * 2.0))) )) 
    v["i112"] = 0.047002*np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"]<0, np.where(data["BURO_DAYS_CREDIT_MAX"]<0, np.where(data["NEW_RATIO_PREV_DAYS_DECISION_MIN"]<0, data["INSTAL_PAYMENT_DIFF_SUM"], data["BURO_DAYS_CREDIT_MAX"] ), data["ACTIVE_AMT_CREDIT_SUM_SUM"] ), ((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) + ((-1.0*((data["NEW_RATIO_PREV_DAYS_DECISION_MIN"]))))) )) 
    v["i113"] = 0.040984*np.tanh(np.where(data["REFUSED_DAYS_DECISION_MEAN"]>0, ((data["PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN"]) * 2.0), np.maximum(((data["CC_AMT_INST_MIN_REGULARITY_VAR"])), ((((np.maximum(((data["BURO_AMT_CREDIT_SUM_OVERDUE_MEAN"])), ((np.where(data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"]>0, data["INSTAL_PAYMENT_DIFF_SUM"], data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] ))))) * 2.0)))) )) 
    v["i114"] = 0.045032*np.tanh((((np.maximum(((data["BURO_AMT_CREDIT_SUM_DEBT_MAX"])), ((((data["BURO_AMT_CREDIT_SUM_MAX"]) * 2.0))))) > (np.maximum(((((((data["PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN"]) + (((data["BURO_AMT_CREDIT_SUM_MEAN"]) * 2.0)))) + (((data["BURO_AMT_CREDIT_SUM_DEBT_MAX"]) * 2.0))))), ((data["BURO_AMT_CREDIT_SUM_MEAN"])))))*1.)) 
    v["i115"] = 0.045600*np.tanh(np.where((((data["NEW_EXT_SOURCES_MEAN"]) + (data["NEW_PHONE_TO_BIRTH_RATIO"]))/2.0)>0, data["APARTMENTS_MEDI"], np.where((((data["POS_COUNT"]) > (data["ORGANIZATION_TYPE_Business_Entity_Type_3"]))*1.)>0, data["NEW_EXT_SOURCES_MEAN"], ((data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"]) * (data["NEW_PHONE_TO_BIRTH_RATIO"])) ) )) 
    v["i116"] = 0.042206*np.tanh(((data["DAYS_BIRTH"]) * ((((((((((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) > (((data["DAYS_BIRTH"]) / 2.0)))*1.)) + (data["PREV_CHANNEL_TYPE_Stone_MEAN"]))) + (data["PREV_NAME_GOODS_CATEGORY_Consumer_Electronics_MEAN"]))) + (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))) + (data["PREV_NAME_GOODS_CATEGORY_Consumer_Electronics_MEAN"]))))) 
    v["i117"] = 0.042064*np.tanh(np.where(data["INSTAL_AMT_INSTALMENT_MEAN"]>0, data["PREV_WEEKDAY_APPR_PROCESS_START_THURSDAY_MEAN"], ((data["PREV_CODE_REJECT_REASON_SCO_MEAN"]) * (np.where(data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"] < -99998, data["NEW_EMPLOY_TO_BIRTH_RATIO"], np.where(data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"] < -99998, data["ACTIVE_CREDIT_DAY_OVERDUE_MAX"], (((data["NEW_SCORES_STD"]) > (data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]))*1.) ) ))) )) 
    v["i118"] = 0.046301*np.tanh(np.where(data["AMT_INCOME_TOTAL"]>0, data["NEW_DOC_IND_KURT"], np.where(data["NAME_EDUCATION_TYPE_Higher_education"]>0, data["AMT_CREDIT"], np.minimum(((((((data["DAYS_BIRTH"]) * 2.0)) * 2.0))), (((((((data["DAYS_BIRTH"]) < (data["PREV_NAME_PORTFOLIO_POS_MEAN"]))*1.)) * 2.0)))) ) )) 
    v["i119"] = 0.049880*np.tanh(np.where((((-1.0) > (data["AMT_GOODS_PRICE"]))*1.)>0, data["CC_AMT_DRAWINGS_ATM_CURRENT_MAX"], ((np.where(data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"]>0, data["INSTAL_AMT_PAYMENT_MAX"], (((((data["WEEKDAY_APPR_PROCESS_START_SUNDAY"]) < (((data["NEW_EXT_SOURCES_MEAN"]) / 2.0)))*1.)) * 2.0) )) * 2.0) )) 
    v["i120"] = 0.048360*np.tanh(np.where(np.tanh(((((data["APPROVED_DAYS_DECISION_MAX"]) + (data["PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN"]))/2.0)))<0, data["PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN"], np.where(data["BURO_AMT_CREDIT_SUM_DEBT_SUM"]>0, (((data["APPROVED_DAYS_DECISION_MAX"]) + (data["REFUSED_APP_CREDIT_PERC_VAR"]))/2.0), ((data["PREV_PRODUCT_COMBINATION_Cash_MEAN"]) - (data["CC_NAME_CONTRACT_STATUS_Active_VAR"])) ) )) 
    v["i121"] = 0.045502*np.tanh(np.where(np.where(((((data["INSTAL_DPD_MAX"]) / 2.0)) - (data["ORGANIZATION_TYPE_Transport__type_3"]))<0, data["NEW_EXT_SOURCES_MEAN"], data["APPROVED_RATE_DOWN_PAYMENT_MEAN"] )<0, (((data["PREV_NAME_PORTFOLIO_Cash_MEAN"]) < (((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)))*1.), data["REGION_RATING_CLIENT"] )) 
    v["i122"] = 0.048897*np.tanh((((data["BURO_DAYS_CREDIT_MAX"]) > (((np.minimum(((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"]) - (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])))), ((((data["BURO_DAYS_CREDIT_MAX"]) - (data["ORGANIZATION_TYPE_Transport__type_3"])))))) + ((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"]) < (((data["ORGANIZATION_TYPE_Transport__type_3"]) * 2.0)))*1.)))))*1.)) 
    v["i123"] = 0.035168*np.tanh(np.where(data["DAYS_EMPLOYED"]>0, np.minimum(((data["REGION_RATING_CLIENT_W_CITY"])), ((data["REGION_POPULATION_RELATIVE"]))), ((np.where(data["HOUR_APPR_PROCESS_START"]<0, data["INSTAL_AMT_PAYMENT_MAX"], np.where(data["CC_AMT_PAYMENT_CURRENT_MEAN"]>0, data["DAYS_EMPLOYED"], data["NEW_DOC_IND_STD"] ) )) * 2.0) )) 
    v["i124"] = 0.048513*np.tanh(np.where(data["INSTAL_DBD_MEAN"]<0, np.where(data["BURO_AMT_CREDIT_SUM_OVERDUE_MEAN"]<0, np.where(data["PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN"]>0, ((data["PREV_NAME_SELLER_INDUSTRY_Consumer_electronics_MEAN"]) * 2.0), (((data["CC_AMT_RECIVABLE_VAR"]) > (data["INSTAL_COUNT"]))*1.) ), data["BURO_AMT_CREDIT_SUM_OVERDUE_MEAN"] ), (-1.0*((data["APPROVED_DAYS_DECISION_MIN"]))) )) 
    v["i125"] = 0.004000*np.tanh((-1.0*((np.maximum(((np.maximum(((((((((data["ORGANIZATION_TYPE_Industry__type_9"]) + (data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"]))/2.0)) + (data["AMT_GOODS_PRICE"]))/2.0))), ((np.where(data["PREV_AMT_GOODS_PRICE_MIN"]<0, data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"], data["PREV_WEEKDAY_APPR_PROCESS_START_SATURDAY_MEAN"] )))))), ((data["NAME_HOUSING_TYPE_Office_apartment"]))))))) 
    v["i126"] = 0.028997*np.tanh(np.where(data["INSTAL_PAYMENT_DIFF_MAX"]>0, data["PREV_CODE_REJECT_REASON_XAP_MEAN"], ((((data["PREV_CODE_REJECT_REASON_HC_MEAN"]) - ((((data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"]) + ((((data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"]) > (data["INSTAL_PAYMENT_DIFF_MAX"]))*1.)))/2.0)))) - ((((data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]) + (data["NAME_FAMILY_STATUS_Married"]))/2.0))) )) 
    v["i127"] = 0.002994*np.tanh((-1.0*(((((((data["PREV_CHANNEL_TYPE_Regional___Local_MEAN"]) > ((((((data["PREV_CHANNEL_TYPE_Regional___Local_MEAN"]) > (((((-1.0*((data["WEEKDAY_APPR_PROCESS_START_MONDAY"])))) > ((((data["REFUSED_CNT_PAYMENT_MEAN"]) > (data["WEEKDAY_APPR_PROCESS_START_MONDAY"]))*1.)))*1.)))*1.)) - (data["AMT_GOODS_PRICE"]))))*1.)) * 2.0))))) 
    v["i128"] = 0.005997*np.tanh(np.where(data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"]>0, data["POS_NAME_CONTRACT_STATUS_Active_MEAN"], np.where(data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]<0, data["NEW_DOC_IND_KURT"], ((((((data["REFUSED_DAYS_DECISION_MAX"]) > (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))*1.)) > (np.where(data["NAME_CONTRACT_TYPE_Cash_loans"]>0, data["NAME_EDUCATION_TYPE_Secondary___secondary_special"], data["APPROVED_AMT_CREDIT_MEAN"] )))*1.) ) )) 
    v["i129"] = 0.046680*np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_OVERDUE_MEAN"]>0, (14.66528892517089844), ((data["NEW_DOC_IND_KURT"]) * (np.where(data["NAME_HOUSING_TYPE_Office_apartment"]>0, -3.0, ((((data["CLOSED_DAYS_CREDIT_MIN"]) * (data["POS_COUNT"]))) - (data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"])) ))) )) 
    v["i130"] = 0.019768*np.tanh(np.where(data["APPROVED_APP_CREDIT_PERC_VAR"] < -99998, data["PREV_WEEKDAY_APPR_PROCESS_START_FRIDAY_MEAN"], np.where(data["PREV_NAME_GOODS_CATEGORY_Photo___Cinema_Equipment_MEAN"]<0, np.where(data["APARTMENTS_MEDI"] < -99998, data["APPROVED_CNT_PAYMENT_MEAN"], np.where(np.tanh((data["PREV_WEEKDAY_APPR_PROCESS_START_FRIDAY_MEAN"]))<0, data["NAME_CONTRACT_TYPE_Cash_loans"], data["PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN"] ) ), data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] ) )) 
    v["i131"] = 0.006600*np.tanh(((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) * (np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"]>0, data["CC_MONTHS_BALANCE_VAR"], np.where(data["CC_AMT_RECEIVABLE_PRINCIPAL_VAR"]>0, data["REFUSED_AMT_CREDIT_MIN"], (((((data["PREV_WEEKDAY_APPR_PROCESS_START_FRIDAY_MEAN"]) < (np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_SUM"])), ((data["REFUSED_AMT_CREDIT_MIN"])))))*1.)) * 2.0) ) )))) 
    v["i132"] = 0.049586*np.tanh((-1.0*((np.where(data["INSTAL_AMT_PAYMENT_MAX"] < -99998, data["YEARS_BEGINEXPLUATATION_MEDI"], np.where(data["INSTAL_AMT_PAYMENT_MAX"]<0, np.maximum(((np.maximum(((data["AMT_CREDIT"])), (((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"]) > (data["CC_AMT_RECIVABLE_VAR"]))*1.)))))), ((data["EXT_SOURCE_2"]))), data["DEF_60_CNT_SOCIAL_CIRCLE"] ) ))))) 
    v["i133"] = 0.049999*np.tanh(np.where(data["PREV_NAME_CLIENT_TYPE_Refreshed_MEAN"]>0, data["BURO_CREDIT_TYPE_Credit_card_MEAN"], ((((np.where((-1.0*(((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MAX"]) > (data["AMT_REQ_CREDIT_BUREAU_QRT"]))*1.))))<0, data["BURO_DAYS_CREDIT_MAX"], ((data["PREV_NAME_GOODS_CATEGORY_Computers_MEAN"]) * (data["BURO_DAYS_CREDIT_MAX"])) )) * 2.0)) * 2.0) )) 
    v["i134"] = 0.047500*np.tanh(np.where(data["REFUSED_HOUR_APPR_PROCESS_START_MEAN"]<0, np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]>0, 3.0, np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, data["FLAG_WORK_PHONE"], (-1.0*((data["FLAG_WORK_PHONE"]))) ) ), np.where(data["ACTIVE_MONTHS_BALANCE_MAX_MAX"]<0, data["PREV_CODE_REJECT_REASON_HC_MEAN"], data["INSTAL_AMT_PAYMENT_MIN"] ) )) 
    v["i135"] = 0.046005*np.tanh(np.where(data["NEW_DOC_IND_STD"]<0, data["INSTAL_PAYMENT_DIFF_SUM"], np.maximum(((np.maximum(((((data["BURO_STATUS_1_MEAN_MEAN"]) * ((-1.0*((data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]))))))), ((data["CC_CNT_DRAWINGS_CURRENT_MEAN"]))))), ((np.where(data["REGION_RATING_CLIENT_W_CITY"]<0, data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"], data["NEW_DOC_IND_STD"] )))) )) 
    v["i136"] = 0.045999*np.tanh(np.where(data["CC_AMT_BALANCE_MAX"]<0, np.where(data["POS_SK_DPD_DEF_MEAN"]>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], np.where(data["BURO_CREDIT_TYPE_Car_loan_MEAN"] < -99998, (-1.0*((data["NEW_CREDIT_TO_INCOME_RATIO"]))), ((data["DEF_30_CNT_SOCIAL_CIRCLE"]) * ((-1.0*((data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__middle_MEAN"]))))) ) ), data["PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN"] )) 
    v["i137"] = 0.049992*np.tanh(((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (np.tanh((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])), ((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"])))))))) - (np.maximum(((np.maximum(((data["CC_AMT_PAYMENT_TOTAL_CURRENT_SUM"])), ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))))), ((np.maximum(((data["NEW_SOURCES_PROD"])), ((data["INSTAL_AMT_PAYMENT_SUM"]))))))))) 
    v["i138"] = 0.050000*np.tanh((((((data["NEW_RATIO_PREV_DAYS_DECISION_MAX"]) > (data["POS_SK_DPD_MEAN"]))*1.)) + (((((data["CC_AMT_DRAWINGS_ATM_CURRENT_SUM"]) - (data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"]))) + ((((data["APPROVED_APP_CREDIT_PERC_MIN"]) < (np.where(data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"]<0, data["POS_SK_DPD_MEAN"], data["APPROVED_APP_CREDIT_PERC_MIN"] )))*1.)))))) 
    v["i139"] = 0.036994*np.tanh(((data["AMT_REQ_CREDIT_BUREAU_QRT"]) * (((((data["ORGANIZATION_TYPE_Military"]) * 2.0)) - ((((data["AMT_CREDIT"]) + (np.maximum(((data["AMT_INCOME_TOTAL"])), ((((np.maximum(((((data["INSTAL_DPD_MEAN"]) * 2.0))), ((data["CC_AMT_RECIVABLE_VAR"])))) * 2.0))))))/2.0)))))) 
    v["i140"] = 0.039200*np.tanh(np.where(data["INSTAL_AMT_PAYMENT_MIN"]>0, np.where(data["PREV_RATE_DOWN_PAYMENT_MAX"]<0, data["PREV_CODE_REJECT_REASON_LIMIT_MEAN"], data["INSTAL_AMT_PAYMENT_MIN"] ), ((data["NEW_RATIO_BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]) * (np.where(data["LIVINGAPARTMENTS_AVG"]>0, data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"], (((data["PREV_AMT_ANNUITY_MEAN"]) > (data["APPROVED_CNT_PAYMENT_MEAN"]))*1.) ))) )) 
    v["i141"] = 0.044997*np.tanh(np.where(data["PREV_AMT_ANNUITY_MIN"] < -99998, data["NEW_DOC_IND_AVG"], ((data["PREV_NAME_YIELD_GROUP_high_MEAN"]) * (((((((data["INSTAL_DBD_SUM"]) * 2.0)) * 2.0)) * ((((data["APPROVED_DAYS_DECISION_MEAN"]) < (np.maximum(((data["AMT_INCOME_TOTAL"])), ((data["INSTAL_DBD_SUM"])))))*1.))))) )) 
    v["i142"] = 0.044000*np.tanh(((data["DAYS_ID_PUBLISH"]) * (np.where(data["POS_MONTHS_BALANCE_MEAN"]<0, data["NEW_RATIO_PREV_DAYS_DECISION_MIN"], np.where(data["INSTAL_AMT_PAYMENT_MIN"]<0, data["CC_AMT_BALANCE_VAR"], ((data["FLOORSMIN_MODE"]) * (data["NEW_RATIO_PREV_APP_CREDIT_PERC_MIN"])) ) )))) 
    v["i143"] = 0.017800*np.tanh(np.where(data["LANDAREA_MEDI"]>0, data["CC_MONTHS_BALANCE_VAR"], (((data["DAYS_ID_PUBLISH"]) + (np.where(data["REFUSED_CNT_PAYMENT_MEAN"]<0, (-1.0*((np.maximum(((data["ORGANIZATION_TYPE_Industry__type_9"])), ((data["ORGANIZATION_TYPE_School"])))))), (-1.0*((data["BASEMENTAREA_MODE"]))) )))/2.0) )) 
    v["i144"] = 0.049495*np.tanh((((((((((data["DAYS_BIRTH"]) * (data["NAME_FAMILY_STATUS_Married"]))) + (((((np.minimum(((((data["DAYS_BIRTH"]) * (data["CODE_GENDER"])))), (((-1.0*((data["BURO_CREDIT_TYPE_Car_loan_MEAN"]))))))) * 2.0)) * 2.0)))/2.0)) * 2.0)) * 2.0)) 
    v["i145"] = 0.047997*np.tanh(((np.maximum(((data["BURO_AMT_CREDIT_SUM_OVERDUE_MEAN"])), (((((data["NAME_FAMILY_STATUS_Married"]) < (np.maximum(((data["ACTIVE_DAYS_CREDIT_MAX"])), ((np.where(data["APPROVED_AMT_APPLICATION_MAX"]>0, np.where(data["ACTIVE_DAYS_CREDIT_MAX"]>0, data["INSTAL_DAYS_ENTRY_PAYMENT_MEAN"], data["APPROVED_AMT_APPLICATION_MAX"] ), data["CLOSED_AMT_CREDIT_SUM_DEBT_MAX"] ))))))*1.))))) * 2.0)) 
    v["i146"] = 0.007402*np.tanh((((-1.0*((np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]>0, data["NEW_RATIO_PREV_RATE_DOWN_PAYMENT_MEAN"], np.where(data["ACTIVE_AMT_CREDIT_SUM_SUM"]>0, np.maximum(((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"])), ((data["LIVE_CITY_NOT_WORK_CITY"]))), (((data["CLOSED_AMT_CREDIT_SUM_MEAN"]) > (np.tanh((data["DEF_60_CNT_SOCIAL_CIRCLE"]))))*1.) ) ))))) * 2.0)) 
    v["i147"] = 0.034997*np.tanh(np.where(data["INSTAL_AMT_PAYMENT_MIN"]<0, ((data["CODE_GENDER"]) * (data["PREV_CNT_PAYMENT_MEAN"])), np.where(data["CODE_GENDER"]<0, ((data["BURO_DAYS_CREDIT_MEAN"]) * (data["PREV_CNT_PAYMENT_MEAN"])), data["BURO_DAYS_CREDIT_MEAN"] ) )) 
    v["i148"] = 0.047001*np.tanh(((data["OBS_30_CNT_SOCIAL_CIRCLE"]) * (((((np.maximum(((np.where(data["REGION_RATING_CLIENT"]<0, ((np.maximum(((data["ORGANIZATION_TYPE_School"])), ((data["REFUSED_CNT_PAYMENT_SUM"])))) * 2.0), (-1.0*((data["APPROVED_AMT_GOODS_PRICE_MEAN"]))) ))), ((data["INSTAL_AMT_INSTALMENT_SUM"])))) * 2.0)) * 2.0)))) 
    v["i149"] = 0.047705*np.tanh(np.where(data["REGION_POPULATION_RELATIVE"]>0, np.where(data["NEW_INC_BY_ORG"]>0, ((data["NEW_DOC_IND_KURT"]) - (data["WEEKDAY_APPR_PROCESS_START_SUNDAY"])), data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"] ), ((((((data["PREV_PRODUCT_COMBINATION_POS_household_without_interest_MEAN"]) * 2.0)) * 2.0)) * (data["NEW_DOC_IND_STD"])) ))
    return Output(v.sum(axis=1)-2.432490)

def GP2(data):
    v = pd.DataFrame()
    v["i0"] = 0.040171*np.tanh(((((((np.where(data["EXT_SOURCE_3"] < -99998, np.tanh((data["DAYS_EMPLOYED"])), (((-1.0*((data["EXT_SOURCE_3"])))) + (np.minimum(((data["DAYS_BIRTH"])), ((data["NEW_CREDIT_TO_GOODS_RATIO"]))))) )) - (data["EXT_SOURCE_2"]))) * 2.0)) * 2.0)) 
    v["i1"] = 0.049975*np.tanh(((((((((((-1.0) - (np.where(data["CC_AMT_PAYMENT_CURRENT_VAR"] < -99998, ((data["NEW_EXT_SOURCES_MEAN"]) * 2.0), ((((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)) * 2.0)) * 2.0) )))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i2"] = 0.040171*np.tanh(((((((((((np.where(data["EXT_SOURCE_2"]>0, data["NEW_RATIO_PREV_DAYS_DECISION_MAX"], ((-2.0) / 2.0) )) - (data["EXT_SOURCE_3"]))) - (data["EXT_SOURCE_2"]))) * 2.0)) * 2.0)) * 2.0)) 
    v["i3"] = 0.040171*np.tanh((((((((data["OCCUPATION_TYPE_Laborers"]) + (data["NEW_CREDIT_TO_GOODS_RATIO"]))/2.0)) + (((((((data["NEW_EXT_SOURCES_MEAN"]) * (-2.0))) + (np.minimum(((data["NEW_CREDIT_TO_GOODS_RATIO"])), (((-1.0*((data["EXT_SOURCE_3"]))))))))) * 2.0)))) * 2.0)) 
    v["i4"] = 0.047400*np.tanh(((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)) * (np.minimum(((np.minimum(((data["NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MAX"])), ((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)))))), ((((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)) * (np.minimum(((data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"])), ((data["NEW_RATIO_PREV_AMT_DOWN_PAYMENT_MAX"]))))))))))) 
    v["i5"] = 0.040181*np.tanh(((((data["DAYS_BIRTH"]) + ((((-1.0*((np.maximum(((data["NAME_EDUCATION_TYPE_Higher_education"])), ((data["EXT_SOURCE_3"]))))))) - (((data["EXT_SOURCE_2"]) + (np.where(data["EXT_SOURCE_3"] < -99998, data["EXT_SOURCE_2"], data["EXT_SOURCE_3"] )))))))) * 2.0)) 
    v["i6"] = 0.047400*np.tanh(((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) - (((((((((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)) - (np.tanh((data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"]))))) - (data["NEW_CREDIT_TO_GOODS_RATIO"]))) - (np.tanh((data["DAYS_EMPLOYED"]))))) * 2.0)))) * 2.0)) 
    v["i7"] = 0.040171*np.tanh(((((np.tanh((data["DAYS_EMPLOYED"]))) + (((((np.where(data["AMT_ANNUITY"]<0, data["AMT_ANNUITY"], np.tanh((np.tanh((data["NEW_CREDIT_TO_GOODS_RATIO"])))) )) + ((-1.0*((data["NEW_EXT_SOURCES_MEAN"])))))) * 2.0)))) * 2.0)) 
    v["i8"] = 0.049660*np.tanh((((((((((-1.0*(((((((data["DAYS_EMPLOYED"]) < (((((data["PREV_AMT_DOWN_PAYMENT_MAX"]) * 2.0)) - (data["NEW_CREDIT_TO_GOODS_RATIO"]))))*1.)) + (data["NEW_EXT_SOURCES_MEAN"])))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i9"] = 0.049975*np.tanh(((((data["NEW_CREDIT_TO_GOODS_RATIO"]) + (((((np.tanh((((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]) + (((np.tanh(((((data["DAYS_EMPLOYED"]) + (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))/2.0)))) * 2.0)))))) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)))) * 2.0)) 
    v["i10"] = 0.048200*np.tanh((((-1.0*((((((data["NEW_EXT_SOURCES_MEAN"]) + (((np.where(data["REFUSED_DAYS_DECISION_MAX"]<0, data["NEW_EXT_SOURCES_MEAN"], data["EXT_SOURCE_3"] )) + (data["CODE_GENDER"]))))) + (np.maximum(((data["APPROVED_APP_CREDIT_PERC_MAX"])), ((data["NAME_EDUCATION_TYPE_Higher_education"]))))))))) * 2.0)) 
    v["i11"] = 0.050000*np.tanh(((((((data["PREV_NAME_CLIENT_TYPE_New_MEAN"]) + (data["PREV_CNT_PAYMENT_MEAN"]))) + (data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))) + (((data["NEW_CREDIT_TO_GOODS_RATIO"]) - (((np.maximum(((data["NEW_EXT_SOURCES_MEAN"])), ((((data["NEW_EXT_SOURCES_MEAN"]) + (data["BURO_CREDIT_ACTIVE_Closed_MEAN"])))))) * 2.0)))))) 
    v["i12"] = 0.049996*np.tanh(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) + (((data["NEW_EXT_SOURCES_MEAN"]) * (np.minimum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), ((np.minimum(((((data["CC_CNT_INSTALMENT_MATURE_CUM_SUM"]) + (np.minimum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), ((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)))))))), ((data["NEW_EXT_SOURCES_MEAN"]))))))))))) 
    v["i13"] = 0.049540*np.tanh(((((((np.maximum(((data["DAYS_EMPLOYED"])), ((data["APPROVED_APP_CREDIT_PERC_MAX"])))) - (data["NEW_EXT_SOURCES_MEAN"]))) + ((((-1.0*((((data["APPROVED_APP_CREDIT_PERC_MAX"]) - (data["PREV_NAME_PRODUCT_TYPE_walk_in_MEAN"])))))) + (np.tanh((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))))))) * 2.0)) 
    v["i14"] = 0.049550*np.tanh(((np.tanh((data["PREV_CNT_PAYMENT_MEAN"]))) + (((((data["NEW_CREDIT_TO_GOODS_RATIO"]) - (data["CODE_GENDER"]))) - (((((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)) + (np.maximum(((data["NAME_EDUCATION_TYPE_Higher_education"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))))) 
    v["i15"] = 0.048505*np.tanh((((-1.0*((((data["NEW_EXT_SOURCES_MEAN"]) - (((((np.tanh((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))) + (np.maximum(((data["DAYS_EMPLOYED"])), ((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])))))) + (((data["INSTAL_PAYMENT_DIFF_MAX"]) - (data["APPROVED_AMT_ANNUITY_MAX"])))))))))) * 2.0)) 
    v["i16"] = 0.049643*np.tanh(((data["FLAG_DOCUMENT_3"]) + ((-1.0*((((data["CODE_GENDER"]) + (((((((data["NEW_EXT_SOURCES_MEAN"]) + ((((data["CC_CNT_DRAWINGS_ATM_CURRENT_SUM"]) < (data["NEW_CAR_TO_EMPLOY_RATIO"]))*1.)))) * 2.0)) + (data["BURO_CREDIT_ACTIVE_Closed_MEAN"])))))))))) 
    v["i17"] = 0.049376*np.tanh(((((((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]) + (np.where(data["NEW_DOC_IND_KURT"]>0, data["DAYS_EMPLOYED"], data["NEW_DOC_IND_KURT"] )))) - ((((((data["NEW_EXT_SOURCES_MEAN"]) + (((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) - (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))))/2.0)) * 2.0)))) * 2.0)) 
    v["i18"] = 0.049800*np.tanh(((((((((data["PREV_CNT_PAYMENT_MEAN"]) - ((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) < (data["EXT_SOURCE_2"]))*1.)))) - (np.where(data["DAYS_EMPLOYED"]>0, data["EXT_SOURCE_1"], (((data["PREV_DAYS_DECISION_MIN"]) < (data["EXT_SOURCE_3"]))*1.) )))) * 2.0)) * 2.0)) 
    v["i19"] = 0.049904*np.tanh(((((data["NEW_DOC_IND_KURT"]) - (((data["NEW_EXT_SOURCES_MEAN"]) - (np.where(data["INSTAL_AMT_PAYMENT_MIN"]<0, data["PREV_DAYS_DECISION_MIN"], data["NEW_RATIO_PREV_DAYS_DECISION_MAX"] )))))) - (((data["POS_COUNT"]) - (((data["APPROVED_CNT_PAYMENT_MEAN"]) - (data["INSTAL_AMT_PAYMENT_MIN"]))))))) 
    v["i20"] = 0.049787*np.tanh(((((((((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (data["CODE_GENDER"]))) + (((data["NEW_CREDIT_TO_GOODS_RATIO"]) + (data["PREV_NAME_YIELD_GROUP_high_MEAN"]))))) - (data["PREV_CODE_REJECT_REASON_XAP_MEAN"]))) + (data["DEF_30_CNT_SOCIAL_CIRCLE"]))) - (data["NEW_EXT_SOURCES_MEAN"]))) * 2.0)) 
    v["i21"] = 0.049800*np.tanh(((np.where(data["PREV_AMT_DOWN_PAYMENT_MAX"]>0, data["CC_AMT_BALANCE_MEAN"], (-1.0*((((((((data["EXT_SOURCE_3"]) * 2.0)) - (np.minimum((((-1.0*((data["CODE_GENDER"]))))), ((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"])))))) - (data["REGION_RATING_CLIENT"]))))) )) * 2.0)) 
    v["i22"] = 0.049000*np.tanh(((((((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]) + (((data["PREV_NAME_CLIENT_TYPE_New_MEAN"]) * 2.0)))) + (((((data["INSTAL_PAYMENT_DIFF_MAX"]) - (data["NEW_EXT_SOURCES_MEAN"]))) + (data["INSTAL_PAYMENT_DIFF_MEAN"]))))) + (((data["PREV_CNT_PAYMENT_MEAN"]) + (data["PREV_NAME_YIELD_GROUP_XNA_MEAN"]))))) 
    v["i23"] = 0.049944*np.tanh(((((np.where(((data["EXT_SOURCE_3"]) - (data["NAME_INCOME_TYPE_Working"]))>0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]>0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], np.where(data["POS_MONTHS_BALANCE_SIZE"]>0, data["REFUSED_CNT_PAYMENT_SUM"], data["NAME_EDUCATION_TYPE_Secondary___secondary_special"] ) ) )) * 2.0)) * 2.0)) 
    v["i24"] = 0.050000*np.tanh(((((((np.where(data["INSTAL_DPD_MEAN"]<0, (((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) + (((data["CC_CNT_INSTALMENT_MATURE_CUM_SUM"]) * (data["NEW_EXT_SOURCES_MEAN"]))))/2.0), (-1.0*((data["NEW_SOURCES_PROD"]))) )) * 2.0)) * 2.0)) * 2.0)) 
    v["i25"] = 0.048001*np.tanh(((((((np.where(data["NEW_CAR_TO_BIRTH_RATIO"]>0, data["NEW_RATIO_BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"], np.maximum(((data["BURO_CREDIT_ACTIVE_Active_MEAN"])), ((np.maximum(((data["REG_CITY_NOT_LIVE_CITY"])), ((np.where(data["INSTAL_DPD_MEAN"]>0, data["INSTAL_DPD_MEAN"], data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"] ))))))) )) * 2.0)) * 2.0)) * 2.0)) 
    v["i26"] = 0.049758*np.tanh(((data["REGION_RATING_CLIENT_W_CITY"]) + (((data["NEW_CREDIT_TO_GOODS_RATIO"]) + (((((((((data["PREV_CNT_PAYMENT_SUM"]) - (np.maximum(((data["POS_COUNT"])), ((data["EXT_SOURCE_1"])))))) * 2.0)) * 2.0)) + (data["NAME_INCOME_TYPE_Working"]))))))) 
    v["i27"] = 0.049720*np.tanh(((((data["NEW_CREDIT_TO_GOODS_RATIO"]) - (data["CODE_GENDER"]))) - (((3.0) * (((data["PREV_AMT_ANNUITY_MIN"]) + (((((data["POS_MONTHS_BALANCE_SIZE"]) - (data["INSTAL_PAYMENT_DIFF_MAX"]))) - (data["INSTAL_PAYMENT_DIFF_MAX"]))))))))) 
    v["i28"] = 0.049994*np.tanh(((data["FLAG_WORK_PHONE"]) + (((((((data["DEF_30_CNT_SOCIAL_CIRCLE"]) + (np.maximum(((data["PREV_CNT_PAYMENT_MEAN"])), ((data["NEW_CREDIT_TO_GOODS_RATIO"])))))) + (np.where(data["CC_AMT_BALANCE_SUM"] < -99998, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"] )))) * 2.0)))) 
    v["i29"] = 0.049830*np.tanh(np.where(data["EXT_SOURCE_1"] < -99998, ((data["DAYS_BIRTH"]) - (np.where(data["CODE_GENDER"]<0, data["NEW_CAR_TO_EMPLOY_RATIO"], data["PREV_NAME_YIELD_GROUP_low_action_MEAN"] ))), ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (data["NEW_EXT_SOURCES_MEAN"]))) - (data["DAYS_BIRTH"])) )) 
    v["i30"] = 0.048802*np.tanh(((((data["INSTAL_PAYMENT_DIFF_MAX"]) - (((data["INSTAL_DBD_SUM"]) - (data["REGION_RATING_CLIENT_W_CITY"]))))) - (((np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((data["PREV_RATE_DOWN_PAYMENT_MAX"])))) - (((((data["INSTAL_AMT_INSTALMENT_MAX"]) - (data["APPROVED_AMT_ANNUITY_MEAN"]))) * 2.0)))))) 
    v["i31"] = 0.049376*np.tanh(((((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) - (data["CC_MONTHS_BALANCE_VAR"]))) + (((((data["INSTAL_PAYMENT_DIFF_MAX"]) - (data["APPROVED_AMT_ANNUITY_MEAN"]))) + (((data["AMT_ANNUITY"]) - (np.maximum(((data["PREV_NAME_YIELD_GROUP_low_action_MEAN"])), ((data["NAME_INCOME_TYPE_State_servant"])))))))))) * 2.0)) 
    v["i32"] = 0.049495*np.tanh(np.where(data["NEW_CAR_TO_EMPLOY_RATIO"]>0, ((data["REFUSED_DAYS_DECISION_MIN"]) * 2.0), ((((((np.maximum(((data["DEF_60_CNT_SOCIAL_CIRCLE"])), ((np.maximum(((data["APPROVED_CNT_PAYMENT_MEAN"])), ((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))))))) + ((-1.0*((data["CODE_GENDER"])))))) * 2.0)) * 2.0) )) 
    v["i33"] = 0.046499*np.tanh(((((((((((np.maximum(((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])), (((-1.0*((np.maximum((((((data["NEW_EXT_SOURCES_MEAN"]) + (((data["NAME_FAMILY_STATUS_Married"]) * 2.0)))/2.0))), ((data["FLOORSMAX_MEDI"])))))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i34"] = 0.049768*np.tanh(((np.where(np.maximum(((data["NEW_RATIO_BURO_DAYS_CREDIT_MAX"])), ((data["PREV_NAME_YIELD_GROUP_low_action_MEAN"])))>0, data["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"], ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (((data["INSTAL_DBD_SUM"]) * 2.0)))) + (((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]) - (data["PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN"])))) )) * 2.0)) 
    v["i35"] = 0.049610*np.tanh(np.where(data["NEW_CAR_TO_BIRTH_RATIO"]>0, data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"], ((((data["PREV_CNT_PAYMENT_MEAN"]) - (data["OCCUPATION_TYPE_Core_staff"]))) - (((data["PREV_AMT_ANNUITY_MEAN"]) - (((data["DAYS_ID_PUBLISH"]) - (((data["PREV_NAME_YIELD_GROUP_low_normal_MEAN"]) - (data["PREV_NAME_YIELD_GROUP_high_MEAN"])))))))) )) 
    v["i36"] = 0.048788*np.tanh(((((((((data["APPROVED_CNT_PAYMENT_MEAN"]) - (((data["POS_MONTHS_BALANCE_MAX"]) - (data["PREV_DAYS_DECISION_MEAN"]))))) - (data["APPROVED_AMT_ANNUITY_MEAN"]))) - (((data["INSTAL_AMT_PAYMENT_MIN"]) - (np.minimum(((data["AMT_ANNUITY"])), ((data["NEW_DOC_IND_KURT"])))))))) * 2.0)) 
    v["i37"] = 0.049586*np.tanh(((((np.where(np.maximum(((np.maximum(((np.maximum(((data["INSTAL_DPD_MEAN"])), ((np.maximum(((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"])), ((data["REG_CITY_NOT_LIVE_CITY"])))))))), ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))))), ((data["BURO_AMT_CREDIT_SUM_DEBT_MAX"])))<0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], (9.0) )) * 2.0)) * 2.0)) 
    v["i38"] = 0.049900*np.tanh(((((((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, (-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (((data["FLAG_DOCUMENT_3"]) * 2.0))) )) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i39"] = 0.049965*np.tanh(((((data["DAYS_LAST_PHONE_CHANGE"]) - (np.maximum(((((data["CODE_GENDER"]) - (data["DAYS_REGISTRATION"])))), ((data["APPROVED_HOUR_APPR_PROCESS_START_MAX"])))))) - (np.where(data["POS_SK_DPD_DEF_MAX"]>0, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_LIMIT_MEAN"], np.maximum(((data["NAME_INCOME_TYPE_State_servant"])), ((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]))) )))) 
    v["i40"] = 0.047400*np.tanh((((((-1.0*((np.where(np.where(data["POS_SK_DPD_DEF_MEAN"]<0, data["PREV_APP_CREDIT_PERC_MEAN"], data["CC_AMT_DRAWINGS_POS_CURRENT_VAR"] )<0, data["LIVINGAREA_AVG"], (((((data["DEF_30_CNT_SOCIAL_CIRCLE"]) * 2.0)) < (np.tanh((data["INSTAL_DBD_SUM"]))))*1.) ))))) * 2.0)) * 2.0)) 
    v["i41"] = 0.047999*np.tanh(((((data["APPROVED_DAYS_DECISION_MIN"]) + (data["INSTAL_PAYMENT_DIFF_MEAN"]))) + (((((((data["APPROVED_CNT_PAYMENT_MEAN"]) - (data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"]))) * 2.0)) + (((data["ORGANIZATION_TYPE_Self_employed"]) + (np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((data["PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN"])))))))))) 
    v["i42"] = 0.049985*np.tanh(np.where(data["NAME_EDUCATION_TYPE_Higher_education"]>0, data["CC_CNT_DRAWINGS_POS_CURRENT_MIN"], ((((data["APPROVED_CNT_PAYMENT_SUM"]) - (((data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]) + (data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"]))))) - (((data["POS_COUNT"]) + ((((data["POS_COUNT"]) > (data["AMT_ANNUITY"]))*1.))))) )) 
    v["i43"] = 0.049000*np.tanh(((np.where(np.where(((((data["BURO_AMT_CREDIT_SUM_DEBT_SUM"]) * 2.0)) + (np.maximum(((data["ACTIVE_DAYS_CREDIT_MAX"])), ((data["NEW_EXT_SOURCES_MEAN"])))))<0, data["NEW_EXT_SOURCES_MEAN"], data["ACTIVE_DAYS_CREDIT_MAX"] )<0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"], (-1.0*((data["NEW_SOURCES_PROD"]))) )) * 2.0)) 
    v["i44"] = 0.033210*np.tanh(np.where((((data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]) + (data["BURO_CREDIT_TYPE_Mortgage_MEAN"]))/2.0)>0, data["CC_CNT_DRAWINGS_CURRENT_MAX"], ((data["AMT_ANNUITY"]) + (((np.where(data["CC_CNT_DRAWINGS_CURRENT_MAX"]>0, data["CC_CNT_DRAWINGS_CURRENT_MAX"], ((data["AMT_ANNUITY"]) - (data["APPROVED_AMT_ANNUITY_MEAN"])) )) * 2.0))) )) 
    v["i45"] = 0.046000*np.tanh(((data["BURO_CREDIT_ACTIVE_Active_MEAN"]) * (((((np.where(data["BURO_CREDIT_ACTIVE_Active_MEAN"]<0, ((np.where(data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"] < -99998, data["BURO_DAYS_CREDIT_MAX"], np.maximum(((data["INSTAL_COUNT"])), ((data["ACTIVE_DAYS_CREDIT_MAX"]))) )) * 2.0), data["ACTIVE_DAYS_CREDIT_MAX"] )) * 2.0)) * 2.0)))) 
    v["i46"] = 0.049161*np.tanh(((((((np.maximum(((((((np.maximum(((data["PREV_PRODUCT_COMBINATION_Cash_Street__high_MEAN"])), ((data["ORGANIZATION_TYPE_Construction"])))) + (np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((data["INSTAL_PAYMENT_DIFF_MEAN"])))))) + (data["DEF_60_CNT_SOCIAL_CIRCLE"])))), ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])))) * 2.0)) * 2.0)) * 2.0)) 
    v["i47"] = 0.043843*np.tanh(((np.where(data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]>0, data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"], ((((data["OCCUPATION_TYPE_Drivers"]) + (np.maximum(((data["INSTAL_DPD_MEAN"])), ((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"])))))) + (((data["FLAG_WORK_PHONE"]) - (data["PREV_HOUR_APPR_PROCESS_START_MEAN"])))) )) - (data["NEW_EMPLOY_TO_BIRTH_RATIO"]))) 
    v["i48"] = 0.045000*np.tanh(((((((((np.where(data["INSTAL_AMT_PAYMENT_SUM"]>0, data["ACTIVE_AMT_CREDIT_SUM_SUM"], np.where(data["NAME_FAMILY_STATUS_Married"]>0, data["ORGANIZATION_TYPE_Business_Entity_Type_3"], (-1.0*((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))) ) )) * 2.0)) + (data["BURO_CREDIT_TYPE_Microloan_MEAN"]))) * 2.0)) + (data["REGION_RATING_CLIENT_W_CITY"]))) 
    v["i49"] = 0.046200*np.tanh(((((np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]<0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], ((np.maximum(((data["NAME_EDUCATION_TYPE_Lower_secondary"])), ((data["PREV_CODE_REJECT_REASON_SCOFR_MEAN"])))) + (((data["ORGANIZATION_TYPE_Self_employed"]) + ((((data["INSTAL_DPD_MEAN"]) > (data["ORGANIZATION_TYPE_Military"]))*1.))))) )) * 2.0)) * 2.0)) 
    v["i50"] = 0.048723*np.tanh(np.where(np.where(data["NEW_EXT_SOURCES_MEAN"]<0, np.where(data["POS_SK_DPD_DEF_MAX"]<0, data["EXT_SOURCE_3"], data["POS_SK_DPD_DEF_MAX"] ), data["POS_SK_DPD_DEF_MAX"] ) < -99998, -3.0, ((((-2.0) - (data["EXT_SOURCE_3"]))) - (data["BURO_DAYS_CREDIT_VAR"])) )) 
    v["i51"] = 0.032400*np.tanh(np.where((((data["REGION_RATING_CLIENT_W_CITY"]) < (data["AMT_REQ_CREDIT_BUREAU_QRT"]))*1.)>0, data["NEW_RATIO_PREV_AMT_DOWN_PAYMENT_MEAN"], np.where(data["ACTIVE_MONTHS_BALANCE_MAX_MAX"]>0, data["AMT_ANNUITY"], np.where(data["APPROVED_AMT_ANNUITY_MAX"]>0, data["APPROVED_CNT_PAYMENT_SUM"], ((data["OWN_CAR_AGE"]) * (data["OCCUPATION_TYPE_Core_staff"])) ) ) )) 
    v["i52"] = 0.048728*np.tanh((((((((-1.0*((((((data["PREV_AMT_ANNUITY_MEAN"]) - ((((data["INSTAL_AMT_PAYMENT_SUM"]) < (data["PREV_AMT_GOODS_PRICE_MAX"]))*1.)))) - (np.where(data["PREV_AMT_GOODS_PRICE_MAX"]<0, data["AMT_CREDIT"], data["INSTAL_AMT_PAYMENT_MAX"] ))))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i53"] = 0.042920*np.tanh(((np.where(data["EXT_SOURCE_3"] < -99998, data["NEW_RATIO_PREV_DAYS_DECISION_MAX"], ((((((((data["BURO_AMT_CREDIT_SUM_DEBT_SUM"]) - (data["BURO_AMT_CREDIT_SUM_MEAN"]))) * 2.0)) * 2.0)) - (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])) )) - (((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]) - (data["FLAG_WORK_PHONE"]))))) 
    v["i54"] = 0.049915*np.tanh(((np.minimum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), (((((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * 2.0))))) + (np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((np.where(data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"] < -99998, np.maximum(((data["INSTAL_PAYMENT_DIFF_MEAN"])), ((data["NEW_CREDIT_TO_GOODS_RATIO"]))), data["BURO_CREDIT_TYPE_Microloan_MEAN"] ))))))) 
    v["i55"] = 0.048997*np.tanh(((np.maximum(((data["APPROVED_CNT_PAYMENT_SUM"])), (((((((data["INSTAL_PAYMENT_PERC_SUM"]) < (data["POS_SK_DPD_DEF_MAX"]))*1.)) * 2.0))))) - (np.maximum(((np.maximum(((np.maximum(((data["ORGANIZATION_TYPE_School"])), ((data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]))))), ((data["PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN"]))))), ((data["CODE_GENDER"])))))) 
    v["i56"] = 0.049560*np.tanh(((((((((np.maximum(((((data["NEW_EXT_SOURCES_MEAN"]) * (data["PREV_PRODUCT_COMBINATION_Cash_Street__middle_MEAN"])))), (((((data["NEW_EXT_SOURCES_MEAN"]) > (data["PREV_NAME_CASH_LOAN_PURPOSE_Urgent_needs_MEAN"]))*1.))))) - (((data["NEW_DOC_IND_STD"]) - (data["PREV_PRODUCT_COMBINATION_Cash_Street__middle_MEAN"]))))) * 2.0)) * 2.0)) * 2.0)) 
    v["i57"] = 0.049800*np.tanh(((((np.where(data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"]<0, ((((data["AMT_ANNUITY"]) - (data["PREV_NAME_GOODS_CATEGORY_Furniture_MEAN"]))) - (data["NEW_DOC_IND_STD"])), data["APPROVED_CNT_PAYMENT_MEAN"] )) - (np.where(data["APPROVED_CNT_PAYMENT_MEAN"]<0, data["INSTAL_AMT_PAYMENT_SUM"], data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"] )))) * 2.0)) 
    v["i58"] = 0.047195*np.tanh(((data["REGION_RATING_CLIENT_W_CITY"]) + (((data["NEW_SCORES_STD"]) + (((((((data["ORGANIZATION_TYPE_Construction"]) + (data["NAME_FAMILY_STATUS_Separated"]))) + (np.maximum(((np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((data["PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN"]))))), ((data["CC_AMT_BALANCE_MIN"])))))) * 2.0)))))) 
    v["i59"] = 0.049000*np.tanh(((data["PREV_CNT_PAYMENT_MEAN"]) - (((((((data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"]) * 2.0)) - (data["PREV_NAME_PAYMENT_TYPE_XNA_MEAN"]))) - (((((((((((data["INSTAL_DPD_MEAN"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * (data["INSTAL_DAYS_ENTRY_PAYMENT_MEAN"]))))))) 
    v["i60"] = 0.048607*np.tanh(((data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__high_MEAN"]) + (((((np.minimum(((data["POS_SK_DPD_DEF_MAX"])), (((-1.0*((data["BURO_AMT_CREDIT_SUM_MEAN"]))))))) - (((np.where(data["NEW_RATIO_BURO_MONTHS_BALANCE_SIZE_MEAN"] < -99998, data["EXT_SOURCE_2"], data["AMT_REQ_CREDIT_BUREAU_QRT"] )) / 2.0)))) - (data["INSTAL_DBD_SUM"]))))) 
    v["i61"] = 0.049003*np.tanh(((((((((((np.where(data["NEW_DOC_IND_KURT"]>0, np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]<0, ((data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"]) - (data["CC_AMT_PAYMENT_TOTAL_CURRENT_SUM"])), data["ACTIVE_DAYS_CREDIT_MEAN"] ), data["INSTAL_PAYMENT_DIFF_MAX"] )) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i62"] = 0.029562*np.tanh(((np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"]>0, data["CC_AMT_RECIVABLE_MIN"], np.where(data["NEW_RATIO_BURO_DAYS_CREDIT_MIN"] < -99998, data["DAYS_LAST_PHONE_CHANGE"], ((np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((np.maximum(((data["BURO_AMT_CREDIT_SUM_DEBT_SUM"])), ((data["CC_AMT_RECIVABLE_MIN"]))))))) + (data["REGION_POPULATION_RELATIVE"])) ) )) * 2.0)) 
    v["i63"] = 0.049969*np.tanh(np.where(data["EXT_SOURCE_1"] < -99998, np.maximum((((-1.0*((data["CODE_GENDER"]))))), ((data["DAYS_BIRTH"]))), np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"], data["BURO_AMT_CREDIT_SUM_DEBT_SUM"] ) )) 
    v["i64"] = 0.034802*np.tanh(((((data["APPROVED_AMT_APPLICATION_MAX"]) - (np.where(data["BURO_AMT_CREDIT_SUM_OVERDUE_MEAN"]>0, -1.0, (((data["POS_COUNT"]) > (data["APPROVED_CNT_PAYMENT_SUM"]))*1.) )))) - (np.maximum(((((((data["INSTAL_AMT_INSTALMENT_MEAN"]) * 2.0)) * 2.0))), ((data["NEW_CAR_TO_EMPLOY_RATIO"])))))) 
    v["i65"] = 0.048998*np.tanh((((((((data["APPROVED_AMT_GOODS_PRICE_MIN"]) < (data["NEW_EXT_SOURCES_MEAN"]))*1.)) + (((((data["NEW_EXT_SOURCES_MEAN"]) * (np.maximum(((data["POS_MONTHS_BALANCE_SIZE"])), ((data["CLOSED_MONTHS_BALANCE_MIN_MIN"])))))) - (np.maximum(((data["NEW_DOC_IND_STD"])), ((data["DAYS_BIRTH"])))))))) * 2.0)) 
    v["i66"] = 0.046998*np.tanh(((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, data["APPROVED_CNT_PAYMENT_SUM"], np.where(data["INSTAL_AMT_PAYMENT_SUM"]>0, data["OCCUPATION_TYPE_Laborers"], (((data["INSTAL_AMT_PAYMENT_SUM"]) < (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) / 2.0)))*1.) ) )) - (np.tanh((data["APPROVED_APP_CREDIT_PERC_MIN"]))))) * 2.0)) 
    v["i67"] = 0.039784*np.tanh(np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"]>0, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_MAX"], ((np.maximum(((data["OCCUPATION_TYPE_Drivers"])), ((np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((data["CC_CNT_DRAWINGS_CURRENT_MAX"]))))))) + (np.where(data["DAYS_LAST_PHONE_CHANGE"]>0, data["ORGANIZATION_TYPE_Business_Entity_Type_3"], data["AMT_ANNUITY"] ))) )) 
    v["i68"] = 0.049100*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]<0, ((((((((((((np.tanh((data["NEW_CREDIT_TO_GOODS_RATIO"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * (((data["AMT_ANNUITY"]) - (data["OCCUPATION_TYPE_High_skill_tech_staff"])))), data["CC_AMT_RECEIVABLE_PRINCIPAL_VAR"] )) 
    v["i69"] = 0.037440*np.tanh(np.where(data["BURO_CREDIT_TYPE_Car_loan_MEAN"]>0, data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"], ((np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((np.maximum(((data["ORGANIZATION_TYPE_Transport__type_3"])), ((np.maximum(((data["INSTAL_AMT_PAYMENT_MAX"])), ((data["CC_CNT_DRAWINGS_CURRENT_VAR"])))))))))) - (np.maximum(((data["CC_AMT_PAYMENT_TOTAL_CURRENT_SUM"])), ((data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"]))))) )) 
    v["i70"] = 0.049961*np.tanh(((((np.where(data["APPROVED_CNT_PAYMENT_SUM"] < -99998, data["NEW_DOC_IND_STD"], ((data["APPROVED_CNT_PAYMENT_SUM"]) - (np.maximum(((data["REFUSED_HOUR_APPR_PROCESS_START_MIN"])), ((np.maximum(((data["CC_AMT_PAYMENT_CURRENT_MEAN"])), ((np.maximum(((data["FLOORSMIN_MEDI"])), ((data["POS_COUNT"]))))))))))) )) * 2.0)) * 2.0)) 
    v["i71"] = 0.049982*np.tanh(((data["POS_SK_DPD_MAX"]) + (((((((((((data["POS_SK_DPD_DEF_MAX"]) + (((data["NAME_EDUCATION_TYPE_Lower_secondary"]) + ((((data["OCCUPATION_TYPE_Core_staff"]) < (((data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"]) * 2.0)))*1.)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) 
    v["i72"] = 0.049400*np.tanh((((((data["NEW_EXT_SOURCES_MEAN"]) > (data["INSTAL_PAYMENT_PERC_MEAN"]))*1.)) - (((((data["AMT_REQ_CREDIT_BUREAU_YEAR"]) + (((data["ORGANIZATION_TYPE_Military"]) + (data["BURO_CREDIT_TYPE_Mortgage_MEAN"]))))) - (np.where(data["INSTAL_PAYMENT_PERC_MEAN"]<0, data["NEW_DOC_IND_KURT"], data["NEW_RATIO_BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"] )))))) 
    v["i73"] = 0.049278*np.tanh(((data["ORGANIZATION_TYPE_Self_employed"]) + (((np.maximum(((data["ORGANIZATION_TYPE_Transport__type_3"])), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"])))) + (((data["INSTAL_DPD_MEAN"]) + (np.maximum(((data["CC_AMT_BALANCE_MIN"])), ((np.where(data["INSTAL_AMT_PAYMENT_MIN"]<0, data["INSTAL_DBD_MAX"], data["PREV_CODE_REJECT_REASON_LIMIT_MEAN"] ))))))))))) 
    v["i74"] = 0.049119*np.tanh(((((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_SUM"]>0, data["BURO_DAYS_CREDIT_MEAN"], np.maximum(((((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_SUM"])), ((data["BURO_AMT_CREDIT_SUM_DEBT_SUM"])))) * 2.0))), ((((data["EXT_SOURCE_3"]) - (data["BURO_DAYS_CREDIT_MEAN"]))))) )) * 2.0)) * 2.0)) 
    v["i75"] = 0.045722*np.tanh(((np.maximum(((np.maximum(((data["BURO_CREDIT_ACTIVE_Sold_MEAN"])), ((np.maximum(((data["WALLSMATERIAL_MODE_Stone__brick"])), ((data["CC_AMT_RECEIVABLE_PRINCIPAL_VAR"])))))))), ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])))) - (np.maximum(((np.maximum(((data["INSTAL_AMT_INSTALMENT_SUM"])), ((data["NEW_RATIO_BURO_DAYS_CREDIT_MAX"]))))), ((data["NEW_CAR_TO_BIRTH_RATIO"])))))) 
    v["i76"] = 0.049090*np.tanh(np.maximum(((data["CC_CNT_DRAWINGS_CURRENT_MEAN"])), ((np.where(data["CLOSED_DAYS_CREDIT_MAX"]<0, np.maximum(((np.where(data["CC_CNT_DRAWINGS_CURRENT_MEAN"] < -99998, data["DEF_30_CNT_SOCIAL_CIRCLE"], data["ORGANIZATION_TYPE_Construction"] ))), ((((data["OBS_60_CNT_SOCIAL_CIRCLE"]) * (data["APPROVED_AMT_CREDIT_MAX"]))))), data["ACTIVE_AMT_CREDIT_SUM_SUM"] ))))) 
    v["i77"] = 0.049867*np.tanh(((data["NEW_RATIO_PREV_AMT_CREDIT_MIN"]) * (((-1.0) + (np.maximum(((data["NEW_EXT_SOURCES_MEAN"])), ((np.where(data["NEW_CAR_TO_BIRTH_RATIO"] < -99998, data["PREV_NAME_PORTFOLIO_Cash_MEAN"], (-1.0*((data["REFUSED_AMT_APPLICATION_MIN"]))) ))))))))) 
    v["i78"] = 0.049750*np.tanh(((np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]>0, data["ACTIVE_DAYS_CREDIT_MAX"], np.where(data["POS_SK_DPD_DEF_MAX"]<0, np.where(data["BURO_DAYS_CREDIT_MEAN"]<0, data["BURO_DAYS_CREDIT_MIN"], np.where(data["PREV_CHANNEL_TYPE_Contact_center_MEAN"]>0, data["PREV_CHANNEL_TYPE_Contact_center_MEAN"], data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"] ) ), data["POS_MONTHS_BALANCE_MEAN"] ) )) * 2.0)) 
    v["i79"] = 0.049999*np.tanh(np.where(data["ACTIVE_MONTHS_BALANCE_MAX_MAX"]>0, data["NEW_RATIO_BURO_AMT_ANNUITY_MAX"], np.where(data["ACTIVE_MONTHS_BALANCE_MAX_MAX"]>0, data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"], np.where(data["BURO_AMT_CREDIT_SUM_MEAN"]>0, data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"], np.maximum(((np.maximum(((data["NEW_CREDIT_TO_GOODS_RATIO"])), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"]))))), ((data["INSTAL_PAYMENT_DIFF_MEAN"]))) ) ) )) 
    v["i80"] = 0.049750*np.tanh(((((((((data["AMT_ANNUITY"]) - (data["WEEKDAY_APPR_PROCESS_START_MONDAY"]))) - (np.where(data["APPROVED_AMT_GOODS_PRICE_MAX"]>0, ((data["NEW_LIVE_IND_SUM"]) * 2.0), ((data["INSTAL_AMT_INSTALMENT_SUM"]) / 2.0) )))) * 2.0)) * 2.0)) 
    v["i81"] = 0.049100*np.tanh(((((((np.where(((data["PREV_PRODUCT_COMBINATION_Cash_Street__high_MEAN"]) * (np.maximum(((data["APPROVED_AMT_GOODS_PRICE_MIN"])), ((data["NAME_CONTRACT_TYPE_Cash_loans"])))))>0, data["POS_MONTHS_BALANCE_MEAN"], np.where(data["PREV_NAME_YIELD_GROUP_high_MEAN"] < -99998, data["REGION_POPULATION_RELATIVE"], data["NEW_EXT_SOURCES_MEAN"] ) )) * 2.0)) * 2.0)) * 2.0)) 
    v["i82"] = 0.048000*np.tanh(((np.where(data["EXT_SOURCE_2"] < -99998, np.tanh((data["PREV_APP_CREDIT_PERC_VAR"])), np.where(data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__middle_MEAN"] < -99998, data["NEW_DOC_IND_AVG"], (-1.0*((((((((data["EXT_SOURCE_2"]) + (data["OCCUPATION_TYPE_High_skill_tech_staff"]))/2.0)) + (data["NEW_DOC_IND_AVG"]))/2.0)))) ) )) * 2.0)) 
    v["i83"] = 0.039998*np.tanh(np.where(data["CC_AMT_PAYMENT_CURRENT_MEAN"]>0, data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"], np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, np.maximum(((data["FLAG_WORK_PHONE"])), ((((data["POS_MONTHS_BALANCE_MEAN"]) - (data["CODE_GENDER"]))))), (-1.0*((np.maximum(((data["FLAG_WORK_PHONE"])), ((data["POS_MONTHS_BALANCE_MEAN"])))))) ) )) 
    v["i84"] = 0.046880*np.tanh(((((np.where(data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]<0, np.where(data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"] < -99998, np.where(data["PREV_CODE_REJECT_REASON_HC_MEAN"]>0, data["PREV_CODE_REJECT_REASON_HC_MEAN"], data["INSTAL_PAYMENT_DIFF_MAX"] ), data["BURO_AMT_CREDIT_SUM_DEBT_SUM"] ), (-1.0*((((data["INSTAL_PAYMENT_DIFF_MAX"]) * 2.0)))) )) * 2.0)) * 2.0)) 
    v["i85"] = 0.036178*np.tanh((-1.0*((((((data["NEW_RATIO_PREV_AMT_CREDIT_MIN"]) - (np.minimum(((data["REFUSED_AMT_DOWN_PAYMENT_MIN"])), ((data["REGION_POPULATION_RELATIVE"])))))) - (np.where(data["EXT_SOURCE_3"] < -99998, data["REFUSED_DAYS_DECISION_MEAN"], (((data["NEW_RATIO_BURO_DAYS_CREDIT_VAR"]) < (data["REGION_POPULATION_RELATIVE"]))*1.) ))))))) 
    v["i86"] = 0.019979*np.tanh(((data["ORGANIZATION_TYPE_Transport__type_3"]) - ((((data["EXT_SOURCE_2"]) + (((np.where((-1.0*((data["EXT_SOURCE_2"])))<0, data["BURO_AMT_CREDIT_SUM_MEAN"], (((data["APPROVED_AMT_CREDIT_MAX"]) < ((((data["OCCUPATION_TYPE_High_skill_tech_staff"]) < (data["APPROVED_AMT_CREDIT_MAX"]))*1.)))*1.) )) * 2.0)))/2.0)))) 
    v["i87"] = 0.047481*np.tanh(((data["ORGANIZATION_TYPE_Transport__type_3"]) + (((data["ORGANIZATION_TYPE_Construction"]) + (np.maximum(((data["BURO_STATUS_1_MEAN_MEAN"])), (((((data["APPROVED_CNT_PAYMENT_SUM"]) + (((data["INSTAL_COUNT"]) * (np.where(data["APPROVED_CNT_PAYMENT_SUM"]<0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_MAX"], data["FLAG_WORK_PHONE"] )))))/2.0))))))))) 
    v["i88"] = 0.046499*np.tanh((-1.0*((np.maximum(((data["YEARS_BUILD_MEDI"])), ((np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"]<0, np.where(data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"]<0, np.where(data["PREV_WEEKDAY_APPR_PROCESS_START_SUNDAY_MEAN"]<0, data["NEW_EMPLOY_TO_BIRTH_RATIO"], data["APPROVED_DAYS_DECISION_MEAN"] ), data["NEW_RATIO_PREV_DAYS_DECISION_MAX"] ), data["BURO_CREDIT_TYPE_Mortgage_MEAN"] )))))))) 
    v["i89"] = 0.047614*np.tanh(np.where(data["BURO_STATUS_1_MEAN_MEAN"]>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], np.where(data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"]<0, np.where(data["ACTIVE_DAYS_CREDIT_ENDDATE_MAX"] < -99998, ((data["PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN"]) + (data["DAYS_REGISTRATION"])), data["PREV_NAME_CONTRACT_TYPE_Revolving_loans_MEAN"] ), ((((data["WALLSMATERIAL_MODE_Stone__brick"]) * 2.0)) * 2.0) ) )) 
    v["i90"] = 0.049597*np.tanh(np.maximum(((np.where(data["APPROVED_AMT_GOODS_PRICE_MIN"]>0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], np.where(data["FLAG_DOCUMENT_8"]>0, data["CC_AMT_INST_MIN_REGULARITY_VAR"], ((data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"]) * (data["APPROVED_CNT_PAYMENT_MEAN"])) ) ))), ((((data["CC_AMT_BALANCE_MEAN"]) - (data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"])))))) 
    v["i91"] = 0.046517*np.tanh(((((((np.where(data["ACTIVE_DAYS_CREDIT_MAX"]>0, ((np.where(data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"]>0, data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"], data["ACTIVE_DAYS_CREDIT_MAX"] )) + (data["NAME_HOUSING_TYPE_Municipal_apartment"])), data["INSTAL_PAYMENT_DIFF_MEAN"] )) + (data["NAME_HOUSING_TYPE_Municipal_apartment"]))) * 2.0)) * 2.0)) 
    v["i92"] = 0.034976*np.tanh(((np.where(data["PREV_AMT_GOODS_PRICE_MEAN"] < -99998, data["NEW_DOC_IND_STD"], ((np.where(data["CLOSED_DAYS_CREDIT_MAX"]>0, data["INSTAL_AMT_PAYMENT_MIN"], np.where(data["ORGANIZATION_TYPE_Transport__type_3"]>0, data["ORGANIZATION_TYPE_Transport__type_3"], ((data["OBS_60_CNT_SOCIAL_CIRCLE"]) * (data["PREV_AMT_GOODS_PRICE_MEAN"])) ) )) * 2.0) )) * 2.0)) 
    v["i93"] = 0.015000*np.tanh(((np.maximum(((np.where(data["WEEKDAY_APPR_PROCESS_START_SATURDAY"]<0, data["ORGANIZATION_TYPE_Business_Entity_Type_3"], data["REFUSED_HOUR_APPR_PROCESS_START_MAX"] ))), ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])))) - (np.maximum(((np.maximum(((data["WEEKDAY_APPR_PROCESS_START_SUNDAY"])), ((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))))), ((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_MEAN"])), ((data["OCCUPATION_TYPE_Accountants"]))))))))) 
    v["i94"] = 0.049002*np.tanh((((data["NEW_EXT_SOURCES_MEAN"]) < (((np.tanh((np.where(data["NEW_EXT_SOURCES_MEAN"]<0, data["NEW_SOURCES_PROD"], np.tanh((np.tanh(((((data["ORGANIZATION_TYPE_Construction"]) > (np.tanh((data["APPROVED_AMT_DOWN_PAYMENT_MIN"]))))*1.))))) )))) * 2.0)))*1.)) 
    v["i95"] = 0.041000*np.tanh(((data["FLAG_WORK_PHONE"]) + (np.where(data["APPROVED_DAYS_DECISION_MIN"]>0, np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, data["CC_AMT_DRAWINGS_CURRENT_MEAN"], ((((data["INSTAL_DPD_MEAN"]) + (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["APPROVED_DAYS_DECISION_MIN"]))))) * 2.0) ), data["NEW_CREDIT_TO_ANNUITY_RATIO"] )))) 
    v["i96"] = 0.048802*np.tanh(np.where(data["EXT_SOURCE_1"] < -99998, ((data["DAYS_BIRTH"]) * 2.0), (((((((-1.0*((((((data["DAYS_BIRTH"]) * 2.0)) * 2.0))))) - (data["EXT_SOURCE_1"]))) - (data["EXT_SOURCE_1"]))) - (data["EXT_SOURCE_1"])) )) 
    v["i97"] = 0.049802*np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_MIN"] < -99998, data["AMT_INCOME_TOTAL"], np.where(data["CC_AMT_RECIVABLE_MEAN"]>0, data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"], ((np.where(data["NEW_INC_BY_ORG"]<0, data["CLOSED_DAYS_CREDIT_MEAN"], data["PREV_NAME_PRODUCT_TYPE_x_sell_MEAN"] )) - (data["DAYS_EMPLOYED"])) ) )) 
    v["i98"] = 0.011046*np.tanh(((np.where((((data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > (data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]))*1.)>0, data["INSTAL_AMT_PAYMENT_MIN"], ((np.maximum(((data["POS_SK_DPD_DEF_MAX"])), (((((data["PREV_DAYS_DECISION_MEAN"]) > ((((data["POS_SK_DPD_DEF_MAX"]) > (data["NEW_RATIO_PREV_DAYS_DECISION_MAX"]))*1.)))*1.))))) * 2.0) )) * 2.0)) 
    v["i99"] = 0.049544*np.tanh(((np.maximum(((data["ORGANIZATION_TYPE_Transport__type_3"])), ((np.maximum(((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"])), ((np.maximum(((data["CC_CNT_DRAWINGS_CURRENT_VAR"])), ((np.where(data["NEW_RATIO_PREV_APP_CREDIT_PERC_MEAN"] < -99998, data["NEW_SCORES_STD"], data["OBS_60_CNT_SOCIAL_CIRCLE"] ))))))))))) + (np.minimum(((data["REGION_RATING_CLIENT_W_CITY"])), ((data["DAYS_ID_PUBLISH"])))))) 
    v["i100"] = 0.047080*np.tanh(np.where(data["EXT_SOURCE_3"] < -99998, ((data["EXT_SOURCE_2"]) * 2.0), ((((data["NAME_FAMILY_STATUS_Married"]) * (data["DAYS_BIRTH"]))) + ((((((data["EXT_SOURCE_3"]) / 2.0)) < (((data["ORGANIZATION_TYPE_Construction"]) - (data["NAME_FAMILY_STATUS_Married"]))))*1.))) )) 
    v["i101"] = 0.049910*np.tanh(np.where(data["BURO_CREDIT_TYPE_Mortgage_MEAN"]>0, data["CC_MONTHS_BALANCE_VAR"], np.where(data["EXT_SOURCE_1"]<0, (((data["NEW_EXT_SOURCES_MEAN"]) > (((data["PREV_CHANNEL_TYPE_Credit_and_cash_offices_MEAN"]) / 2.0)))*1.), ((((data["REGION_RATING_CLIENT_W_CITY"]) - (data["NEW_EXT_SOURCES_MEAN"]))) - (data["NAME_INCOME_TYPE_Working"])) ) )) 
    v["i102"] = 0.049000*np.tanh(((((np.tanh((np.tanh((data["NEW_EXT_SOURCES_MEAN"]))))) - (np.maximum((((((data["WEEKDAY_APPR_PROCESS_START_SUNDAY"]) > (((data["NEW_EXT_SOURCES_MEAN"]) / 2.0)))*1.))), ((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), ((data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]))))))))) - (data["NEW_EXT_SOURCES_MEAN"]))) 
    v["i103"] = 0.047481*np.tanh(((((((((data["FLAG_DOCUMENT_3"]) * (np.where(data["BURO_MONTHS_BALANCE_SIZE_SUM"]<0, np.maximum(((((data["PREV_PRODUCT_COMBINATION_POS_household_without_interest_MEAN"]) * 2.0))), ((data["REGION_POPULATION_RELATIVE"]))), data["PREV_PRODUCT_COMBINATION_POS_household_without_interest_MEAN"] )))) * 2.0)) - (data["WEEKDAY_APPR_PROCESS_START_SUNDAY"]))) * 2.0)) 
    v["i104"] = 0.037198*np.tanh(np.maximum(((data["BURO_CREDIT_TYPE_Microloan_MEAN"])), ((((data["EXT_SOURCE_3"]) * (np.where(((data["EXT_SOURCE_3"]) * (data["EXT_SOURCE_2"])) < -99998, data["EXT_SOURCE_2"], np.tanh((((data["NEW_CREDIT_TO_GOODS_RATIO"]) * (data["EXT_SOURCE_2"])))) ))))))) 
    v["i105"] = 0.049050*np.tanh(np.where(data["PREV_AMT_DOWN_PAYMENT_MIN"]>0, data["INSTAL_AMT_PAYMENT_MIN"], ((((np.where(data["PREV_NAME_PRODUCT_TYPE_walk_in_MEAN"]>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["PREV_AMT_DOWN_PAYMENT_MIN"] )) - (np.where(data["TOTALAREA_MODE"] < -99998, data["PREV_NAME_YIELD_GROUP_middle_MEAN"], data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"] )))) * 2.0) )) 
    v["i106"] = 0.043662*np.tanh(np.where(data["NEW_DOC_IND_KURT"]>0, np.where(data["BURO_DAYS_CREDIT_MIN"]>0, np.maximum(((data["PREV_WEEKDAY_APPR_PROCESS_START_FRIDAY_MEAN"])), ((data["NEW_RATIO_BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]))), np.where(data["NEW_INC_PER_CHLD"]<0, data["ACTIVE_MONTHS_BALANCE_MIN_MIN"], (-1.0*((data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]))) ) ), ((data["NEW_DOC_IND_KURT"]) * 2.0) )) 
    v["i107"] = 0.023067*np.tanh((((((((data["EXT_SOURCE_2"]) < ((-1.0*((((((((-1.0*((data["EXT_SOURCE_2"])))) > (((data["PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN"]) * ((((data["BURO_DAYS_CREDIT_MAX"]) < (data["PREV_AMT_GOODS_PRICE_MEAN"]))*1.)))))*1.)) * 2.0))))))*1.)) * 2.0)) * 2.0)) 
    v["i108"] = 0.050000*np.tanh((-1.0*((np.where(data["APPROVED_AMT_CREDIT_MAX"]>0, data["NEW_RATIO_PREV_AMT_CREDIT_MAX"], np.where(np.where(data["CLOSED_AMT_CREDIT_SUM_SUM"]>0, data["NEW_EMPLOY_TO_BIRTH_RATIO"], (((data["CODE_GENDER"]) + (data["POS_MONTHS_BALANCE_SIZE"]))/2.0) )<0, data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"], (-1.0*((data["REFUSED_AMT_CREDIT_MIN"]))) ) ))))) 
    v["i109"] = 0.001990*np.tanh(((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"]) - (np.where(data["APPROVED_AMT_ANNUITY_MEAN"]>0, data["PREV_NAME_SELLER_INDUSTRY_XNA_MEAN"], ((np.maximum(((((np.maximum(((data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"])), ((np.minimum(((data["BURO_DAYS_CREDIT_UPDATE_MEAN"])), ((data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"]))))))) * 2.0))), ((data["INSTAL_AMT_INSTALMENT_MEAN"])))) * 2.0) )))) 
    v["i110"] = 0.035600*np.tanh((((-1.0*((np.maximum((((((data["NAME_FAMILY_STATUS_Married"]) + (data["NAME_INCOME_TYPE_Commercial_associate"]))/2.0))), ((data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"]))))))) - (np.maximum((((((data["PREV_WEEKDAY_APPR_PROCESS_START_SATURDAY_MEAN"]) + (data["OCCUPATION_TYPE_High_skill_tech_staff"]))/2.0))), ((((data["OCCUPATION_TYPE_Accountants"]) + (data["ORGANIZATION_TYPE_Medicine"])))))))) 
    v["i111"] = 0.026365*np.tanh(((((((((data["ORGANIZATION_TYPE_Medicine"]) > (data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"]))*1.)) > (data["DAYS_BIRTH"]))*1.)) - (np.where(data["DAYS_BIRTH"]>0, data["NEW_INC_BY_ORG"], ((data["PREV_NAME_PORTFOLIO_POS_MEAN"]) + ((((data["POS_SK_DPD_MEAN"]) > (data["CC_AMT_BALANCE_VAR"]))*1.))) )))) 
    v["i112"] = 0.049648*np.tanh(np.where(data["NEW_RATIO_PREV_AMT_APPLICATION_MIN"] < -99998, (((((data["PREV_NAME_CASH_LOAN_PURPOSE_XNA_MEAN"]) * ((-1.0*((data["NAME_HOUSING_TYPE_House___apartment"])))))) + (((data["POS_SK_DPD_MAX"]) - (data["NEW_DOC_IND_AVG"]))))/2.0), ((data["PREV_CHANNEL_TYPE_Stone_MEAN"]) - ((-1.0*((data["PREV_NAME_TYPE_SUITE_Spouse__partner_MEAN"]))))) )) 
    v["i113"] = 0.028480*np.tanh(np.where(data["PREV_CHANNEL_TYPE_Credit_and_cash_offices_MEAN"] < -99998, data["NEW_DOC_IND_STD"], np.where(data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"] < -99998, data["POS_SK_DPD_DEF_MAX"], np.where(data["PREV_CHANNEL_TYPE_Credit_and_cash_offices_MEAN"]>0, data["BURO_CREDIT_TYPE_Credit_card_MEAN"], (((data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > ((((data["PREV_NAME_PAYMENT_TYPE_Cash_through_the_bank_MEAN"]) < (data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]))*1.)))*1.) ) ) )) 
    v["i114"] = 0.048956*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_SUM"]<0, np.where(data["AMT_GOODS_PRICE"]<0, np.where(data["NEW_RATIO_BURO_MONTHS_BALANCE_SIZE_SUM"] < -99998, data["OCCUPATION_TYPE_Laborers"], data["POS_COUNT"] ), data["NAME_EDUCATION_TYPE_Higher_education"] ), ((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) + ((-1.0*((data["ACTIVE_AMT_CREDIT_SUM_MAX"]))))) )) 
    v["i115"] = 0.039000*np.tanh(np.where(data["NEW_RATIO_BURO_DAYS_CREDIT_MEAN"]>0, np.where(data["NEW_EXT_SOURCES_MEAN"]<0, ((data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"]) * 2.0), data["PREV_CODE_REJECT_REASON_SCO_MEAN"] ), (-1.0*((np.where(data["NEW_EXT_SOURCES_MEAN"]<0, data["INSTAL_AMT_PAYMENT_SUM"], ((data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"]) + (data["PREV_CODE_REJECT_REASON_SCO_MEAN"])) )))) )) 
    v["i116"] = 0.049997*np.tanh((-1.0*((((np.where(data["CC_AMT_RECIVABLE_VAR"]>0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"], (-1.0*(((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) > (data["BURO_AMT_CREDIT_SUM_MEAN"]))*1.)))) )) + ((((data["BURO_AMT_CREDIT_SUM_MEAN"]) > (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))*1.))))))) 
    v["i117"] = 0.049001*np.tanh(np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_VAR"]>0, data["CLOSED_DAYS_CREDIT_MIN"], np.where(data["NEW_CREDIT_TO_GOODS_RATIO"]>0, np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["APPROVED_AMT_APPLICATION_MAX"]))), (((data["NEW_CREDIT_TO_GOODS_RATIO"]) < (np.where(data["APPROVED_AMT_APPLICATION_MEAN"]<0, data["APPROVED_AMT_APPLICATION_MEAN"], data["CC_NAME_CONTRACT_STATUS_Active_SUM"] )))*1.) ) )) 
    v["i118"] = 0.029997*np.tanh(np.where(data["NEW_RATIO_PREV_APP_CREDIT_PERC_MEAN"]>0, data["NEW_RATIO_PREV_APP_CREDIT_PERC_MEAN"], np.where(data["CC_AMT_RECEIVABLE_PRINCIPAL_VAR"] < -99998, np.where(data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]<0, data["NEW_CREDIT_TO_GOODS_RATIO"], data["BURO_AMT_CREDIT_SUM_OVERDUE_MEAN"] ), np.where(data["CC_CNT_DRAWINGS_CURRENT_MAX"]<0, data["CLOSED_DAYS_CREDIT_VAR"], data["PREV_NAME_PAYMENT_TYPE_XNA_MEAN"] ) ) )) 
    v["i119"] = 0.049942*np.tanh(np.where(data["PREV_NAME_CLIENT_TYPE_Refreshed_MEAN"]>0, data["BURO_CREDIT_TYPE_Credit_card_MEAN"], np.maximum(((data["BURO_CREDIT_ACTIVE_Sold_MEAN"])), ((np.where(data["PREV_AMT_DOWN_PAYMENT_MIN"]>0, data["FLOORSMIN_MODE"], np.where(data["NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM"]>0, data["BURO_CREDIT_TYPE_Credit_card_MEAN"], ((data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"]) - (data["CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN"])) ) )))) )) 
    v["i120"] = 0.014008*np.tanh((-1.0*((np.where(data["NONLIVINGAREA_AVG"]>0, data["NEW_INC_BY_ORG"], ((np.maximum(((data["AMT_REQ_CREDIT_BUREAU_QRT"])), (((((np.maximum(((data["WEEKDAY_APPR_PROCESS_START_SATURDAY"])), ((data["NEW_RATIO_PREV_AMT_CREDIT_MAX"])))) + (np.maximum(((data["NAME_HOUSING_TYPE_Office_apartment"])), ((data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"])))))/2.0))))) * 2.0) ))))) 
    v["i121"] = 0.047508*np.tanh(np.where(data["FLAG_PHONE"]<0, ((data["INSTAL_DBD_SUM"]) * (np.where(data["NEW_DOC_IND_STD"]>0, np.where(data["BURO_DAYS_CREDIT_MAX"]>0, data["PREV_NAME_YIELD_GROUP_high_MEAN"], data["FLAG_PHONE"] ), 3.0 ))), data["OWN_CAR_AGE"] )) 
    v["i122"] = 0.022005*np.tanh(((np.maximum(((data["CC_AMT_RECIVABLE_VAR"])), ((np.maximum((((((data["POS_MONTHS_BALANCE_MEAN"]) < (np.minimum(((data["EXT_SOURCE_3"])), ((data["INSTAL_PAYMENT_DIFF_SUM"])))))*1.))), (((((data["PREV_NAME_CASH_LOAN_PURPOSE_Medicine_MEAN"]) > (((data["BURO_AMT_CREDIT_SUM_OVERDUE_MEAN"]) * (data["REFUSED_AMT_GOODS_PRICE_MAX"]))))*1.)))))))) * 2.0)) 
    v["i123"] = 0.024998*np.tanh(np.where(data["LIVE_CITY_NOT_WORK_CITY"]>0, data["ACTIVE_AMT_ANNUITY_MEAN"], np.where(data["ACTIVE_AMT_CREDIT_SUM_DEBT_MAX"]<0, np.where(data["CLOSED_DAYS_CREDIT_VAR"]<0, np.where(data["PREV_NAME_CONTRACT_STATUS_Approved_MEAN"]<0, data["NEW_RATIO_PREV_AMT_GOODS_PRICE_MIN"], data["PREV_NAME_CONTRACT_STATUS_Approved_MEAN"] ), data["APPROVED_APP_CREDIT_PERC_VAR"] ), data["NEW_CREDIT_TO_INCOME_RATIO"] ) )) 
    v["i124"] = 0.037322*np.tanh(np.maximum(((np.where(data["BURO_AMT_CREDIT_SUM_OVERDUE_MEAN"] < -99998, data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"], data["BURO_AMT_CREDIT_SUM_OVERDUE_MEAN"] ))), ((np.where(data["APPROVED_AMT_GOODS_PRICE_MIN"]>0, data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"], (((data["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"]) < (data["APPROVED_AMT_GOODS_PRICE_MIN"]))*1.) ))))) 
    v["i125"] = 0.049402*np.tanh(((np.where(data["CC_AMT_DRAWINGS_POS_CURRENT_SUM"]>0, data["PREV_AMT_GOODS_PRICE_MIN"], np.where(data["PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN"]>0, (-1.0*((data["PREV_NAME_YIELD_GROUP_middle_MEAN"]))), ((data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__middle_MEAN"]) * (data["APPROVED_CNT_PAYMENT_MEAN"])) ) )) - ((((data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"]) > (data["REG_CITY_NOT_LIVE_CITY"]))*1.)))) 
    v["i126"] = 0.049398*np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_OVERDUE_MEAN"] < -99998, data["DEF_30_CNT_SOCIAL_CIRCLE"], np.where(data["EXT_SOURCE_3"] < -99998, data["EXT_SOURCE_2"], ((np.maximum(((data["DAYS_BIRTH"])), ((data["NAME_EDUCATION_TYPE_Higher_education"])))) - (((data["NAME_EDUCATION_TYPE_Higher_education"]) * (data["DAYS_BIRTH"])))) ) )) 
    v["i127"] = 0.043856*np.tanh(((np.where(data["DAYS_ID_PUBLISH"]<0, ((data["DAYS_BIRTH"]) - (data["NEW_EMPLOY_TO_BIRTH_RATIO"])), (-1.0*((np.maximum(((data["PREV_NAME_CONTRACT_TYPE_Consumer_loans_MEAN"])), ((data["DAYS_BIRTH"])))))) )) + (data["DAYS_ID_PUBLISH"]))) 
    v["i128"] = 0.047562*np.tanh(((data["NAME_EDUCATION_TYPE_Lower_secondary"]) + (((data["POS_SK_DPD_MEAN"]) + (np.maximum(((((np.where(data["ACTIVE_DAYS_CREDIT_ENDDATE_MAX"] < -99998, data["INSTAL_PAYMENT_DIFF_SUM"], data["PREV_AMT_ANNUITY_MIN"] )) + (data["ORGANIZATION_TYPE_Transport__type_3"])))), ((((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) + (data["BURO_CREDIT_TYPE_Microloan_MEAN"])))))))))) 
    v["i129"] = 0.041886*np.tanh(((data["DAYS_REGISTRATION"]) * ((((((data["POS_NAME_CONTRACT_STATUS_Signed_MEAN"]) + (data["REGION_RATING_CLIENT_W_CITY"]))/2.0)) + (((((((data["PREV_CODE_REJECT_REASON_SCOFR_MEAN"]) + ((-1.0*((data["ORGANIZATION_TYPE_Military"])))))/2.0)) + ((((data["POS_SK_DPD_DEF_MEAN"]) > (data["DAYS_BIRTH"]))*1.)))/2.0)))))) 
    v["i130"] = 0.049340*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]<0, np.where(data["CC_AMT_INST_MIN_REGULARITY_SUM"]<0, np.where(data["CC_AMT_INST_MIN_REGULARITY_SUM"] < -99998, data["NAME_HOUSING_TYPE_Rented_apartment"], data["APPROVED_CNT_PAYMENT_MEAN"] ), np.where(data["CC_CNT_INSTALMENT_MATURE_CUM_MEAN"]>0, data["FLOORSMIN_AVG"], data["CC_AMT_CREDIT_LIMIT_ACTUAL_MIN"] ) ), 3.141593 )) 
    v["i131"] = 0.044198*np.tanh(np.where(((data["APPROVED_APP_CREDIT_PERC_VAR"]) - (data["EXT_SOURCE_3"])) < -99998, ((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) * 2.0), np.maximum(((((data["EXT_SOURCE_3"]) - (data["APPROVED_APP_CREDIT_PERC_VAR"])))), ((((((data["CC_AMT_RECEIVABLE_PRINCIPAL_VAR"]) - (data["CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN"]))) * 2.0)))) )) 
    v["i132"] = 0.049514*np.tanh((-1.0*((np.where(data["LIVINGAREA_AVG"]>0, data["NEW_RATIO_PREV_APP_CREDIT_PERC_MEAN"], np.maximum(((data["NAME_HOUSING_TYPE_Office_apartment"])), ((((np.maximum((((((data["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"]) > (np.maximum(((data["REGION_POPULATION_RELATIVE"])), ((data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"])))))*1.))), ((data["ORGANIZATION_TYPE_Industry__type_9"])))) * 2.0)))) ))))) 
    v["i133"] = 0.049862*np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MAX"]>0, data["CC_NAME_CONTRACT_STATUS_Active_SUM"], ((np.where((((data["ENTRANCES_MEDI"]) > (data["INSTAL_DPD_MEAN"]))*1.)>0, data["CC_NAME_CONTRACT_STATUS_Active_SUM"], (((data["NEW_RATIO_PREV_DAYS_DECISION_MAX"]) > (data["ORGANIZATION_TYPE_Military"]))*1.) )) * 2.0) )) 
    v["i134"] = 0.045502*np.tanh((-1.0*((np.where(data["EXT_SOURCE_3"]>0, data["BURO_DAYS_CREDIT_UPDATE_MEAN"], np.where(data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]>0, data["INSTAL_PAYMENT_DIFF_MAX"], np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"]<0, data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"], np.where(data["ACTIVE_AMT_CREDIT_SUM_SUM"]<0, data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"], data["EXT_SOURCE_3"] ) ) ) ))))) 
    v["i135"] = 0.020008*np.tanh((-1.0*((np.where(data["APPROVED_CNT_PAYMENT_SUM"]>0, ((np.tanh((data["DAYS_EMPLOYED"]))) - (np.where(data["APPROVED_AMT_GOODS_PRICE_MIN"]<0, (((data["INSTAL_DPD_MEAN"]) > (data["DAYS_EMPLOYED"]))*1.), data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] ))), data["OCCUPATION_TYPE_Medicine_staff"] ))))) 
    v["i136"] = 0.035000*np.tanh(np.where(data["NEW_EXT_SOURCES_MEAN"]>0, (-1.0*((data["NEW_PHONE_TO_BIRTH_RATIO"]))), np.where(data["EXT_SOURCE_3"] < -99998, data["REFUSED_AMT_CREDIT_MAX"], ((data["PREV_NAME_GOODS_CATEGORY_Photo___Cinema_Equipment_MEAN"]) * (np.where(data["PREV_NAME_GOODS_CATEGORY_Photo___Cinema_Equipment_MEAN"] < -99998, data["EXT_SOURCE_3"], data["ACTIVE_DAYS_CREDIT_ENDDATE_MEAN"] ))) ) )) 
    v["i137"] = 0.048501*np.tanh(np.where(data["NEW_RATIO_BURO_MONTHS_BALANCE_SIZE_MEAN"]>0, data["PREV_NAME_YIELD_GROUP_middle_MEAN"], np.maximum(((data["PREV_NAME_GOODS_CATEGORY_Audio_Video_MEAN"])), ((((np.where(data["INSTAL_PAYMENT_DIFF_MAX"]>0, data["OCCUPATION_TYPE_Drivers"], data["APPROVED_DAYS_DECISION_MAX"] )) * (data["NEW_CREDIT_TO_GOODS_RATIO"]))))) )) 
    v["i138"] = 0.048902*np.tanh(np.where(data["NEW_RATIO_BURO_DAYS_CREDIT_VAR"] < -99998, (((np.where(data["BURO_DAYS_CREDIT_MEAN"]>0, data["PREV_CHANNEL_TYPE_Stone_MEAN"], data["PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN"] )) + (np.where(data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"] < -99998, data["PREV_CHANNEL_TYPE_Stone_MEAN"], data["ACTIVE_DAYS_CREDIT_MEAN"] )))/2.0), ((data["NEW_RATIO_PREV_AMT_ANNUITY_MAX"]) + (data["BURO_DAYS_CREDIT_MEAN"])) )) 
    v["i139"] = 0.043840*np.tanh(np.where(data["AMT_INCOME_TOTAL"]>0, data["FLAG_DOCUMENT_3"], np.maximum(((data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"])), ((np.maximum(((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"])), (((-1.0*(((((((data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"]) > (data["NEW_EXT_SOURCES_MEAN"]))*1.)) + (data["NEW_CREDIT_TO_GOODS_RATIO"]))))))))))) )) 
    v["i140"] = 0.049910*np.tanh(np.where(data["PREV_NAME_PAYMENT_TYPE_Cash_through_the_bank_MEAN"]>0, data["PREV_CODE_REJECT_REASON_XAP_MEAN"], np.where(data["CC_AMT_PAYMENT_CURRENT_SUM"]>0, data["OCCUPATION_TYPE_Drivers"], (((data["OCCUPATION_TYPE_Drivers"]) > (np.where(data["APPROVED_AMT_ANNUITY_MIN"] < -99998, data["NEW_RATIO_BURO_DAYS_CREDIT_MAX"], ((data["APPROVED_AMT_GOODS_PRICE_MIN"]) / 2.0) )))*1.) ) )) 
    v["i141"] = 0.049610*np.tanh(((np.where(data["LANDAREA_AVG"]>0, data["CC_NAME_CONTRACT_STATUS_Active_SUM"], ((np.where(data["BURO_STATUS_0_MEAN_MEAN"]>0, data["PREV_PRODUCT_COMBINATION_POS_household_without_interest_MEAN"], np.where(data["BASEMENTAREA_AVG"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (((data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > (data["POS_SK_DPD_DEF_MEAN"]))*1.) ) )) * 2.0) )) * 2.0)) 
    v["i142"] = 0.042842*np.tanh(np.where(data["CODE_GENDER"]>0, np.minimum(((data["CODE_GENDER"])), ((((data["DAYS_BIRTH"]) * 2.0)))), (-1.0*((((data["DAYS_BIRTH"]) - ((-1.0*((((data["NEW_EXT_SOURCES_MEAN"]) - ((-1.0*((data["DAYS_BIRTH"]))))))))))))) )) 
    v["i143"] = 0.045501*np.tanh(np.where(data["REGION_RATING_CLIENT"]>0, ((data["PREV_RATE_DOWN_PAYMENT_MIN"]) * 2.0), np.where(((data["CC_AMT_PAYMENT_CURRENT_SUM"]) / 2.0)>0, data["PREV_NAME_TYPE_SUITE_Family_MEAN"], np.where(data["NEW_PHONE_TO_EMPLOY_RATIO"]>0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], (((data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"]) > (data["NEW_PHONE_TO_EMPLOY_RATIO"]))*1.) ) ) )) 
    v["i144"] = 0.000197*np.tanh((((data["POS_NAME_CONTRACT_STATUS_Completed_MEAN"]) < (np.tanh((np.where(data["PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN"]<0, np.where(data["PREV_AMT_GOODS_PRICE_MAX"]<0, data["FLOORSMIN_MODE"], data["APPROVED_AMT_APPLICATION_MAX"] ), np.where(data["BURO_STATUS_1_MEAN_MEAN"]<0, data["CC_AMT_DRAWINGS_ATM_CURRENT_SUM"], (-1.0*((data["PREV_NAME_SELLER_INDUSTRY_XNA_MEAN"]))) ) )))))*1.)) 
    v["i145"] = 0.049703*np.tanh(np.maximum(((data["POS_NAME_CONTRACT_STATUS_Returned_to_the_store_MEAN"])), ((np.maximum(((((((data["BURO_AMT_CREDIT_SUM_OVERDUE_MEAN"]) * 2.0)) * 2.0))), (((((data["DAYS_EMPLOYED"]) < (np.where(np.maximum(((data["BURO_DAYS_CREDIT_ENDDATE_MIN"])), ((data["ACTIVE_DAYS_CREDIT_ENDDATE_MIN"])))>0, data["BURO_CREDIT_ACTIVE_Sold_MEAN"], data["CC_CNT_DRAWINGS_POS_CURRENT_MIN"] )))*1.)))))))) 
    v["i146"] = 0.049482*np.tanh(np.where(data["NEW_CREDIT_TO_GOODS_RATIO"]<0, data["INSTAL_DBD_MAX"], ((np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]<0, np.where(data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"]>0, data["APPROVED_AMT_GOODS_PRICE_MAX"], ((data["PREV_NAME_PORTFOLIO_Cash_MEAN"]) * (data["POS_NAME_CONTRACT_STATUS_Completed_MEAN"])) ), data["NEW_RATIO_BURO_AMT_ANNUITY_MEAN"] )) - (data["PREV_NAME_TYPE_SUITE_Children_MEAN"])) )) 
    v["i147"] = 0.040002*np.tanh(np.where(data["APARTMENTS_AVG"]<0, np.where(data["BURO_AMT_CREDIT_SUM_OVERDUE_MEAN"]<0, ((((data["APPROVED_AMT_ANNUITY_MEAN"]) * (data["OCCUPATION_TYPE_Laborers"]))) / 2.0), 1.0 ), (-1.0*((np.where(data["NAME_FAMILY_STATUS_Married"]<0, data["BURO_STATUS_0_MEAN_MEAN"], data["OCCUPATION_TYPE_Laborers"] )))) )) 
    v["i148"] = 0.048200*np.tanh(np.where(data["PREV_NAME_SELLER_INDUSTRY_Clothing_MEAN"]>0, data["CC_CNT_DRAWINGS_POS_CURRENT_VAR"], np.where(data["INSTAL_DBD_MEAN"]>0, data["POS_COUNT"], ((data["INSTAL_DAYS_ENTRY_PAYMENT_MEAN"]) * ((((np.where(data["REFUSED_AMT_GOODS_PRICE_MEAN"] < -99998, data["PREV_NAME_SELLER_INDUSTRY_Clothing_MEAN"], data["REFUSED_AMT_GOODS_PRICE_MEAN"] )) > (data["POS_COUNT"]))*1.))) ) )) 
    v["i149"] = 0.047494*np.tanh((-1.0*((np.maximum((((((data["APPROVED_HOUR_APPR_PROCESS_START_MIN"]) < (np.where(data["INSTAL_AMT_PAYMENT_MIN"]<0, data["REFUSED_DAYS_DECISION_MIN"], np.where(data["REFUSED_AMT_ANNUITY_MIN"] < -99998, data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"], data["NONLIVINGAPARTMENTS_AVG"] ) )))*1.))), ((np.maximum(((data["ORGANIZATION_TYPE_Industry__type_9"])), ((data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]))))))))))
    return Output(v.sum(axis=1)-2.432490)
roc_auc_score(train_df.TARGET,GP1(train_df))
roc_auc_score(train_df.TARGET,GP2(train_df))
x = test_df[['SK_ID_CURR']].copy()
x['TARGET'] = .5*GP1(test_df)+.5*GP2(test_df)
x.to_csv('pure_submission.csv', index = False)