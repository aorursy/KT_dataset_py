import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
warnings.simplefilter(action='ignore', category=FutureWarning)
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
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
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
    df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    df= df.drop(dropcolum,axis=1)
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
        'DAYS_CREDIT': [ 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': [ 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum'],
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
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
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
        'AMT_ANNUITY': [ 'max', 'mean'],
        'AMT_APPLICATION': [ 'max','mean'],
        'AMT_CREDIT': [ 'max', 'mean'],
        'APP_CREDIT_PERC': [ 'max', 'mean'],
        'AMT_DOWN_PAYMENT': [ 'max', 'mean'],
        'AMT_GOODS_PRICE': [ 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': [ 'max', 'mean'],
        'RATE_DOWN_PAYMENT': [ 'max', 'mean'],
        'DAYS_DECISION': [ 'max', 'mean'],
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
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
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
        'DPD': ['max', 'mean', 'sum','min','std' ],
        'DBD': ['max', 'mean', 'sum','min','std'],
        'PAYMENT_PERC': [ 'max','mean',  'var','min','std'],
        'PAYMENT_DIFF': [ 'max','mean', 'var','min','std'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum','min','std'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum','std'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum','std']
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
    cc_agg = cc.groupby('SK_ID_CURR').agg([ 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg
num_rows = None
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
df.drop('index',inplace=True,axis=1)
df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '_') #My GP doesn't like funky names!
features = list(set(df.columns).difference(['SK_ID_CURR','TARGET']))
for c in features:
    ss = StandardScaler()
    df.loc[~np.isfinite(df[c]),c] = np.nan
    df.loc[~df[c].isnull(),c] = ss.fit_transform(df.loc[~df[c].isnull(),c].values.reshape(-1,1))
    df[c].fillna(df[c].mean(),inplace=True)
train_df = df[df['TARGET'].notnull()].copy()
test_df = df[df['TARGET'].isnull()].copy()
def GPCluster1(data):
    v = pd.DataFrame()
    v["i0"] = 0.050000*np.tanh(((-2.0) * (data["EXT_SOURCE_2"])))  
    v["i1"] = 0.050000*np.tanh((((2.0) > (data["DAYS_BIRTH"]))*1.))  
    v["i2"] = 0.050000*np.tanh(((np.where(((data["NEW_EXT_SOURCES_MEAN"]) - (data["NEW_SOURCES_PROD"]))>0, ((((data["NEW_EXT_SOURCES_MEAN"]) + ((((1.0) > (data["NEW_EXT_SOURCES_MEAN"]))*1.)))) * 2.0), 0.318310 )) + (data["NEW_EXT_SOURCES_MEAN"])))  
    v["i3"] = 0.050000*np.tanh(((((((0.636620) + (data["NEW_EXT_SOURCES_MEAN"]))/2.0)) + ((((1.570796) + (data["EXT_SOURCE_3"]))/2.0)))/2.0))  
    v["i4"] = 0.050000*np.tanh(((0.636620) * (np.minimum((((9.0))), ((np.tanh(((-1.0*((((data["EXT_SOURCE_3"]) + (data["DAYS_BIRTH"])))))))))))))  
    v["i5"] = 0.050000*np.tanh(((data["EXT_SOURCE_1"]) + (((((((((data["BURO_DAYS_CREDIT_MEAN"]) / 2.0)) > (data["EXT_SOURCE_2"]))*1.)) < (np.where(data["EXT_SOURCE_1"]>0, data["EXT_SOURCE_3"], data["EXT_SOURCE_3"] )))*1.))))  
    v["i6"] = 0.050000*np.tanh((-1.0*((((data["EXT_SOURCE_3"]) - (np.where(data["EXT_SOURCE_1"]>0, ((np.where(-2.0>0, data["DAYS_EMPLOYED"], -2.0 )) - (data["EXT_SOURCE_2"])), ((data["DAYS_EMPLOYED"]) - (data["EXT_SOURCE_2"])) )))))))  
    v["i7"] = 0.049976*np.tanh(((((data["DAYS_EMPLOYED"]) + (((((data["DAYS_BIRTH"]) + (data["NEW_CREDIT_TO_GOODS_RATIO"]))) + (((((data["BURO_DAYS_CREDIT_MEAN"]) - (data["EXT_SOURCE_1"]))) - (data["EXT_SOURCE_1"]))))))) + (data["DAYS_EMPLOYED"])))  
    v["i8"] = 0.050000*np.tanh(np.minimum(((np.minimum(((data["NEW_SOURCES_PROD"])), ((np.maximum(((1.570796)), ((np.minimum((((3.0))), ((data["EXT_SOURCE_2"]))))))))))), ((np.tanh((data["EXT_SOURCE_1"]))))))  
    v["i9"] = 0.050000*np.tanh((-1.0*(((((((((((data["DAYS_EMPLOYED"]) - ((((0.0) < (0.636620))*1.)))) * 2.0)) + (data["DAYS_EMPLOYED"]))) < (data["NEW_SOURCES_PROD"]))*1.)))))  
    v["i10"] = 0.050000*np.tanh((((((data["EXT_SOURCE_2"]) + (((data["EXT_SOURCE_3"]) + (data["EXT_SOURCE_3"]))))/2.0)) + (np.maximum(((data["EXT_SOURCE_3"])), ((data["EXT_SOURCE_2"]))))))  
    v["i11"] = 0.050000*np.tanh(((((data["NEW_SOURCES_PROD"]) + (((data["EXT_SOURCE_2"]) + (data["EXT_SOURCE_3"]))))) + (((data["NAME_EDUCATION_TYPE_Higher_education"]) + (((data["EXT_SOURCE_3"]) + (data["EXT_SOURCE_2"])))))))  
    v["i12"] = 0.050000*np.tanh(((((((((3.141593) / 2.0)) + (((data["NEW_EXT_SOURCES_MEAN"]) + (data["NEW_EXT_SOURCES_MEAN"]))))) + (np.minimum(((data["NAME_EDUCATION_TYPE_Higher_education"])), (((((data["NEW_EXT_SOURCES_MEAN"]) < (data["NEW_EXT_SOURCES_MEAN"]))*1.))))))) - (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])))  
    v["i13"] = 0.049951*np.tanh((((-1.0*(((-1.0*((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))))))) * 2.0))  
    v["i14"] = 0.050000*np.tanh(((((((1.0) > ((((2.0) > (0.318310))*1.)))*1.)) > (np.maximum(((0.318310)), ((data["NEW_CREDIT_TO_GOODS_RATIO"])))))*1.))  
    v["i15"] = 0.050000*np.tanh(np.tanh((((((((((data["NEW_SOURCES_PROD"]) > (np.tanh((0.636620))))*1.)) + (((1.0) / 2.0)))) < ((((((((2.66290140151977539)) / 2.0)) * 2.0)) * 2.0)))*1.))))  
    v["i16"] = 0.050000*np.tanh(((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) / 2.0)) * (np.where(((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) / 2.0)>0, ((0.636620) - (np.tanh((1.570796)))), data["NAME_EDUCATION_TYPE_Higher_education"] ))))  
    v["i17"] = 0.050000*np.tanh(np.minimum((((7.51910972595214844))), (((((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]) + (((np.maximum(((np.tanh((((data["EXT_SOURCE_3"]) / 2.0))))), ((data["PREV_NAME_PRODUCT_TYPE_walk_in_MEAN"])))) * 2.0)))/2.0)))))  
    v["i18"] = 0.050000*np.tanh(((((((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) + ((((data["INSTAL_PAYMENT_PERC_MIN"]) + (1.0))/2.0)))) + ((((data["INSTAL_PAYMENT_PERC_MIN"]) + (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]))/2.0)))) + (((data["INSTAL_PAYMENT_PERC_MIN"]) + (data["NEW_EXT_SOURCES_MEAN"])))))  
    v["i19"] = 0.050000*np.tanh(((np.tanh((((((data["CODE_GENDER"]) + (((data["NEW_SOURCES_PROD"]) + (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]))))) + (data["CODE_GENDER"]))))) + ((((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) + (((data["NEW_SOURCES_PROD"]) + (data["NEW_SOURCES_PROD"]))))/2.0))))  
    v["i20"] = 0.050000*np.tanh(((((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) - ((((np.tanh(((((-3.0) < ((((7.0)) * 2.0)))*1.)))) > (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))*1.)))) - (data["NEW_EXT_SOURCES_MEAN"]))) - (data["NEW_EXT_SOURCES_MEAN"])))  
    v["i21"] = 0.050000*np.tanh(((((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) - (((-1.0) - (data["NEW_EMPLOY_TO_BIRTH_RATIO"]))))) + (((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]) - (((((data["DAYS_EMPLOYED"]) - ((1.45568525791168213)))) - (data["NEW_EXT_SOURCES_MEAN"])))))))  
    v["i22"] = 0.050000*np.tanh(((np.tanh((data["NEW_SOURCES_PROD"]))) - (data["PREV_NAME_PRODUCT_TYPE_walk_in_MEAN"])))  
    v["i23"] = 0.050000*np.tanh(((((((data["CC_AMT_RECIVABLE_MEAN"]) + (data["FLAG_DOCUMENT_3"]))) + (np.minimum(((data["CC_AMT_RECIVABLE_MEAN"])), ((data["CC_AMT_RECIVABLE_MEAN"])))))) + (((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) + (((data["CC_AMT_BALANCE_MEAN"]) + (data["CC_AMT_RECIVABLE_MEAN"])))))))  
    v["i24"] = 0.050000*np.tanh(((((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) + (((data["NEW_CREDIT_TO_GOODS_RATIO"]) - (data["NEW_EXT_SOURCES_MEAN"]))))) + (data["NEW_CREDIT_TO_GOODS_RATIO"]))) - ((-1.0*((data["BURO_CREDIT_TYPE_Microloan_MEAN"]))))))  
    v["i25"] = 0.050000*np.tanh(((data["PREV_CNT_PAYMENT_MEAN"]) + (((((((((1.0) + (np.minimum(((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"])), (((((-1.0*((data["PREV_CNT_PAYMENT_MEAN"])))) * 2.0))))))) * 2.0)) + (data["NEW_EMPLOY_TO_BIRTH_RATIO"]))) + (data["EXT_SOURCE_3"])))))  
    v["i26"] = 0.050000*np.tanh(((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) - (data["NAME_EDUCATION_TYPE_Higher_education"]))) - (np.where(((data["BURO_CREDIT_TYPE_Microloan_MEAN"]) * 2.0)>0, ((data["NAME_EDUCATION_TYPE_Higher_education"]) - (((data["BURO_CREDIT_TYPE_Microloan_MEAN"]) * 2.0))), data["CODE_GENDER"] ))))  
    v["i27"] = 0.050000*np.tanh(((((((data["DEF_30_CNT_SOCIAL_CIRCLE"]) > (0.0))*1.)) < ((((((0.0) * ((8.73927879333496094)))) + (((0.0) * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))/2.0)))*1.))  
    v["i28"] = 0.050000*np.tanh(((data["CC_CNT_DRAWINGS_CURRENT_MAX"]) + ((((data["PREV_NAME_YIELD_GROUP_high_MEAN"]) + (((data["PREV_NAME_YIELD_GROUP_high_MEAN"]) - (data["PREV_NAME_YIELD_GROUP_high_MEAN"]))))/2.0))))  
    v["i29"] = 0.050000*np.tanh(np.minimum(((data["NAME_INCOME_TYPE_Working"])), (((2.63944816589355469)))))  
    v["i30"] = 0.050000*np.tanh(((data["PREV_NAME_PRODUCT_TYPE_walk_in_MEAN"]) + (((data["BURO_CREDIT_TYPE_Microloan_MEAN"]) + ((((data["BURO_CREDIT_TYPE_Microloan_MEAN"]) + (np.minimum(((-1.0)), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"])))))/2.0))))))  
    v["i31"] = 0.050000*np.tanh(np.minimum(((data["INSTAL_PAYMENT_PERC_MIN"])), (((((((np.tanh((data["INSTAL_PAYMENT_PERC_MIN"]))) * (data["INSTAL_PAYMENT_PERC_MIN"]))) + (data["INSTAL_PAYMENT_PERC_MIN"]))/2.0)))))  
    v["i32"] = 0.050000*np.tanh(((((((-1.0) < (np.minimum(((2.0)), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"])))))*1.)) + (-1.0))/2.0))  
    v["i33"] = 0.050000*np.tanh(np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]>0, np.tanh((np.minimum(((-2.0)), ((-2.0))))), (((1.570796) + (data["NEW_EXT_SOURCES_MEAN"]))/2.0) ))  
    v["i34"] = 0.050000*np.tanh((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["PREV_NAME_YIELD_GROUP_high_MEAN"]))/2.0))  
    v["i35"] = 0.050000*np.tanh(((((np.maximum(((np.maximum(((data["PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN"])), ((((data["PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN"]) * 2.0)))))), (((((data["PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN"]) + (-2.0))/2.0))))) * 2.0)) / 2.0))  
    v["i36"] = 0.050000*np.tanh(((((data["APPROVED_CNT_PAYMENT_MEAN"]) - (1.570796))) - (data["APPROVED_CNT_PAYMENT_MEAN"])))  
    v["i37"] = 0.050000*np.tanh(((((data["BURO_CREDIT_TYPE_Microloan_MEAN"]) + (data["NAME_EDUCATION_TYPE_Higher_education"]))) + (((((((((((data["ORGANIZATION_TYPE_Self_employed"]) + (data["NEW_DOC_IND_KURT"]))) - (data["POS_MONTHS_BALANCE_SIZE"]))) - (data["NAME_EDUCATION_TYPE_Higher_education"]))) - (data["POS_MONTHS_BALANCE_SIZE"]))) * 2.0))))  
    v["i38"] = 0.050000*np.tanh(np.where(data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]>0, data["CC_AMT_TOTAL_RECEIVABLE_MEAN"], np.minimum(((((data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]) * (1.0)))), ((((data["FLAG_OWN_CAR"]) + (((data["FLAG_OWN_CAR"]) - (data["FLAG_OWN_CAR"]))))))) ))  
    v["i39"] = 0.050000*np.tanh((((((data["BURO_CREDIT_ACTIVE_Closed_MEAN"]) > (0.0))*1.)) + (np.minimum((((8.0))), ((data["PREV_APP_CREDIT_PERC_MEAN"]))))))  
    v["i40"] = 0.050000*np.tanh(np.maximum(((((data["OCCUPATION_TYPE_Drivers"]) + (((data["PREV_CODE_REJECT_REASON_SCOFR_MEAN"]) + (((data["PREV_NAME_YIELD_GROUP_XNA_MEAN"]) + (((data["OCCUPATION_TYPE_Drivers"]) / 2.0))))))))), ((((np.maximum(((data["OCCUPATION_TYPE_Drivers"])), ((data["PREV_CODE_REJECT_REASON_SCOFR_MEAN"])))) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))))))  
    v["i41"] = 0.050000*np.tanh(np.where(np.minimum((((((((1.570796) / 2.0)) + (data["INSTAL_AMT_PAYMENT_MAX"]))/2.0))), ((data["INSTAL_AMT_INSTALMENT_MAX"])))>0, data["PREV_CNT_PAYMENT_MEAN"], data["BURO_CREDIT_TYPE_Microloan_MEAN"] ))  
    v["i42"] = 0.049976*np.tanh((((((((-1.0*((((data["REG_CITY_NOT_LIVE_CITY"]) + (0.318310)))))) - (data["INSTAL_PAYMENT_DIFF_MEAN"]))) + (data["INSTAL_DBD_MIN"]))) - (data["NEW_ANNUITY_TO_INCOME_RATIO"])))  
    v["i43"] = 0.050000*np.tanh((((np.tanh((1.570796))) + (np.minimum(((data["OCCUPATION_TYPE_Core_staff"])), ((((((-1.0) - (0.0))) * (0.318310)))))))/2.0))  
    v["i44"] = 0.048681*np.tanh(((np.minimum(((np.minimum(((3.0)), ((3.0))))), (((((data["CC_AMT_RECEIVABLE_PRINCIPAL_MEAN"]) < (0.0))*1.))))) + (0.318310)))  
    v["i45"] = 0.050000*np.tanh(np.maximum(((np.maximum(((data["CC_AMT_TOTAL_RECEIVABLE_MEAN"])), ((data["INSTAL_AMT_PAYMENT_MAX"]))))), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"]))))  
    v["i46"] = 0.050000*np.tanh(np.tanh((((((data["OCCUPATION_TYPE_Laborers"]) + ((((((((((data["PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN"]) + (data["BURO_STATUS_1_MEAN_MEAN"]))/2.0)) + (data["ORGANIZATION_TYPE_Self_employed"]))) * 2.0)) + (data["FLAG_WORK_PHONE"]))))) + (data["ORGANIZATION_TYPE_Self_employed"])))))  
    v["i47"] = 0.050000*np.tanh(np.where((-1.0*((((data["BURO_CREDIT_TYPE_Microloan_MEAN"]) * 2.0))))>0, 2.0, -2.0 ))  
    v["i48"] = 0.050000*np.tanh(((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]) + ((((((data["ORGANIZATION_TYPE_Construction"]) + (((np.tanh((np.tanh((0.636620))))) * ((10.0)))))/2.0)) * 2.0))))  
    v["i49"] = 0.050000*np.tanh((-1.0*(((((((np.minimum(((data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"])), ((data["APPROVED_AMT_ANNUITY_MEAN"])))) * 2.0)) + ((((data["OCCUPATION_TYPE_Drivers"]) < (np.maximum(((3.141593)), ((-1.0)))))*1.)))/2.0)))))  
    v["i50"] = 0.050000*np.tanh(np.maximum(((3.0)), ((1.0))))  
    v["i51"] = 0.050000*np.tanh(np.tanh((np.minimum((((11.29484272003173828))), ((((data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"]) * 2.0)))))))  
    v["i52"] = 0.050000*np.tanh(((((((data["INSTAL_AMT_PAYMENT_MAX"]) / 2.0)) + (((data["INSTAL_AMT_PAYMENT_MAX"]) * 2.0)))) * (data["APPROVED_CNT_PAYMENT_SUM"])))  
    v["i53"] = 0.050000*np.tanh(((np.maximum(((np.minimum(((((1.0) * 2.0))), ((((((data["INSTAL_PAYMENT_DIFF_MAX"]) * 2.0)) / 2.0)))))), ((np.minimum(((data["INSTAL_PAYMENT_DIFF_MAX"])), ((-1.0))))))) / 2.0))  
    v["i54"] = 0.050000*np.tanh((((0.0) < ((((data["APPROVED_CNT_PAYMENT_MEAN"]) < (data["OWN_CAR_AGE"]))*1.)))*1.))  
    v["i55"] = 0.050000*np.tanh(np.minimum(((((np.tanh(((((0.318310) > (-1.0))*1.)))) * (1.0)))), (((((((data["REGION_RATING_CLIENT_W_CITY"]) / 2.0)) > (((data["OWN_CAR_AGE"]) / 2.0)))*1.)))))  
    v["i56"] = 0.050000*np.tanh(np.maximum(((data["ORGANIZATION_TYPE_Transport__type_3"])), ((data["ORGANIZATION_TYPE_Transport__type_3"]))))  
    v["i57"] = 0.050000*np.tanh(((np.minimum(((((((((((data["PREV_CODE_REJECT_REASON_SCOFR_MEAN"]) + (data["PREV_CODE_REJECT_REASON_SCOFR_MEAN"]))) + (data["ACTIVE_DAYS_CREDIT_MEAN"]))) + (data["DAYS_LAST_PHONE_CHANGE"]))) + (data["ACTIVE_DAYS_CREDIT_MEAN"])))), ((data["DAYS_LAST_PHONE_CHANGE"])))) * 2.0))  
    v["i58"] = 0.050000*np.tanh(np.minimum(((1.0)), (((-1.0*(((((data["APPROVED_CNT_PAYMENT_MEAN"]) < (np.where(data["PREV_CNT_PAYMENT_SUM"]>0, data["PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN"], 1.0 )))*1.))))))))  
    v["i59"] = 0.049585*np.tanh(((((np.maximum(((-2.0)), ((((data["PREV_CNT_PAYMENT_SUM"]) * (data["APPROVED_CNT_PAYMENT_MEAN"])))))) + (data["APPROVED_CNT_PAYMENT_MEAN"]))) + (((((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) + (0.0))) * ((12.97596836090087891))))))  
    v["i60"] = 0.050000*np.tanh((((((((np.minimum(((data["INSTAL_PAYMENT_DIFF_MEAN"])), ((data["FLAG_WORK_PHONE"])))) + (data["INSTAL_PAYMENT_DIFF_MEAN"]))/2.0)) + (data["INSTAL_PAYMENT_DIFF_MEAN"]))) + (data["INSTAL_PAYMENT_DIFF_MEAN"])))  
    v["i61"] = 0.050000*np.tanh(np.where(data["BURO_CREDIT_ACTIVE_Closed_MEAN"]>0, data["NAME_FAMILY_STATUS_Married"], np.minimum(((np.minimum(((((np.minimum(((data["NAME_FAMILY_STATUS_Married"])), ((-3.0)))) + (0.0)))), ((0.636620))))), ((1.0))) ))  
    v["i62"] = 0.049951*np.tanh((((-1.0*(((((((np.tanh((np.minimum((((-1.0*((-3.0))))), ((((0.0) * (3.141593)))))))) < (0.0))*1.)) * (data["APPROVED_AMT_ANNUITY_MEAN"])))))) * 2.0))  
    v["i63"] = 0.050000*np.tanh(((((np.tanh(((5.51910448074340820)))) / 2.0)) + (np.minimum(((data["PREV_NAME_PAYMENT_TYPE_XNA_MEAN"])), ((data["PREV_CNT_PAYMENT_MEAN"]))))))  
    v["i64"] = 0.050000*np.tanh((-1.0*(((-1.0*((-3.0)))))))  
    v["i65"] = 0.050000*np.tanh(((((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) * 2.0)) + (((((((data["ACTIVE_AMT_CREDIT_SUM_SUM"]) + (((data["INSTAL_DBD_STD"]) + (((((3.0) * (data["ACTIVE_AMT_CREDIT_SUM_SUM"]))) * 2.0)))))) * 2.0)) * 2.0))))  
    v["i66"] = 0.050000*np.tanh((((-1.0*((np.tanh((data["CC_AMT_PAYMENT_CURRENT_SUM"])))))) * (np.tanh(((((((-3.0) + (((3.141593) * (3.0))))) + (data["CC_AMT_PAYMENT_TOTAL_CURRENT_SUM"]))/2.0))))))  
    v["i67"] = 0.050000*np.tanh(((np.maximum(((((data["DEF_30_CNT_SOCIAL_CIRCLE"]) + (data["PREV_CNT_PAYMENT_MEAN"])))), ((((data["DEF_60_CNT_SOCIAL_CIRCLE"]) + (((data["ORGANIZATION_TYPE_Transport__type_3"]) + (data["POS_SK_DPD_MEAN"])))))))) / 2.0))  
    v["i68"] = 0.049047*np.tanh(((-2.0) * (2.0)))  
    v["i69"] = 0.048803*np.tanh((((1.570796) + (-2.0))/2.0))  
    v["i70"] = 0.050000*np.tanh(np.maximum(((((data["ACTIVE_AMT_ANNUITY_MAX"]) + (data["ORGANIZATION_TYPE_Construction"])))), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"]))))  
    v["i71"] = 0.050000*np.tanh(((((-1.0*((-2.0)))) + ((((data["ORGANIZATION_TYPE_Military"]) + (data["ORGANIZATION_TYPE_Transport__type_3"]))/2.0)))/2.0))  
    v["i72"] = 0.049780*np.tanh((((-1.0*((data["AMT_ANNUITY"])))) * 2.0))  
    v["i73"] = 0.050000*np.tanh(np.where(2.0>0, ((((data["CODE_GENDER"]) - (0.318310))) / 2.0), ((((np.tanh((0.0))) * (data["CODE_GENDER"]))) * (2.0)) ))  
    v["i74"] = 0.050000*np.tanh(np.maximum(((data["NAME_INCOME_TYPE_Unemployed"])), ((0.0))))  
    v["i75"] = 0.0*np.tanh((((data["ACTIVE_AMT_ANNUITY_MAX"]) + (data["ACTIVE_AMT_ANNUITY_MAX"]))/2.0))  
    v["i76"] = 0.050000*np.tanh(((((data["OWN_CAR_AGE"]) - ((((((data["ACTIVE_AMT_ANNUITY_MAX"]) + ((((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]) + (data["POS_MONTHS_BALANCE_SIZE"]))/2.0)))) + (data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]))/2.0)))) - (data["PREV_NAME_YIELD_GROUP_low_action_MEAN"])))  
    v["i77"] = 0.050000*np.tanh(((0.636620) - (0.636620)))  
    v["i78"] = 0.050000*np.tanh(((data["APPROVED_CNT_PAYMENT_SUM"]) - (((1.0) - (data["APPROVED_CNT_PAYMENT_SUM"])))))  
    v["i79"] = 0.0*np.tanh(((-1.0) * ((12.60724353790283203))))  
    v["i80"] = 0.049976*np.tanh((((np.maximum(((data["POS_SK_DPD_MEAN"])), ((-3.0)))) < (data["PREV_CODE_REJECT_REASON_SCOFR_MEAN"]))*1.))  
    v["i81"] = 0.049976*np.tanh(np.tanh((np.maximum(((((data["CC_AMT_PAYMENT_CURRENT_SUM"]) * (data["ACTIVE_AMT_ANNUITY_MAX"])))), ((data["ORGANIZATION_TYPE_Military"]))))))  
    v["i82"] = 0.050000*np.tanh((((((data["CC_AMT_BALANCE_MEAN"]) < (data["CC_AMT_BALANCE_MAX"]))*1.)) + (np.maximum(((data["ACTIVE_AMT_ANNUITY_MAX"])), (((((data["CC_AMT_BALANCE_MEAN"]) > (data["CC_AMT_TOTAL_RECEIVABLE_MEAN"]))*1.)))))))  
    v["i83"] = 0.050000*np.tanh(((data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]) - (data["OWN_CAR_AGE"])))  
    v["i84"] = 0.050000*np.tanh(((data["ORGANIZATION_TYPE_Realtor"]) * (np.minimum(((data["ORGANIZATION_TYPE_Realtor"])), ((data["POS_SK_DPD_MEAN"]))))))  
    v["i85"] = 0.050000*np.tanh(((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) + (((data["AMT_ANNUITY"]) + (data["AMT_ANNUITY"])))))  
    v["i86"] = 0.049951*np.tanh((((0.318310) < (np.where(-1.0>0, data["PREV_NAME_GOODS_CATEGORY_Direct_Sales_MEAN"], (((0.636620) < (np.minimum((((((((2.0) * 2.0)) < (3.141593))*1.))), (((0.51556718349456787))))))*1.) )))*1.))  
    v["i87"] = 0.049878*np.tanh(np.minimum(((1.0)), ((((data["POS_SK_DPD_MEAN"]) - (data["ORGANIZATION_TYPE_Transport__type_3"]))))))  
    v["i88"] = 0.049902*np.tanh(np.minimum(((((data["POS_MONTHS_BALANCE_SIZE"]) - ((-1.0*((data["ORGANIZATION_TYPE_Transport__type_3"]))))))), (((8.34731864929199219)))))  
    v["i89"] = 0.050000*np.tanh(((0.318310) - (np.minimum(((2.0)), ((0.318310))))))  
    v["i90"] = 0.050000*np.tanh(((((((((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) + (data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"]))/2.0)) + (-2.0))) + (data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"]))/2.0)) + (((data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"]) + (np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), ((data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]))))))))  
    v["i91"] = 0.049951*np.tanh(np.maximum(((((((0.636620) * (data["PREV_NAME_GOODS_CATEGORY_Furniture_MEAN"]))) * (((((0.0) + ((-1.0*((-3.0)))))) - (np.minimum(((data["NAME_INCOME_TYPE_Unemployed"])), ((data["PREV_NAME_GOODS_CATEGORY_Furniture_MEAN"]))))))))), ((data["PREV_NAME_GOODS_CATEGORY_Furniture_MEAN"]))))  
    v["i92"] = 0.049951*np.tanh((((((np.tanh((3.141593))) / 2.0)) + (2.0))/2.0))  
    v["i93"] = 0.037396*np.tanh(np.where(np.maximum(((-3.0)), ((data["NAME_INCOME_TYPE_Unemployed"])))>0, ((data["POS_SK_DPD_MEAN"]) + (0.0)), 2.0 ))  
    v["i94"] = 0.049976*np.tanh(((0.318310) - (data["ORGANIZATION_TYPE_Transport__type_3"])))  
    v["i95"] = 0.049927*np.tanh(np.maximum((((2.56927442550659180))), ((-2.0))))  
    v["i96"] = 0.049976*np.tanh((((data["INSTAL_DBD_SUM"]) + (data["ORGANIZATION_TYPE_Industry__type_9"]))/2.0))  
    v["i97"] = 0.049707*np.tanh(((((data["PREV_CNT_PAYMENT_SUM"]) + ((((3.141593) < (((((data["PREV_CNT_PAYMENT_SUM"]) * 2.0)) - (((2.0) * 2.0)))))*1.)))) - (data["BURO_CREDIT_TYPE_Car_loan_MEAN"])))  
    v["i98"] = 0.050000*np.tanh(np.minimum(((0.318310)), ((0.636620))))  
    v["i99"] = 0.049976*np.tanh(np.maximum(((2.0)), ((-2.0))))  
    return v.sum(axis=1)

def GPCluster2(data):
    v = pd.DataFrame()
    v["i0"] = 0.050000*np.tanh(np.minimum(((data["BURO_CREDIT_ACTIVE_Closed_MEAN"])), ((((data["EXT_SOURCE_3"]) - (0.318310))))))  
    v["i1"] = 0.050000*np.tanh(((data["DAYS_BIRTH"]) - (data["EXT_SOURCE_3"])))  
    v["i2"] = 0.050000*np.tanh(((((np.tanh((np.tanh((data["NEW_EXT_SOURCES_MEAN"]))))) - (((((0.318310) - (data["EXT_SOURCE_1"]))) + (((data["NEW_EXT_SOURCES_MEAN"]) * 2.0)))))) - (data["NEW_EXT_SOURCES_MEAN"])))  
    v["i3"] = 0.050000*np.tanh((((((((((data["NEW_EXT_SOURCES_MEAN"]) < (((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]) - (1.0))))*1.)) - (np.tanh((0.318310))))) - (data["NEW_EXT_SOURCES_MEAN"]))) - (data["NEW_EXT_SOURCES_MEAN"])))  
    v["i4"] = 0.050000*np.tanh((((data["NEW_EXT_SOURCES_MEAN"]) < ((-1.0*((((np.minimum((((14.50290107727050781))), ((data["NEW_EXT_SOURCES_MEAN"])))) + ((((((data["NEW_EXT_SOURCES_MEAN"]) > (data["NEW_SOURCES_PROD"]))*1.)) - ((-1.0*((data["NEW_EXT_SOURCES_MEAN"]))))))))))))*1.))  
    v["i5"] = 0.050000*np.tanh(((((((data["NEW_CREDIT_TO_GOODS_RATIO"]) - (np.where((((((-1.0*(((-1.0*((data["EXT_SOURCE_2"]))))))) - (data["EXT_SOURCE_3"]))) * 2.0)>0, data["EXT_SOURCE_2"], data["EXT_SOURCE_2"] )))) - (data["EXT_SOURCE_3"]))) * 2.0))  
    v["i6"] = 0.050000*np.tanh(((data["EXT_SOURCE_2"]) - (np.maximum((((1.0))), ((data["BURO_DAYS_CREDIT_MEAN"]))))))  
    v["i7"] = 0.049976*np.tanh(((((data["PREV_CODE_REJECT_REASON_XAP_MEAN"]) + (((np.minimum(((data["BURO_CREDIT_ACTIVE_Closed_MEAN"])), ((np.tanh(((((data["BURO_CREDIT_ACTIVE_Closed_MEAN"]) + (data["BURO_CREDIT_ACTIVE_Closed_MEAN"]))/2.0))))))) / 2.0)))) + (data["EXT_SOURCE_1"])))  
    v["i8"] = 0.050000*np.tanh(((((((((data["PREV_CODE_REJECT_REASON_XAP_MEAN"]) + (data["EXT_SOURCE_2"]))/2.0)) - (data["NEW_CREDIT_TO_GOODS_RATIO"]))) + (data["EXT_SOURCE_3"]))/2.0))  
    v["i9"] = 0.050000*np.tanh((((((((data["NEW_EXT_SOURCES_MEAN"]) + (data["NEW_EXT_SOURCES_MEAN"]))/2.0)) * (np.maximum(((0.636620)), ((np.maximum(((data["NEW_EXT_SOURCES_MEAN"])), ((data["NEW_SOURCES_PROD"]))))))))) + (np.minimum(((data["NEW_EXT_SOURCES_MEAN"])), ((3.141593))))))  
    v["i10"] = 0.050000*np.tanh(((((((((data["NEW_CREDIT_TO_GOODS_RATIO"]) / 2.0)) + (data["NEW_CREDIT_TO_GOODS_RATIO"]))/2.0)) + ((((np.minimum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), ((1.0)))) + (data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))/2.0)))/2.0))  
    v["i11"] = 0.050000*np.tanh(((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]) + (((((((((data["PREV_CODE_REJECT_REASON_XAP_MEAN"]) - (0.636620))) < (data["BURO_DAYS_CREDIT_MEAN"]))*1.)) < ((((((-2.0) - (data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))) < (1.570796))*1.)))*1.))))  
    v["i12"] = 0.050000*np.tanh(np.tanh((np.tanh((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"])))))  
    v["i13"] = 0.049951*np.tanh((((((((data["EXT_SOURCE_3"]) + (data["NEW_EXT_SOURCES_MEAN"]))/2.0)) * 2.0)) + (((((data["NEW_EXT_SOURCES_MEAN"]) + (((3.141593) + (data["NEW_EXT_SOURCES_MEAN"]))))) + (data["NEW_EXT_SOURCES_MEAN"])))))  
    v["i14"] = 0.050000*np.tanh((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) + (np.maximum(((data["CC_CNT_DRAWINGS_CURRENT_MAX"])), ((((((-1.0) - ((((0.636620) < (0.636620))*1.)))) - (data["NEW_EXT_SOURCES_MEAN"])))))))/2.0))  
    v["i15"] = 0.050000*np.tanh(((np.minimum(((data["DAYS_EMPLOYED"])), ((data["NEW_EXT_SOURCES_MEAN"])))) + (((((data["NEW_CREDIT_TO_GOODS_RATIO"]) - (data["NEW_EXT_SOURCES_MEAN"]))) + ((((-1.0*((data["CODE_GENDER"])))) - (data["NEW_EXT_SOURCES_MEAN"])))))))  
    v["i16"] = 0.050000*np.tanh(((((((((np.minimum(((data["NAME_INCOME_TYPE_Working"])), ((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"])))) - (data["EXT_SOURCE_3"]))) / 2.0)) * 2.0)) - (data["EXT_SOURCE_1"])))  
    v["i17"] = 0.050000*np.tanh((((((((((data["PREV_NAME_PRODUCT_TYPE_walk_in_MEAN"]) + (((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) - (data["NEW_EXT_SOURCES_MEAN"]))) - (data["NEW_EXT_SOURCES_MEAN"]))))) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))/2.0)) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))) - (data["NAME_EDUCATION_TYPE_Higher_education"])))  
    v["i18"] = 0.050000*np.tanh(((((((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]) - ((((data["NEW_EXT_SOURCES_MEAN"]) + (3.141593))/2.0)))) - (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"]))) - (((data["EXT_SOURCE_3"]) + (data["INSTAL_PAYMENT_PERC_MIN"])))))  
    v["i19"] = 0.050000*np.tanh(((np.maximum((((((((7.41638469696044922)) * (data["CC_CNT_DRAWINGS_CURRENT_MAX"]))) + (data["PREV_NAME_PRODUCT_TYPE_walk_in_MEAN"])))), ((np.tanh((data["PREV_NAME_PRODUCT_TYPE_walk_in_MEAN"])))))) + (data["NEW_CREDIT_TO_GOODS_RATIO"])))  
    v["i20"] = 0.050000*np.tanh(((((data["INSTAL_PAYMENT_PERC_MIN"]) + (data["INSTAL_PAYMENT_PERC_MIN"]))) + (((np.minimum(((data["NEW_SOURCES_PROD"])), ((data["EXT_SOURCE_3"])))) + (data["INSTAL_PAYMENT_PERC_MIN"])))))  
    v["i21"] = 0.050000*np.tanh(np.maximum(((((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]) + (data["DAYS_EMPLOYED"])))), ((((((((np.maximum(((data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"])), ((data["CC_CNT_DRAWINGS_CURRENT_MAX"])))) + (data["CC_CNT_DRAWINGS_CURRENT_MAX"]))) * 2.0)) + (((data["CC_CNT_DRAWINGS_CURRENT_MAX"]) * 2.0)))))))  
    v["i22"] = 0.050000*np.tanh(((((data["REGION_RATING_CLIENT_W_CITY"]) - (data["CODE_GENDER"]))) - (data["NEW_EXT_SOURCES_MEAN"])))  
    v["i23"] = 0.050000*np.tanh(np.minimum((((((np.minimum(((data["NEW_EXT_SOURCES_MEAN"])), ((data["NEW_EXT_SOURCES_MEAN"])))) + (data["NEW_EXT_SOURCES_MEAN"]))/2.0))), ((data["NEW_EXT_SOURCES_MEAN"]))))  
    v["i24"] = 0.050000*np.tanh((((data["NAME_EDUCATION_TYPE_Higher_education"]) > (np.tanh((np.maximum(((1.0)), ((data["EXT_SOURCE_2"])))))))*1.))  
    v["i25"] = 0.050000*np.tanh(np.maximum(((data["PREV_CNT_PAYMENT_MEAN"])), ((((data["PREV_CNT_PAYMENT_MEAN"]) + (np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), ((data["BURO_CREDIT_TYPE_Microloan_MEAN"])))))))))  
    v["i26"] = 0.050000*np.tanh(np.maximum(((data["NEW_EXT_SOURCES_MEAN"])), (((((data["CODE_GENDER"]) + ((((((((1.0) + (-2.0))) + (data["NEW_EXT_SOURCES_MEAN"]))/2.0)) * (((data["NAME_EDUCATION_TYPE_Higher_education"]) * (data["INSTAL_PAYMENT_PERC_MIN"]))))))/2.0)))))  
    v["i27"] = 0.050000*np.tanh((((((-1.0*((data["APPROVED_RATE_DOWN_PAYMENT_MAX"])))) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (data["INSTAL_PAYMENT_PERC_MIN"]))))) + (((data["DEF_30_CNT_SOCIAL_CIRCLE"]) + ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]))/2.0))))))  
    v["i28"] = 0.050000*np.tanh((((((data["FLAG_OWN_CAR"]) + (((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) + (data["FLAG_OWN_CAR"]))))) + ((((((((((data["NEW_EMPLOY_TO_BIRTH_RATIO"]) + (data["NEW_EMPLOY_TO_BIRTH_RATIO"]))) > (data["BURO_CREDIT_TYPE_Microloan_MEAN"]))*1.)) / 2.0)) * 2.0)))/2.0))  
    v["i29"] = 0.050000*np.tanh(np.maximum(((data["EXT_SOURCE_2"])), ((0.636620))))  
    v["i30"] = 0.050000*np.tanh(((((data["APPROVED_AMT_ANNUITY_MEAN"]) + (((1.0) - ((-1.0*((((np.minimum(((data["EXT_SOURCE_3"])), ((data["INSTAL_DAYS_ENTRY_PAYMENT_STD"])))) - (data["PREV_NAME_PRODUCT_TYPE_walk_in_MEAN"])))))))))) + (data["INSTAL_DAYS_ENTRY_PAYMENT_STD"])))  
    v["i31"] = 0.050000*np.tanh(np.tanh((((data["PREV_CNT_PAYMENT_MEAN"]) * 2.0))))  
    v["i32"] = 0.050000*np.tanh(((((0.636620) - (data["REGION_RATING_CLIENT_W_CITY"]))) - ((((-1.0*(((((data["POS_COUNT"]) + (data["POS_COUNT"]))/2.0))))) - (data["POS_COUNT"])))))  
    v["i33"] = 0.050000*np.tanh(np.maximum(((((data["CC_AMT_BALANCE_MEAN"]) * 2.0))), ((data["PREV_CNT_PAYMENT_MEAN"]))))  
    v["i34"] = 0.050000*np.tanh(((((data["DEF_60_CNT_SOCIAL_CIRCLE"]) - (data["CODE_GENDER"]))) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) - (((data["CODE_GENDER"]) - (((data["BURO_CREDIT_TYPE_Microloan_MEAN"]) + (((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * 2.0)) - (data["APPROVED_AMT_ANNUITY_MEAN"])))))))))))  
    v["i35"] = 0.050000*np.tanh(((((((((data["REG_CITY_NOT_LIVE_CITY"]) + (data["DAYS_ID_PUBLISH"]))) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]))) + (((data["BURO_CREDIT_TYPE_Microloan_MEAN"]) / 2.0)))) - (data["APPROVED_AMT_ANNUITY_MEAN"])))  
    v["i36"] = 0.050000*np.tanh(((((((data["PREV_NAME_YIELD_GROUP_high_MEAN"]) + (((data["DEF_30_CNT_SOCIAL_CIRCLE"]) - (((data["PREV_APP_CREDIT_PERC_MEAN"]) - (-1.0))))))) - (data["APPROVED_HOUR_APPR_PROCESS_START_MAX"]))) + (data["APPROVED_CNT_PAYMENT_MEAN"])))  
    v["i37"] = 0.050000*np.tanh((-1.0*((((2.0) - ((((1.0) + (((((data["NEW_DOC_IND_KURT"]) * ((((-3.0) < (-1.0))*1.)))) / 2.0)))/2.0)))))))  
    v["i38"] = 0.050000*np.tanh(((((((((data["CC_AMT_BALANCE_MEAN"]) * 2.0)) - (data["FLAG_OWN_CAR"]))) - (data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]))) + (((((data["PREV_CNT_PAYMENT_MEAN"]) - (data["INSTAL_PAYMENT_PERC_MIN"]))) - (data["PREV_NAME_YIELD_GROUP_low_action_MEAN"])))))  
    v["i39"] = 0.050000*np.tanh(((data["INSTAL_PAYMENT_DIFF_MEAN"]) + (np.maximum(((data["INSTAL_PAYMENT_DIFF_MEAN"])), ((((data["INSTAL_PAYMENT_DIFF_MEAN"]) + (((data["DAYS_EMPLOYED"]) + (data["INSTAL_PAYMENT_DIFF_MEAN"]))))))))))  
    v["i40"] = 0.050000*np.tanh(((data["APPROVED_AMT_ANNUITY_MEAN"]) + (np.maximum(((((data["WALLSMATERIAL_MODE_Panel"]) + (np.maximum(((((data["CODE_GENDER"]) + (data["CODE_GENDER"])))), ((data["APPROVED_AMT_ANNUITY_MEAN"]))))))), ((data["CODE_GENDER"]))))))  
    v["i41"] = 0.050000*np.tanh(((data["NAME_FAMILY_STATUS_Married"]) - ((((0.636620) + (np.tanh((((0.636620) + (data["NAME_FAMILY_STATUS_Married"]))))))/2.0))))  
    v["i42"] = 0.049976*np.tanh(((data["INSTAL_PAYMENT_DIFF_MEAN"]) + (((np.where(data["INSTAL_PAYMENT_DIFF_MEAN"]>0, ((data["REG_CITY_NOT_LIVE_CITY"]) + (data["INSTAL_PAYMENT_DIFF_MEAN"])), ((data["APPROVED_CNT_PAYMENT_MEAN"]) + (data["FLAG_DOCUMENT_3"])) )) + (data["APPROVED_CNT_PAYMENT_MEAN"])))))  
    v["i43"] = 0.050000*np.tanh(((data["PREV_NAME_YIELD_GROUP_low_action_MEAN"]) + (((data["POS_COUNT"]) + (((((((1.23316550254821777)) + (data["OCCUPATION_TYPE_Core_staff"]))/2.0)) + (((data["APPROVED_AMT_ANNUITY_MEAN"]) + (data["INSTAL_DBD_SUM"])))))))))  
    v["i44"] = 0.048681*np.tanh(((np.maximum(((np.minimum(((data["INSTAL_AMT_INSTALMENT_MAX"])), ((2.0))))), ((((1.570796) * (data["BURO_CREDIT_TYPE_Microloan_MEAN"])))))) * (2.0)))  
    v["i45"] = 0.050000*np.tanh((((((data["APPROVED_CNT_PAYMENT_MEAN"]) * 2.0)) < ((((((1.0) * 2.0)) < (1.0))*1.)))*1.))  
    v["i46"] = 0.050000*np.tanh((((((data["PREV_NAME_YIELD_GROUP_low_normal_MEAN"]) + (0.318310))) + (((data["PREV_NAME_YIELD_GROUP_low_normal_MEAN"]) / 2.0)))/2.0))  
    v["i47"] = 0.050000*np.tanh(np.maximum(((data["DEF_30_CNT_SOCIAL_CIRCLE"])), ((((((data["NEW_CREDIT_TO_GOODS_RATIO"]) * 2.0)) + ((((((data["REGION_RATING_CLIENT_W_CITY"]) + (data["DEF_60_CNT_SOCIAL_CIRCLE"]))/2.0)) * 2.0)))))))  
    v["i48"] = 0.050000*np.tanh((((data["BURO_STATUS_1_MEAN_MEAN"]) + ((((np.maximum(((data["PREV_CODE_REJECT_REASON_HC_MEAN"])), ((((data["ORGANIZATION_TYPE_Construction"]) / 2.0))))) + (data["CC_CNT_DRAWINGS_CURRENT_MAX"]))/2.0)))/2.0))  
    v["i49"] = 0.050000*np.tanh((-1.0*((np.maximum((((7.0))), ((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])))))))  
    v["i50"] = 0.050000*np.tanh(((data["WALLSMATERIAL_MODE_Panel"]) - (data["AMT_ANNUITY"])))  
    v["i51"] = 0.050000*np.tanh(((((data["PREV_RATE_DOWN_PAYMENT_MAX"]) + (((((data["PREV_RATE_DOWN_PAYMENT_MAX"]) + (np.where(data["PREV_RATE_DOWN_PAYMENT_MAX"]>0, data["PREV_RATE_DOWN_PAYMENT_MAX"], data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"] )))) + (data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"]))))) + (((data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"]) + (data["BURO_CREDIT_TYPE_Mortgage_MEAN"])))))  
    v["i52"] = 0.050000*np.tanh(np.tanh((np.maximum(((np.maximum(((-1.0)), ((np.maximum((((((data["PREV_CNT_PAYMENT_SUM"]) < (((3.141593) - (data["INSTAL_PAYMENT_DIFF_MAX"]))))*1.))), ((0.636620)))))))), ((0.0))))))  
    v["i53"] = 0.050000*np.tanh((((-1.0*((np.maximum(((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])), ((data["INSTAL_DBD_MIN"]))))))) * ((((((-1.0*((0.0)))) / 2.0)) + (-3.0)))))  
    v["i54"] = 0.050000*np.tanh(((1.570796) - (data["APPROVED_CNT_PAYMENT_MEAN"])))  
    v["i55"] = 0.050000*np.tanh(((((((((data["OWN_CAR_AGE"]) + ((((data["REGION_RATING_CLIENT_W_CITY"]) + (data["ORGANIZATION_TYPE_Business_Entity_Type_3"]))/2.0)))) + (((((data["OWN_CAR_AGE"]) + (data["INSTAL_PAYMENT_DIFF_MEAN"]))) - (data["BURO_CREDIT_TYPE_Mortgage_MEAN"]))))) * 2.0)) + (data["PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN"])))  
    v["i56"] = 0.050000*np.tanh((((((((data["AMT_ANNUITY"]) * (((np.tanh((np.minimum(((data["OCCUPATION_TYPE_Core_staff"])), ((data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"])))))) - ((((13.42193794250488281)) - (data["AMT_ANNUITY"]))))))) > ((13.42193794250488281)))*1.)) - (data["AMT_ANNUITY"])))  
    v["i57"] = 0.050000*np.tanh(np.maximum((((((((data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]) > (np.minimum(((1.570796)), (((((data["DAYS_LAST_PHONE_CHANGE"]) + (data["NEW_PHONE_TO_BIRTH_RATIO"]))/2.0))))))*1.)) * 2.0))), ((((((data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]) / 2.0)) * 2.0)))))  
    v["i58"] = 0.050000*np.tanh((((data["PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN"]) + ((((data["ORGANIZATION_TYPE_Transport__type_3"]) + ((((((((data["PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN"]) * 2.0)) + (data["ORGANIZATION_TYPE_Transport__type_3"]))/2.0)) + (((data["APPROVED_CNT_PAYMENT_MEAN"]) * (data["APPROVED_CNT_PAYMENT_SUM"]))))))/2.0)))/2.0))  
    v["i59"] = 0.049585*np.tanh((((data["PREV_CNT_PAYMENT_MEAN"]) > ((((2.0) > (((np.minimum(((data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"])), (((((((0.0) - (1.570796))) < (0.0))*1.))))) * 2.0)))*1.)))*1.))  
    v["i60"] = 0.050000*np.tanh(np.maximum(((data["OCCUPATION_TYPE_Core_staff"])), ((((data["OCCUPATION_TYPE_Core_staff"]) + (np.maximum((((((((((((data["OCCUPATION_TYPE_Core_staff"]) > (data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]))*1.)) > (data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]))*1.)) + (data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]))/2.0))), ((data["NAME_INCOME_TYPE_State_servant"])))))))))  
    v["i61"] = 0.050000*np.tanh(((np.tanh((np.where(data["BURO_CREDIT_ACTIVE_Closed_MEAN"]>0, np.minimum(((-2.0)), ((data["BURO_CREDIT_ACTIVE_Closed_MEAN"]))), data["NAME_FAMILY_STATUS_Married"] )))) - (data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"])))  
    v["i62"] = 0.049951*np.tanh(((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]) + (((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]) + ((((((np.where(data["POS_MONTHS_BALANCE_MAX"]>0, data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"], data["POS_MONTHS_BALANCE_MAX"] )) + (data["POS_MONTHS_BALANCE_MAX"]))/2.0)) + (data["PREV_AMT_ANNUITY_MEAN"])))))))  
    v["i63"] = 0.050000*np.tanh((-1.0*(((((((0.636620) * 2.0)) + ((13.48252677917480469)))/2.0)))))  
    v["i64"] = 0.050000*np.tanh(np.maximum(((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), ((((np.minimum(((data["INSTAL_AMT_INSTALMENT_MAX"])), ((((data["ORGANIZATION_TYPE_Transport__type_3"]) + (data["ORGANIZATION_TYPE_Transport__type_3"])))))) + (((data["INSTAL_AMT_INSTALMENT_MAX"]) * (((data["INSTAL_AMT_INSTALMENT_MAX"]) * 2.0)))))))))  
    v["i65"] = 0.050000*np.tanh(np.minimum(((data["POS_SK_DPD_MEAN"])), ((np.minimum(((-1.0)), (((8.0))))))))  
    v["i66"] = 0.050000*np.tanh(((((data["PREV_CNT_PAYMENT_SUM"]) * (0.318310))) * ((((data["POS_SK_DPD_MEAN"]) > (2.0))*1.))))  
    v["i67"] = 0.050000*np.tanh(np.where(((3.141593) + ((-1.0*((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) / 2.0))))))>0, data["PREV_PRODUCT_COMBINATION_Cash_Street__low_MEAN"], ((np.maximum(((data["ORGANIZATION_TYPE_Transport__type_3"])), ((-2.0)))) / 2.0) ))  
    v["i68"] = 0.049047*np.tanh(((((data["INSTAL_DBD_SUM"]) - (np.minimum(((0.318310)), (((((2.0)) * 2.0))))))) + (-3.0)))  
    v["i69"] = 0.048803*np.tanh(((np.where(data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]>0, (5.37185430526733398), 0.318310 )) - (np.maximum(((np.maximum(((0.0)), ((3.0))))), (((2.04461622238159180)))))))  
    v["i70"] = 0.050000*np.tanh(((data["ORGANIZATION_TYPE_Military"]) - (data["ORGANIZATION_TYPE_Self_employed"])))  
    v["i71"] = 0.050000*np.tanh(((((((((data["ORGANIZATION_TYPE_Transport__type_3"]) - ((((data["FLAG_OWN_CAR"]) + (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]))/2.0)))) - (data["FLAG_OWN_CAR"]))) - ((-1.0*((0.636620)))))) - (data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])))  
    v["i72"] = 0.049780*np.tanh(np.maximum((((((-1.0*((3.0)))) + ((-1.0*((((((((4.0)) < (1.570796))*1.)) * (np.maximum(((np.tanh((1.0)))), ((1.570796)))))))))))), ((3.0))))  
    v["i73"] = 0.050000*np.tanh(np.minimum(((data["AMT_ANNUITY"])), ((((np.maximum(((-2.0)), ((1.0)))) - (np.minimum(((((data["AMT_ANNUITY"]) + (data["AMT_ANNUITY"])))), ((data["AMT_ANNUITY"])))))))))  
    v["i74"] = 0.050000*np.tanh(((((data["ORGANIZATION_TYPE_Realtor"]) - (0.0))) * (data["POS_SK_DPD_MEAN"])))  
    v["i75"] = 0.0*np.tanh((((np.minimum(((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])), ((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])))) < ((8.0)))*1.))  
    v["i76"] = 0.050000*np.tanh(np.minimum(((data["POS_MONTHS_BALANCE_SIZE"])), ((((data["POS_COUNT"]) / 2.0)))))  
    v["i77"] = 0.050000*np.tanh(((((data["ORGANIZATION_TYPE_Realtor"]) - (((((1.570796) / 2.0)) * 2.0)))) * (data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__high_MEAN"])))  
    v["i78"] = 0.050000*np.tanh(np.minimum(((data["PREV_CODE_REJECT_REASON_SCOFR_MEAN"])), ((data["NAME_INCOME_TYPE_Unemployed"]))))  
    v["i79"] = 0.0*np.tanh((((np.maximum(((1.570796)), ((0.318310)))) + (np.minimum(((((-2.0) - (-2.0)))), ((((0.318310) * ((((data["PREV_CODE_REJECT_REASON_SCOFR_MEAN"]) + (data["ACTIVE_AMT_ANNUITY_MAX"]))/2.0))))))))/2.0))  
    v["i80"] = 0.049976*np.tanh((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) < (np.maximum(((0.318310)), ((data["POS_SK_DPD_MEAN"])))))*1.))  
    v["i81"] = 0.049976*np.tanh((((((np.maximum(((data["CC_AMT_PAYMENT_CURRENT_SUM"])), (((((data["ORGANIZATION_TYPE_Military"]) + (data["ORGANIZATION_TYPE_Bank"]))/2.0))))) / 2.0)) + (data["PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN"]))/2.0))  
    v["i82"] = 0.050000*np.tanh((((11.54250621795654297)) + (data["ACTIVE_AMT_ANNUITY_MAX"])))  
    v["i83"] = 0.050000*np.tanh((-1.0*((((data["BURO_CREDIT_TYPE_Mortgage_MEAN"]) + (data["FLOORSMAX_MODE"]))))))  
    v["i84"] = 0.050000*np.tanh(np.minimum(((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])), (((((data["ORGANIZATION_TYPE_Business_Entity_Type_3"]) > (-2.0))*1.)))))  
    v["i85"] = 0.050000*np.tanh((((((np.minimum(((np.tanh((-3.0)))), ((1.0)))) + (np.minimum(((data["ORGANIZATION_TYPE_Business_Entity_Type_3"])), ((data["POS_SK_DPD_MEAN"])))))/2.0)) - (data["ORGANIZATION_TYPE_Business_Entity_Type_3"])))  
    v["i86"] = 0.049951*np.tanh((-1.0*((((data["REGION_RATING_CLIENT_W_CITY"]) - ((-1.0*((np.tanh((2.0)))))))))))  
    v["i87"] = 0.049878*np.tanh(np.maximum(((data["NAME_INCOME_TYPE_Unemployed"])), ((-3.0))))  
    v["i88"] = 0.049902*np.tanh(((((0.636620) + (((data["POS_MONTHS_BALANCE_SIZE"]) + (((((data["POS_MONTHS_BALANCE_SIZE"]) / 2.0)) + (data["POS_MONTHS_BALANCE_SIZE"]))))))) + (np.maximum(((((0.636620) + (data["POS_MONTHS_BALANCE_SIZE"])))), ((data["POS_MONTHS_BALANCE_SIZE"]))))))  
    v["i89"] = 0.050000*np.tanh(((((((data["CC_AMT_BALANCE_MEAN"]) + (np.maximum(((data["INSTAL_DBD_STD"])), ((data["INSTAL_DBD_STD"])))))) + (data["PREV_CNT_PAYMENT_SUM"]))) + (data["INSTAL_DBD_STD"])))  
    v["i90"] = 0.050000*np.tanh((-1.0*((np.minimum((((0.0))), ((((data["CC_AMT_DRAWINGS_POS_CURRENT_MAX"]) * ((((((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) > (np.tanh((0.0))))*1.)) + (((2.0) + (0.0)))))))))))))  
    v["i91"] = 0.049951*np.tanh((-1.0*((np.tanh((np.minimum(((data["ACTIVE_AMT_ANNUITY_MAX"])), ((((1.570796) * (data["NAME_INCOME_TYPE_Unemployed"])))))))))))  
    v["i92"] = 0.049951*np.tanh((((np.where((5.0)>0, data["APPROVED_CNT_PAYMENT_SUM"], 3.0 )) > ((((data["ACTIVE_AMT_ANNUITY_MEAN"]) + (((1.570796) * 2.0)))/2.0)))*1.))  
    v["i93"] = 0.037396*np.tanh(np.maximum((((((data["ORGANIZATION_TYPE_Construction"]) + (data["ACTIVE_AMT_ANNUITY_MEAN"]))/2.0))), ((data["ACTIVE_AMT_ANNUITY_MEAN"]))))  
    v["i94"] = 0.049976*np.tanh((((data["ORGANIZATION_TYPE_Construction"]) > (data["ACTIVE_AMT_ANNUITY_MEAN"]))*1.))  
    v["i95"] = 0.049927*np.tanh(((data["CC_AMT_BALANCE_MEAN"]) - (data["ORGANIZATION_TYPE_Realtor"])))  
    v["i96"] = 0.049976*np.tanh(((((((data["INSTAL_DBD_SUM"]) < ((8.0)))*1.)) > (1.0))*1.))  
    v["i97"] = 0.049707*np.tanh((((np.tanh((np.minimum(((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])), ((data["BURO_CREDIT_TYPE_Car_loan_MEAN"])))))) + (data["BURO_CREDIT_TYPE_Mortgage_MEAN"]))/2.0))  
    v["i98"] = 0.050000*np.tanh(((data["PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN"]) * (0.318310)))  
    v["i99"] = 0.049976*np.tanh(np.minimum(((data["APPROVED_CNT_PAYMENT_SUM"])), ((3.141593))))  
    return v.sum(axis=1)   
pos = train_df[train_df.TARGET==1]
neg = train_df[train_df.TARGET==0][::12]#Makes it more balanced between negative and positive otherwise we will get swamped by the negatives
plt.figure(figsize=(15,15))
plt.scatter(GPCluster1(pos),GPCluster2(pos),s=30, alpha=.5)
plt.scatter(GPCluster1(neg),GPCluster2(neg),s=30, alpha=.5)
plt.show()
traincluster = pd.DataFrame({'x':GPCluster1(train_df),'y':GPCluster2(train_df)})
traincluster.to_csv('traincluster.csv',index=False)
testcluster = pd.DataFrame({'x':GPCluster1(test_df),'y':GPCluster2(test_df)})
testcluster.to_csv('testcluster.csv',index=False)
x1 = GPCluster1(train_df).values
x2 = GPCluster2(train_df).values
gptraindata = np.hstack([x1.reshape(-1,1),x2.reshape(-1,1)])
folds = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(gptraindata)):
    trn_x, trn_y = gptraindata[trn_idx], train_df.TARGET.values[trn_idx]
    val_x, val_y = gptraindata[val_idx], train_df.TARGET.values[val_idx]
    clf = KNeighborsClassifier(n_neighbors=100)
    clf.fit(trn_x,trn_y)
    score = roc_auc_score(val_y,clf.predict_proba(val_x)[:,1])
    print('Fold:', n_fold,score)
    scores.append(score)
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()
print('Mean Score:',np.mean(scores))
print('Std Score:',np.std(scores))