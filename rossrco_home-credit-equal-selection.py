#numeric
import numpy as np
import pandas as pd

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
import folium

from IPython.display import display

plt.style.use('bmh')
%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['axes.titlepad'] = 25
sns.set_color_codes('pastel')

#Pandas warnings
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 100)

#system
import os
import gc
print(os.listdir('../input'))
int_types = [np.int64, np.int32, np.int16, np.int8]
float_types = [np.float64, np.float32, np.float16]

def reduce_int(columns, frame, dtypes):
    for c in frame[columns].columns:
        for i_type in dtypes:
            if frame[c].max() < np.iinfo(i_type).max and frame[c].min() > np.iinfo(i_type).min:
                prev = i_type
            else:
                frame[c] = frame[c].astype(prev)

def reduce_float(columns, frame, dtypes):
    for c in frame[columns].columns:
        for i_type in dtypes:
            if frame[c].max() < np.finfo(i_type).max and frame[c].min() > np.finfo(i_type).min:
                prev = i_type
            else:
                frame[c] = frame[c].astype(prev)                

def reduce_categorical(columns, frame):
    for c in frame[columns].columns:
        frame[c] = frame[c].astype('category')
        gc.collect()

def reduce_columns(columns, frame, dcat):
    if dcat == 'integer':
        reduce_int(columns, frame, int_types)
    if dcat == 'float':
        reduce_float(columns, frame, float_types)
    if dcat == 'category':
        reduce_categorical(columns, frame)

def reduce_columns_a(columns, frame):
    for c in frame[columns].columns:
        if frame[c].dtypes in int_types:
            reduce_int([c], frame, int_types)
        if frame[c].dtypes in float_types:
            reduce_float([c], frame, float_types)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn import tree
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
app_train =  pd.read_csv('../input/application_train.csv', index_col = 0)
app_train_s_1 = app_train[app_train.TARGET == 1]
app_train_s_0 = app_train[app_train.TARGET == 0].sample(len(app_train_s_1))
app_train = app_train_s_1.append(app_train_s_0)
gc.collect()
app_train.info()
Y = app_train[['TARGET']]
reduce_columns_a(Y.columns, Y)
app_train.drop('TARGET', axis = 1, inplace = True)
Y.head()
app_train.head()
all_cat_col = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
               'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
               'NAME_HOUSING_TYPE', 'ORGANIZATION_TYPE', 'OCCUPATION_TYPE',
               'WEEKDAY_APPR_PROCESS_START', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',
               'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
for c in all_cat_col:
    col_dummies = pd.get_dummies(app_train[c], drop_first = True, prefix = '%s_' % c)
    col_dummies.fillna(0, inplace = True)
    col_dummies = col_dummies.astype(np.int8)
    #reduce_columns(col_dummies.columns, col_dummies, 'integer')
    app_train = app_train.join(col_dummies)
    app_train.drop([c], axis = 1, inplace = True)
app_train.head()
imp = SimpleImputer(strategy = 'median')
app_train = pd.DataFrame(imp.fit_transform(app_train), columns = app_train.columns, index = app_train.index)
app_train['DAYS_EMPLOYED'] = app_train['DAYS_EMPLOYED'].replace({365243 : 0})
app_train['family_inc_ratio'] = app_train['AMT_INCOME_TOTAL'] / app_train['CNT_FAM_MEMBERS']
app_train['prc_employment_lifetime'] = app_train['DAYS_EMPLOYED'] / app_train['DAYS_BIRTH']
app_train['debt_income_ratio'] = app_train['AMT_INCOME_TOTAL'] / app_train['AMT_CREDIT']
app_train['financing_ratio'] = app_train['AMT_GOODS_PRICE'] / app_train['AMT_CREDIT']
app_train['annuity_pct_credit'] = app_train['AMT_ANNUITY'] / app_train['AMT_CREDIT']
app_train['id_change_to_registration'] = np.vectorize(lambda a, b : a / b if b != 0 else 0)(app_train['DAYS_ID_PUBLISH'].values, app_train['DAYS_REGISTRATION'].values)
app_train['pct_phone_lifetime'] = np.vectorize(lambda a, b : a / b if b != 0 else 0)(app_train['DAYS_LAST_PHONE_CHANGE'].values, app_train['DAYS_EMPLOYED'].values)
app_train['mean_ext_score'] = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].fillna(0).mean(axis = 1)
app_train.head()
prev_app_inst = pd.read_csv('../input/installments_payments.csv', index_col = 0)
prev_app_inst = prev_app_inst[prev_app_inst.SK_ID_CURR.isin(app_train.index)]
prev_app_inst.fillna(0, inplace = True)
prev_app_inst.head()
reduce_columns(['SK_ID_CURR', 'NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT'], prev_app_inst, 'integer')
reduce_columns(['DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT'], prev_app_inst, 'float')
prev_app_installment_agg = pd.DataFrame(index = prev_app_inst.SK_ID_CURR.unique())
num_payments = pd.pivot_table(data = prev_app_inst, index = 'SK_ID_CURR', values = 'NUM_INSTALMENT_NUMBER', aggfunc = 'count', fill_value = 0)
num_payments.rename(columns = {'NUM_INSTALMENT_NUMBER' : 'bureau_inst_num_hist_loan_periods'}, inplace = True)
prev_app_installment_agg = prev_app_installment_agg.join(num_payments)
prev_app_installment_agg.head()
prev_app_inst['payment_delay'] = prev_app_inst['DAYS_ENTRY_PAYMENT'] - prev_app_inst['DAYS_INSTALMENT']
prev_app_inst['paid_pct_inst'] = np.vectorize(lambda a, b : a / b if b != 0 else 0)(prev_app_inst['AMT_PAYMENT'].values, prev_app_inst['AMT_INSTALMENT'].values)
for m in ['mean', 'max', 'min', 'sum']:
    for p in ['ly', 'lifetime']:
        if p == 'ly':
            frame = pd.pivot_table(data = prev_app_inst[prev_app_inst.DAYS_INSTALMENT < 365],
                                   index = 'SK_ID_CURR',
                                   values = ['payment_delay', 'paid_pct_inst'],
                                   aggfunc = m,
                                   fill_value = 0)
        else:
            frame = pd.pivot_table(data = prev_app_inst,
                                   index = 'SK_ID_CURR',
                                   values = ['payment_delay', 'paid_pct_inst'],
                                   aggfunc = m,
                                   fill_value = 0)
        
        frame.rename(columns = {'paid_pct_inst' : 'bureau_inst_%s_paid_pct_inst_%s' % (m, p),
                                'payment_delay' : 'bureau_inst_%s_payment_delay_%s' % (m, p)},
                     inplace = True)
        prev_app_installment_agg = prev_app_installment_agg.join(frame)
prev_app_installment_agg.head()
app_train = app_train.join(prev_app_installment_agg, how = 'left')
del prev_app_installment_agg, frame, num_payments, prev_app_inst
gc.collect()
prev_app_card_bal = pd.read_csv('../input/credit_card_balance.csv', index_col = 0)
prev_app_card_bal = prev_app_card_bal[prev_app_card_bal.SK_ID_CURR.isin(app_train.index)]
prev_app_card_bal.fillna(0, inplace = True)
reduce_columns(['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'CNT_DRAWINGS_CURRENT', 'SK_DPD', 'SK_DPD_DEF'], prev_app_card_bal, 'integer')
reduce_columns(['AMT_BALANCE', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
                'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT',
                'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE',
                'CNT_DRAWINGS_ATM_CURRENT'],
               prev_app_card_bal,
               'float')
num_payments = pd.pivot_table(data = prev_app_card_bal, index = 'SK_ID_CURR', values = 'MONTHS_BALANCE', aggfunc = 'count', fill_value = 0)
num_payments.rename(columns = {'MONTHS_BALANCE' : 'bureau_card_num_hist_card_periods'}, inplace = True)
prev_app_card_agg = pd.DataFrame(index = prev_app_card_bal.SK_ID_CURR.unique())
prev_app_card_agg = prev_app_card_agg.join(num_payments)
prev_app_card_agg.head()
prev_app_card_bal['int_gen_bal_pct_limit'] = np.vectorize(lambda a, b : a / b if b != 0 else 0)(prev_app_card_bal['AMT_BALANCE'].values, prev_app_card_bal['AMT_CREDIT_LIMIT_ACTUAL'].values)
prev_app_card_bal['limit_utilization'] = np.vectorize(lambda a, b : a / b if b != 0 else 0)(prev_app_card_bal['AMT_DRAWINGS_CURRENT'].values, prev_app_card_bal['AMT_CREDIT_LIMIT_ACTUAL'].values)
prev_app_card_bal['payment_pct_minimum'] = np.vectorize(lambda a, b : a / b if b != 0 else 0)(prev_app_card_bal['AMT_PAYMENT_TOTAL_CURRENT'].values, prev_app_card_bal['AMT_INST_MIN_REGULARITY'].values)
prev_app_card_bal['principal_pct_receivable'] = np.vectorize(lambda a, b : a / b if b != 0 else 0)(prev_app_card_bal['AMT_RECEIVABLE_PRINCIPAL'].values, prev_app_card_bal['AMT_TOTAL_RECEIVABLE'].values)
prev_app_card_bal.head()
'''card_agg_value_cols = ['AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_CURRENT', 'AMT_INST_MIN_REGULARITY',
                       'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_TOTAL_RECEIVABLE', 'AMT_RECEIVABLE_PRINCIPAL',
                       'CNT_DRAWINGS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM', 'SK_DPD', 'SK_DPD_DEF',
                       'int_gen_bal_pct_limit', 'limit_utilization', 'payment_pct_minimum', 'principal_pct_receivable']'''

card_agg_value_cols = ['int_gen_bal_pct_limit', 'limit_utilization', 'payment_pct_minimum', 'principal_pct_receivable']
for m in ['mean', 'max', 'min', 'sum']:
    for p in ['ly', 'lifetime']:
        if p == 'ly':
            frame = pd.pivot_table(data = prev_app_card_bal[(prev_app_card_bal.NAME_CONTRACT_STATUS == 'Active') &
                                                            (prev_app_card_bal.MONTHS_BALANCE >= -12)],
                                   index = 'SK_ID_CURR',
                                   values = card_agg_value_cols,
                                   aggfunc = m,
                                   fill_value = 0)
        else:
            frame = pd.pivot_table(data = prev_app_card_bal[prev_app_card_bal.NAME_CONTRACT_STATUS == 'Active'],
                                   index = 'SK_ID_CURR',
                                   values = card_agg_value_cols,
                                   aggfunc = m,
                                   fill_value = 0)
        
        for c in card_agg_value_cols:
            frame.rename(columns = {c : 'bureau_card_%s_%s_%s' % (m, c, p)}, inplace = True)
        prev_app_card_agg = prev_app_card_agg.join(frame)
prev_app_card_agg.head()
app_train = app_train.join(prev_app_card_agg, how = 'left')
del prev_app_card_agg, frame, num_payments, prev_app_card_bal
gc.collect()
bureau = pd.read_csv('../input/bureau.csv', index_col = 0)
bureau = bureau[bureau.index.isin(app_train.index)]

common_loan_types1 = ['Credit card', 'Consumer credit', 'Microloan']
common_loan_types2 = ['Credit card', 'Consumer credit', 'Car loan', 'Mortgage', 'Microloan']
common_loan_statuses = ['Active', 'Closed']
bureau = bureau[(bureau.CREDIT_TYPE.isin(common_loan_types1)) & (bureau.CREDIT_ACTIVE.isin(common_loan_statuses))]

bureau.fillna(0, inplace = True)
bureau.drop('CREDIT_CURRENCY', axis = 1, inplace = True)
bureau.head()
reduce_columns(['SK_ID_BUREAU', 'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'CNT_CREDIT_PROLONG', 'DAYS_CREDIT_UPDATE'],
               bureau,
               'integer')
reduce_columns(['DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM',
                'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY'],
               bureau,
               'float')
reduce_columns(['CREDIT_ACTIVE', 'CREDIT_TYPE'], bureau, 'category')
#bureau_bal = pd.read_csv('../input/bureau_balance.csv', index_col = 0)
#bureau_bal = bureau_bal[bureau_bal.index.isin(bureau.SK_ID_BUREAU)]
#reduce_columns(['MONTHS_BALANCE'], bureau_bal, 'integer')
#reduce_columns(['STATUS'], bureau_bal, 'category')
#bureau_bal.info()
#bureau_bal.head()
#bureau_bal_agg = pd.DataFrame(index = bureau_bal.index.unique())
#bureau_bal_pivot = pd.pivot_table(data = bureau_bal, index = bureau_bal.index, columns = 'STATUS', values = 'MONTHS_BALANCE', aggfunc = 'min', fill_value = 0)
#bba_months_columns = []
#
#for c in bureau_bal_pivot.columns:
#    bureau_bal_agg = bureau_bal_agg.join(bureau_bal_pivot[c])
#    c_name = 'months_in_status_%s' % c
#    bureau_bal_agg.rename(columns = {'%s' % c : c_name}, inplace = True)
#    bba_months_columns.append(c_name)
#reduce_columns(bba_months_columns, bureau_bal_agg, 'integer')
#bureau = bureau.merge(bureau_bal_agg, how = 'left', left_on = 'SK_ID_BUREAU', right_index = True)
#del bureau_bal_agg, bureau_bal_pivot, bureau_bal
#gc.collect()
bureau.head()
bureau_agg = pd.DataFrame(index = bureau.index.unique())
value_cols = ['AMT_CREDIT_SUM', 'AMT_ANNUITY']

for m in ['mean', 'sum']:
    frame = pd.pivot_table(data = bureau,
                               index = bureau.index,
                               columns = ['CREDIT_ACTIVE'],
                               values = value_cols,
                               aggfunc = m,
                               fill_value = 0)
    
    for (c_0, c_1) in frame.columns:
        bureau_agg = bureau_agg.join(frame[(c_0, c_1)]).rename(columns = {(c_0, c_1) : 'bureau_data_%s_%s_%s' % ((c_0, c_1, m))})
value_cols = ['DAYS_CREDIT']

for m in ['max', 'min']:
    frame = pd.pivot_table(data = bureau,
                               index = bureau.index,
                               columns = ['CREDIT_ACTIVE'],
                               values = value_cols,
                               aggfunc = m,
                               fill_value = 0)
    
    for (c_0, c_1) in frame.columns:
        bureau_agg = bureau_agg.join(frame[(c_0, c_1)]).rename(columns = {(c_0, c_1) : 'bureau_data_%s_%s_%s' % ((c_0, c_1, m))})
app_train = app_train.join(bureau_agg, how = 'left')
del bureau, bureau_agg, frame
gc.collect()
app_train.head()
pos_bal = pd.read_csv('../input/POS_CASH_balance.csv', index_col = 0)
pos_bal = pos_bal[pos_bal.SK_ID_CURR.isin(app_train.index)]
pos_bal.fillna(0, inplace = True)
reduce_columns(['SK_ID_CURR', 'MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF', 'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE'], pos_bal, 'integer')
reduce_columns(['NAME_CONTRACT_STATUS'], pos_bal, 'category')
pos_bal_agg = pd.DataFrame(index = pos_bal.SK_ID_CURR.unique())
for m in ['max', 'sum']:
    for p in ['ly', 'lifetime']:
        if p == 'ly':
            frame = pd.pivot_table(data = pos_bal[pos_bal.MONTHS_BALANCE > -12],
                                   index = pos_bal[pos_bal.MONTHS_BALANCE > -12].SK_ID_CURR,
                                   columns = 'NAME_CONTRACT_STATUS',
                                   values = ['MONTHS_BALANCE', 'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE', 'SK_DPD', 'SK_DPD_DEF'],
                                   aggfunc = m)
        else:
            frame = pd.pivot_table(data = pos_bal,
                                   index = pos_bal.SK_ID_CURR,
                                   columns = 'NAME_CONTRACT_STATUS',
                                   values = ['MONTHS_BALANCE', 'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE', 'SK_DPD', 'SK_DPD_DEF'],
                                   aggfunc = m)
        
        len_all_apps = len(frame)

        for (c_0, c_1) in frame.columns:
            if len(frame[frame[(c_0, c_1)].isnull()]) / len_all_apps > 0.3:
                frame.drop((c_0, c_1), axis = 1, inplace = True)
        
        for c in card_agg_value_cols:
            frame.rename(columns = {c : 'bureau_card_%s_%s_%s' % (m, c, p)}, inplace = True)
        
        frame.fillna(0, inplace = True)
        
        for (c_0, c_1) in frame:
            pos_bal_agg = pos_bal_agg.join(frame[(c_0, c_1)]).rename(columns = {(c_0, c_1) : 'pos_bal_%s_%s_%s_%s' % (c_0, c_1, m, p)})
app_train = app_train.join(pos_bal_agg, how = 'left')
del pos_bal, pos_bal_agg, frame
gc.collect()
prev_app = pd.read_csv('../input/previous_application.csv', index_col = 0)
prev_app = prev_app[prev_app.SK_ID_CURR.isin(app_train.index)]
prev_app = prev_app[prev_app.NAME_CONTRACT_TYPE.isin(['Cash loans', 'Consumer loans', 'Revolving loans'])]
prev_app.fillna(0, inplace = True)
reduce_columns(['SK_ID_CURR', 'AMT_ANNUITY', 'AMT_DOWN_PAYMENT',
                'AMT_GOODS_PRICE', 'HOUR_APPR_PROCESS_START', 'NFLAG_LAST_APPL_IN_DAY',
                'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY',
                'RATE_INTEREST_PRIVILEGED', 'DAYS_DECISION', 'SELLERPLACE_AREA',
                'CNT_PAYMENT', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
                'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION',
                'NFLAG_INSURED_ON_APPROVAL'],
               prev_app,
               'integer')
reduce_columns(['AMT_APPLICATION', 'AMT_CREDIT'], prev_app, 'float')
reduce_columns(['NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT',
                'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE',
                'CODE_REJECT_REASON', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE',
                'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE',
                'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION'],
               prev_app,
               'category')
prev_app['granted_pct_requested'] = np.vectorize(lambda a, b : a / b if b != 0 else 0)(prev_app['AMT_CREDIT'].values, prev_app['AMT_APPLICATION'].values)
prev_app_agg = pd.DataFrame(index = prev_app.SK_ID_CURR.unique())
for p in ['ly', 'lifetime']:
    if p == 'ly':
        frame = pd.pivot_table(data = prev_app[prev_app.DAYS_FIRST_DUE > -365],
                                   index = 'SK_ID_CURR',
                                   #columns = ['NAME_CONTRACT_TYPE', 'NAME_CONTRACT_STATUS'],
                                   values = 'AMT_CREDIT',
                                   aggfunc = 'count',
                                   fill_value = 0)
    else:
        frame = pd.pivot_table(data = prev_app,
                                   index = 'SK_ID_CURR',
                                   #columns = ['NAME_CONTRACT_TYPE', 'NAME_CONTRACT_STATUS'],
                                   values = 'AMT_CREDIT',
                                   aggfunc = 'count',
                                   fill_value = 0)
    
    for c in frame.columns:
            prev_app_agg = prev_app_agg.join(frame[c]).rename(columns = {c : 'prev_app_count_%s_%s' % (c, p)})
    
    #for (c_0, c_1) in frame:
    #        prev_app_agg = prev_app_agg.join(frame[(c_0, c_1)]).rename(columns = {(c_0, c_1) : 'prev_app_count_%s_%s_%s' % (c_0, c_1, p)})
for m in ['sum', 'mean', 'max']:
    for p in ['ly', 'lifetime']:
        if p == 'ly':
            frame = pd.pivot_table(data = prev_app[(prev_app.NAME_CONTRACT_STATUS == 'Approved') &
                                                   (prev_app.DAYS_FIRST_DUE > -365)],
                                   index = 'SK_ID_CURR',
                                   #columns = 'NAME_CONTRACT_TYPE',
                                   values = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE', 'HOUR_APPR_PROCESS_START', 'RATE_DOWN_PAYMENT', 'DAYS_DECISION', 'CNT_PAYMENT', 'granted_pct_requested'],
                                   aggfunc = 'sum',
                                   fill_value = 0)
        else:
            frame = pd.pivot_table(data = prev_app[(prev_app.NAME_CONTRACT_STATUS == 'Approved') &
                                                   (prev_app.DAYS_FIRST_DUE > -365)],
                                   index = 'SK_ID_CURR',
                                   #columns = 'NAME_CONTRACT_TYPE',
                                   values = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE', 'HOUR_APPR_PROCESS_START', 'RATE_DOWN_PAYMENT', 'DAYS_DECISION', 'CNT_PAYMENT', 'granted_pct_requested'],
                                   aggfunc = 'sum',
                                   fill_value = 0)
        
        for c in frame.columns:
            prev_app_agg = prev_app_agg.join(frame[c]).rename(columns = {c : 'prev_app_approved_%s_%s_%s' % (c, m, p)})
        
        #for (c_0, c_1) in frame:
        #    prev_app_agg = prev_app_agg.join(frame[(c_0, c_1)]).rename(columns = {(c_0, c_1) : 'prev_app_%s_%s_%s_%s' % (c_0, c_1, m, p)})
app_train = app_train.join(prev_app_agg, how = 'left')
del prev_app, prev_app_agg, frame
gc.collect()
app_train.fillna(0, inplace = True)
imp = SimpleImputer(strategy = 'median')

for c in app_train.columns:
    app_train[c] = imp.fit_transform(app_train[c].values.reshape(-1, 1))

#app_train = pd.DataFrame(imp.fit_transform(app_train), columns = app_train.columns, index = app_train.index)
app_train['bureau_data_active_pct_closed_credit'] = np.vectorize(lambda a, b : a / b if b != 0 else 0)(app_train['bureau_data_AMT_CREDIT_SUM_Active_sum'].values, app_train['bureau_data_AMT_CREDIT_SUM_Closed_sum'].values)
app_train['bureau_data_active_bureau_credit_pct_current_credit'] = np.vectorize(lambda a, b : a / b if b != 0 else 0)(app_train['bureau_data_AMT_CREDIT_SUM_Active_sum'].values, app_train['AMT_CREDIT'].values)
app_train['bureau_data_active_bureau_credit_pct_income'] = np.vectorize(lambda a, b : a / b if b != 0 else 0)(app_train['bureau_data_AMT_CREDIT_SUM_Active_sum'].values, app_train['AMT_INCOME_TOTAL'].values)
app_train['bureau_data_first_closed_credit_pct_age'] = np.vectorize(lambda a, b : a / b if b != 0 else 0)(app_train['bureau_data_DAYS_CREDIT_Closed_min'].values, app_train['DAYS_BIRTH'].values)
app_train['bureau_data_days_between_low_high_closed_credit'] = app_train['bureau_data_DAYS_CREDIT_Closed_min'] - app_train['bureau_data_DAYS_CREDIT_Closed_max']
app_train['bureau_data_days_between_low_high_active_credit'] = app_train['bureau_data_DAYS_CREDIT_Active_min'] - app_train['bureau_data_DAYS_CREDIT_Active_max']
rob_scaler = RobustScaler()
scaler = MinMaxScaler((0, 1))
for c in app_train.columns:
    app_train[c] = rob_scaler.fit_transform(app_train[c].values.reshape(-1, 1))
    #app_train[c] = normalize(app_train[c].values.reshape(-1, 1))
    app_train[c] = scaler.fit_transform(app_train[c].values.reshape(-1, 1))

#app_train = pd.DataFrame(scaler.fit_transform(app_train), columns = app_train.columns, index = app_train.index)
app_train.head()
X_train, X_valid, y_train, y_valid = train_test_split(app_train, Y.TARGET, test_size = 0.2, random_state = 42)

X_train.head()
models = {'ada_boost' : AdaBoostClassifier(),
          'logisitc_regression_l1' : LogisticRegression(penalty = 'l1'),
          'logisitc_regression_l2' : LogisticRegression(penalty = 'l2'),
          'lgbm' : LGBMClassifier(),
          'xgboost' : XGBClassifier(),
          'decision_tree_clf' : tree.DecisionTreeClassifier(),
          'naive_bayes' : GaussianNB(),
          'random_forest' : RandomForestClassifier()}

'''models = {'lgbm1' : LGBMClassifier(),
          'lgbm2' : LGBMClassifier(learning_rate = 0.085),
          'lgbm3' : LGBMClassifier(random_state = 42, learning_rate = 0.085)}'''

#valid_res = pd.DataFrame(columns = ['classifier', 'area_under_curve', 'roc_curve'])
valid_res = pd.DataFrame(columns = ['classifier', 'area_under_curve'])
for m_name, m in models.items():
    clf = m
    clf.fit(X_train, y_train)
    to_append = {}
    to_append['classifier'] = m_name
    res_p = clf.predict_proba(X_valid)
    to_append['area_under_curve'] = roc_auc_score(y_valid, res_p[:, 1])
    valid_res = valid_res.append(to_append, ignore_index = True)
    
    print('%s done!' % m_name)
valid_res
n_best = 10
n_folds = 4
splitter = KFold(n_splits= n_folds, shuffle = True, random_state = 42)
feature_importances = {}

for i, (train_ind, valid_ind) in zip(range(n_folds), splitter.split(app_train, Y)):
    X_train, y_train = app_train.iloc[train_ind], Y.iloc[train_ind].TARGET
    X_valid, y_valid = app_train.iloc[valid_ind], Y.iloc[valid_ind].TARGET
    
    clf = LGBMClassifier(random_state = 42, learning_rate = 0.085)
    clf.fit(X_train, y_train)
    res_p = clf.predict_proba(X_valid)
    auc = roc_auc_score(y_valid, res_p[:, 1])
    print('The AUC result for fold {} is: {:.3f}'.format(i, auc))
    ffi = pd.DataFrame({'feature_importance' : clf.feature_importances_}, index = X_train.columns)
    feature_importances['fold_%s' % i] = ffi
    
    del clf, X_train, y_train, X_valid, y_valid
    gc.collect()
for f, fi in feature_importances.items():
    print(f)
    display(fi.sort_values(by = 'feature_importance', ascending = False)[:n_best])
X_train, X_valid, y_train, y_valid = train_test_split(app_train, Y.TARGET, test_size = 0.2, random_state = 42)

param = {'learning_rate' : np.arange(0.001, 0.1, 0.005)}

clf = GridSearchCV(estimator = LGBMClassifier(random_state = 42),
                   param_grid = param,
                   cv = KFold(n_splits = 3),
                   scoring = make_scorer(score_func = roc_auc_score,
                                         greater_is_better = True,
                                         needs_proba = True),#,
                                         #needs_threshold = True),
                   #n_jobs = -1,
                   verbose = 1)
clf.fit(X_train, y_train)
clf.best_params_
clf.best_score_
feature_importances = pd.DataFrame({'feature_importance' : clf.best_estimator_.feature_importances_}, index = X_train.columns)
feature_importances.sort_values(by = 'feature_importance', ascending = False)[:n_best]
