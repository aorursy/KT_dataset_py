%matplotlib inline



import os

import numpy as np

import pandas as pd

import time

import gc



import matplotlib.pyplot as plt

import seaborn as sns



import cufflinks as cf

cf.go_offline()



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
print(os.listdir("../input"))
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024 ** 2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        col_type = df[col].dtype



        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024 ** 2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
start_time = time.time()

train = pd.read_csv('../input/application_train.csv')

test = pd.read_csv('../input/application_test.csv')



POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')

bureau_balance = pd.read_csv('../input/bureau_balance.csv')

previous_application = pd.read_csv('../input/previous_application.csv')

installments_payments = pd.read_csv('../input/installments_payments.csv')

credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')

bureau = pd.read_csv('../input/bureau.csv')



print('all data loaded in {:4f} sec'.format(time.time()-start_time))
def replace_to_nan(df):

    df = df[df['CODE_GENDER'] != 'XNA']

    df.replace(to_replace={'XNA': np.nan, 'XAP': np.nan}, value=None, inplace=True)

    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    return df
train = replace_to_nan(train)

test = replace_to_nan(test)
print('train data shape ', train.shape)

print('test data shape ', test.shape)
print('column type')

train.dtypes.value_counts()
TARGET = 'TARGET' 

ID = 'SK_ID_CURR'

train[TARGET].hist()

train[TARGET].value_counts()
for col in train.select_dtypes(['object', 'category']).columns:

    min_op = set(train[col]) - set(test[col])

    if len(min_op) > 0:

        print('{} have some row that not in test table, will treat'.format(col), min_op,'as nan')

        for not_in_test in min_op:

            train[col].replace(not_in_test, np.nan, inplace=True)

    min_op = set(test[col]) - set(train[col])

    if len(min_op) > 0:

        print('{} have some row that not in train table, will treat'.format(col), min_op,'as nan')

        for not_in_train in min_op:

            test[col].replace(not_in_train, np.nan, inplace=True)
# this function is used to get the dataframe containing column with missing value with its total and percentage

def missing_col(df):

    total = df.isnull().sum().sort_values(ascending = False)

    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)

    missing_col  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_col[missing_col['Total'] > 0]
missing_col(train)
missing_col(POS_CASH_balance)
missing_col(bureau_balance)
missing_col(previous_application)
missing_col(installments_payments)
missing_col(credit_card_balance)
missing_col(bureau)
# plot distribution target=1 vs target=0

def plot_dist(col, train=train, target=TARGET):

    plt.figure(figsize=(12,8))

    target_0 = train[train[target]==0].dropna()

    target_1 = train[train[target]==1].dropna()

    sns.distplot(target_0[col].values, label='target: 0')

    sns.distplot(target_1[col].values, color='red', label='target: 1')

    plt.xlabel(col)

    plt.legend()

    plt.show()



# plot value vs its index and TARGET

def plot_val_vs_idx(col, train=train, target=TARGET):

    plt.figure(figsize=(12,8))

    plt.scatter(range(train.shape[0]), train[col], c=train[target], cmap='viridis')

    plt.ylabel(col)

    plt.xlabel('index')

    plt.colorbar()

    plt.show()



def pie_i_plot(col, train=train, hole=0.5, title=None):

    if title is None:

        title = col

    temp = train[col].value_counts()

    df = pd.DataFrame({'labels': temp.index, 'values': temp.values})

    df.iplot(kind='pie',labels='labels',values='values', title=title, hole = 0.5)
train.drop([ID, TARGET], axis=1).describe()
print('see their age')

(train['DAYS_BIRTH'] /-365).describe()
train[train['DAYS_EMPLOYED'] < train['DAYS_BIRTH']]
(train['DAYS_REGISTRATION'] /-365).describe()
print('check if there are some mistake on data', train[train['DAYS_EMPLOYED'] < train['DAYS_BIRTH']].shape[0])
# create a new features (ESTIMATED AGE)

train['EST_AGE'] = (train['DAYS_BIRTH'] / -365)
plot_dist('EST_AGE')
correlations = train.drop(ID, axis=1).corr()
plt.figure(figsize=(20,20))

sns.heatmap(correlations)
correlations = correlations.sort_values(by=TARGET)

correlations.head(10)['TARGET']
correlations.tail(10)['TARGET']
del(correlations)

gc.collect()
plt.figure(figsize=(5,5))

sns.heatmap(train[[TARGET,'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].corr(), annot = True)
for ext_col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:

    plot_dist(ext_col)
train = train.drop('EST_AGE', axis=1)
# this plot is used to get probability of a categorical column that loan is not repaid (target=1)

def create_plot_prob(df, column, rotate=False, limit=None):

    display_name = column.replace('_', ' ').title()

    if rotate:

        rotation = 45

    else:

        rotation = 0

    if limit is not None:

        selected = list(df[col].value_counts().head(limit).index)

        df = df[df[col].isin(selected)]

    grouped_data = df[[column, TARGET]].groupby(column).agg(['mean', 'count'])

    grouped_data.reset_index(inplace=True)

    grouped_data.columns = [column, 'mean', 'sum']

    grouped_data = grouped_data.sort_values(by='mean')

    sns.set(style="whitegrid")



    fig = plt.figure(figsize=(20, 10))

#     plt.tight_layout()

    ax1 = fig.add_subplot(111)

    ax1.set_title('Occurance of Loan not repaid per {}'.format(display_name))

    label = grouped_data[column]

    g = sns.barplot(x=column, y='sum', data=grouped_data, order=label, ax=ax1)

    g.set_xticklabels(labels=label, rotation=rotation)

    g.set(xlabel=display_name, ylabel='Number of Case')

    ax2 = ax1.twinx()

    h = sns.pointplot(x=column, y="mean", data=grouped_data, ax=ax2)

    h.set(xlabel='index', ylabel='Loan not repaid Prior Probability')

    ax2.grid(False)

    plt.show()
for col in train.select_dtypes('object').columns:

    create_plot_prob(train, col, limit=10)
for col in train.select_dtypes(['object', 'category']).columns:

    pie_i_plot(col, title=col)
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)



POS_CASH_balance = reduce_mem_usage(POS_CASH_balance)

bureau_balance = reduce_mem_usage(bureau_balance)

del previous_application

gc.collect()

# previous_application = reduce_mem_usage(previous_application)

installments_payments = reduce_mem_usage(installments_payments)

credit_card_balance = reduce_mem_usage(credit_card_balance)

bureau = reduce_mem_usage(bureau)
def one_hot_encoder(df, nan_as_category=True):

    original_columns = list(df.columns)

    categorical_columns = [col for col in df.columns if str(df[col].dtype) in ['object', 'category']]

    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)

    new_columns = [c for c in df.columns if c not in original_columns]

    return df, new_columns
df = train.append(test)

df.head(1)
for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:

    df[bin_feature], uniques = pd.factorize(df[bin_feature])

df, cat_cols = one_hot_encoder(df, False)



df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']

df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

del test

del train

gc.collect()
grp = credit_card_balance.groupby(ID)['SK_ID_PREV'].nunique().reset_index().rename(index=str, columns={'SK_ID_PREV':'NO_LOANS'})

credit_card_balance = credit_card_balance.merge(grp, on=ID, how='left')

del(grp)

gc.collect()
# No of Installments paid per Loan per Customer 

grp = credit_card_balance.groupby(by = ['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].max().reset_index().rename(index = str, columns = {'CNT_INSTALMENT_MATURE_CUM': 'NO_INSTALMENTS'})

grp1 = grp.groupby(by = ['SK_ID_CURR'])['NO_INSTALMENTS'].sum().reset_index().rename(index = str, columns = {'NO_INSTALMENTS': 'TOTAL_INSTALMENTS'})

credit_card_balance = credit_card_balance.merge(grp1, on = ['SK_ID_CURR'], how = 'left')

del grp, grp1

gc.collect()
#AVERAGE NUMBER OF TIMES DAYS PAST DUE HAS OCCURRED PER CUSTOMER

def f(DPD):

    

    # DPD is a series of values of SK_DPD for each of the groupby combination 

    # We convert it to a list to get the number of SK_DPD values NOT EQUALS ZERO

    x = DPD.tolist()

    c = 0

    for i,j in enumerate(x):

        if j != 0:

            c += 1

    

    return c 



grp = credit_card_balance.groupby(by = ['SK_ID_CURR', 'SK_ID_PREV']).apply(lambda x: f(x.SK_DPD)).reset_index().rename(index = str, columns = {0: 'NO_DPD'})

grp1 = grp.groupby(by = ['SK_ID_CURR'])['NO_DPD'].mean().reset_index().rename(index = str, columns = {'NO_DPD' : 'DPD_COUNT'})



credit_card_balance = credit_card_balance.merge(grp1, on = ['SK_ID_CURR'], how = 'left')

del grp1

del grp 

gc.collect()
#% of MINIMUM PAYMENTS MISSED

def f(min_pay, total_pay):

    

    M = min_pay.tolist()

    T = total_pay.tolist()

    P = len(M)

    c = 0 

    # Find the count of transactions when Payment made is less than Minimum Payment 

    for i in range(len(M)):

        if T[i] < M[i]:

            c += 1  

    return (100*c)/P



grp = credit_card_balance.groupby(by = ['SK_ID_CURR']).apply(lambda x: f(x.AMT_INST_MIN_REGULARITY, x.AMT_PAYMENT_CURRENT)).reset_index().rename(index = str, columns = { 0 : 'PERCENTAGE_MISSED_PAYMENTS'})

credit_card_balance = credit_card_balance.merge(grp, on = ['SK_ID_CURR'], how = 'left')

del grp 

gc.collect()
grp = credit_card_balance.groupby(by = ['SK_ID_CURR'])['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(index = str, columns = {'AMT_DRAWINGS_ATM_CURRENT' : 'DRAWINGS_ATM'})

credit_card_balance = credit_card_balance.merge(grp, on = ['SK_ID_CURR'], how = 'left')

del grp

gc.collect()



grp = credit_card_balance.groupby(by = ['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index = str, columns = {'AMT_DRAWINGS_CURRENT' : 'DRAWINGS_TOTAL'})

credit_card_balance = credit_card_balance.merge(grp, on = ['SK_ID_CURR'], how = 'left')

del grp

gc.collect()



credit_card_balance['CASH_CARD_RATIO1'] = (credit_card_balance['DRAWINGS_ATM']/credit_card_balance['DRAWINGS_TOTAL'])*100

del credit_card_balance['DRAWINGS_ATM']

del credit_card_balance['DRAWINGS_TOTAL']

gc.collect()



grp = credit_card_balance.groupby(by = ['SK_ID_CURR'])['CASH_CARD_RATIO1'].mean().reset_index().rename(index = str, columns ={ 'CASH_CARD_RATIO1' : 'CASH_CARD_RATIO'})

credit_card_balance = credit_card_balance.merge(grp, on = ['SK_ID_CURR'], how = 'left')

del grp 

gc.collect()



del credit_card_balance['CASH_CARD_RATIO1']

gc.collect()
credit_card_balance.shape
df.shape
credit_card_balance, cat_cols = one_hot_encoder(credit_card_balance, nan_as_category=False)
credit_card_balance.drop(['SK_ID_PREV'], axis= 1, inplace = True)
df = df.merge(credit_card_balance, on=ID, how='left')

del(credit_card_balance)

gc.collect()

df.shape
# NUMBER OF PAST LOANS PER CUSTOMER

grp = bureau[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT'].count().reset_index().rename(index=str, columns={'DAYS_CREDIT': 'BUREAU_LOAN_COUNT'})

bureau = bureau.merge(grp, on = ['SK_ID_CURR'], how = 'left')
# NUMBER OF TYPES OF PAST LOANS PER CUSTOMER

grp = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(index=str, columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})

bureau = bureau.merge(grp, on = ['SK_ID_CURR'], how = 'left')
# Is the Customer diversified in taking multiple types of Loan or Focused on a single type of loan

bureau['AVERAGE_LOAN_TYPE'] = bureau['BUREAU_LOAN_COUNT']/bureau['BUREAU_LOAN_TYPES']

del bureau['BUREAU_LOAN_COUNT'], bureau['BUREAU_LOAN_TYPES'], grp

gc.collect()
#% OF ACTIVE LOANS FROM BUREAU DATA

bureau['CREDIT_ACTIVE_BINARY'] = bureau['CREDIT_ACTIVE']



def f(x):

    if x == 'Closed':

        y = 0

    else:

        y = 1    

    return y



bureau['CREDIT_ACTIVE_BINARY'] = bureau.apply(lambda x: f(x.CREDIT_ACTIVE), axis = 1)



# Calculate mean number of loans that are ACTIVE per CUSTOMER 

grp = bureau.groupby(by = ['SK_ID_CURR'])['CREDIT_ACTIVE_BINARY'].mean().reset_index().rename(index=str, columns={'CREDIT_ACTIVE_BINARY': 'ACTIVE_LOANS_PERCENTAGE'})

bureau = bureau.merge(grp, on = ['SK_ID_CURR'], how = 'left')

del bureau['CREDIT_ACTIVE_BINARY'], grp

gc.collect()
bureau_balance, bureau_balance_cat = one_hot_encoder(bureau_balance, True)

bureau, bureau_cat = one_hot_encoder(bureau, True)
bureau_balance_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}

for col in bureau_balance_cat:

    bureau_balance_aggregations[col] = ['mean']

bureau_balance_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(bureau_balance_aggregations)

bureau_balance_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_balance_agg.columns.tolist()])

bureau = bureau.join(bureau_balance_agg, how='left', on='SK_ID_BUREAU')

bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)

del bureau_balance, bureau_balance_agg

gc.collect()
bureau.shape
num_aggregations = {

    'DAYS_CREDIT': ['mean'],

    'DAYS_CREDIT_ENDDATE': ['mean'],

    'DAYS_CREDIT_UPDATE': ['mean'],

    'CREDIT_DAY_OVERDUE': ['mean'],

    'AMT_CREDIT_MAX_OVERDUE': ['mean'],

    'AMT_CREDIT_SUM': ['mean',],

    'AMT_CREDIT_SUM_DEBT': ['mean'],

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

for cat in bureau_balance_cat: cat_aggregations[cat + "_MEAN"] = ['mean']



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
bureau_agg.shape
df.shape
df = df.merge(bureau_agg, on=ID, how='left')

del(bureau_agg)

gc.collect()

df.shape
ins, cat_cols = one_hot_encoder(installments_payments, nan_as_category= True)

del installments_payments

gc.collect()

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
df = df.merge(ins_agg, on=ID, how='left')

del(ins_agg)

gc.collect()

df.shape
pos, cat_cols = one_hot_encoder(POS_CASH_balance, nan_as_category= True)

del(POS_CASH_balance)

gc.collect()

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
df = df.merge(pos_agg, on=ID, how='left')

del(pos_agg)

gc.collect()

df.shape
df = reduce_mem_usage(df)