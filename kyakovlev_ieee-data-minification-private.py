# General imports

import numpy as np

import pandas as pd

import os, warnings, datetime, math



from sklearn.preprocessing import LabelEncoder



warnings.filterwarnings('ignore')
########################### Helpers

#################################################################################

## -------------------

## Memory Reducer

# :df pandas dataframe to reduce size             # type: pd.DataFrame()

# :verbose                                        # type: bool

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

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

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

## -------------------
########################### Vars

#################################################################################

START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
########################### DATA LOAD

#################################################################################

print('Load Data')

train_df = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

test_df = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

test_df['isFraud'] = 0



train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
########################### Base check

#################################################################################



for df in [train_df, test_df, train_identity, test_identity]:

    original = df.copy()

    df = reduce_mem_usage(df)



    for col in list(df):

        if df[col].dtype!='O':

            if (df[col]-original[col]).sum()!=0:

                df[col] = original[col]

                print('Bad transformation', col)
########################### TransactionDT

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')

us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())



for df in [train_df, test_df]:

    

    # Temporary variables for aggregation

    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

    df['DT_M'] = ((df['DT'].dt.year-2017)*12 + df['DT'].dt.month).astype(np.int8)

    df['DT_W'] = ((df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear).astype(np.int8)

    df['DT_D'] = ((df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear).astype(np.int16)

    

    df['DT_hour'] = (df['DT'].dt.hour).astype(np.int8)

    df['DT_day_week'] = (df['DT'].dt.dayofweek).astype(np.int8)

    df['DT_day_month'] = (df['DT'].dt.day).astype(np.int8)

    df['DT_week_month'] = (df['DT'].dt.day)/7

    df['DT_week_month'] = df['DT_week_month'].apply(lambda x: math.ceil(x))



    # Possible solo feature

    df['is_december'] = df['DT'].dt.month

    df['is_december'] = (df['is_december']==12).astype(np.int8)



    # Holidays

    df['is_holiday'] = (df['DT'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)



# Total transactions per timeblock

for col in ['DT_M','DT_W','DT_D']:

    temp_df = pd.concat([train_df[[col]], test_df[[col]]])

    fq_encode = temp_df[col].value_counts().to_dict()

            

    train_df[col+'_total'] = train_df[col].map(fq_encode)

    test_df[col+'_total']  = test_df[col].map(fq_encode)
########################### card4, card6, ProductCD

#################################################################################

# Converting Strings to ints(or floats if nan in column) using frequency encoding

# We will be able to use these columns as category or as numerical feature



for col in ['card4', 'card6', 'ProductCD']:

    print('Encoding', col)

    temp_df = pd.concat([train_df[[col]], test_df[[col]]])

    col_encoded = temp_df[col].value_counts().to_dict()   

    train_df[col] = train_df[col].map(col_encoded)

    test_df[col]  = test_df[col].map(col_encoded)

    print(col_encoded)
########################### M columns

#################################################################################

# Converting Strings to ints(or floats if nan in column)



for col in ['M1','M2','M3','M5','M6','M7','M8','M9']:

    train_df[col] = train_df[col].map({'T':1, 'F':0})

    test_df[col]  = test_df[col].map({'T':1, 'F':0})



for col in ['M4']:

    print('Encoding', col)

    temp_df = pd.concat([train_df[[col]], test_df[[col]]])

    col_encoded = temp_df[col].value_counts().to_dict()   

    train_df[col] = train_df[col].map(col_encoded)

    test_df[col]  = test_df[col].map(col_encoded)

    print(col_encoded)
########################### Identity columns

#################################################################################



def minify_identity_df(df):



    df['id_12'] = df['id_12'].map({'Found':1, 'NotFound':0})

    df['id_15'] = df['id_15'].map({'New':2, 'Found':1, 'Unknown':0})

    df['id_16'] = df['id_16'].map({'Found':1, 'NotFound':0})



    df['id_23'] = df['id_23'].map({'TRANSPARENT':4, 'IP_PROXY':3, 'IP_PROXY:ANONYMOUS':2, 'IP_PROXY:HIDDEN':1})



    df['id_27'] = df['id_27'].map({'Found':1, 'NotFound':0})

    df['id_28'] = df['id_28'].map({'New':2, 'Found':1})



    df['id_29'] = df['id_29'].map({'Found':1, 'NotFound':0})



    df['id_35'] = df['id_35'].map({'T':1, 'F':0})

    df['id_36'] = df['id_36'].map({'T':1, 'F':0})

    df['id_37'] = df['id_37'].map({'T':1, 'F':0})

    df['id_38'] = df['id_38'].map({'T':1, 'F':0})



    df['id_34'] = df['id_34'].fillna(':0')

    df['id_34'] = df['id_34'].apply(lambda x: x.split(':')[1]).astype(np.int8)

    df['id_34'] = np.where(df['id_34']==0, np.nan, df['id_34'])

    

    df['id_33'] = df['id_33'].fillna('0x0')

    df['id_33_0'] = df['id_33'].apply(lambda x: x.split('x')[0]).astype(int)

    df['id_33_1'] = df['id_33'].apply(lambda x: x.split('x')[1]).astype(int)

    df['id_33'] = np.where(df['id_33']=='0x0', np.nan, df['id_33'])



    df['DeviceType'].map({'desktop':1, 'mobile':0})

    return df



train_identity = minify_identity_df(train_identity)

test_identity = minify_identity_df(test_identity)



for col in ['id_33']:

    train_identity[col] = train_identity[col].fillna('unseen_before_label')

    test_identity[col]  = test_identity[col].fillna('unseen_before_label')

    

    le = LabelEncoder()

    le.fit(list(train_identity[col])+list(test_identity[col]))

    train_identity[col] = le.transform(train_identity[col])

    test_identity[col]  = le.transform(test_identity[col])
########################### Deltas



for df in [train_df, test_df]:

    for col in ['D'+str(i) for i in range(1,16) if i!=9]: 

        new_col = 'uid_td_'+str(col)

        df[new_col] = df[col].fillna(0).astype(int)

        df[new_col] = df[new_col].apply(lambda x: pd.Timedelta(x, unit='D'))

        df[new_col] = (df['DT'] - df[new_col]).dt.date

        df[new_col] = df[new_col].astype(str)

        df[new_col] = np.where(df[col].isna(), np.nan, df[new_col])
########################### Final check

#################################################################################



for df in [train_df, test_df, train_identity, test_identity]:

    original = df.copy()

    df = reduce_mem_usage(df)



    for col in list(df):

        if df[col].dtype!='O':

            if (df[col]-original[col]).sum()!=0:

                df[col] = original[col]

                print('Bad transformation', col)
########################### Export

#################################################################################



train_df.to_pickle('train_transaction.pkl')

test_df.to_pickle('test_transaction.pkl')



train_identity.to_pickle('train_identity.pkl')

test_identity.to_pickle('test_identity.pkl')
########################### Full minification for fast tests

#################################################################################

for df in [train_df, test_df, train_identity, test_identity]:

    df = reduce_mem_usage(df)
########################### Export

#################################################################################



train_df.to_pickle('train_transaction_mini.pkl')

test_df.to_pickle('test_transaction_mini.pkl')



train_identity.to_pickle('train_identity_mini.pkl')

test_identity.to_pickle('test_identity_mini.pkl')
########################### Export

#################################################################################



possible_goups = [['V1', 'V2', 'V6', 'V7', 'V8', 'V9'],

 ['V1', 'V2', 'V3', 'V6', 'V7', 'V8', 'V9'],

 ['V2', 'V3', 'V6', 'V7', 'V8', 'V9'],

 ['V4', 'V5'],

 ['V10', 'V11'],

 ['V12', 'V13'],

 ['V14', 'V65'],

 ['V15', 'V16', 'V33', 'V34', 'V57', 'V58', 'V79', 'V94'],

 ['V15', 'V16', 'V33', 'V34', 'V57'],

 ['V17', 'V18', 'V21', 'V22'],

 ['V19', 'V20'],

 ['V17', 'V18', 'V21', 'V22', 'V63', 'V84'],

 ['V23', 'V24'],

 ['V25', 'V26'],

 ['V27', 'V28', 'V68', 'V89'],

 ['V29', 'V30', 'V69', 'V90', 'V91'],

 ['V29', 'V30', 'V70', 'V90', 'V91'],

 ['V31', 'V32', 'V50', 'V71', 'V92', 'V93'],

 ['V31', 'V32', 'V92'],

 ['V15', 'V16', 'V33', 'V34', 'V51', 'V94'],

 ['V15', 'V16', 'V33', 'V34'],

 ['V35', 'V36'],

 ['V37', 'V38'],

 ['V39', 'V40'],

 ['V41', 'V46', 'V47'],

 ['V42', 'V43', 'V84'],

 ['V42', 'V43'],

 ['V44', 'V45'],

 ['V48', 'V49'],

 ['V31', 'V50', 'V71', 'V92'],

 ['V33', 'V51', 'V52', 'V73', 'V94'],

 ['V51', 'V52'],

 ['V53', 'V54'],

 ['V15', 'V16', 'V57', 'V58', 'V73', 'V79', 'V94'],

 ['V15', 'V57', 'V58', 'V79'],

 ['V59', 'V60', 'V63'],

 ['V59', 'V60'],

 ['V61', 'V62'],

 ['V21', 'V59', 'V63', 'V64', 'V84'],

 ['V63', 'V64'],

 ['V66', 'V67'],

 ['V29', 'V69', 'V70', 'V90'],

 ['V30', 'V69', 'V70', 'V90', 'V91'],

 ['V31', 'V50', 'V71', 'V72', 'V92', 'V93'],

 ['V71', 'V72', 'V92', 'V93'],

 ['V51', 'V57', 'V73', 'V74', 'V94'],

 ['V73', 'V74'],

 ['V75', 'V76'],

 ['V80', 'V81', 'V84'],

 ['V80', 'V81'],

 ['V82', 'V83'],

 ['V21', 'V42', 'V63', 'V80', 'V84', 'V85'],

 ['V84', 'V85'],

 ['V86', 'V87'],

 ['V29', 'V30', 'V69', 'V70', 'V90', 'V91'],

 ['V31', 'V32', 'V50', 'V71', 'V72', 'V92', 'V93'],

 ['V31', 'V71', 'V72', 'V92', 'V93'],

 ['V15', 'V33', 'V51', 'V57', 'V73', 'V94'],

 ['V101',

  'V102',

  'V103',

  'V143',

  'V167',

  'V168',

  'V177',

  'V178',

  'V179',

  'V279',

  'V280',

  'V293',

  'V295',

  'V322',

  'V323',

  'V324',

  'V95',

  'V96',

  'V97'],

 ['V101',

  'V102',

  'V103',

  'V143',

  'V167',

  'V168',

  'V177',

  'V178',

  'V179',

  'V279',

  'V280',

  'V293',

  'V294',

  'V295',

  'V322',

  'V323',

  'V324',

  'V95',

  'V96',

  'V97'],

 ['V105', 'V106', 'V296', 'V298', 'V299', 'V329', 'V330'],

 ['V105', 'V106', 'V298', 'V299', 'V329', 'V330'],

 ['V111', 'V113'],

 ['V126', 'V128', 'V132', 'V134'],

 ['V127', 'V128', 'V133', 'V134'],

 ['V126', 'V127', 'V128', 'V132', 'V133', 'V134', 'V332'],

 ['V129', 'V266', 'V269', 'V309', 'V334'],

 ['V130', 'V310'],

 ['V131', 'V312'],

 ['V126', 'V128', 'V132', 'V133', 'V134'],

 ['V127', 'V128', 'V132', 'V133', 'V134'],

 ['V126', 'V127', 'V128', 'V132', 'V133', 'V134', 'V318', 'V332'],

 ['V136', 'V137'],

 ['V101',

  'V102',

  'V103',

  'V143',

  'V167',

  'V177',

  'V178',

  'V179',

  'V279',

  'V280',

  'V293',

  'V295',

  'V322',

  'V323',

  'V324',

  'V95',

  'V96',

  'V97'],

 ['V144', 'V145', 'V150', 'V151'],

 ['V148', 'V149', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158'],

 ['V144', 'V145', 'V150', 'V151', 'V152'],

 ['V151', 'V152'],

 ['V161', 'V163'],

 ['V162', 'V163'],

 ['V161', 'V162', 'V163'],

 ['V101',

  'V102',

  'V103',

  'V167',

  'V168',

  'V177',

  'V178',

  'V179',

  'V279',

  'V280',

  'V293',

  'V295',

  'V322',

  'V323',

  'V324',

  'V95',

  'V96',

  'V97'],

 ['V176', 'V190', 'V199', 'V228', 'V246', 'V257'],

 ['V180', 'V182', 'V183'],

 ['V181', 'V328'],

 ['V180', 'V182', 'V183', 'V330'],

 ['V186', 'V191', 'V196'],

 ['V187', 'V192'],

 ['V187', 'V192', 'V193'],

 ['V192', 'V193', 'V196'],

 ['V194', 'V197'],

 ['V195', 'V198'],

 ['V186', 'V191', 'V193', 'V196'],

 ['V202', 'V204', 'V211', 'V213'],

 ['V203', 'V204', 'V212'],

 ['V202', 'V203', 'V204', 'V213'],

 ['V202', 'V211', 'V213'],

 ['V203', 'V212', 'V213'],

 ['V202', 'V204', 'V211', 'V212', 'V213'],

 ['V214', 'V276', 'V337'],

 ['V215', 'V216', 'V277', 'V278', 'V338', 'V339'],

 ['V217', 'V219', 'V231', 'V233'],

 ['V218', 'V219', 'V232', 'V233'],

 ['V217', 'V218', 'V219', 'V231', 'V232', 'V233'],

 ['V222', 'V230'],

 ['V224', 'V225'],

 ['V229', 'V230', 'V258'],

 ['V222', 'V229', 'V230', 'V258'],

 ['V236', 'V237'],

 ['V238', 'V239'],

 ['V240', 'V241', 'V247', 'V252', 'V260'],

 ['V242', 'V244'],

 ['V245', 'V259'],

 ['V240', 'V241', 'V247', 'V249', 'V252'],

 ['V248', 'V249', 'V254'],

 ['V247', 'V248', 'V249', 'V252'],

 ['V250', 'V251'],

 ['V248', 'V254'],

 ['V255', 'V256'],

 ['V240', 'V241', 'V260'],

 ['V263', 'V265', 'V273', 'V274', 'V275'],

 ['V264', 'V265'],

 ['V263', 'V264', 'V265'],

 ['V129', 'V266', 'V269', 'V309', 'V334', 'V336'],

 ['V268', 'V336'],

 ['V270', 'V272'],

 ['V263', 'V273', 'V274', 'V275'],

 ['V291', 'V292'],

 ['V102', 'V280', 'V294', 'V295', 'V323', 'V96'],

 ['V105', 'V296', 'V298', 'V299', 'V329'],

 ['V105', 'V106', 'V296', 'V298', 'V299', 'V329'],

 ['V105', 'V106', 'V296', 'V298', 'V299', 'V330'],

 ['V300', 'V301'],

 ['V302', 'V304'],

 ['V303', 'V304'],

 ['V302', 'V303', 'V304'],

 ['V306', 'V308', 'V316', 'V318'],

 ['V307', 'V308', 'V317'],

 ['V306', 'V307', 'V308', 'V318'],

 ['V313', 'V315'],

 ['V306', 'V316', 'V318'],

 ['V307', 'V317', 'V318'],

 ['V134', 'V306', 'V308', 'V316', 'V317', 'V318'],

 ['V320', 'V321'],

 ['V326', 'V327'],

 ['V105', 'V106', 'V296', 'V298', 'V329', 'V330'],

 ['V105', 'V106', 'V183', 'V299', 'V329', 'V330'],

 ['V331', 'V332', 'V333'],

 ['V128', 'V134', 'V331', 'V332', 'V333'],

 ['V335', 'V336'],

 ['V266', 'V268', 'V269', 'V334', 'V335', 'V336']]



with open('possible_goups.pickle', 'wb') as f:

    pickle.dump(possible_goups, f, pickle.HIGHEST_PROTOCOL)