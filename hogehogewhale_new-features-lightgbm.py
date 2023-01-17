import pandas as pd

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

import numpy as np



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder



import gc, warnings, datetime



import lightgbm as lgb



warnings.filterwarnings('ignore')
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
def corret_card_id(x): 

    x=x.replace('.0','')

    x=x.replace('-999','nan')

    return x
def define_indexes(df):

    cards_cols= ['card1', 'card2', 'card3', 'card5']

    for card in cards_cols: 

        if '1' in card: 

            df['card_id']= df[card].map(str)

        else : 

            df['card_id']+= ' '+df[card].map(str)

    df['card_id']=df['card_id'].apply(corret_card_id)

    

    df['uid'] = df['card1'].astype(str)+'_'+df['card2'].astype(str)+'_'+df['card3'].astype(str)+'_'+df['card4'].astype(str)



    df['uid2'] = df['uid'].astype(str)+'_'+df['addr1'].astype(str)+'_'+df['addr2'].astype(str)



    df['uid3'] = df['uid2'].astype(str)+'_'+df['addr1'].astype(str)+'_'+df['addr2'].astype(str)



    df['uid4'] = df['card1'].astype(str)+'_'+df['card2'].astype(str)



    df['uid5'] = df['uid'].astype(str)+'_'+df['card3'].astype(str)+'_'+df['card5'].astype(str)



    df['uid6'] = df['uid2'].astype(str)+'_'+df['addr1'].astype(str)+'_'+df['addr2'].astype(str)



    df['uid7'] = df['uid3'].astype(str)+'_'+df['P_emaildomain'].astype(str)



    df['uid8'] = df['uid3'].astype(str)+'_'+df['R_emaildomain'].astype(str)

    

    return df
def define_time(df):

    START_DATE = '2017-12-01'

    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')

    df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

    df['year'] = df['TransactionDT'].dt.year

    df['month'] = df['TransactionDT'].dt.month

    df['dow'] = df['TransactionDT'].dt.dayofweek

    df['hour'] = df['TransactionDT'].dt.hour

    df['day'] = df['TransactionDT'].dt.day

    return df
def id_split(dataframe):

    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]

    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]



    dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]

    dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]



    dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]

    dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]



    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0]

    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1]



    dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1]

    dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1]



    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'

    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'

    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'

    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'

    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'

    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'

    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'



    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"

    dataframe['had_id'] = 1

    gc.collect()

    

    return dataframe
def setbrowser(df):

    df.loc[df["id_31"]=="samsung browser 7.0",'lastest_browser']=1

    df.loc[df["id_31"]=="opera 53.0",'lastest_browser']=1

    df.loc[df["id_31"]=="mobile safari 10.0",'lastest_browser']=1

    df.loc[df["id_31"]=="google search application 49.0",'lastest_browser']=1

    df.loc[df["id_31"]=="firefox 60.0",'lastest_browser']=1

    df.loc[df["id_31"]=="edge 17.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 69.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 67.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 63.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 63.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 64.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 64.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 64.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 65.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 65.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 65.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 66.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 66.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 66.0 for ios",'lastest_browser']=1

    return df
def fix_emails(df):

    df['P_emaildomain'] = df['P_emaildomain'].fillna('email_not_provided')

    df['R_emaildomain'] = df['R_emaildomain'].fillna('email_not_provided')

    df['email_match_not_nan'] = np.where((df['P_emaildomain']==df['R_emaildomain'])&(df['P_emaildomain']!='email_not_provided'),1,0)

    return df
train_transaction = reduce_mem_usage(pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv'))

train_identity = reduce_mem_usage(pd.read_csv('../input/ieee-fraud-detection/train_identity.csv'))



test_transaction = reduce_mem_usage(pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv'))

test_identity = reduce_mem_usage(pd.read_csv('../input/ieee-fraud-detection/test_identity.csv'))



sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
train_identity = id_split(train_identity)

test_identity = id_split(test_identity)
train_identity['DeviceInfo'] = train_identity['DeviceInfo'].fillna('unknown_device').str.lower()

test_identity['DeviceInfo'] = test_identity['DeviceInfo'].fillna('unknown_device').str.lower()



train_identity['DeviceInfo_c'] = train_identity['DeviceInfo']

test_identity['DeviceInfo_c'] = test_identity['DeviceInfo']



device_match_dict = {

    'sm':'sm-',

    'sm':'samsung',

    'huawei':'huawei',

    'moto':'moto',

    'rv':'rv:',

    'trident':'trident',

    'lg':'lg-',

    'htc':'htc',

    'blade':'blade',

    'windows':'windows',

    'lenovo':'lenovo',

    'linux':'linux',

    'f3':'f3',

    'f5':'f5'

}

for dev_type_s, dev_type_o in device_match_dict.items():

    train_identity['DeviceInfo_c'] = train_identity['DeviceInfo_c'].apply(lambda x: dev_type_s if dev_type_o in x else x)

    test_identity['DeviceInfo_c'] = test_identity['DeviceInfo_c'].apply(lambda x: dev_type_s if dev_type_o in x else x)



train_identity['DeviceInfo_c'] = train_identity['DeviceInfo_c'].apply(lambda x: 'other_d_type' if x not in device_match_dict else x)

test_identity['DeviceInfo_c'] = test_identity['DeviceInfo_c'].apply(lambda x: 'other_d_type' if x not in device_match_dict else x)
train_identity['id_30'] = train_identity['id_30'].fillna('unknown_device').str.lower()

test_identity['id_30'] = test_identity['id_30'].fillna('unknown_device').str.lower()



train_identity['id_30_c'] = train_identity['id_30']

test_identity['id_30_c'] = test_identity['id_30']



device_match_dict = {

    'ios':'ios',

    'windows':'windows',

    'mac':'mac',

    'android':'android',

    'linux':'linux'

}

for dev_type_s, dev_type_o in device_match_dict.items():

    train_identity['id_30_c'] = train_identity['id_30_c'].apply(lambda x: dev_type_s if dev_type_o in x else x)

    test_identity['id_30_c'] = test_identity['id_30_c'].apply(lambda x: dev_type_s if dev_type_o in x else x)

    

train_identity['id_30_v'] = train_identity['id_30'].apply(lambda x: ''.join([i for i in x if i.isdigit()]))

test_identity['id_30_v'] = test_identity['id_30'].apply(lambda x: ''.join([i for i in x if i.isdigit()]))

        

train_identity['id_30_v'] = np.where(train_identity['id_30_v']!='', train_identity['id_30_v'], 0).astype(int)

test_identity['id_30_v'] = np.where(test_identity['id_30_v']!='', test_identity['id_30_v'], 0).astype(int)
train_identity['id_31'] = train_identity['id_31'].fillna('unknown_br').str.lower()

test_identity['id_31']  = test_identity['id_31'].fillna('unknown_br').str.lower()



train_identity['id_31'] = train_identity['id_31'].apply(lambda x: x.replace('webview','webvw'))

test_identity['id_31']  = test_identity['id_31'].apply(lambda x: x.replace('webview','webvw'))



train_identity['id_31'] = train_identity['id_31'].apply(lambda x: x.replace('for',' '))

test_identity['id_31']  = test_identity['id_31'].apply(lambda x: x.replace('for',' '))



browser_list = set(list(train_identity['id_31'].unique()) + list(test_identity['id_31'].unique()))

browser_list2 = []

for item in browser_list:

    browser_list2 += item.split(' ')

browser_list2 = list(set(browser_list2))



browser_list3 = []

for item in browser_list2:

    browser_list3 += item.split('/')

browser_list3 = list(set(browser_list3))

        

train_identity['id_31_v'] = train_identity['id_31'].apply(lambda x: ''.join([i for i in x if i.isdigit()]))

test_identity['id_31_v'] = test_identity['id_31'].apply(lambda x: ''.join([i for i in x if i.isdigit()]))



train_identity['id_31_v'] = np.where(train_identity['id_31_v']!='', train_identity['id_31_v'], 0).astype(int)

test_identity['id_31_v'] = np.where(test_identity['id_31_v']!='', test_identity['id_31_v'], 0).astype(int)
train_transaction['nulls_transaction'] = train_transaction.isna().sum(axis=1)

train_transaction['nulls_cards'] = train_transaction[[

    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',

    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',

]].isna().sum(axis=1)

train_transaction['nulls_C'] = train_transaction[[

    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',

    'C10', 'C11', 'C12', 'C13', 'C14',

]].isna().sum(axis=1)

train_transaction['nulls_D'] = train_transaction[[

    'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',

    'D10', 'D11', 'D12', 'D13', 'D14', 'D15',

]].isna().sum(axis=1)

train_transaction['nulls_M'] = train_transaction[[

    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',

]].isna().sum(axis=1)

train_transaction['nulls_V'] = train_transaction.iloc[:, 54:393].isna().sum(axis=1)



train_identity['nulls_identity'] = train_identity.isna().sum(axis=1)

train_identity['nulls_device'] = train_identity[[

    'DeviceType', 'DeviceInfo'

]].isna().sum(axis=1)

train_identity['nulls_id'] = train_identity.iloc[:, 0:38].isna().sum(axis=1)







test_transaction['nulls_transaction'] = test_transaction.isna().sum(axis=1)

test_transaction['nulls_cards'] = test_transaction[[

    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',

    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',

]].isna().sum(axis=1)

test_transaction['nulls_C'] = test_transaction[[

    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',

    'C10', 'C11', 'C12', 'C13', 'C14',

]].isna().sum(axis=1)

test_transaction['nulls_D'] = test_transaction[[

    'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',

    'D10', 'D11', 'D12', 'D13', 'D14', 'D15',

]].isna().sum(axis=1)

test_transaction['nulls_M'] = test_transaction[[

    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',

]].isna().sum(axis=1)

test_transaction['nulls_V'] = test_transaction.iloc[:, 54:393].isna().sum(axis=1)



test_identity['nulls_identity'] = test_identity.isna().sum(axis=1)

test_identity['nulls_device'] = test_identity[[

    'DeviceType', 'DeviceInfo'

]].isna().sum(axis=1)

test_identity['nulls_id'] = test_identity.iloc[:, 0:38].isna().sum(axis=1)
train = train_transaction.merge(train_identity,on='TransactionID', how='left')

test = test_transaction.merge(test_identity,on='TransactionID', how='left')
train['nulls'] = train.isna().sum(axis=1)

test['nulls'] = test.isna().sum(axis=1)
train = define_indexes(train)

test = define_indexes(test)
for col in ['ProductCD','M4']:

    temp_dict = train.groupby([col])['isFraud'].agg(['mean']).reset_index().rename(columns={'mean': col+'_target_mean'})

    temp_dict.index = temp_dict[col].values

    temp_dict = temp_dict[col+'_target_mean'].to_dict()



    train[col+'_target_mean'] = train[col].map(temp_dict)

    test[col+'_target_mean']  = test[col].map(temp_dict)
train['bank_type'] = train['card3'].astype(str)+'_'+train['card5'].astype(str)

test['bank_type']  = test['card3'].astype(str)+'_'+test['card5'].astype(str)



train['address_match'] = train['bank_type'].astype(str)+'_'+train['addr2'].astype(str)

test['address_match']  = test['bank_type'].astype(str)+'_'+test['addr2'].astype(str)



for col in ['address_match','bank_type']:

    temp = pd.concat([train[[col]], test[[col]]])

    temp[col] = np.where(temp[col].str.contains('nan'), np.nan, temp[col])

    temp = temp.dropna()

    fq_encode = temp[col].value_counts().to_dict()   

    train[col] = train[col].map(fq_encode)

    test[col]  = test[col].map(fq_encode)



train['address_match'] = train['address_match']/train['bank_type'] 

test['address_match']  = test['address_match']/test['bank_type']
for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 

                'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:

    f1, f2 = feature.split('__')

    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)

    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)

    le = LabelEncoder()

    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))

    train[feature] = le.transform(list(train[feature].astype(str).values))

    test[feature] = le.transform(list(test[feature].astype(str).values))
for feature in ['id_34', 'id_36']:

    train[feature + '_count_full'] = train[feature].map(

        pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

    test[feature + '_count_full'] = test[feature].map(

        pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
for feature in ['id_01', 'id_31', 'id_33', 'id_35', 'id_36']:

    train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))

    test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))
train['card1_card2'] = train['card1'].astype(str) + '_' + train['card2'].astype(str)

train['addr1_dist1'] = train['addr1'].astype(str) + '_' + train['dist1'].astype(str)

train['card1_addr2'] = train['card1'].astype(str) + '_' + train['addr2'].astype(str)

train['card2_addr1'] = train['card2'].astype(str) + '_' + train['addr1'].astype(str)

train['card2_addr2'] = train['card2'].astype(str) + '_' + train['addr2'].astype(str)

train['card4_addr1'] = train['card4'].astype(str) + '_' + train['addr1'].astype(str)

train['card4_addr2'] = train['card4'].astype(str) + '_' + train['addr2'].astype(str)

train['P_emaildomain_addr1'] = train['P_emaildomain'].astype(str) + '_' + train['addr1'].astype(str)

train['id01_addr1'] = train['id_01'].astype(str) + '_' + train['addr1'].astype(str)



test['card1_card2'] = test['card1'].astype(str) + '_' + test['card2'].astype(str)

test['addr1_dist1'] = test['addr1'].astype(str) + '_' + test['dist1'].astype(str)

test['card1_addr2'] = test['card1'].astype(str) + '_' + test['addr2'].astype(str)

test['card2_addr1'] = test['card2'].astype(str) + '_' + test['addr1'].astype(str)

test['card2_addr2'] = test['card2'].astype(str) + '_' + test['addr2'].astype(str)

test['card4_addr1'] = test['card4'].astype(str) + '_' + test['addr1'].astype(str)

test['card4_addr2'] = test['card4'].astype(str) + '_' + test['addr2'].astype(str)

test['P_emaildomain_addr1'] = test['P_emaildomain'].astype(str) + '_' + test['addr1'].astype(str)

test['id01_addr1'] = test['id_01'].astype(str) + '_' + test['addr1'].astype(str)
a = np.zeros(train.shape[0])

train["lastest_browser"] = a

train=setbrowser(train)



a = np.zeros(test.shape[0])

test["lastest_browser"] = a

test=setbrowser(test)
train['card1_count_full'] = train['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))

train['card2_count_full'] = train['card2'].map(pd.concat([train['card2'], test['card2']], ignore_index=True).value_counts(dropna=False))

train['card3_count_full'] = train['card3'].map(pd.concat([train['card3'], test['card3']], ignore_index=True).value_counts(dropna=False))

train['card4_count_full'] = train['card4'].map(pd.concat([train['card4'], test['card4']], ignore_index=True).value_counts(dropna=False))

train['card5_count_full'] = train['card5'].map(pd.concat([train['card5'], test['card5']], ignore_index=True).value_counts(dropna=False))

train['card6_count_full'] = train['card6'].map(pd.concat([train['card6'], test['card6']], ignore_index=True).value_counts(dropna=False))

train['addr1_count_full'] = train['addr1'].map(pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False))

train['addr2_count_full'] = train['addr2'].map(pd.concat([train['addr2'], test['addr2']], ignore_index=True).value_counts(dropna=False))

train['P_emaildomain_count_full'] = train['P_emaildomain'].map(pd.concat([train['P_emaildomain'], test['P_emaildomain']], ignore_index=True).value_counts(dropna=False))

train['R_emaildomain_count_full'] = train['R_emaildomain'].map(pd.concat([train['R_emaildomain'], test['R_emaildomain']], ignore_index=True).value_counts(dropna=False))

train['uid_count_full'] = train['uid'].map(pd.concat([train['uid'], test['uid']], ignore_index=True).value_counts(dropna=False))

train['uid2_count_full'] = train['uid2'].map(pd.concat([train['uid2'], test['uid2']], ignore_index=True).value_counts(dropna=False))

train['uid3_count_full'] = train['uid3'].map(pd.concat([train['uid3'], test['uid3']], ignore_index=True).value_counts(dropna=False))

train['uid4_count_full'] = train['uid4'].map(pd.concat([train['uid4'], test['uid4']], ignore_index=True).value_counts(dropna=False))

train['uid5_count_full'] = train['uid5'].map(pd.concat([train['uid5'], test['uid5']], ignore_index=True).value_counts(dropna=False))

train['uid6_count_full'] = train['uid6'].map(pd.concat([train['uid6'], test['uid6']], ignore_index=True).value_counts(dropna=False))

train['uid7_count_full'] = train['uid7'].map(pd.concat([train['uid7'], test['uid7']], ignore_index=True).value_counts(dropna=False))

train['uid8_count_full'] = train['uid8'].map(pd.concat([train['uid8'], test['uid8']], ignore_index=True).value_counts(dropna=False))

train['card_id_count_full'] = train['card_id'].map(pd.concat([train['card_id'], test['card_id']], ignore_index=True).value_counts(dropna=False))



test['card1_count_full'] = test['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))

test['card2_count_full'] = test['card2'].map(pd.concat([train['card2'], test['card2']], ignore_index=True).value_counts(dropna=False))

test['card3_count_full'] = test['card3'].map(pd.concat([train['card3'], test['card3']], ignore_index=True).value_counts(dropna=False))

test['card4_count_full'] = test['card4'].map(pd.concat([train['card4'], test['card4']], ignore_index=True).value_counts(dropna=False))

test['card5_count_full'] = test['card5'].map(pd.concat([train['card5'], test['card5']], ignore_index=True).value_counts(dropna=False))

test['card6_count_full'] = test['card6'].map(pd.concat([train['card6'], test['card6']], ignore_index=True).value_counts(dropna=False))

test['addr1_count_full'] = test['addr1'].map(pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False))

test['addr2_count_full'] = test['addr2'].map(pd.concat([train['addr2'], test['addr2']], ignore_index=True).value_counts(dropna=False))

test['P_emaildomain_count_full'] = test['P_emaildomain'].map(pd.concat([train['P_emaildomain'], test['P_emaildomain']], ignore_index=True).value_counts(dropna=False))

test['R_emaildomain_count_full'] = test['R_emaildomain'].map(pd.concat([train['R_emaildomain'], test['R_emaildomain']], ignore_index=True).value_counts(dropna=False))

test['uid_count_full'] = test['uid'].map(pd.concat([train['uid'], test['uid']], ignore_index=True).value_counts(dropna=False))

test['uid2_count_full'] = test['uid2'].map(pd.concat([train['uid2'], test['uid2']], ignore_index=True).value_counts(dropna=False))

test['uid3_count_full'] = test['uid3'].map(pd.concat([train['uid3'], test['uid3']], ignore_index=True).value_counts(dropna=False))

test['uid4_count_full'] = test['uid4'].map(pd.concat([train['uid4'], test['uid4']], ignore_index=True).value_counts(dropna=False))

test['uid5_count_full'] = test['uid5'].map(pd.concat([train['uid5'], test['uid5']], ignore_index=True).value_counts(dropna=False))

test['uid6_count_full'] = test['uid6'].map(pd.concat([train['uid6'], test['uid6']], ignore_index=True).value_counts(dropna=False))

test['uid7_count_full'] = test['uid7'].map(pd.concat([train['uid7'], test['uid7']], ignore_index=True).value_counts(dropna=False))

test['uid8_count_full'] = test['uid8'].map(pd.concat([train['uid8'], test['uid8']], ignore_index=True).value_counts(dropna=False))

test['card_id_count_full'] = test['card_id'].map(pd.concat([train['card_id'], test['card_id']], ignore_index=True).value_counts(dropna=False))
valid_card = train['TransactionAmt'].value_counts()

valid_card = valid_card[valid_card>10]

valid_card = list(valid_card.index)



train['TransactionAmt_check'] = np.where(train['TransactionAmt'].isin(test['TransactionAmt']), 1, 0)

test['TransactionAmt_check']  = np.where(test['TransactionAmt'].isin(train['TransactionAmt']), 1, 0)



i_cols = ['card1','card2','card3','card4','card5','card6','uid','uid2','uid3','uid4','uid5','uid6','uid7','uid8']



for col in i_cols:

    for agg_type in ['mean', 'std']:

        new_col_name = col+'_TransactionAmt_'+agg_type

        temp = pd.concat([train[[col, 'TransactionAmt']], test[[col,'TransactionAmt']]])

        temp = temp.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})

        

        temp.index = list(temp[col])

        temp = temp[new_col_name].to_dict()   

    

        train[new_col_name] = train[col].map(temp)

        test[new_col_name]  = test[col].map(temp)
train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')

train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')



train['TransactionAmt_to_mean_card_id'] = train['TransactionAmt'] / train.groupby(['card_id'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_std_card_id'] = train['TransactionAmt_to_mean_card_id'] / train.groupby(['card_id'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')



test['TransactionAmt_to_mean_card_id'] = test['TransactionAmt'] / test.groupby(['card_id'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_std_card_id'] = test['TransactionAmt_to_mean_card_id'] / test.groupby(['card_id'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_mean_card5'] = train['TransactionAmt'] / train.groupby(['card5'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_mean_addr1'] = train['TransactionAmt'] / train.groupby(['addr1'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_mean_id31'] = train['TransactionAmt'] / train.groupby(['id_31'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_mean_devicename'] = train['TransactionAmt'] / train.groupby(['device_name'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_std_card5'] = train['TransactionAmt'] / train.groupby(['card5'])['TransactionAmt'].transform('std')

train['TransactionAmt_to_std_addr1'] = train['TransactionAmt'] / train.groupby(['addr1'])['TransactionAmt'].transform('std')

train['TransactionAmt_to_std_id31'] = train['TransactionAmt'] / train.groupby(['id_31'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_mean_card5'] = test['TransactionAmt'] / test.groupby(['card5'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_mean_addr1'] = test['TransactionAmt'] / test.groupby(['addr1'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_mean_id31'] = test['TransactionAmt'] / test.groupby(['id_31'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_mean_devicename'] = test['TransactionAmt'] / test.groupby(['device_name'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_std_card5'] = test['TransactionAmt'] / test.groupby(['card5'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_std_addr1'] = test['TransactionAmt'] / test.groupby(['addr1'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_std_id31'] = test['TransactionAmt'] / test.groupby(['id_31'])['TransactionAmt'].transform('std')
train['Trans_min_mean'] = train['TransactionAmt'] - train['TransactionAmt'].mean()

train['Trans_min_std'] = train['Trans_min_mean'] / train['TransactionAmt'].std()



test['Trans_min_mean'] = test['TransactionAmt'] - test['TransactionAmt'].mean()

test['Trans_min_std'] = test['Trans_min_mean'] / test['TransactionAmt'].std()
train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)

test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)
train['TransactionAmt'] = np.log1p(train['TransactionAmt'])

test['TransactionAmt'] = np.log1p(test['TransactionAmt'])
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')



for df in [train, test]:

    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

    df['DT_M'] = (df['DT'].dt.year-2017)*12 + df['DT'].dt.month

    df['DT_W'] = (df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear

    df['DT_D'] = (df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear

    

    # D9 column

    df['D9'] = np.where(df['D9'].isna(),0,1)



periods = ['DT_M','DT_W','DT_D']

i_cols = ['card1','card2','card3','card4','card5','card6','uid','uid2','uid3','uid4','uid5','uid6','uid7','uid8']

for period in periods:

    for col in i_cols:

        new_column = col + '_' + period

            

        temp = pd.concat([train[[col,period]], test[[col,period]]])

        temp[new_column] = temp[col].astype(str) + '_' + (temp[period]).astype(str)

        fq_encode = temp[new_column].value_counts().to_dict()

            

        train[new_column] = (train[col].astype(str) + '_' + train[period].astype(str)).map(fq_encode)

        test[new_column]  = (test[col].astype(str) + '_' + test[period].astype(str)).map(fq_encode)
dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')

us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())



for df in [train, test]:

    df['is_december'] = df['DT'].dt.month

    df['is_december'] = (df['is_december']==12).astype(np.int8)



    df['is_holiday'] = (df['DT'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)
i_cols = ['D'+str(i) for i in range(1,16)]

uids = ['card1','card2','card3','card4','card5','card6','uid','uid2','uid3','uid4','uid5','uid6','uid7','uid8']



for df in [train, test]:

    for col in i_cols:

        df[col] = df[col].clip(0) 

    df['D9_not_na'] = np.where(df['D9'].isna(),0,1)

    df['D8_not_same_day'] = np.where(df['D8']>=1,1,0)

    df['D8_D9_decimal_dist'] = df['D8'].fillna(0)-df['D8'].fillna(0).astype(int)

    df['D8_D9_decimal_dist'] = ((df['D8_D9_decimal_dist']-df['D9'])**2)**0.5

    df['D8'] = df['D8'].fillna(-1).astype(int)
train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')

train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')

train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')

train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')



train['id_02_to_mean_card_id'] = train['id_02'] / train.groupby(['card_id'])['id_02'].transform('mean')

train['id_02_to_std_card_id'] = train['id_02_to_mean_card_id'] / train.groupby(['card_id'])['id_02'].transform('std')
test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')

test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')

test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')

test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')



test['id_02_to_mean_card_id'] = test['id_02'] / test.groupby(['card_id'])['id_02'].transform('mean')

test['id_02_to_std_card_id'] = test['id_02_to_mean_card_id'] / test.groupby(['card_id'])['id_02'].transform('std')
train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')

train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')

train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')

train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')

train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')



train['D15_to_mean_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('mean')

train['D15_to_std_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('std')



train['D15_to_mean_card_id'] = train['D15'] / train.groupby(['card_id'])['D15'].transform('mean')

train['D15_to_std_card_id'] = train['D15_to_mean_card_id'] / train.groupby(['card_id'])['D15'].transform('std')
test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')

test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')

test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')

test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')

test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')



test['D15_to_mean_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('mean')

test['D15_to_std_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('std')



test['D15_to_mean_card_id'] = test['D15'] / test.groupby(['card_id'])['D15'].transform('mean')

test['D15_to_std_card_id'] = test['D15_to_mean_card_id'] / test.groupby(['card_id'])['D15'].transform('std')
train['dist1_to_mean_card1'] = train['dist1'] / train.groupby(['card1'])['dist1'].transform('mean')

train['dist1_to_mean_card4'] = train['dist1'] / train.groupby(['card4'])['dist1'].transform('mean')

train['dist1_to_std_card1'] = train['dist1'] / train.groupby(['card1'])['dist1'].transform('std')

train['dist1_to_std_card4'] = train['dist1'] / train.groupby(['card4'])['dist1'].transform('std')



train['dist1_to_mean_addr1'] = train['dist1'] / train.groupby(['addr1'])['dist1'].transform('mean')

train['dist1_to_std_addr1'] = train['dist1'] / train.groupby(['addr1'])['dist1'].transform('std')



train['dist1_to_mean_card_id'] = train['dist1'] / train.groupby(['card_id'])['dist1'].transform('mean')

train['dist1_to_std_card_id'] = train['dist1_to_mean_card_id'] / train.groupby(['card_id'])['dist1'].transform('std')
test['dist1_to_mean_card1'] = test['dist1'] / test.groupby(['card1'])['dist1'].transform('mean')

test['dist1_to_mean_card4'] = test['dist1'] / test.groupby(['card4'])['dist1'].transform('mean')

test['dist1_to_std_card1'] = test['dist1'] / test.groupby(['card1'])['dist1'].transform('std')

test['dist1_to_std_card4'] = test['dist1'] / test.groupby(['card4'])['dist1'].transform('std')



test['dist1_to_mean_addr1'] = test['dist1'] / test.groupby(['addr1'])['dist1'].transform('mean')

test['dist1_to_std_addr1'] = test['dist1'] / test.groupby(['addr1'])['dist1'].transform('std')





test['dist1_to_mean_card_id'] = test['dist1'] / test.groupby(['card_id'])['dist1'].transform('mean')

test['dist1_to_std_card_id'] = test['dist1_to_mean_card_id'] / test.groupby(['card_id'])['dist1'].transform('std')
train['first_value_card1'] = train.loc[~train['card1'].isnull(), 'card1'].astype(str).str[0:1].astype(float)

train['two_value_card1'] = train.loc[~train['card1'].isnull(), 'card1'].astype(str).str[0:2].astype(float)

train['card2'] = train['card2'].fillna(0)

train['first_value_card2'] = train['card2'].astype(str).str[0:1].astype(float)

train['two_value_card2'] = train['card2'].astype(str).str[0:2].astype(float)
test['first_value_card1'] = test.loc[~test['card1'].isnull(), 'card1'].astype(str).str[0:1].astype(float)

test['two_value_card1'] = test.loc[~test['card1'].isnull(), 'card1'].astype(str).str[0:2].astype(float)

test['card2'] = test['card2'].fillna(0)

test['first_value_card2'] = test['card2'].astype(str).str[0:1].astype(float)

test['two_value_card2'] = test['card2'].astype(str).str[0:2].astype(float)
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum',

          'scranton.edu': 'other', 'netzero.net': 'other',

          'optonline.net': 'other', 'comcast.net': 'other', 

          'cfl.rr.com': 'other', 'sc.rr.com': 'other',

          'suddenlink.net': 'other', 'windstream.net': 'other',

          'gmx.de': 'other', 'earthlink.net': 'other', 

          'servicios-ta.com': 'other', 'bellsouth.net': 'other', 

          'web.de': 'other', 'mail.com': 'other',

          'cableone.net': 'other', 'roadrunner.com': 'other', 

          'protonmail.com': 'other', 'anonymous.com': 'other',

          'juno.com': 'other', 'ptd.net': 'other',

          'netzero.com': 'other', 'cox.net': 'other', 

          'hotmail.co.uk': 'microsoft', 

          'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 

          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 

          'live.com': 'microsoft', 'aim.com': 'aol',

          'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',

          'gmail.com': 'google', 'me.com': 'apple', 

          'hotmail.com': 'microsoft',  

          'hotmail.fr': 'microsoft',

          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 

          'yahoo.de': 'yahoo', 

          'live.fr': 'microsoft', 'verizon.net': 'yahoo', 

          'msn.com': 'microsoft', 'q.com': 'centurylink',

          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 

           'rocketmail.com': 'yahoo', 

          'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 

          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 

          'embarqmail.com': 'centurylink', 

          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo',

          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft',

           'aol.com': 'aol', 'icloud.com': 'apple'}



us_emails = ['gmail', 'net', 'edu']
for c in ['P_emaildomain', 'R_emaildomain']:

    train[c + '_bin'] = train[c].map(emails)

    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])

    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    

    test[c + '_bin'] = test[c].map(emails)

    test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

    test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
train['email_domain_comp'] = (train['P_emaildomain'].values == train['R_emaildomain'].values).astype(int)

train['email_domain_suffix_bin'] = (train['P_emaildomain_bin'].values == train['R_emaildomain_bin'].values).astype(int)

train['email_domain_suffix_comp'] = (train['P_emaildomain_suffix'].values == train['R_emaildomain_suffix'].values).astype(int)



test['email_domain_comp'] = (test['P_emaildomain'].values == test['R_emaildomain'].values).astype(int)

test['email_domain_suffix_bin'] = (test['P_emaildomain_bin'].values == test['R_emaildomain_bin'].values).astype(int)

test['email_domain_suffix_comp'] = (test['P_emaildomain_suffix'].values == test['R_emaildomain_suffix'].values).astype(int)
train['P_isprotonmail'] = 0

train.loc[train['P_emaildomain']=='protonmail.com', 'P_isprotonmail'] = 1

train['R_isprotonmail'] = 0

train.loc[train['R_emaildomain']=='protonmail.com', 'R_isprotonmail'] = 1



test['P_isprotonmail'] = 0

test.loc[test['P_emaildomain']=='protonmail.com', 'P_isprotonmail'] = 1

test['R_isprotonmail'] = 0

test.loc[test['R_emaildomain']=='protonmail.com', 'R_isprotonmail'] = 1
train['email_check_nan_all'] = np.where((train['P_emaildomain'].isna())&(train['R_emaildomain'].isna()),1,0)

test['email_check_nan_all']  = np.where((test['P_emaildomain'].isna())&(test['R_emaildomain'].isna()),1,0)



train['email_check_nan_any'] = np.where((train['P_emaildomain'].isna())|(train['R_emaildomain'].isna()),1,0)

test['email_check_nan_any']  = np.where((test['P_emaildomain'].isna())|(test['R_emaildomain'].isna()),1,0)



train = fix_emails(train)

test = fix_emails(test)
train['local_hour'] = train['D9']*24

test['local_hour']  = test['D9']*24



train['local_hour'] = train['local_hour'] - (train['TransactionDT']/(60*60))%24

test['local_hour']  = test['local_hour'] - (test['TransactionDT']/(60*60))%24



train['local_hour_dist'] = train['local_hour']/train['dist2']

test['local_hour_dist']  = test['local_hour']/test['dist2']
i_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']



train['M_sum'] = train[i_cols].sum(axis=1).astype(np.int8)

test['M_sum']  = test[i_cols].sum(axis=1).astype(np.int8)



train['M_na'] = train[i_cols].isna().sum(axis=1).astype(np.int8)

test['M_na']  = test[i_cols].isna().sum(axis=1).astype(np.int8)



train['M_type'] = ''

test['M_type']  = ''



for col in i_cols:

    train['M_type'] = '_'+train[col].astype(str)

    test['M_type'] = '_'+test[col].astype(str)



i_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']



for df in [train, test]:

    df['M_sum'] = df[i_cols].sum(axis=1).astype(np.int8)

    df['M_na'] = df[i_cols].isna().sum(axis=1).astype(np.int8)
i_cols = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']



train['C_sum'] = 0

test['C_sum']  = 0



train['C_null'] = 0

test['C_null']  = 0



for col in i_cols:

    train['C_sum'] += np.where(train[col]==1,1,0)

    test['C_sum']  += np.where(test[col]==1,1,0)



    train['C_null'] += np.where(train[col]==0,1,0)

    test['C_null']  += np.where(test[col]==0,1,0)

    

    valid_values = train[col].value_counts()

    valid_values = valid_values[valid_values>1000]

    valid_values = list(valid_values.index)

    

    train[col+'_valid'] = np.where(train[col].isin(valid_values),1,0)

    test[col+'_valid']  = np.where(test[col].isin(valid_values),1,0)
train = define_time(train)

test = define_time(test)
i_cols = ['card1','card2','card3','card4','card5','card6',

          'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',

          'D1','D2','D3','D4','D5','D6','D7','D8','D9',

          'addr1','addr2',

          'dist1','dist2',

          'P_emaildomain', 'R_emaildomain',

          'id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10',

          'id_11','id_13','id_14','id_17','id_18','id_19','id_20','id_21','id_22','id_24',

          'id_25','id_26','id_30','id_31','id_32','id_33',

          'DeviceInfo','DeviceInfo_c','id_30_c','id_30_v','id_31_v',

          'uid','uid2','uid3','uid4','uid5','uid6','uid7','uid8','card_id',

          'year','month','dow','hour','day',

          'DT_M','DT_W','DT_D',

         ]



for col in i_cols:

    temp = pd.concat([train[[col]], test[[col]]])

    fq_encode = temp[col].value_counts().to_dict()   

    train[col+'_fq_enc'] = train[col].map(fq_encode)

    test[col+'_fq_enc']  = test[col].map(fq_encode)
for f in train.select_dtypes(include='category').columns.tolist() + train.select_dtypes(include='object').columns.tolist():

    lbl = LabelEncoder()

    lbl.fit(list(train[f].astype(str).values) + list(test[f].astype(str).values))

    train[f] = lbl.transform(list(train[f].astype(str).values))

    test[f] = lbl.transform(list(test[f].astype(str).values)) 
one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]

one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]



cols_to_drop = list(set(

    one_value_cols+ one_value_cols_test

))



train.drop(cols_to_drop, axis=1, inplace=True)

test.drop(cols_to_drop, axis=1, inplace=True)
rm_cols = [

    'TransactionID',

    'TransactionDT',

    'DT','DT_M','DT_W','DT_D',

]

        

train = train.drop(rm_cols, axis=1)

test = test.drop(rm_cols, axis=1)
del train_transaction, train_identity, test_transaction, test_identity



X_train = train.drop('isFraud', axis=1)

y_train = train['isFraud'].copy()



X_test = test



del train, test
X_train = X_train.fillna(-999)

X_test = X_test.fillna(-999)
X_train = reduce_mem_usage(X_train)

X_test = reduce_mem_usage(X_test)
gc.collect()
#lgb_params = {

#       'num_leaves': int(419.6798218900234),

#       'min_child_weight': 0.025498779263792414, 

#       'feature_fraction': 0.22767740299561248,

#       'bagging_fraction': 0.10759876163848486,

#       'min_data_in_leaf': int(50.93596650321186),

#       'learning_rate': 0.0034567443022639277,

#       'reg_alpha': 0.030397185517009623,

#       'reg_lambda': 0.2865231477445093,

#        # constant

#       'objective': 'binary',

#       'max_depth': -1,

#       'boosting_type': 'gbdt',

#       'bagging_seed': 11,

#       'metric': 'auc',

#       'verbosity': -1,

#       'random_state': 47,

#   }
#splits = 5

#folds = KFold(n_splits = splits)

#oof = np.zeros(len(X_train))

#predictions = np.zeros(len(X_test))
#cv_score = []



#for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, y_train.values)):

#    print("\nFold {}".format(fold_))

#    X_fit, y_fit = X_train.iloc[trn_idx], y_train.iloc[trn_idx]

#    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

    

#    trn_data = lgb.Dataset(X_fit, label=y_fit)

#    val_data = lgb.Dataset(X_val, label=y_val)

    

#    lgb_clf = lgb.train(

#        lgb_params,

#        trn_data,

#        10000,

#        valid_sets = [trn_data, val_data],

#        verbose_eval=500,

#        early_stopping_rounds=500

#    )



#    pred = lgb_clf.predict(X_val)

#    oof[val_idx] = pred

#    roc_auc = roc_auc_score(y_val, pred)

#    print( "\tauc = ", roc_auc)

#    cv_score.append(roc_auc)

#    predictions += lgb_clf.predict(X_test) / splits
#print(cv_score)

#print('\nCV score : ', np.mean(cv_score))
#sample_submission = sample_submission.reset_index()

#sample_submission["isFraud"] = predictions

#sample_submission.to_csv("../submission/submission65.csv", index=False)