# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler 

from sklearn.metrics import confusion_matrix, classification_report, f1_score



import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.
os.listdir('/kaggle/input')
!python3 /kaggle/input/itis-hackathon/make_dataset_kaggle.py /kaggle/input/itis-hackathon/
interim_test = pd.read_csv('/kaggle/working/interim_test.csv')

interim_train_09 = pd.read_csv('/kaggle/working/interim_train_09.csv')

interim_train_10 = pd.read_csv('/kaggle/working/interim_train_10.csv')
balance_train_09 = pd.read_csv('/kaggle/input/itis-hackathon/train_09_raw_feat/balance_train_09.csv')

balance_train_10 = pd.read_csv('/kaggle/input/itis-hackathon/train_10_raw_feat/balance_train_10.csv')

balance_test = pd.read_csv('/kaggle/input/itis-hackathon/test_raw_feat/balance_test.csv')
tariff_train_09 = pd.read_csv('/kaggle/input/itis-hackathon/train_09_raw_feat/tariff_train_09.csv')

tariff_train_10 = pd.read_csv('/kaggle/input/itis-hackathon/train_10_raw_feat/tariff_train_10.csv')

tariff_test = pd.read_csv('/kaggle/input/itis-hackathon/test_raw_feat/tariff_test.csv')
payments_train_09 = pd.read_csv('/kaggle/input/itis-hackathon/train_09_raw_feat/payments_train_09.csv')

payments_train_10 = pd.read_csv('/kaggle/input/itis-hackathon/train_10_raw_feat/payments_train_10.csv')

payments_test = pd.read_csv('/kaggle/input/itis-hackathon/test_raw_feat/payments_test.csv')
interim_train_09
#interim_train_09 = interim_train_09.drop(columns=['ID', 'SERVICE_INT_ID', 'ADMIN_QUESTION_INT_ID', 'FEATURE_INT_ID', 'CHANNEL_INT_ID',

#                         'BAL_BELOW_ZERO','BALANCE_DOWNTIME','PROMISED_PAYMENT'])
#interim_train_10 = interim_train_10.drop(columns=['ID', 'SERVICE_INT_ID', 'ADMIN_QUESTION_INT_ID', 'FEATURE_INT_ID', 'CHANNEL_INT_ID',

#                         'BAL_BELOW_ZERO','BALANCE_DOWNTIME','PROMISED_PAYMENT'])
#interim_test = interim_test.drop(columns=['ACTIVATE_DATE', 'PHYZ_TYPE',

#                                                  'CITY_NAME'])

columns = ['TRAFFIC_0_In',

 'TRAFFIC_0_Out',

 'TRAFFIC_1_In',

 'TRAFFIC_1_Out',

 'TRAFFIC_2_In',

 'TRAFFIC_2_Out',

 'TRAFFIC_3_In',

 'TRAFFIC_3_Out',

 'TRAFFIC_4_In',

 'TRAFFIC_4_Out',

 'TRAFFIC_5_In',

 'TRAFFIC_5_Out']
def null_PHYZ_TYPE(df):

    df['PHYZ_TYPE'] = df['PHYZ_TYPE'].fillna(df['PHYZ_TYPE'].value_counts().keys()[0])

    return df
def del_null_big(df):

    df = df.drop(['ID', 'SERVICE_INT_ID', 'ADMIN_QUESTION_INT_ID', 'FEATURE_INT_ID', 'CHANNEL_INT_ID',

                         'BAL_BELOW_ZERO','BALANCE_DOWNTIME','PROMISED_PAYMENT'],axis=1)

    return df



def fillna_null(df):

    Ethernet100 = df['PHYZ_TYPE'] == 'Ethernet 100M'

    GPON = df['PHYZ_TYPE']        == 'GPON'

    GePON = df['PHYZ_TYPE']       == 'GePON'

    Ethernet1 = df['PHYZ_TYPE']   == 'Ethernet 1G'

    ADSL = df['PHYZ_TYPE']        == 'ADSL'

    

    df.loc[Ethernet100, columns] =  df.loc[Ethernet100,columns].apply(lambda x: x.fillna(x.median()),axis=0)

    df.loc[GPON, columns] =  df.loc[GPON,columns].apply(lambda x: x.fillna(x.median()),axis=0)

    df.loc[GePON, columns] =  df.loc[GePON,columns].apply(lambda x: x.fillna(x.median()),axis=0)

    df.loc[Ethernet1, columns] =  df.loc[Ethernet1,columns].apply(lambda x: x.fillna(x.median()),axis=0)

    df.loc[ADSL, columns] =  df.loc[ADSL,columns].apply(lambda x:  x.fillna(x.median()),axis=0)

    return df                                                                         
def new_AVG_TRAFFIC(df):

    df['AVG_TRAFFIC_In'] = (df['TRAFFIC_0_In'] + df['TRAFFIC_1_In'] + df['TRAFFIC_2_In'] + df['TRAFFIC_3_In'] + df['TRAFFIC_4_In'] + df['TRAFFIC_5_In'])/6 

    df['AVG_TRAFFIC_Out'] = (df['TRAFFIC_0_Out'] + df['TRAFFIC_1_Out'] + df['TRAFFIC_2_Out'] + df['TRAFFIC_3_Out'] + df['TRAFFIC_4_Out'] + df['TRAFFIC_5_Out'])/6

    

    return df
def new_cat_TRAFFIC(df):

    bins = [0]

    bins = np.append(bins, df.describe()['AVG_TRAFFIC_In'][4:-1].values)

    bins = np.append(bins, df['AVG_TRAFFIC_In'].max())

    df['Category_In'] = pd.cut(df['AVG_TRAFFIC_In'], bins=bins, labels=['Low', 'Mid', 'High_Mid', 'High'])

    

    bins = [0]

    bins = np.append(bins, df.describe()['AVG_TRAFFIC_Out'][4:-1].values)

    bins = np.append(bins, df['AVG_TRAFFIC_Out'].max())

    df['Category_Out'] = pd.cut(df['AVG_TRAFFIC_Out'], bins=bins, labels=['Low', 'Mid', 'High_Mid', 'High'])

    

    return df

        
def Category_null(df):

    df['Category_In'] = df['Category_In'].fillna(df['Category_In'].value_counts().keys()[0])

    df['Category_Out'] = df['Category_Out'].fillna(df['Category_Out'].value_counts().keys()[0])

    df['Category_In'] = df['Category_In'].astype('object')

    df['Category_Out'] = df['Category_Out'].astype('object')

    

    return df
def Date(df):

    df['ACTIVATE_DATE'] = pd.to_datetime(df['ACTIVATE_DATE'])

    df['year'] = df.ACTIVATE_DATE.dt.year

    df['month'] = df.ACTIVATE_DATE.dt.month

    df['day'] = df.ACTIVATE_DATE.dt.day

    df = df.drop('ACTIVATE_DATE',axis=1)

    

    return df
def one_hot_encoder(df, nan_as_category = True):

    original_columns = list(df.columns)

    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']

    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)

    new_columns = [c for c in df.columns if c not in original_columns]

    return df, new_columns
print(interim_train_09.shape, interim_train_10.shape, interim_test.shape)

train = null_PHYZ_TYPE(interim_train_09)

valid = null_PHYZ_TYPE(interim_train_10)

test = null_PHYZ_TYPE(interim_test)

print(interim_train_09.shape, interim_train_10.shape, interim_test.shape)
print(train.shape, valid.shape, test.shape)

train = del_null_big(train)

valid = del_null_big(valid)

test = test.drop(['BAL_BELOW_ZERO','BALANCE_DOWNTIME','PROMISED_PAYMENT'],axis=1)

print(train.shape, valid.shape, test.shape)
print(train.shape, valid.shape, test.shape)

train = fillna_null(train)

valid = fillna_null(valid)

test = fillna_null(test)

print(train.shape, valid.shape, test.shape)
print(train.shape, valid.shape, test.shape)

train = new_AVG_TRAFFIC(train)

valid = new_AVG_TRAFFIC(valid)

test = new_AVG_TRAFFIC(test)

print(train.shape, valid.shape, test.shape)
print(train.shape, valid.shape, test.shape)

train = new_cat_TRAFFIC(train)

valid = new_cat_TRAFFIC(valid)

test = new_cat_TRAFFIC(test)

print(train.shape, valid.shape, test.shape)
print(train.shape, valid.shape, test.shape)

train = Category_null(train)

valid = Category_null(valid)

test = Category_null(test)

print(train.shape, valid.shape, test.shape)
print(train.shape, valid.shape, test.shape)

train = Date(train)

valid = Date(valid)

test = Date(test)

print(train.shape, valid.shape, test.shape)
print(train.shape, valid.shape, test.shape)

train, tr_col = one_hot_encoder(train)

valid, v_col = one_hot_encoder(valid)

test, te_col  = one_hot_encoder(test)

print(train.shape, valid.shape, test.shape)
def balance(df):

    print(df.shape)

    df['CHANGE_DATE'] = pd.to_datetime(df['CHANGE_DATE'])

    df['day'] = df.CHANGE_DATE.dt.day

    df = df.drop('CHANGE_DATE',axis=1)

    df['day'] =  df['day'].astype(str) 

    df, cat_b_9 = one_hot_encoder(df)

    

    agg_balance = {'BALANCE':['mean','median'],

                  'day_1':['count'],'day_2':['count'],'day_3':['count'],'day_4':['count'],'day_5':['count'],

                  'day_6':['count'],'day_7':['count'],'day_8':['count'],'day_9':['count'],'day_10':['count'],

                  'day_11':['count'],'day_12':['count'],'day_13':['count'],'day_14':['count'],'day_15':['count'],

                  'day_16':['count'],'day_17':['count'],'day_18':['count'],'day_19':['count'],'day_20':['count'],

                  'day_21':['count'],'day_22':['count'],'day_23':['count'],'day_24':['count']}

    

    df_agg = df.groupby('USER_ID').agg(agg_balance)

    

    df_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in  df_agg.columns.tolist()])

    df_09 = df_agg.reset_index()

    return df_09
balance_train_09 = balance(balance_train_09)

balance_train_10 = balance(balance_train_10)

balance_test = balance(balance_test)
train = train.merge(balance_train_09, how='left')

valid = valid.merge(balance_train_10, how='left')

test = test.merge(balance_test, how='left')
def agg_tariff(df):

    print(df.shape)

    df['BTIME'] = pd.to_datetime(df['BTIME'])

    df['year_B'] = df.BTIME.dt.year

    df['month_B'] = df.BTIME.dt.month

    df['day_B'] = df.BTIME.dt.day

    

    df = df.drop('BTIME',axis=1)

    

    df['year_B'] =  df['year_B'].astype(str)

    df['month_B'] =  df['month_B'].astype(str) 

    df['day_B'] =  df['day_B'].astype(str) 

    

    df['ETIME'] = pd.to_datetime(df['ETIME'],errors='coerce')

    df['year_E'] = df.ETIME.dt.year

    df['month_E'] = df.ETIME.dt.month

    df['day_E'] = df.ETIME.dt.day

    

    df = df.drop('ETIME',axis=1)

    

    df['year_E'] =  df['year_E'].astype(str)

    df['month_E'] =  df['month_E'].astype(str) 

    df['day_E'] =  df['day_E'].astype(str) 

    

    dic = {}

    for col in [col for col in df.columns if ('day' in col) or ('month' in col) or ('year' in col)]:

        dic[str(col)] = ['count']

    dic['MONTH_PAY']  = ['median']   

    df_agg = df.groupby('USER_ID').agg(dic)

    

    df_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in  df_agg.columns.tolist()])

    df = df_agg.reset_index()

    print(df.shape)

    return df    
tariff_train_09 = agg_tariff(tariff_train_09)

tariff_train_10 = agg_tariff(tariff_train_10)

tariff_test = agg_tariff(tariff_test)
train = train.merge(tariff_train_09, how='left')

valid = valid.merge(tariff_train_10, how='left')

test = test.merge(tariff_test, how='left',on='USER_ID')
def payment_agg(df):

    df['NEW_PAY_DATE'] = pd.to_datetime(df['NEW_PAY_DATE'])

    df['month'] = df.NEW_PAY_DATE.dt.month

    df['day'] = df.NEW_PAY_DATE.dt.day

    df = df.drop('NEW_PAY_DATE',axis=1)



    

    dict_agg = {}

    dict_agg['NEW_VALUE'] = ['median']

    dict_agg['month'] = ['count']

    dict_agg['day'] = ['count']

    

    df_agg = df.groupby('USER_ID').agg(dict_agg)

    df_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in  df_agg.columns.tolist()])

    df = df_agg.reset_index()

    

    return df
payments_train_09 = payment_agg(payments_train_09)

payments_train_10 = payment_agg(payments_train_10)

payments_test = payment_agg(payments_test)
train = train.merge(payments_train_09, how='left')

valid = valid.merge(payments_train_10, how='left')

test = test.merge(payments_test, how='left',on='USER_ID')
train = train.fillna(0)

valid = valid.fillna(0)

test =  test.fillna(0)
interim_train_09.fillna(0, inplace=True)

interim_train_10.fillna(0, inplace=True)

interim_test.fillna(0, inplace=True)
def align_data(train: pd.DataFrame, test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

    """Согласование признаков у train и test датасетов

    

    Arguments:

        train {pd.DataFrame} -- train датасет

        test {pd.DataFrame} -- test датасет

    Returns:

        train {pd.DataFrame}, test {pd.DataFrame} - датасеты с одинаковыми признаками

    """

    intersect_list = np.intersect1d(train.columns, test.columns)

    if "TARGET" not in intersect_list:

        train = train[np.append(intersect_list, "TARGET")]

    else:

        train = train[intersect_list]

    test = test[intersect_list]

    return train, test
train_09, test_10 = align_data(train, valid)
train_09.shape, test_10.shape
def fit_and_pred_logreg(train, test):

    """Fit and predict LogisticRegression

    

    Arguments:

        train {pd.DataFrame} -- processed train dataset

        test {pd.DataFrame} -- processed test dataset

    

    Returns:

        model {sklearn.BaseEstimator} -- fit sklearn model

        y_pred {np.array} -- predictions

    """

    model = LogisticRegression(class_weight="balanced", random_state=17, n_jobs=-1)

    x_train = train.drop(columns=["TARGET"])

    y_train = train.TARGET

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)

    x_test = scaler.transform(test)



    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)



    return model, y_pred
y_test = test_10.TARGET

model, y_pred = fit_and_pred_logreg(train_09, test_10.drop(columns='TARGET'))
train_09
print(classification_report(y_true=y_test, y_pred=y_pred))

print(confusion_matrix(y_true=y_test, y_pred=y_pred))
train_10, test = align_data(test_10, test)
test = test.fillna(0)
model, y_pred = fit_and_pred_logreg(train_10, test)
y_pred
interim_test['PREDICT'] = y_pred

interim_test[['USER_ID', 'PREDICT']].to_csv('baseline_submission.csv', index=False) # В папке output. Выгрузить ручками