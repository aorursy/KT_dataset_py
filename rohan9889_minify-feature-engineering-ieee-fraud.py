import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
train_dataset = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

test_dataset = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
del train_identity

del train_transaction

del test_identity

del test_transaction
TransactionID = test_dataset['TransactionID']
test_dataset.drop(['TransactionID'],axis=1,inplace=True)
train_dataset.drop(['TransactionID'],axis=1,inplace=True)
y = train_dataset['isFraud']
train_dataset.drop(['isFraud'],axis=1,inplace=True)
# by https://www.kaggle.com/dimartinot



def clean_inf_nan(df):

    return df.replace([np.inf, -np.inf], np.nan)   



# Cleaning infinite values to NaN

train_dataset = clean_inf_nan(train_dataset)

test_dataset = clean_inf_nan(test_dataset)
list_of_num = []

list_of_obj = []

for i in train_dataset.columns :

    if train_dataset[i].dtypes == 'object':

        list_of_obj.append(i)

    else:

        list_of_num.append(i)
for i in list_of_num :

    train_dataset[i] = train_dataset[i].fillna(0)

    test_dataset[i] = test_dataset[i].fillna(0)
for i in list_of_obj :

    temp = str('no_'+i)

    train_dataset[i] = train_dataset[i].fillna(temp)

    test_dataset[i] = test_dataset[i].fillna(temp)
def minify_identity_df(df):

    df['M1']  = df['M1'].map({'T':2, 'F':1, 'no_M1':0})

    df['M2']  = df['M2'].map({'T':2, 'F':1, 'no_M2':0})

    df['M3']  = df['M3'].map({'T':2, 'F':1, 'no_M3':0})

    df['M5']  = df['M5'].map({'T':2, 'F':1, 'no_M5':0})

    df['M6']  = df['M6'].map({'T':2, 'F':1, 'no_M6':0})

    df['M7']  = df['M7'].map({'T':2, 'F':1, 'no_M7':0})

    df['M8']  = df['M8'].map({'T':2, 'F':1, 'no_M8':0})

    df['M9']  = df['M9'].map({'T':2, 'F':1, 'no_M9':0})

    df['id_12'] = df['id_12'].map({'Found':1, 'NotFound':2, 'no_id_12':0})

    df['id_15'] = df['id_15'].map({'New':3, 'Found':2, 'Unknown':1, 'no_id_15':0})

    df['id_16'] = df['id_16'].map({'Found':2, 'NotFound':1, 'no_id_16':0})

    df['id_23'] = df['id_23'].map({'IP_PROXY:TRANSPARENT':3, 'no_id_23':0,

                                   'IP_PROXY:ANONYMOUS':2, 'IP_PROXY:HIDDEN':1})

    df['id_27'] = df['id_27'].map({'Found':1, 'NotFound':2, 'no_id_27':0})

    df['id_28'] = df['id_28'].map({'New':2, 'Found':1, 'no_id_28':0})

    df['id_29'] = df['id_29'].map({'Found':1, 'NotFound':2, 'no_id_29':0})

    df['id_35'] = df['id_35'].map({'T':1, 'F':2, 'no_id_35':0})

    df['id_36'] = df['id_36'].map({'T':1, 'F':2, 'no_id_36':0})

    df['id_37'] = df['id_37'].map({'T':1, 'F':2, 'no_id_37':0})

    df['id_38'] = df['id_38'].map({'T':1, 'F':2, 'no_id_38':0})

    df['DeviceType'].map({'desktop':2, 'mobile':1, 'no_DeviceType':0})

    return df
#https://www.kaggle.com/kyakovlev/ieee-internal-blend

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

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



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
train_dataset =  minify_identity_df(train_dataset)

test_dataset = minify_identity_df(test_dataset)
train_dataset =  reduce_mem_usage(train_dataset)

test_dataset = reduce_mem_usage(test_dataset)
def convert_id31(val):

    try:

        val = val.lower()

        x = val.split()

        if 'chrome' in x:

            return 'chrome'

        elif 'safari' in x:

            return 'safari'

        elif 'ie' in x:

            return 'ie'

        elif 'edge' in x:

            return 'edge'

        elif 'firefox' in x:

            return 'firefox'

        elif 'samsung' in x:

            return 'samsung'

        else:

            return 'no_id_31'

    except:

        return 'no_id_31'
train_dataset['id_31'] = train_dataset['id_31'].apply(convert_id31)

test_dataset['id_31'] = test_dataset['id_31'].apply(convert_id31)
def convert_id30(val):

            try:

                val = val.lower()

                x = val.split()[0]

                if x == 'ios':

                    return 'ios'

                elif x == 'android':

                    return 'android'

                elif x == 'mac':

                    return 'mac'

                elif x == 'windows':

                    return 'windows'

                elif x == 'linux':

                    return 'linux'

                else:

                    return 'no_id_30'

            except:

                return 'no_id_30'
train_dataset['id_30'] = train_dataset['id_30'].apply(convert_id30)

test_dataset['id_30'] = test_dataset['id_30'].apply(convert_id30)
for X in [train_dataset,test_dataset]:

    X['device_name'] = X['DeviceInfo'].str.split('/', expand=True)[0]



    X.loc[X['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'

    X.loc[X['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'

    X.loc[X['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'

    X.loc[X['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'

    X.loc[X['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'

    X.loc[X['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'

    X.loc[X['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'

    X.loc[X['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'

    X.loc[X['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'

    X.loc[X['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'

    X.loc[X['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'

    X.loc[X['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'

    X.loc[X['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'

    X.loc[X['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'

    X.loc[X['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'

    X.loc[X['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

    X.loc[X['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    X.loc[X.device_name.isin(X.device_name.value_counts()[X.device_name.value_counts() < 200].index), 'device_name'] = "Others"

    X.drop(['DeviceInfo'],axis=1,inplace=True)
temp = train_dataset['P_emaildomain'].str.split('.', expand=True)

train_dataset['P_emailF'] = temp[0]

train_dataset['P_emailL'] = temp[1]

temp = train_dataset['R_emaildomain'].str.split('.', expand=True)

train_dataset['R_emailF'] = temp[0]

train_dataset['R_emailL'] = temp[1]

temp = test_dataset['P_emaildomain'].str.split('.', expand=True)

test_dataset['P_emailF'] = temp[0]

test_dataset['P_emailL'] = temp[1]

temp = test_dataset['R_emaildomain'].str.split('.', expand=True)

test_dataset['R_emailF'] = temp[0]

test_dataset['R_emailL'] = temp[1]

del temp

del train_dataset['P_emaildomain']

del test_dataset['P_emaildomain']

del train_dataset['R_emaildomain']

del test_dataset['R_emaildomain']
train_dataset['P_emailF'] = train_dataset['P_emailF'].fillna('no_P_F')

train_dataset['P_emailL'] = train_dataset['P_emailL'].fillna('no_P_L')

train_dataset['R_emailF'] = train_dataset['R_emailF'].fillna('no_R_F')

train_dataset['R_emailL'] = train_dataset['R_emailL'].fillna('no_R_L')



test_dataset['P_emailF'] = test_dataset['P_emailF'].fillna('no_P_F')

test_dataset['P_emailL'] = test_dataset['P_emailL'].fillna('no_P_L')

test_dataset['R_emailF'] = test_dataset['R_emailF'].fillna('no_R_F')

test_dataset['R_emailL'] = test_dataset['R_emailL'].fillna('no_R_L')
from sklearn.preprocessing import LabelEncoder
for i in train_dataset.select_dtypes(include=['category', 'object']).columns.values:

        encoder = LabelEncoder()

        encoder.fit(list(train_dataset[i].values) + list(test_dataset[i].values))

        train_dataset[i] = encoder.transform(list(train_dataset[i].values))

        test_dataset[i] = encoder.transform(list(test_dataset[i].values))
import datetime

START_DATE = '1800-01-01'

startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
# code from https://www.kaggle.com/kimchiwoong/simple-eda-ensemble-for-xgboost-and-lgbm

train_dataset["Date"] = train_dataset['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

train_dataset['TransactionDT_Weekdays'] = train_dataset['Date'].dt.dayofweek

train_dataset['TransactionDT_Days'] = train_dataset['Date'].dt.day

train_dataset['TransactionDT_Hours'] = train_dataset['Date'].dt.hour

train_dataset.drop(columns='Date', inplace=True)



test_dataset["Date"] = test_dataset['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

test_dataset['TransactionDT_Weekdays'] = test_dataset['Date'].dt.dayofweek

test_dataset['TransactionDT_Days'] = test_dataset['Date'].dt.day

test_dataset['TransactionDT_Hours'] = test_dataset['Date'].dt.hour

test_dataset.drop(columns='Date', inplace=True)

train_dataset.drop(['TransactionDT'],axis=1,inplace=True)

test_dataset.drop(['TransactionDT'],axis=1,inplace=True)
def quasi_constant(data):

    quasi = [col for col in train_dataset.columns if train_dataset[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

    return quasi
quasi_features = quasi_constant(train_dataset)
train_dataset.drop(quasi_features,axis=1,inplace=True)

test_dataset.drop(quasi_features,axis=1,inplace=True)
train_dataset =  reduce_mem_usage(train_dataset)

test_dataset = reduce_mem_usage(test_dataset)
train_dataset['isFraud'] = y
train_dataset.to_pickle('Train.pkl')

test_dataset.to_pickle('Test.pkl')