import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



#df_members = pd.read_csv('../input/members_v3.csv')

df_transactions = pd.read_csv('../input/transactions.csv')
print(df_transactions.shape)

df_transactions.head()
mem = df_transactions.memory_usage(index=True).sum()

print(mem/ 1024**2," MB")
def change_datatype(df):

    int_cols = list(df.select_dtypes(include=['int']).columns)

    for col in int_cols:

        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):

            df[col] = df[col].astype(np.int8)

        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):

            df[col] = df[col].astype(np.int16)

        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):

            df[col] = df[col].astype(np.int32)

        else:

            df[col] = df[col].astype(np.int64)



change_datatype(df_transactions)



def change_datatype_float(df):

    float_cols = list(df.select_dtypes(include=['float']).columns)

    for col in float_cols:

        df[col] = df[col].astype(np.float32)

        

change_datatype_float(df_transactions)



mem = df_transactions.memory_usage(index=True).sum()

print(mem/ 1024**2," MB")
#--- Members dataframe

mem = df_members.memory_usage(index=True).sum()

print(mem/ 1024**2," MB")



change_datatype(df_members)

change_datatype_float(df_members)



#--- Recheck memory of Members dataframe

mem = df_members.memory_usage(index=True).sum()

print(mem/ 1024**2," MB")
print(df_transactions.dtypes, '\n')

print(df_members.dtypes)
len(df_transactions.columns)
df_transactions['discount'] = df_transactions['plan_list_price'] - df_transactions['actual_amount_paid']



df_transactions['discount'].unique()
df_transactions['is_discount'] = df_transactions.discount.apply(lambda x: 1 if x > 0 else 0)

print(df_transactions['is_discount'].head())

print(df_transactions['is_discount'].unique())
df_transactions['amt_per_day'] = df_transactions['actual_amount_paid'] / df_transactions['payment_plan_days']

df_transactions['amt_per_day'].head()
date_cols = ['transaction_date', 'membership_expire_date']

print(df_transactions[date_cols].dtypes)

for col in date_cols:

    df_transactions[col] = pd.to_datetime(df_transactions[col], format='%Y%m%d')

    

df_transactions.head()
#--- difference in days ---

df_transactions['membership_duration'] = df_transactions.membership_expire_date - df_transactions.transaction_date

df_transactions['membership_duration'] = df_transactions['membership_duration'] / np.timedelta64(1, 'D')

df_transactions['membership_duration'] = df_transactions['membership_duration'].astype(int)



 

#---difference in months ---

#df_transactions['membership_duration_M'] = (df_transactions.membership_expire_date - df_transactions.transaction_date)/ np.timedelta64(1, 'M')

#df_transactions['membership_duration_M'] = round(df_transactions['membership_duration_M']).astype(int)

#df_transactions['membership_duration_M'].head()

len(df_transactions.columns)
change_datatype(df_transactions)

change_datatype_float(df_transactions)
df_members.head()
#--- Number of columns 

len(df_members.columns)
date_cols = ['registration_init_time', 'expiration_date']



for col in date_cols:

    df_members[col] = pd.to_datetime(df_members[col], format='%Y%m%d')
#--- difference in days ---

df_members['registration_duration'] = df_members.expiration_date - df_members.registration_init_time

df_members['registration_duration'] = df_members['registration_duration'] / np.timedelta64(1, 'D')

df_members['registration_duration'] = df_members['registration_duration'].astype(int)



#---difference in months ---

#df_members['registration_duration_M'] = (df_members.expiration_date - df_members.registration_init_time)/ np.timedelta64(1, 'M')

#df_members['registration_duration_M'] = round(df_members['registration_duration_M']).astype(int)

#--- Reducing and checking memory again ---

change_datatype(df_members)

change_datatype_float(df_members)



#--- Recheck memory of Members dataframe

mem = df_members.memory_usage(index=True).sum()

print(mem/ 1024**2," MB")
#-- merging the two dataframes---

df_comb = pd.merge(df_transactions, df_members, on='msno', how='inner')



#--- deleting the dataframes to save memory

del df_transactions

del df_members



df_comb.head()
#df_comb = df_comb.drop('msno', 1)

mem = df_comb.memory_usage(index=True).sum()

print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))
df_comb['reg_mem_duration'] = df_comb['registration_duration'] - df_comb['membership_duration']

#df_comb['reg_mem_duration_M'] = df_comb['registration_duration_M'] - df_comb['membership_duration_M']

df_comb.head()
df_comb['autorenew_&_not_cancel'] = ((df_comb.is_auto_renew == 1) == (df_comb.is_cancel == 0)).astype(np.int8)

df_comb['autorenew_&_not_cancel'].unique()
df_comb['notAutorenew_&_cancel'] = ((df_comb.is_auto_renew == 0) == (df_comb.is_cancel == 1)).astype(np.int8)

df_comb['notAutorenew_&_cancel'].unique()
df_comb['long_time_user'] = (((df_comb['registration_duration'] / 365).astype(int)) > 1).astype(int)
datetime_cols = list(df_comb.select_dtypes(include=['datetime64[ns]']).columns)
df_comb = df_comb.drop([datetime_cols], 1)