import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib.pyplot import xticks, figure

import os

print(os.listdir("../input"))



from subprocess import check_output



# Any results you write to the current directory are saved as output.
print(check_output(['ls', '../input']).decode('utf8'))
transaction_df = pd.read_csv('../input/transactions_v2.csv')

transaction_df.head()
transaction_df.duplicated(subset='msno').sum()
transaction_df.isna().sum()
figure(num=None, figsize=(16,8), dpi=60)

sns.countplot(x='payment_method_id', data=transaction_df)

xticks(rotation=90)
transaction_df.payment_method_id = transaction_df.payment_method_id.replace([2,3,5,6,8,10,11,12,14,16,17,18,19,21,23,24,25,26], 1)
transaction_df.payment_method_id = pd.factorize(transaction_df.payment_method_id)[0]
figure(num=None, figsize=(16,8), dpi=60)

sns.countplot(x='payment_method_id', data=transaction_df)

xticks(rotation=90)
transaction_df.payment_plan_days.describe()
sns.boxplot(x='payment_plan_days', data=transaction_df)
transaction_df.payment_plan_days.quantile([0.05,0.98])
transaction_df.plan_list_price.describe()
sns.boxplot(x='plan_list_price', data=transaction_df)
transaction_df.plan_list_price.quantile([0.05,0.98])
sns.countplot(x='is_auto_renew', data=transaction_df)
dublicate_msno_df = transaction_df[transaction_df.duplicated(subset='msno')]

dublicated_msno = dublicate_msno_df.msno.unique()
len(dublicated_msno)
transaction_group = transaction_df.groupby('msno')
new_transaction_df = pd.DataFrame(data=None, columns=['msno','payment_mode','total_plan_days','total_plan_price','total_ammount_paid','last_plan_days','last_plan_price',

                                                     'is_auto_renewal','first_transaction_date','last_transaction_date','membership_expire_date','is_cancel','no_of_record'])
for msno in dublicated_msno:

    group_df = transaction_group.get_group(msno)

    payment_mode = group_df.payment_method_id.mode()[0]

    total_plan_days = group_df.payment_plan_days.sum()

    total_plan_price = group_df.plan_list_price.sum()

    total_ammount_paid = group_df.actual_amount_paid.sum()

    is_auto_renewal = group_df.is_auto_renew.mode()[0]

    first_transaction_date = group_df.transaction_date.min()

    last_transaction_date = group_df.transaction_date.max()

    membership_expire_date = group_df.membership_expire_date.max()

    last_date = group_df[group_df.transaction_date==last_transaction_date]

    last_plan_days = 0

    last_plan_price = 0

    if len(last_date) > 0:

        last_plan_days = last_date['payment_plan_days'].values.max()

        last_plan_price = last_date['plan_list_price'].values.max()

    

    cancled = group_df.is_cancel.values

    is_cancle = 1 if 1 in cancled else 0

    no_of_record = len(group_df)

    pointer = len(new_transaction_df)

    new_transaction_df.loc[pointer] = [msno, payment_mode, total_plan_days, total_plan_price, total_ammount_paid, last_plan_days, last_plan_price,

                                       is_auto_renewal, first_transaction_date, last_transaction_date, membership_expire_date,

                                       is_cancle, no_of_record]
new_transaction_df.head()
print(check_output(['ls', '.']).decode('utf8'))
import os

os.mkdir('data')
new_transaction_df.to_csv('data/new_transaction_v2.csv', index=False)
unique_msno_df = transaction_df[~transaction_df.msno.isin(dublicated_msno)]

new_transaction_df_1 = new_transaction_df
unique_msno_df.duplicated(subset='msno').sum()
len(unique_msno_df)
rename_columns = {'payment_method_id': 'payment_mode', 'payment_plan_days': 'total_plan_days', 'plan_list_price': 'total_plan_price',

          'actual_amount_paid': 'total_ammount_paid', 'transaction_date': 'first_transaction_date','is_auto_renew': 'is_auto_renewal'}

unique_msno_df['last_transaction_date'] = unique_msno_df.transaction_date

unique_msno_df['last_plan_days'] = unique_msno_df['payment_plan_days']

unique_msno_df['last_plan_price'] = unique_msno_df['plan_list_price']

unique_msno_df['no_of_record'] = 1

unique_msno_df = unique_msno_df.rename(columns=rename_columns)
unique_msno_df.head()
new_transaction_df.head()
unique_msno_df.to_csv('data/unique_mnso.csv', index=False)
new_transaction_df_1 = new_transaction_df_1.append(unique_msno_df, ignore_index=True, sort=False)
new_transaction_df_1.to_csv('data/new_transaction_df_1.csv', index=False)
del(new_transaction_df)

del(unique_msno_df)
member_df = pd.read_csv('../input/members_v3.csv')

member_df.head()
member_df.duplicated(subset='msno').sum()
member_df.isna().sum() / len(member_df) * 100
member_df.gender.describe()
sns.countplot(x='gender', data=member_df)
member_details_df = new_transaction_df_1.merge(member_df, how='left', on='msno')
member_details_df.head()
member_details_df.first_transaction_date = pd.to_datetime(member_details_df.first_transaction_date, format='%Y%m%d') 

member_details_df.last_transaction_date = pd.to_datetime(member_details_df.last_transaction_date, format='%Y%m%d')

member_details_df.membership_expire_date = pd.to_datetime(member_details_df.membership_expire_date, format='%Y%m%d')

member_details_df.registration_init_time = pd.to_datetime(member_details_df.registration_init_time, format='%Y%m%d.0')
del(new_transaction_df_1)

del(member_df)
member_details_df.isna().sum() / len(member_details_df) * 100
member_details_df.gender.describe()
member_details_df = member_details_df.drop(columns='gender')
member_details_df.city.describe()
figure(figsize=(10,6))

sns.countplot('city', data=member_details_df)

xticks(rotation=90)
member_details_df.city = member_details_df.city.fillna(1.)
figure(figsize=(10,6))

sns.countplot('city', data=member_details_df)

xticks(rotation=90)
member_details_df.bd.describe()
sns.boxplot(x='bd', data=member_details_df)
percentiles = member_details_df.bd.quantile([0.51,0.999]).values

member_details_df.loc[member_details_df.bd < percentiles[0], 'bd'] = percentiles[0]

member_details_df.loc[member_details_df.bd > percentiles[1], 'bd'] = percentiles[1]
percentiles
sns.boxplot(x='bd', data=member_details_df)
member_details_df.bd.fillna(member_details_df.bd.mean(), inplace=True)
member_details_df.registered_via.describe()
sns.countplot(x='registered_via', data=member_details_df)
member_details_df.registered_via.fillna(7., inplace=True)
sns.countplot(x='registered_via', data=member_details_df)
member_details_df.registration_init_time.describe()
print(member_details_df.registration_init_time.min())

print(member_details_df.registration_init_time.max())
date_ftd = member_details_df.first_transaction_date

date_reginit = member_details_df.registration_init_time



total_years = ((date_ftd - date_reginit) / np.timedelta64(1,'Y')).dropna().values

print(sum(total_years) / len(total_years))

del(date_ftd)

del(date_reginit)
from dateutil.relativedelta import relativedelta
temp_init_time = member_details_df.first_transaction_date - pd.Timedelta(days=365*3.44)
member_details_df.registration_init_time = member_details_df.registration_init_time.fillna(temp_init_time).dt.date

# del(temp_init_time)
member_details_df.registration_init_time = pd.to_datetime(member_details_df.registration_init_time)
member_details_df.isna().sum() / len(member_details_df) * 100
member_details_df.to_csv('data/member_details_df_clean.csv', index=False)
member_details_df.dtypes
# membership_expire_in = (member_details_df.membership_expire_date - member_details_df.last_transaction_date) / np.timedelta64(1, 'M')

# membership_expire_in = membership_expire_in.round().astype(int)

# member_details_df['membership_expire_in'] = membership_expire_in

# del(membership_expire_in)
diff_init_first_trans =  (member_details_df.first_transaction_date - member_details_df.registration_init_time) / np.timedelta64(1, 'Y')

diff_init_first_trans = diff_init_first_trans.round().astype(int)

member_details_df['diff_init_first_trans'] = diff_init_first_trans

del(diff_init_first_trans)
average_plan_days = member_details_df.total_plan_days / member_details_df.no_of_record

average_plan_days = average_plan_days.astype(float).round(0).astype(int)

member_details_df['average_plan_days'] = average_plan_days
member_details_df['average_plan_price'] = member_details_df.total_plan_price / member_details_df.no_of_record

member_details_df['average_amount_paid'] = member_details_df.total_ammount_paid / member_details_df.no_of_record
member_details_df['ammount_due'] = member_details_df.total_plan_price - member_details_df.total_ammount_paid
drop_columns = ['total_plan_days', 'total_plan_price', 'total_ammount_paid', 'first_transaction_date', 'last_transaction_date',

               'membership_expire_date', 'registration_init_time']

member_details_df = member_details_df.drop(columns=drop_columns)
member_details_df.head()
member_details_df_final = member_details_df
train_df = pd.read_csv('../input/train_v2.csv')

train_df.head()
len(train_df)
test_df = pd.read_csv('../input/sample_submission_v2.csv')

test_df.head()
len(test_df)
member_details_df_final = member_details_df_final.merge(train_df, how='left', on='msno')

member_details_df_final = member_details_df_final.merge(test_df, how='left', on='msno')

member_details_df_final.head()
member_details_df_final.is_churn_x.fillna(member_details_df_final.is_churn_y, inplace=True)

member_details_df_final.rename(columns={'is_churn_x': 'is_churn'}, inplace=True)

member_details_df_final.drop(columns=['is_churn_y'], inplace=True)
member_details_df_final.dropna(inplace=True)

member_details_df_final.head()
member_details_df_final.to_csv('data/member_details_df_final.csv', index=False)
del(member_details_df_final)
train_df = train_df.merge(member_details_df, how='left', on='msno')
train_df.isna().sum()
train_df.dropna(inplace=True)
test_df = test_df.merge(member_details_df, how='left', on='msno')
test_df.isna().sum()
test_df.dropna(inplace=True)
train_df.head()
train_X = train_df.iloc[:,2:]

train_y = train_df['is_churn']
test_df.head()
test_X = test_df.iloc[:,2:]

test_y = test_df['is_churn']
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=50, n_estimators=100, random_state=0)
rfc.fit(train_X, train_y)
predict_y = rfc.predict(test_X)
test_y = test_y.values
correct = 0

for i, val in enumerate(predict_y):

    if val == test_y[i]:

#         print('actual : {}, predicted : {}'.format(test_y[i], val))

        correct += 1

print(correct)
correct / len(test_y) * 100