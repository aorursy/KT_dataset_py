import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
%matplotlib inline
train = pd.read_csv('../input/hse-pml-2/train_resort.csv')
test = pd.read_csv('../input/hse-pml-2/test_resort.csv')
col_names = pd.read_excel('../input/hse-pml-2/column_names.xlsx')
train.info()
test.info()
train.season_holidayed_code.unique(), test.season_holidayed_code.unique()
train.state_code_residence.unique(), test.state_code_residence.unique()
train.resort_type_code.unique()
train.head()
test.head()
col_names
plt.figure(figsize=(10, 10))
train.amount_spent_per_room_night_scaled.hist()
plt.title('train target')
plt.show()
q = train.amount_spent_per_room_night_scaled.quantile(0.01)
train[train.amount_spent_per_room_night_scaled < q][:5]
train.checkin_date = pd.to_datetime(train.checkin_date)
train.checkout_date = pd.to_datetime(train.checkout_date)
test.checkin_date = pd.to_datetime(test.checkin_date)
test.checkout_date = pd.to_datetime(test.checkout_date)
train.booking_date = pd.to_datetime(train.booking_date)
test.booking_date = pd.to_datetime(test.booking_date)
train.checkin_date.min(), train.checkout_date.max()
test.checkin_date.min(), test.checkout_date.max()
test.checkin_date.sort_values()[:20]
train[train.booking_date > train.checkin_date]
test[test.booking_date > test.checkin_date]
def intersects(train, test, cols):
    res = pd.DataFrame(index=cols,
                       columns=['# common', 'intersection fraction (train)', 'intersection fraction (test)'])
    for col in cols:
        unq_test = set(test[col].unique())
        unq_train = set(train[col].unique())
        n_common = len(unq_test & unq_train) 
        res.loc[col, '# common'] = n_common
        res.loc[col, 'intersection fraction (train)'] = n_common / len(unq_train)
        res.loc[col, 'intersection fraction (test)'] = n_common / len(unq_test)
    return res
cat_cols = list(filter(lambda x: 'id' in x or 'code' in x, train.columns))
intersects(train, test, cat_cols)
cols_to_vis = [x for x in cat_cols if x not in ('memberid', 'reservation_id', 'cluster_code', 'reservationstatusid_code', 'resort_id')]
for col in cols_to_vis:
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title(f'train {col}')
    ax[0].hist(train[col], bins=len(train[col].unique()))
    ax[0].grid()
    ax[1].hist(test[col], bins=len(test[col].unique()))
    ax[1].set_title(f'test {col}')
    ax[1].grid()
plt.show()
numeric_cols = list(filter(
    lambda x: not ('code' in x or 'id' in x or 'date' in x or 'amount' in x or 'buck' in x), train))
numeric_cols
for col in numeric_cols:
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title(f'train {col}')
    ax[0].hist(train[col])
    ax[0].grid()
    ax[1].hist(test[col])
    ax[1].set_title(f'test {col}')
    ax[1].grid()
plt.show()
train.roomnights.sort_values()[:10]
train['adults+children'] = train.numberofadults + train.numberofchildren
test['adults+children'] = test.numberofadults + test.numberofchildren
for col in ['numberofchildren','numberofadults','adults+children', 'total_pax']:
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title(f'train {col}')
    ax[0].hist(train[col])
    ax[0].grid()
    ax[1].hist(test[col])
    ax[1].set_title(f'test {col}')
    ax[1].grid()
plt.show()
common_membid = list(set(train.memberid) & set(test.memberid))
train[train.memberid == common_membid[0]]
test[test.memberid == common_membid[0]]
train[train.memberid == common_membid[1]]
test[test.memberid == common_membid[1]]
test.shape
train.shape
train[['amount_spent_per_room_night_scaled', 'total_pax', 'numberofadults', 'numberofchildren']].corr()
