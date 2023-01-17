from scipy.stats import ks_2samp
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime as dt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

pd.options.display.max_columns = 300
df_train = pd.read_csv("../input/hse-pml-2/train_resort.csv")
df_test = pd.read_csv("../input/hse-pml-2/test_resort.csv")
cols = pd.read_excel("../input/hse-pml-2/column_names.xlsx")
tr_size = df_train.shape[0]
test_size = df_test.shape[0]
print("Train size: {} obs.\nTest size: {} obs.".format(tr_size, test_size))
df = pd.concat([df_train, df_test], axis=0)
mode_1 = df["season_holidayed_code"].value_counts().index[0]
mode_2 = df["state_code_residence"].value_counts().index[0]
df["season_holidayed_code"].fillna(mode_1, inplace=True)
df["state_code_residence"].fillna(mode_2, inplace=True)
df["book_to_checkin"] = df[["booking_date", "checkin_date"]].apply(lambda x: (dt.strptime(x[1], "%Y-%m-%d") - dt.strptime(x[0], "%Y-%m-%d")).total_seconds() // 86400, axis=1)
df["checkin_to_checkout"] = df[["checkin_date", "checkout_date"]].apply(lambda x: (dt.strptime(x[1], "%Y-%m-%d") - dt.strptime(x[0], "%Y-%m-%d")).total_seconds() // 86400, axis=1)

df["book_year"] = df["booking_date"].apply(lambda x: int(x[:4]))
df["book_month"] = df["booking_date"].apply(lambda x: int(x[5:7]))

df["checkin_year"] = df["checkin_date"].apply(lambda x: int(x[:4]))
df["checkin_month"] = df["checkin_date"].apply(lambda x: int(x[5:7]))
df_train
df.nunique().sort_values(ascending=False)
df.drop(["reservation_id",
        "amount_spent_per_room_night_scaled",
        "memberid"] + [col for col in df.columns if "date" in col],
       axis=1, inplace=True)
df.nunique()
def create_buckets(feat, bin_density, name=0):
    final_vals = dict()
    indexes = df[feat].value_counts().index.tolist()
    values = (df[feat].value_counts() / df.shape[0]).tolist()
    cntr = 0
    aux_list = []
    for num, val in enumerate(values):
        cntr += values[num]
        aux_list.append(indexes[num])
        if cntr >= bin_density:
            final_vals["val_"+str(num)] = aux_list
            aux_list = []
            cntr = 0
            
    if list(final_vals.items())[-1][1] != aux_list:
        final_vals["val_last"] = aux_list   
    
    dict_vals = dict()
    
    for it1 in list(final_vals.items()):
        for it2 in it1[1]:
            dict_vals[it2] = it1[0]
    if name == 0:
        df[feat] = df[feat].apply(lambda x: dict_vals[x])
    else:
        df[name] = df[feat].apply(lambda x: dict_vals[x])
for col in tqdm(df.columns):
    if col not in ["book_to_checkin", "roomnights"]:
        create_buckets(col, 0.05)
df.nunique().sort_values(ascending=False)
df_train = df.iloc[:tr_size, :]
df_test = df.iloc[tr_size:, :]
for col in ["book_to_checkin", "roomnights"]:
    pval = ks_2samp(df_train[col], df_test[col])[1]
    if pval > 0.05:
        print("P-value = " + str(pval) + ": нет свидетельств разницы между распределениями признака " + col + " в тренировочной и тестовой выборке.")
    else:
        print("P-value = " + str(pval) + ": распределения признака " + col + " в тренировочной и тестовой выборке отличаются.")
def calculate_psi(feat):
    all_vals = df[feat].unique()
    psi = 0
    for val in all_vals:
        tr_ = df_train[df_train[feat]==val].shape[0] / df_train.shape[0]
        test_ = df_test[df_test[feat]==val].shape[0] / df_test.shape[0]
        term_to_add = (tr_ - test_) * np.log((tr_ + 0.001)/(test_ + 0.001))
        psi += term_to_add
    return psi
psi_table = pd.DataFrame(columns = ["psi"], 
                         index = [col for col in df.columns if col not in ["book_to_checkin", "roomnights"]])
for feat in [col for col in df.columns if col not in ["book_to_checkin", "roomnights"]]:
    psi_table.loc[feat, "psi"] = calculate_psi(feat)
psi_table.sort_values(by="psi", ascending=False)
data = pd.read_csv('../input/hse-pml-2/train_resort.csv')
test = pd.read_csv('../input/hse-pml-2/test_resort.csv')

target = data['amount_spent_per_room_night_scaled']
data.drop(['reservation_id', 'memberid', 'amount_spent_per_room_night_scaled'], axis=1, inplace=True)
test.drop(['reservation_id', 'memberid'], axis=1, inplace=True)

data['booking_year'] = data.booking_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').year)
data['booking_month'] = data.booking_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').month)
data['booking_day'] = data.booking_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').day)

data['checkin_year'] = data.checkin_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').year)
data['checkin_month'] = data.checkin_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').month)
data['checkin_day'] = data.checkin_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').day)

data['checkout_year'] = data.checkout_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').year)
data['checkout_month'] = data.checkout_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').month)
data['checkout_day'] = data.checkout_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').day)

test['booking_year'] = test.booking_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').year)
test['booking_month'] = test.booking_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').month)
test['booking_day'] = test.booking_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').day)

test['checkin_year'] = test.checkin_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').year)
test['checkin_month'] = test.checkin_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').month)
test['checkin_day'] = test.checkin_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').day)

test['checkout_year'] = test.checkout_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').year)
test['checkout_month'] = test.checkout_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').month)
test['checkout_day'] = test.checkout_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d').day)

data.drop(['booking_date', 'checkin_date', 'checkout_date'], axis=1, inplace=True)
test.drop(['booking_date', 'checkin_date', 'checkout_date'], axis=1, inplace=True)
print('train:')
print('number of adults:', sorted(data.numberofadults.unique()))
print('number of children:', sorted(data.numberofchildren.unique()))
print('total pax:', sorted(data.total_pax.unique()))

print('\ntest:')
print('number of adults:', sorted(test.numberofadults.unique()))
print('number of children:', sorted(test.numberofchildren.unique()))
print('total pax:', sorted(test.total_pax.unique()))
print('person travelling id:', sorted(Counter(data.persontravellingid.unique())))
Counter(data.persontravellingid)
print('train:')
print('booking year:', sorted(data.booking_year.unique()))
print('checkin year:', sorted(data.checkin_year.unique()))
print('checkin month:', sorted(data[data.checkin_year == 2017].checkin_month.unique()))

print('\ntest:')
print('booking year:', sorted(test.booking_year.unique()))
print('checkin year:', sorted(test.checkin_year.unique()))
print('checkin month:', sorted(test[test.checkin_year == 2019].checkin_month.unique()))
print('roomnights:', sorted(data.roomnights.unique()))
print('state_code_residence:', sorted(data['state_code_residence'].unique()))
train = pd.read_csv('../input/hse-pml-2/train_resort.csv')
plt.figure(figsize=(18,6))
train['amount_spent_per_room_night_scaled'].hist(bins=200)
plt.show();
train['checkin_date'] = pd.to_datetime(train['checkin_date'])
train.sort_values(by='checkin_date', inplace=True)
plt.figure(figsize=(18,6))
plt.scatter(train['checkin_date'], train['amount_spent_per_room_night_scaled'])
plt.show();

print('Среднее значение метрики за первые 1000 наблюдений = {}'.format(train.iloc[1000:]['amount_spent_per_room_night_scaled'].mean()))
print('Среднее значение метрики за последние 1000 наблюдений = {}'.format(train.iloc[:-1000:-1]['amount_spent_per_room_night_scaled'].mean()))