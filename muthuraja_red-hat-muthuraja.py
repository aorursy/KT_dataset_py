# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

%matplotlib inline

# Any results you write to the current directory are saved as output.
act_train = pd.read_csv('../input/act_train.csv', parse_dates = ['date'])

act_test = pd.read_csv('../input/act_test.csv', parse_dates = ['date'])

people = pd.read_csv('../input/people.csv', parse_dates = ['date'])
people.head()
act_train.head()
act_test.head()
train_df = pd.merge(act_train, people, on = ['people_id'], how = 'left', suffixes = ('_t', '_p'))

test_df = pd.merge(act_test, people, on = ['people_id'], how = 'left', suffixes = ('_t', '_p'))
train_df.head()
test_df.head()
train_df.isna().sum()/train_df.shape[0] * 100
test_df.isna().sum()/test_df.shape[0] * 100
del act_train,act_test,people
train_df = train_df.assign(year=lambda train_df: train_df.date_t.dt.year,

                           month=lambda train_df: train_df.date_t.dt.month,

                           day=lambda train_df: train_df.date_t.dt.day)
train_df = train_df.drop('date_t', axis = 1)
test_df = test_df.assign(year=lambda test_df: test_df.date_t.dt.year,

                           month=lambda test_df: test_df.date_t.dt.month,

                           day=lambda test_df: test_df.date_t.dt.day)

test_df = test_df.drop('date_t', axis = 1)
train_df.activity_id.unique()
test_df.activity_id.unique()
train_df.activity_id.str.startswith('act1').sum()
train_df.activity_id.str.startswith('act2').sum()
assert(train_df.activity_id.nunique()==train_df.activity_id.str.startswith('act1').sum()+train_df.activity_id.str.startswith('act2').sum()), "other activities exist"
test_df.activity_id.str.startswith('act1').sum()
test_df.activity_id.str.startswith('act2').sum()
assert(test_df.activity_id.nunique()==test_df.activity_id.str.startswith('act1').sum()+test_df.activity_id.str.startswith('act2').sum()), "other activities exist"
train_df.activity_id.str.startswith('act1').sum()/train_df.activity_id.str.startswith('act2').sum()*100
train_df.activity_id = train_df.activity_id.astype(str)
test_df.activity_id = test_df.activity_id.astype(str)
new_df = train_df.copy()

new_df = new_df.assign(activity_new = new_df['activity_id'])
new_df = new_df.assign(activity_new2 = new_df['activity_id'])
new_df.activity_new = new_df.activity_new.str[3:4]
new_df.activity_new2 = new_df.activity_new2.str[5:]
new_df.activity_new.unique()
new_df.activity_new2.unique()
train_df = new_df

train_df = train_df.drop('activity_id', axis = 1)
new_df = test_df.copy()

new_df = new_df.assign(activity_new = new_df['activity_id'])

new_df = new_df.assign(activity_new2 = new_df['activity_id'])

new_df.activity_new = new_df.activity_new.str[3:4]

new_df.activity_new2 = new_df.activity_new2.str[5:]
test_df = new_df

test_df = test_df.drop('activity_id', axis = 1)

del new_df
new_df = train_df.copy()

new_df.activity_category = new_df.activity_category.str.lstrip('type ').astype(np.float)
new_df.loc[new_df.activity_new2=='4e+06','activity_new2'] = '4000000'
new_df.loc[new_df.activity_new2=='2e+06','activity_new2'] = '2000000'
new_df.loc[new_df.activity_new2=='2e+05','activity_new2'] = '200000'
new_df.loc[new_df.activity_new2=='5e+05','activity_new2'] = '500000'
new_df.activity_new = new_df.activity_new.replace('','-999')

new_df.activity_new = new_df.activity_new.astype('int32')

new_df.activity_new2 = new_df.activity_new2.replace('','-999')

new_df.activity_new2 = new_df.activity_new2.astype('int32')
new_df.loc[new_df.activity_new==-999,'activity_new'] = np.nan
new_df.loc[new_df.activity_new2==-999,'activity_new2'] = np.nan
train_df=new_df
new_df = test_df.copy()

new_df.activity_category = new_df.activity_category.str.lstrip('type ').astype(np.float)

new_df.loc[new_df.activity_new2=='4e+06','activity_new2'] = '4000000'

new_df.loc[new_df.activity_new2=='2e+06','activity_new2'] = '2000000'

new_df.loc[new_df.activity_new2=='2e+05','activity_new2'] = '200000'

new_df.loc[new_df.activity_new2=='5e+05','activity_new2'] = '500000'

new_df.loc[new_df.activity_new2=='3e+05','activity_new2'] = '300000'

new_df.loc[new_df.activity_new2=='9e+05','activity_new2'] = '900000'

new_df.activity_new = new_df.activity_new.replace('','-999')

new_df.activity_new = new_df.activity_new.astype('int32')

new_df.activity_new2 = new_df.activity_new2.replace('','-999')

new_df.activity_new2 = new_df.activity_new2.astype('int32')

new_df.loc[new_df.activity_new==-999,'activity_new'] = np.nan

new_df.loc[new_df.activity_new2==-999,'activity_new2'] = np.nan

test_df=new_df

del new_df
temp_df = train_df.copy()

col_range1 = 11

col_range2 = 10

char_columns1 = ['char_' + str(i) for i in range(1,col_range1)]

char_columns1 = [s + '_t' for s in char_columns1]

char_columns2 = ['char_' + str(i) for i in range(1,col_range2)]

char_columns2 = [s + '_p' for s in char_columns2]

char_columns3 = ['char_10_p']

char_columns3 = char_columns3+['char_' + str(i) for i in range(col_range2+1,39)]



temp_df[char_columns1] = temp_df[char_columns1].fillna('type -999')

temp_df[char_columns1] = temp_df[char_columns1].apply(lambda col: col.str.lstrip('type '))

temp_df[char_columns1] = temp_df[char_columns1].astype(np.int32)



temp_df[char_columns2] = temp_df[char_columns2].fillna('type -999').apply(lambda col: col.str.lstrip('type ')).astype(np.int32)



temp_df[char_columns3] = temp_df[char_columns3].fillna(-999)*1
char_columns = char_columns1+char_columns2+char_columns3

print(char_columns)
temp_df[char_columns] = temp_df[char_columns].replace(-999, np.NaN)
train_df = temp_df

del temp_df
temp_df = test_df.copy()

col_range1 = 11

col_range2 = 10

char_columns1 = ['char_' + str(i) for i in range(1,col_range1)]

char_columns1 = [s + '_t' for s in char_columns1]

char_columns2 = ['char_' + str(i) for i in range(1,col_range2)]

char_columns2 = [s + '_p' for s in char_columns2]

char_columns3 = ['char_10_p']

char_columns3 = char_columns3+['char_' + str(i) for i in range(col_range2+1,39)]



temp_df[char_columns1] = temp_df[char_columns1].fillna('type -999')

temp_df[char_columns1] = temp_df[char_columns1].apply(lambda col: col.str.lstrip('type '))

temp_df[char_columns1] = temp_df[char_columns1].astype(np.int32)



temp_df[char_columns2] = temp_df[char_columns2].fillna('type -999').apply(lambda col: col.str.lstrip('type ')).astype(np.int32)



temp_df[char_columns3] = temp_df[char_columns3].fillna(-999)*1



char_columns = char_columns1+char_columns2+char_columns3



temp_df[char_columns] = temp_df[char_columns].replace(-999, np.NaN)



test_df = temp_df

del temp_df
new_df = train_df.copy()

new_df.group_1 = new_df.group_1.str.lstrip('group ').astype(np.float)
train_df = new_df
new_df = test_df.copy()

new_df.group_1 = new_df.group_1.str.lstrip('group ').astype(np.float)

test_df = new_df

del new_df
test_df.group_1.unique()
train_df = train_df.assign(year_p=lambda train_df: train_df.date_p.dt.year,

                           month_p=lambda train_df: train_df.date_p.dt.month,

                           day_p=lambda train_df: train_df.date_p.dt.day)

train_df = train_df.drop('date_p', axis = 1)
test_df = test_df.assign(year_p=lambda test_df: test_df.date_p.dt.year,

                           month_p=lambda test_df: test_df.date_p.dt.month,

                           day_p=lambda test_df: test_df.date_p.dt.day)

test_df = test_df.drop('date_p', axis = 1)
train_df.head()
new_train_df = train_df[-(train_df.activity_new.isnull())]

new_test_df = test_df[-(test_df.activity_new.isnull())]
test_df[(test_df.activity_new.isnull())].count()
new_train_df.isna().sum()/new_train_df.shape[0] * 100
new_test_df.isna().sum()/new_test_df.shape[0] * 100
char_columns_drop = char_columns1.copy()

char_columns_drop.pop()

print(char_columns_drop)
train_df = train_df.drop(char_columns_drop, axis = 1)

test_df = test_df.drop(char_columns_drop, axis = 1)
new_train_df = new_train_df.drop(char_columns_drop, axis = 1)

new_test_df = new_test_df.drop(char_columns_drop, axis = 1)
temp_df = new_train_df[new_train_df.char_10_t.isna()]
temp_df['activity_new'].unique()
temp_df['activity_new2'].unique()
temp_df['activity_new2'].nunique()
temp_df['activity_category'].unique()
temp_df['group_1'].unique()
temp_df['group_1'].nunique()
temp_df.head()
grouped = temp_df.groupby(['activity_new'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['activity_new2'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['year'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['month'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['day'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['year_p'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['month_p'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['day_p'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['activity_category'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_10_t'])

grouped['outcome'].sum()
grouped = temp_df.groupby(['char_1_p'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_2_p'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_3_p'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_4_p'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_5_p'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_6_p'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_7_p'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_8_p'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_9_p'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_10_p'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_10_p'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_11'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_12'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_13'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_14'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_15'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_16'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_17'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_18'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
grouped = temp_df.groupby(['char_19'])

grouped['outcome'].mean()*100
grouped = temp_df.groupby(['char_20'])

grouped['outcome'].mean()*100
grouped = temp_df.groupby(['group_1'])

grouped['outcome'].sum()/grouped['outcome'].count()*100
temp_df.head()
temp_df.outcome.unique()
del temp_df
train_df['outcome'].sum()/train_df.shape[0]*100
input_features = list(set(train_df.columns.values) & set(test_df.columns.values))
not_useful = ['people_id']

useful = [itm for itm in input_features if itm not in not_useful]
print(input_features)
Y_train = train_df['outcome']

X_train = train_df[useful]

X_test = test_df[useful]
not_categorical = ['year', 'month', 'day', 'year_p', 'month_p', 'day_p']

categorical = [itm for itm in X_train.columns.values if itm not in not_categorical]
X_train[categorical].head()
dummies = pd.get_dummies(X_train[categorical], sparse = True)
new_X_train = X_train.drop(categorical, axis = 1)
new_X_train = pd.concat([new_X_train, dummies], axis = 1)
new_X_train.head()
dummies2 = pd.get_dummies(X_test[categorical], sparse = True)
new_X_test = X_test.drop(categorical, axis = 1)

new_X_test = pd.concat([new_X_test, dummies2], axis = 1)

new_X_test.head()
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score



# split data into train and validation sets

seed = 123

val_size = 0.33

new_X_train2, X_val, Y_train2, Y_val = train_test_split(new_X_train, Y_train, test_size=val_size, random_state=seed)
# fit model no training data

model = XGBClassifier()

model.fit(new_X_train2, Y_train2)
# make predictions for validation data

y_pred = model.predict(X_val)

predictions = [round(value) for value in y_pred]
# evaluate predictions

roc_auc = roc_auc_score(Y_val, predictions)

print("Area under ROC Curve: %f" % (roc_auc * 100.0))
# make predictions for test data

y_pred_test = model.predict(new_X_test)

test_predictions = [round(value) for value in y_pred_test]
act_test = pd.read_csv('../input/act_test.csv', parse_dates = ['date'])
test_pred_series = pd.Series(v for v in test_predictions)
sub_file = 'submission.csv'

print('Writing submission: ', sub_file)

f = open(sub_file, 'w')

f.write('activity_id,outcome\n')

i = 0

for id in act_test['activity_id']:

    str1 = str(id) + ',' + str(test_predictions[i])

    str1 += '\n'

    i += 1

    f.write(str1)

f.close()