def get_null(train):

  columns = train.columns

  values = []

  features = []



  for column in columns:

    size = train[train[column].isnull()].shape[0]

    if(size != 0):

      features.append(column)

      values.append(size)



  return pd.DataFrame({

                       'features': features,

                       'values': values

  })
def range_quantile(column):

  q1 = train[column].quantile(0.25)

  q2 = train[column].quantile(0.5)

  q3 = train[column].quantile(0.75)

  print(train[train[column] == q1]['isChurned'].value_counts())

  print(train[(train[column] > q1) & (train[column] <= q2)]['isChurned'].value_counts())

  print(train[(train[column] > q2) & (train[column] <= q3)]['isChurned'].value_counts())

  print(train[train[column] > q3]['isChurned'].value_counts())

  return q1, q2, q3
def corr_top(data, target, top):

  ret = pd.DataFrame({

      'features': data.corr()[target].sort_values(ascending=False)[1:top+1].index,

      'corr': data.corr()[target].sort_values(ascending=False)[1:top+1].values

  })

  return ret
import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv('../input/seleksidukungaib/train.csv')

test = pd.read_csv('../input/seleksidukungaib/test.csv')

print(train.shape)

print(test.shape)
train.head()
test.head()
train.dtypes
train['isChurned'].value_counts()
train.describe()
test.describe()
get_null(train)
get_null(test)
train.drop(columns=['idx'], inplace=True)

test.drop(columns=['idx'], inplace=True)
print(train[train['userId'].duplicated()]['userId'].unique().shape[0])

print(test[test['userId'].duplicated()]['userId'].unique().shape[0])
ddc = train[['date_collected', 'date']]
print(ddc.shape[0])

print(ddc[ddc.duplicated()].shape[0])
train.drop(columns=['date'], inplace=True)

test.drop(columns=['date'], inplace=True)
train['userId'].value_counts()
duplicate = train[(train.duplicated(subset=['userId', 'date_collected'], keep=False))]

null_row = duplicate[(duplicate['max_recharge_trx'].isnull()) | (duplicate['average_topup_trx'].isnull()) | (duplicate['total_transaction'].isnull())]

count = 0

for i, el in null_row.iterrows():

  temp_df = duplicate[(duplicate['userId'] == el['userId']) & (duplicate['date_collected'] == el['date_collected'])]

  idx_recharge, idx_topup, idx_transaction = -1, -1, -1

  for j, row in temp_df.iterrows():

    if(~np.isnan(row['max_recharge_trx'])):

      idx_recharge = j

    if(~np.isnan(row['average_topup_trx'])):

      idx_topup = j

    if(~np.isnan(row['total_transaction'])):

      idx_transaction = j



  if(idx_recharge != -1):

    train.at[i, 'max_recharge_trx'] = train.at[idx_recharge, 'max_recharge_trx']

  if(idx_topup != -1):

    train.at[i, 'average_topup_trx'] = train.at[idx_topup, 'average_topup_trx']

  if(idx_transaction != -1):

    train.at[i, 'total_transaction'] = train.at[idx_transaction, 'total_transaction']

  count += 1

  if(count % 1000 == 0): 

    print(count)
get_null(train)
def update_max_recharge_trx(row):

  if(np.isnan(row['max_recharge_trx'])):

    row['max_recharge_trx'] = row['average_recharge_trx'] * 2 - row['min_recharge_trx'] 

  return row

  

train = train.apply(update_max_recharge_trx, axis=1)
def update_average_topup_trx(row):

  if(np.isnan(row['average_topup_trx'])):

    if(row['num_topup_trx'] < 2):

      row['average_topup_trx'] = row['min_topup_trx']

    else:

      row['average_topup_trx'] = (row['min_topup_trx'] + row['max_topup_trx'])/2

  return row



train = train.apply(update_average_topup_trx, axis=1)
def update_num_transfer(x):

  return x['num_transaction'] - x['num_topup_trx'] - x['num_recharge_trx']
train['num_transfer_trx'] = train.apply(update_num_transfer, axis=1)

test['num_transfer_trx'] = test.apply(update_num_transfer, axis=1)
def update_average_trx(x):

  if(x['num_transfer_trx'] == 0):

    return 0

  if(np.isnan(x['total_transaction'])):

    return np.nan

  topup = x['average_topup_trx'] * x['num_topup_trx']

  recharge = x['average_recharge_trx'] * x['num_recharge_trx']

  total = x['total_transaction']

  return (total-topup-recharge)/x['num_transfer_trx']
train['average_transfer_trx'] = train.apply(update_average_trx, axis=1)

test['average_transfer_trx'] = test.apply(update_average_trx, axis=1)
train.head()
# Rumus dari total transaction adalah

# average_topup * num_topup + average_recharge * num_recharge



def fill_total_transaction(x):

  if(np.isnan(x['total_transaction'])):

    if(np.isnan(x['average_transfer_trx'])):

      return x

    x['total_transaction'] = (

        (x['average_topup_trx'] * x['num_topup_trx']) + 

        (x['average_recharge_trx'] * x['num_recharge_trx']) +

        (x['average_transfer_trx'] * x['num_transfer_trx'])

    ) 

  return x



train = train.apply(fill_total_transaction, axis=1)
train[train['isActive'].isnull()]
train.drop(train[train['isActive'].isnull()].index, inplace=True)
test['premium'].value_counts()
test['premium'] = test['premium'].fillna(False)
train['isUpgradedUser'].unique()
train.drop(columns='isUpgradedUser', inplace=True)

test.drop(columns='isUpgradedUser', inplace=True)
train.drop(columns=['random_number'], inplace=True)

test.drop(columns=['random_number'], inplace=True)
# train.to_csv('cleaned_train.csv', index=False)

# test.to_csv('cleaned_test.csv', index=False)
# train = pd.read_csv('cleaned_train.csv')

# test = pd.read_csv('cleaned_test.csv')
print(train[train.duplicated()].shape[0])

print(train[train['total_transaction'].isnull()].shape[0])

print(train[train['average_transfer_trx'] < 0].shape[0])
train.drop(train[train['average_transfer_trx'] < 0].index, inplace=True)
train.drop(train[train.duplicated()].index, inplace=True)
train.drop(train[train['total_transaction'].isnull()].index, inplace=True)
numerical_features = ['num_recharge_trx', 'num_topup_trx', 'num_transfer_trx', 'num_transaction']

fix, ax = plt.subplots(2, 2, figsize=(15,6))

train[train['isChurned'] == 0][numerical_features].hist(bins=20, color='blue', alpha=0.5, ax=ax)

train[train['isChurned'] == 1][numerical_features].hist(bins=20, color='orange', alpha=0.5, ax=ax)

plt.show()
fig, ax = plt.subplots(4, 2, figsize=(12, 12))

bool_cols = ['isActive', 'isVerifiedPhone', 'isVerifiedEmail', 'blocked', 'premium', 'super', 'userLevel', 'pinEnabled']



i, j = 0, 0

sns.set(style='darkgrid')

for col in bool_cols:

  sns.countplot(data=train, x=col, hue='isChurned', ax=ax[i][j])

  j += 1

  if(j == 2):

    j = 0

    i += 1



plt.tight_layout()

plt.show()
print(train[train['num_transfer_trx'] != 0]['isChurned'].value_counts())

print(train[train['num_transfer_trx'] == 0]['isChurned'].value_counts())
def update_transaction(x):

  if(x['num_transaction'] != 0):

    x['num_transaction'] -= x['num_transfer_trx']

    x['total_transaction'] -= (x['average_transfer_trx'] * x['num_transfer_trx'])

  if(x['total_transaction'] > -1e-8 and x['total_transaction'] < 1e-8):

    x['total_transaction'] = 0

  return x



train = train.apply(update_transaction, axis=1)

test = test.apply(update_transaction, axis=1)
train.drop(columns=['num_transfer_trx', 'average_transfer_trx', 'min_transfer_trx', 'max_transfer_trx'], inplace=True)

test.drop(columns=['num_transfer_trx', 'average_transfer_trx', 'min_transfer_trx', 'max_transfer_trx'], inplace=True)
train['average_recharge_trx'] /= 15000

train['min_recharge_trx'] /= 15000

train['max_recharge_trx'] /= 15000

train['average_topup_trx'] /= 15000

train['min_topup_trx'] /= 15000

train['max_topup_trx'] /= 15000

train['total_transaction'] /= 15000
test['average_recharge_trx'] /= 15000

test['min_recharge_trx'] /= 15000

test['max_recharge_trx'] /= 15000

test['average_topup_trx'] /= 15000

test['min_topup_trx'] /= 15000

test['max_topup_trx'] /= 15000

test['total_transaction'] /= 15000
train['total_recharge'] = train['average_recharge_trx'] * train['num_recharge_trx']

train['total_topup'] = train['average_topup_trx'] * train['num_topup_trx']

test['total_recharge'] = test['average_recharge_trx'] * test['num_recharge_trx']

test['total_topup'] = test['average_topup_trx'] * test['num_topup_trx']
train.columns
cols = ['userId', 

        'date_collected', 

        'num_recharge_trx', 

        'average_recharge_trx',

        'max_recharge_trx',

        'min_recharge_trx',

        'num_topup_trx',

        'average_topup_trx',

        'max_topup_trx',

        'min_topup_trx',

        'total_recharge',

        'total_topup',

        'num_transaction',

        'total_transaction',

        'isActive',

        'isVerifiedPhone',

        'isVerifiedEmail',

        'blocked',

        'premium',

        'super',

        'userLevel',

        'pinEnabled',

        'isChurned']



train = train[cols]

test = test[cols[:-1]]
from sklearn.preprocessing import LabelEncoder

encode = ['premium', 'super', 'pinEnabled']

lbl = LabelEncoder()

for col in encode:

  train[col] = lbl.fit_transform(train[col].values)

  test[col] = lbl.transform(test[col].values)
matrix = np.triu(train.corr())

plt.figure(figsize=(16,12))

plt.title('Correlation')

sns.heatmap(train.corr(), linewidth=.5, annot=True, cmap='BrBG', mask=matrix)

plt.show()
train.drop(columns=[

                    'userId',

                    'date_collected',

], inplace=True)



test.drop(columns=[

                    'userId',

                    'date_collected',

], inplace=True)  
no_dup_train = train.drop(train[train.duplicated()].index).copy()

dup_index = no_dup_train[no_dup_train.drop(columns='isChurned').duplicated()].index
len(dup_index)
def get_same_row(index, data=train):

  row = data.loc[index]

  return train[(train['num_recharge_trx'] == row['num_recharge_trx']) &

        (train['average_recharge_trx'] == row['average_recharge_trx']) &

        (train['max_recharge_trx'] == row['max_recharge_trx']) &

        (train['min_recharge_trx'] == row['min_recharge_trx']) &

        (train['total_recharge'] == row['total_recharge']) &

        (train['num_topup_trx'] == row['num_topup_trx']) &

        (train['average_topup_trx'] == row['average_topup_trx']) &

        (train['max_topup_trx'] == row['max_topup_trx']) &

        (train['min_topup_trx'] == row['min_topup_trx']) &

        (train['total_topup'] == row['total_topup']) &

        (train['num_transaction'] == row['num_transaction']) &

        (train['total_transaction'] == row['total_transaction']) &

        (train['isActive'] == row['isActive']) &

        (train['isVerifiedPhone'] == row['isVerifiedPhone']) &

        (train['isVerifiedEmail'] == row['isVerifiedEmail']) &

        (train['blocked'] == row['blocked']) &

        (train['premium'] == row['premium']) &

        (train['super'] == row['super']) &

        (train['userLevel'] == row['userLevel']) &

        (train['pinEnabled'] == row['pinEnabled'])]
def update_row(threshold=0.3, dup_index=dup_index):

  for a, i in enumerate(dup_index):

    dup_row = get_same_row(i)

    index = dup_row.index

    zero, one = dup_row['isChurned'].value_counts()[0], dup_row['isChurned'].value_counts()[1]



    if(min(zero, one) / (zero+one) <= threshold):

      mode = dup_row.mode()['isChurned'][0]

      for idx in index:

        train.at[idx, 'isChurned'] = mode



    if(a % 100 == 0):

      print(a)
train['isChurned'].value_counts()
update_row()
train['isChurned'].value_counts()
print(train.shape)

print(test.shape)
df_x = train.drop(['isChurned'], axis=1)

df_y = train['isChurned']

x = df_x

y = df_y
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x = scaler.fit_transform(x)

new_test = scaler.transform(test)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
from sklearn.metrics import confusion_matrix, f1_score
def train_model(models, x=x_train, y=y_train):

  for key, model in models.items():

    print(f'Currently Training {key}')

    model.fit(x, y)
def preds(models, x_test=x_test, y_test=y_test, new_test=new_test):

  preds = {

      'models': [],

      'val_f1_score': [],

      'train_f1_score': [],

      'confusion_matrix': [],

      'prediction': []

  }

  for key, model in models.items():

    preds['models'].append(key)

    test_pred = model.predict(x_test)

    train_pred = model.predict(x_train)

    preds['val_f1_score'].append(f1_score(test_pred, y_test))

    preds['train_f1_score'].append(f1_score(train_pred, y_train))

    preds['confusion_matrix'].append(confusion_matrix(test_pred, y_test))    

    preds['prediction'].append(model.predict(new_test))

  

  return pd.DataFrame(preds)
evaluation = []
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
linear_models = {

    'LogisticRegression': LogisticRegression(),

    'Perceptron': Perceptron(),

    'RidgeClassifier': RidgeClassifier(),

    'SGDClassifier': SGDClassifier()

}
train_model(linear_models)
evaluation.append(preds(linear_models))
from sklearn.naive_bayes import GaussianNB
naive_bayes = {

    'GaussianNB': GaussianNB()

}
train_model(naive_bayes)
evaluation.append(preds(naive_bayes))
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
tree_model = {

    'DecisionTree': DecisionTreeClassifier(),

    'ExtraTree': ExtraTreeClassifier()

}
train_model(tree_model)
evaluation.append(preds(tree_model))
!pip install catboost

from catboost import CatBoostClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.ensemble import AdaBoostClassifier
boosting_model = {

    'CatBoost': CatBoostClassifier(eval_metric='F1', logging_level='Silent'),

    'XGBoost': XGBClassifier(),

    'LGBM': LGBMClassifier(),

    'AdaBoost': AdaBoostClassifier()

}
train_model(boosting_model)
evaluation.append(preds(boosting_model))
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
ensemble_model = {

    'RandomForest': RandomForestClassifier(),

    'ExtraTrees': ExtraTreesClassifier()

}
train_model(ensemble_model)
evaluation.append(preds(ensemble_model))
eval = pd.concat(evaluation).sort_values('val_f1_score',ascending=False).reset_index(drop=True)

eval
for i in range(eval.shape[0]):

  unique, counts = np.unique(eval.loc[i]['prediction'], return_counts=True)

  print(eval.loc[i]['models'])

  for el_unique, el_counts in zip(unique, counts):

    print(f'{el_unique} {el_counts}')

  print()
from sklearn.model_selection import GridSearchCV
params = {

    'C': np.logspace(-2,2,20),

    'solver': ['liblinear', 'lbfgs'],

    'penalty': ['l2']

}
tuning_model = GridSearchCV(estimator = linear_models['LogisticRegression'],

                            param_grid = params,

                            verbose=True,

                            scoring = 'f1',

                            n_jobs=5)
tuning_model.fit(x, y)

predict = pd.DataFrame(tuning_model.best_estimator_.predict(new_test))

predict[0].value_counts()
print(tuning_model.best_estimator_)

print(tuning_model.best_params_)
# first = pd.read_csv('1st_sub.csv')

# second = pd.read_csv('2nd_sub.csv')
y_pred = predict[0]
# comparison = pd.DataFrame({

#     '1st_sub': first['isChurned'],

#     '2nd_sub': second['isChurned'],

#     'prediction': y_pred.astype(int)

# })
def compare(col):

  count_diff = 0

  for i, row in comparison.iterrows():

    if(row['prediction'] != row[col]):

      count_diff += 1

  return count_diff
# compare('1st_sub')
# compare('2nd_sub')
idx = pd.read_csv('../input/seleksidukungaib/test.csv')['idx']
submission = pd.DataFrame({

    'idx': idx,

    'isChurned': y_pred.astype(int)

})
submission['isChurned'].value_counts()
submission.to_csv('submission.csv', index=False)
!kaggle competitions submit -c seleksidukungaib -f submission.csv -m "No Comment"