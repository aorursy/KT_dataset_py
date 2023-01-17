# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



plt.style.use('seaborn-darkgrid')



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')
train.head()
train.drop(columns=['id'], inplace=True)
train.info()
for col in train.columns:

    print(train[col].value_counts())

    print()
binary_columns = []

for i in range(5):

    binary_columns.append('bin_{}'.format(i))



binary_columns
bin_3 = {'T': 1, 'F': 0}

bin_4 = {'Y': 1, 'N': 0}
train['bin_3'] = train['bin_3'].map(bin_3)

train['bin_4'] = train['bin_4'].map(bin_4);
train[binary_columns]
from sklearn.preprocessing import LabelEncoder
nominal_columns = []

for i in range(10):

    nominal_columns.append('nom_{}'.format(i))



nominal_columns
label_encoders = {}



for c in nominal_columns:

    label_encoders[c] = LabelEncoder()

    label_encoders[c].fit(train[c])

    train[c] = label_encoders[c].transform(train[c])
label_encoders['nom_0'].inverse_transform([0, 1, 2])
train[nominal_columns]
ordinal_columns = []

for i in range(6):

    ordinal_columns.append('ord_{}'.format(i))



ordinal_columns
def to_category(column, categories):

    train[column] = train[column].astype('category')

    train[column] = train[column].cat.set_categories(categories, ordered=True)

    train[column] = train[column].cat.codes
train['ord_0'].unique()
to_category('ord_0', [1, 2, 3])
train['ord_1'].unique()
to_category('ord_1', ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'])
train['ord_2'].unique()
to_category('ord_2', ['Freezing', 'Cold', 'Warm', 'Hot', 'Boining Hot', 'Lava Hot'])
train['ord_3'].unique()
np.sort(train['ord_3'].unique())
to_category('ord_3', np.sort(train['ord_3'].unique()))
train['ord_4'].unique()
to_category('ord_4', np.sort(train['ord_4'].unique()))
train['ord_5'].unique()
ord(train['ord_5'].unique()[0][0]) + ord(train['ord_5'].unique()[0][1])
train['ord_5'] = train['ord_5'].apply(lambda x: ord(x[0]) + ord(x[1]))
train[ordinal_columns]
cyclical_features = ['day', 'month']
plt.plot(np.arange(0,7));
days_sin = np.sin(2 * np.pi * np.arange(0,7)/7)

days_cos = np.cos(2 * np.pi * np.arange(0,7)/7)
plt.plot(days_sin)

plt.plot(days_cos)

plt.show()
plt.xlabel('days_sin')

plt.ylabel('days_cos')

plt.scatter(days_sin, days_cos)

plt.show()
train['day_sin'] = np.sin(2 * np.pi * (train['day'] - 1)/7)

train['day_cos'] = np.cos(2 * np.pi * (train['day'] - 1)/7)

train.drop(columns=['day'], inplace=True)
plt.plot(np.arange(0,12));
months_sin = np.sin(2 * np.pi * np.arange(0,12)/12)

months_cos = np.cos(2 * np.pi * np.arange(0,12)/12)
plt.plot(months_sin)

plt.plot(months_cos)

plt.show()
plt.xlabel('months_sin')

plt.ylabel('months_cos')

plt.scatter(months_sin, months_cos)

plt.show()
train['month_sin'] = np.sin(2 * np.pi * (train['month'] - 1)/12)

train['month_cos'] = np.cos(2 * np.pi * (train['month'] - 1)/12)

train.drop(columns=['month'], inplace=True)
cyclical_features = ['day_sin', 'day_cos', 'month_sin', 'month_cos']



train[cyclical_features]
cat_features = binary_columns + ordinal_columns + nominal_columns

cat_features
for cat in cat_features:

    print('The feature {} has {} categories'.format(cat, train[cat].unique().shape[0]))
X = train.drop(columns=['target'])

y = train['target']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
verbose = 0 # Define a quantidade de informação fornecida a cada iteração
import lightgbm
lightgbm_no_cat = lightgbm.LGBMClassifier(learning_rate=0.1, n_estimators=500, random_state=42)

lightgbm_no_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=5, verbose=verbose)

print('Accuracy for {} {} categories: {}'.format('LightGBM', 'without', lightgbm_no_cat.score(X_test, y_test)))
lightgbm_with_cat = lightgbm.LGBMClassifier(learning_rate=0.1, n_estimators=500, random_state=42)

lightgbm_with_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=5, categorical_feature=cat_features, verbose=verbose)

print('Accuracy for {} {} categories: {}'.format('LightGBM', 'with', lightgbm_with_cat.score(X_test, y_test)))
import catboost
catboost_no_cat = catboost.CatBoostClassifier(learning_rate=0.1, n_estimators=500, random_state=42)

catboost_no_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=5, verbose=verbose)

print('Accuracy for {} {} categories: {}'.format('Catboost', 'without', catboost_no_cat.score(X_test, y_test)))
%%time

catboost_with_cat = catboost.CatBoostClassifier(learning_rate=0.1, n_estimators=500, random_state=42)

catboost_with_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=5, cat_features=cat_features, verbose=verbose)

print('Accuracy for {} {} categories: {}'.format('Catboost', 'with', catboost_with_cat.score(X_test, y_test)))
catboost_with_cat_gpu = catboost.CatBoostClassifier(learning_rate=0.1, n_estimators=500, random_state=42, task_type='GPU', devices='0:1')
%%time

catboost_with_cat_gpu.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=5, cat_features=cat_features, verbose=verbose)

print('Accuracy for {} {} categories: {}'.format('Catboost', 'with', catboost_with_cat_gpu.score(X_test, y_test)))
import xgboost
%%time

xgb = xgboost.XGBClassifier(learning_rate=0.1, n_estimators=500, random_state=42, tree_method='hist')

xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=verbose)

print('Accuracy for {} {} categories: {}'.format('XGBoost', 'without', xgb.score(X_test, y_test)))
%%time

xgb = xgboost.XGBClassifier(learning_rate=0.1, n_estimators=500, random_state=42, tree_method='gpu_hist')

xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=verbose)

print('Accuracy for {} {} categories: {}'.format('XGBoost', 'without', xgb.score(X_test, y_test)))