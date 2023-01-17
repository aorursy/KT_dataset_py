# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from itertools import combinations
from catboost import CatBoostClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from datetime import datetime
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/bank-classification.csv')
def examine_columns(df: pd.DataFrame):
    for col in df.columns:
        print(col, df[col].nunique())
        if df[col].nunique() < 100:
            print(df[col].value_counts(dropna=False))
        else:
            print(df[col].agg(['mean', 'std', 'min', 'max', 'kurtosis', 'skew']))
        print()
examine_columns(df)
dt_birth = pd.to_datetime(df['birth_date'])
dt_contact = pd.to_datetime(df['contact_date'])

df['age_days'] = (datetime.now() - dt_birth).dt.days
df['contacted_age_days'] = (dt_contact - dt_birth).dt.days
del df['birth_date']

df['contact_year'] = dt_contact.dt.year
df['contact_month'] = dt_contact.dt.month
df['contact_weekday'] = dt_contact.dt.weekday
df['since_contacted'] = (datetime.now() - dt_contact).dt.days
del df['contact_date']
df.head()
df.shape
examine_columns(df)
from sklearn.preprocessing import PolynomialFeatures, minmax_scale
df['lots_of_pdays'] = (df['pdays'] == 999).astype(np.int)
numeric_features = [
    'age_days',
    'contacted_age_days',
    'since_contacted',
    'pdays',
]
for col in numeric_features:
    df[f'{col}_log'] = np.log(df[col].values).astype(df[col].dtype)
poly_colnames = list(set(numeric_features) | set(map(lambda col: col+'_log', numeric_features)))
poly = PolynomialFeatures(degree=3)
df_poly = pd.DataFrame(poly.fit_transform(df[poly_colnames].values), index=df.index).add_prefix('poly_')
df_poly.head()
%%time
df = df.join(df_poly)
df_cols_num = [
    *[col for col in df.columns if 'poly' in col or 'log' in col],
    *list(set(numeric_features) & set(df.columns))
]

df_cols_unknown_to_na = [
    'y',
    'age_days',
    'contacted_age_days',
    'contact_year',
    'contact_month',
    'contact_weekday',
    'since_contacted',
    *df_cols_num
]  # other columns will treat 'unknown' as a separate category

df[df_cols_unknown_to_na] = df[df_cols_unknown_to_na].replace('unknown', np.nan)
df.index = df['id']
del df['id']
test_df = df[df['y'].isna()]
train_df = df[~df['y'].isna()]
labels = train_df['y'].map({'yes': 1, 'no': 0}).copy()
del train_df['y']
del test_df['y']
test_df.shape
train_df.shape
labels.shape
df.head()
assert(train_df[list(set(df_cols_unknown_to_na) & set(train_df.columns))].isna().sum().sum() == 0)
from catboost import cv, Pool
cat_features_nunique_threshold = 1000  # keeping days since contacted as a categorical feature
below_threshold = train_df.nunique() < cat_features_nunique_threshold
numerical = pd.Series(map(lambda col: col in set(df_cols_num), train_df.columns), index=below_threshold.index)
cat_features_ids = np.where(below_threshold & ~numerical)[0].tolist()
pool = Pool(train_df, label=labels, cat_features=cat_features_ids)
len(cat_features_ids)
params = {}
params['loss_function'] = 'Logloss'
params['iterations'] = 128
params['custom_loss'] = 'AUC'
params['random_seed'] = 42
params['learning_rate'] = 0.2137
params['early_stopping_rounds'] = 24
cv_data = cv(
    params = params,
    pool = pool,
    fold_count=5,
    shuffle=True,
    partition_random_seed=42,
    plot=True,
    stratified=False,
    verbose=False
)
cv_data['test-AUC-mean'].agg(['mean', 'std'])
model_save_filename = f"catboost-{cv_data['test-AUC-mean'].mean()}.csv"
model_save_filename
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_df, labels, test_size=0.2, shuffle=True, random_state=42)

model = CatBoostClassifier(**params)
model.fit(Pool(X_train, y_train, cat_features=cat_features_ids), eval_set=(X_val, y_val), verbose=False, plot=True, use_best_model=True)
preds = model.predict_proba(test_df)
pd.read_csv('../input/sample_submission.csv').head()
subm = pd.DataFrame({'id': test_df.index.values, 'y': preds[:, 1]})
subm.head()
subm.to_csv(model_save_filename, index=False)
