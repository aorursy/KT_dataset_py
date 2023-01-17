import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import roc_auc_score
# download train and test files

! wget -q https://datahack-prod.s3.amazonaws.com/train_file/train.csv_VsW9EGx.zip

! unzip -q train.csv_VsW9EGx.zip

! wget -q https://datahack-prod.s3.amazonaws.com/test_file/test.csv_yAFwdy2.zip

! unzip -q test.csv_yAFwdy2.zip

! wget -q https://datahack-prod.s3.amazonaws.com/sample_submission/sample_submission_iA3afxn.csv
# dropping region_code

df = pd.read_csv('train.csv').drop(['id', 'Region_Code'], 1)

df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype(int)

print(df.shape)

df.head(4)
# checking nans

df.isnull().sum()
# checking datatypes

df.dtypes
# checking unique values in all columns

for i in df.columns:

  print(f'{i} -> {df[i].nunique()}')

  print(df[i].value_counts(dropna=False))

  print('---------------')
# defining categorical and continous columns

categorical = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']

continous = ['Age', 'Annual_Premium', 'Vintage']

y = 'Response'
# encoding categorical columns

cat_enc = {}

for col in categorical:

  cat_enc[f'{col}_enc'] = LabelEncoder()

  df[col] = cat_enc[f'{col}_enc'].fit_transform(df[col])

print(cat_enc)
# binning age column

age_enc = KBinsDiscretizer(n_bins=5, encode='ordinal')

df['Age'] = age_enc.fit_transform(df['Age'].values.reshape(-1, 1)).astype(int)
# encoding policy_sales_channel

channel_map = {}

channel_count = df.groupby('Policy_Sales_Channel')['Response'].value_counts(normalize=True).to_dict()

for i, channel in enumerate(df['Policy_Sales_Channel'].unique()):

  count = channel_count.get((channel, 0))

  if count is not None:

    if count > 0.95:

      channel_map[channel] = 0

    elif count > 0.9:

      channel_map[channel] = 1

    elif count > 0.8:

      channel_map[channel] = 2

    else:

      channel_map[channel] = 3

  else:

    channel_map[channel] = 3

    print(channel)

df['Policy_Sales_Channel'] = [channel_map[i] for i in df['Policy_Sales_Channel'].values.tolist()]
# scaling continous columns

scaler = StandardScaler()

df[['Annual_Premium', 'Vintage']] = scaler.fit_transform(df[['Annual_Premium', 'Vintage']])
df.head()
# train validation split

x, y = df.drop('Response', 1), df['Response']

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.05, stratify = y, random_state=36)

print(x_train.shape, x_val.shape)
lr = LogisticRegression(class_weight='balanced')

lr.fit(x_train, y_train)

y_pred = lr.predict_proba(x_val)

roc_auc_score(y_val, y_pred[:, 1])
test_df = pd.read_csv('test.csv').drop(['Region_Code'], 1)

test_df['Policy_Sales_Channel'] = test_df['Policy_Sales_Channel'].astype(int)

print(test_df.shape)

test_df.head(4)
for col in categorical:

  test_df[col] = cat_enc[f'{col}_enc'].transform(test_df[col])

test_df['Age'] = age_enc.transform(test_df['Age'].values.reshape(-1, 1)).astype(int)

test_df['Policy_Sales_Channel'] = [channel_map[i] if i in channel_map else 1 for i in test_df['Policy_Sales_Channel'].values.tolist()]

test_df[['Annual_Premium', 'Vintage']] = scaler.transform(test_df[['Annual_Premium', 'Vintage']])

test_df.head(2)
# training LR on whole dataset

lr = LogisticRegression(class_weight='balanced')

lr.fit(x, y)
x_test = test_df.drop('id', 1).values

y_test = lr.predict_proba(x_test)

submit_df = pd.DataFrame(data={'id': test_df['id'], 'Response': y_test[:, 1]})

submit_df['Response'] = submit_df['Response'].astype('float16')

submit_df['id'] = submit_df['id'].astype('int32')

submit_df.head()
submit_df.to_csv('lr_submission.csv', index=False)