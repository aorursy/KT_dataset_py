import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Loading data

train_df = pd.read_csv('/kaggle/input/ltfs-2/train_fwYjLYX.csv', parse_dates=['application_date'])

test_df = pd.read_csv('/kaggle/input/ltfs-2/test_1eLl9Yf.csv', parse_dates=['application_date'])
print(f'Shape of training data is {train_df.shape}')
print(f'Shape of testing data is {test_df.shape}')
train_df.head()
train_df.info()
train_df.describe()
# Oldest application date in training data

train_df.application_date.min()
# Newest application date in training data

train_df.application_date.max()
# No. of applications per segment

case_count_by_segment = pd.DataFrame(train_df.groupby(['segment'])['case_count'].agg(sum))

case_count_by_segment['%_Of_Total'] = case_count_by_segment['case_count'].apply \

(lambda x: x/case_count_by_segment['case_count'].sum()*100)
case_count_by_segment
# No. of zones

list(train_df.zone.unique())
# List of states

print(f'Total no. of states in the data are {train_df.state.unique().size}')

list(train_df.state.unique())
# Case count by date and segment

case_count_by_date_segment = pd.DataFrame(train_df.groupby(['application_date','segment'])['case_count'].agg(sum))

case_count_by_date_segment.reset_index(drop=False,inplace=True)
plt.figure(figsize = (16,5))

sns.lineplot(x="application_date",

             y="case_count",

             hue="segment",

             data=case_count_by_date_segment,

             palette=sns.color_palette('hls', n_colors=2))
# Extracting features from application_date

train_df['month'] = train_df['application_date'].dt.month

train_df['day_of_month'] = train_df['application_date'].dt.day

train_df['day_of_week'] = train_df['application_date'].dt.dayofweek

train_df['year'] = train_df['application_date'].dt.year

train_df['year_month'] = train_df.apply(lambda x: str(x['year']) + '_' + str(x['month']), axis=1)
train_df.head()
# Case count by date and segment

case_count_by_year_month = pd.DataFrame(train_df.groupby(['year_month','segment'])['case_count'].agg(sum))

case_count_by_year_month.reset_index(drop=False,inplace=True)

plt.figure(figsize = (25,6))

sns.lineplot(x="year_month",

             y="case_count",

             hue="segment",

             data=case_count_by_year_month,

             palette=sns.color_palette('hls', n_colors=2))
# Case count by date and segment

case_count_by_month = pd.DataFrame(train_df.groupby(['month','segment'])['case_count'].agg(sum))

case_count_by_month.reset_index(drop=False,inplace=True)

plt.figure(figsize = (16,5))

sns.lineplot(x="month",

             y="case_count",

             hue="segment",

             data=case_count_by_month,

             palette=sns.color_palette('hls', n_colors=2))
# Case count by date and segment

case_count_by_day_of_month = pd.DataFrame(train_df.groupby(['day_of_month','segment'])['case_count'].agg(sum))

case_count_by_day_of_month.reset_index(drop=False,inplace=True)

plt.figure(figsize = (16,5))

sns.lineplot(x="day_of_month",

             y="case_count",

             hue="segment",

             data=case_count_by_day_of_month,

             palette=sns.color_palette('hls', n_colors=2))
# Case count by date and segment

case_count_by_day_of_week = pd.DataFrame(train_df.groupby(['day_of_week','segment'])['case_count'].agg(sum))

case_count_by_day_of_week.reset_index(drop=False,inplace=True)

plt.figure(figsize = (16,5))

sns.lineplot(x="day_of_week",

             y="case_count",

             hue="segment",

             data=case_count_by_day_of_week,

             palette=sns.color_palette('hls', n_colors=2))
train_df.head()
train_new_df = pd.DataFrame(train_df.groupby(['segment','month','day_of_month','day_of_week','year'])\

                            ['case_count'].agg(sum).reset_index(drop=False))
train_new_df.head()
# Separate the dataframes segment wise

#train_seg_1_df = train_df[train_df['segment']==1].reset_index(drop=True)

#train_seg_2_df = train_df[train_df['segment']==2].reset_index(drop=True)
#train_seg_1_df.head()
#from sklearn.preprocessing import LabelEncoder

#le = LabelEncoder()

#train_seg_1_df['state'] = le.fit_transform(train_seg_1_df['state'])

#train_seg_1_df['zone'] = le.fit_transform(train_seg_1_df['zone'])
#train_seg_1_df.head()
categoricals = ['segment','month','day_of_month','day_of_week']

target = train_new_df.pop('case_count')

feat_cols = list(train_new_df.columns)

#remove_cols = ['application_date','segment']

#feat_cols = [cols for cols in feat_cols if cols not in remove_cols]

feat_cols.remove('year')

feat_cols
import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold



params = {

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': {'mape'},

            'subsample': 0.25,

            'subsample_freq': 1,

            'learning_rate': 0.3,

            'num_leaves': 20,

            'feature_fraction': 0.9,

            'lambda_l1': 1,  

            'lambda_l2': 1

            }



folds = 4

seed = 555



kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)



models = []



for train_index, val_index in kf.split(train_new_df, train_new_df['month']):

    train_X = train_new_df[feat_cols].iloc[train_index]

    val_X = train_new_df[feat_cols].iloc[val_index]

    train_y = target.iloc[train_index]

    val_y = target.iloc[val_index]

    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)

    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)

    gbm = lgb.train(params,

                lgb_train,

                num_boost_round=500,

                valid_sets=(lgb_train, lgb_eval),

                early_stopping_rounds=100,

                verbose_eval = 100)

    models.append(gbm)
for model in models:

    lgb.plot_importance(model)

    plt.show()
test_df.head()
test_df['month'] = test_df['application_date'].dt.month

test_df['day_of_month'] = test_df['application_date'].dt.day

test_df['day_of_week'] = test_df['application_date'].dt.dayofweek
test_df.shape
temp_df = test_df[['id','application_date']]

test_df = test_df[feat_cols]
predictions = []

predictions = (sum([model.predict(test_df) for model in models])/folds)
len(predictions)
test_df = pd.read_csv('/kaggle/input/ltfs-2/test_1eLl9Yf.csv', parse_dates=['application_date'])

test_df['case_count'] = pd.Series(predictions)
test_df.to_csv('submission_1.csv', index=False)