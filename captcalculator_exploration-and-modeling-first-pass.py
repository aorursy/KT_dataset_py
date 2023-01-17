import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv', index_col='Id')
test = pd.read_csv('../input/test.csv', index_col='Id')
train.head()
test.head()
missing = train.isnull().sum().sort_values(ascending=False)
missing= (missing[missing > 0] / train.shape[0])
ax = missing.round(3).plot.bar();
ax.set_title('% Missing values')
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals]);
plt.figure(figsize=(10,5));
train.groupby('idhogar')['idhogar'].count().hist(bins=10);
plt.title('Distribution of household size');
plt.xlabel('Household size');
fig, (ax1, ax2) = plt.subplots(1, 2);
sns.boxplot(x='Target', y='v2a1', data=train, ax=ax1);
sns.boxplot(x='Target', y='v2a1', data=train[train.v2a1 < 2000000], ax=ax2);
ax1.set_title('With outlier');
ax2.set_title('Without outlier');
ax1.set_xlabel('Poverty level');
ax2.set_xlabel('Poverty level');
ax1.set_ylabel('Monthyl rent payment');
ax2.set_ylabel('');
fig.suptitle('Distribution of monthly rent payment by poverty level', x=1, y=1, fontsize=16);
fig.subplots_adjust(left=0.1, right=2, wspace=0.3);
train.groupby('Target')['v2a1'].median()
train_by_hhid = train.groupby('idhogar')
rm_by_id = train_by_hhid['Target', 'rooms'].first()

plt.figure(figsize=(10, 5));
sns.countplot(x='rooms', hue='Target', data=rm_by_id);
plt.title('Distribution of rooms in house by poverty level');
plt.xlabel('Number of rooms');
bathroom_by_id = train_by_hhid['Target', 'v14a', 'refrig'].first()

fig, (ax1, ax2) = plt.subplots(1, 2)
sns.countplot(x='v14a', hue='Target', data=bathroom_by_id, ax=ax1);
sns.countplot(x='refrig', hue='Target', data=bathroom_by_id, ax=ax2);
ax1.set_xticklabels(['No', 'Yes']);
ax2.set_xticklabels(['No', 'Yes']);
ax1.set_xlabel('Has bathroom');
ax2.set_ylabel('Has Refrigerator');
fig.subplots_adjust(left=0.1, right=2)
fig.suptitle('Distribution of Bathrooms and Refrigerators by Poverty Level', x=1, y=1, fontsize=16);
fig, (ax1, ax2) = plt.subplots(1, 2);
sns.boxplot(x='Target', y='meaneduc', data=train, ax=ax1);
sns.violinplot(x='Target', y='meaneduc', data=train, ax=ax2);

ax1.set_xlabel('Poverty level');
ax2.set_xlabel('Poverty level');
ax1.set_ylabel('Mean years education');
ax2.set_ylabel('');
fig.suptitle('Distribution of mean years education by poverty level', x=1, y=1, fontsize=16);
fig.subplots_adjust(left=0.1, right=2, wspace=0.3);
#outlier in test set which rez_esc is 99.0
test.loc[test['rez_esc'] == 99.0 , 'rez_esc'] = 5
# correct entries that were not delabeled as per discussion thread
relabel_cols = ['edjefe', 'edjefa', 'dependency']

train[relabel_cols] = train[relabel_cols].replace({'yes':1, 'no':1}).astype(float)
test[relabel_cols] = test[relabel_cols].replace({'yes':1, 'no':1}).astype(float)
# set monthly rent payment to 0 where hh owns home
train.loc[train.tipovivi1 == 1, 'v2a1'] = 0
test.loc[test.tipovivi1 == 1, 'v2a1'] = 0
# dictionary of columns and aggregation method
agg_dict = {'escolari':np.sum, 'rez_esc':np.sum, 'age':np.sum, 'estadocivil1':np.sum}

# group by household and apply aggregtion methods
train_by_hh = train.groupby('idhogar').agg(agg_dict)
test_by_hh = test.groupby('idhogar').agg(agg_dict)

# join household level data with individual level data
train = train.join(train_by_hh, on='idhogar', rsuffix='_hh')
test = test.join(test_by_hh, on='idhogar', rsuffix='_hh')
# per capita monthly rent
train['rent_per_cap'] = train['v2a1'] / train['tamhog']
test['rent_per_cap'] = test['v2a1'] / test['tamhog']

# per capital tablets
train['tab_per_cap'] = train['v18q'] / train['tamhog']
test['tab_per_cap'] = test['v18q'] / test['tamhog']

# male-female ratio of hh
train['mf_rat'] = train['r4h3'] / train['r4m3']
test['mf_rat'] = test['r4h3'] / test['r4m3']

train['walls_roof_bad'] = train['epared1'] + train['eviv1']
test['walls_roof_bad'] = test['epared1'] + test['eviv1']

# percent of hh under 12 years old
train['child_perc'] = ( train['r4h1'] + train['r4m1'] ) / train['r4t3']
test['child_perc'] = ( test['r4h1'] + test['r4m1'] ) / test['r4t3']

#share of children under 19 that are 12 or under
train['young_perc'] = train['r4t1'] / train['hogar_nin']
test['young_perc'] = test['r4t1'] / test['hogar_nin']

#number of children per adult
train['child_per_adult'] = train['hogar_nin'] / train['hogar_adul']
test['child_per_adult'] = test['hogar_nin'] / test['hogar_adul']

# number of 65+ as percent of total
train['older_perc'] = train['hogar_mayor'] / train['tamviv']
test['older_perc'] = test['hogar_mayor'] / test['tamviv']

# difference between number of poeple living in hh and hh members
train['tamdiff'] = train['tamhog'] - train['tamviv']
test['tamdiff'] = test['tamhog'] - test['tamviv']

## hh has computer and/or television
train['comp_tv'] = train['computer'] + train['television']
test['comp_tv'] = test['computer'] + test['television']
# replace NaNs in train and test data with -1
train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)

train.replace([np.inf, -np.inf], -1, inplace=True)
test.replace([np.inf, -np.inf], -1, inplace=True)
# Create the feature and target arrays for sklearn
keep_cols = [col for col in train.columns if col[:3] != 'SQB']
keep_cols = [col for col in keep_cols if col not in ['idhogar', 'agesq', 'Target']]

X = train.loc[:, keep_cols].values
y = train.Target.values

X_test = test.loc[:, keep_cols].values
gbm = GradientBoostingClassifier(n_estimators=100).fit(X,y)
fi = pd.DataFrame({
    'importance':gbm.feature_importances_.round(5)
}, index=train.loc[:, keep_cols].columns)

fi.sort_values('importance', ascending=False, inplace=True)

fi.iloc[1:30, ].plot.bar(legend=None, figsize=(17,5));
plt.title('Feature Importance');
skf = StratifiedKFold(n_splits=3)
skf.split(X, y)
logmod = LogisticRegression(class_weight='balanced')

param_grid = { 'C': [0.5, 1] }

log_grid = GridSearchCV(logmod, cv=skf, param_grid=param_grid, scoring='f1_macro')
log_grid.fit(X, y);
print(log_grid.best_params_)
print(log_grid.best_score_)
params = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 4, 5, 6],
    'n_estimators': [500, 1000],
    'colsample_bytree':[0.5, 1],
}
lgb_mod = lgb.LGBMClassifier(class_weight='balanced')

lgbm_grid = GridSearchCV(estimator=lgb_mod, cv=skf, param_grid=params, scoring='f1_macro', verbose=1, n_jobs=2)
lgbm_grid.fit(X, y);
print('Best lightgbm parameters: {}'.format(lgbm_grid.best_params_))
print('Best tuning score: {}'.format(lgbm_grid.best_score_))
preds_log = log_grid.predict(X_test)
preds_lgbm = lgbm_grid.predict(X_test)
# plot distribution of predictions from models
plt.subplot(2, 2, 1)
sns.countplot(y, color='red')
plt.title('Distribution of Training Labels')
plt.subplot(2, 2, 2)
sns.countplot(preds_log, color='red')
plt.title('Distribution of Logistic Predictions')
plt.subplot(2, 2, 3)
sns.countplot(preds_lgbm, color='red')
plt.title('Distribution of LightGBM Predictions')
plt.subplots_adjust(top=2, right=2)
# create final submission dataframe and write to csv
sub_log = pd.DataFrame({'Id':test.index, 'Target':preds_log})
sub_lgbm = pd.DataFrame({'Id':test.index, 'Target':preds_lgbm})

#sub_log.to_csv('sub_log1.csv', index=False)
sub_lgbm.to_csv('sub_lgbm1.csv', index=False)