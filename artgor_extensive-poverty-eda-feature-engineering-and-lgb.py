import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
pd.set_option('max_columns', 150)
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=10)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
train.shape, test.shape
train.head(10)
for col in train.columns[:-1]:
    if train[col].isnull().any():
        print('Column {0} has {1:.2f}% null values in train set.'.format(col, np.sum(train[col].isnull()) * 100 / train.shape[0]))
    if test[col].isnull().any():
        print('Column {0} has {1:.2f}% null values in test set.'.format(col, np.sum(test[col].isnull()) * 100 / test.shape[0]))
one_value_column = [i for i in train.columns if train[i].nunique() == 1][0]
print(f'Column "{one_value_column}" has only one unique value in train set.')
print('{0} columns in train set are binary.'.format(sum([1 for i in train.columns if train[i].nunique() == 2])))
idhogars = train.groupby(['idhogar']).agg({'Target': ['count', 'min', 'max']}).reset_index()
idhogars[idhogars['Target']['min'] != idhogars['Target']['max']].head(20)
for i in idhogars[idhogars['Target']['min'] != idhogars['Target']['max']]['idhogar'].unique():
    correct_value = train.loc[(train['idhogar'] == i) & (train['parentesco1'] == 1), 'Target'].values[0]
    train.loc[train['idhogar'] == i, 'Target'] = correct_value
sns.countplot(x="Target", data=train);
# Let's create a short train set which contains only one line per household for correct analysis and visualization of household features.
train_short = train.drop_duplicates('idhogar')
train_short.groupby('Target')['v2a1'].mean()
sns.boxplot(x="Target", y="v2a1", data=train_short);
print('Mean monthly rate for households with different sizes of household separately by poverty level.')
sns.factorplot(x="tamhog", y="v2a1", col="Target", data=train_short, kind="bar");
print('Counts of households with different sizes of household separately by poverty level.')
sns.factorplot(x="tamhog", col="Target", data=train_short, kind="count");
print('Overcrowding rate by bedrooms and rooms.')
train_short.groupby('tamhog').agg({'hacdor': 'mean', 'hacapo': 'mean'}).style.background_gradient(cmap='bwr', low=.5, high=0)
train_short.groupby('tamhog').agg({'v14a': 'mean', 'refrig': 'mean', 'v18q': 'mean'})
sns.boxplot(x='Target', y='escolari', data = train);
plt.title('Years of schooling per household poverty level.')
def combine_features(data, cols=[], name=''):
    df = data.copy()
    for i, col in enumerate(cols):
        print(i + 1, col)
    df[cols] = df[cols].multiply([i for i in range(1, len(cols) + 1)], axis=1)
    df[name] = df[cols].sum(axis=1)
    df.drop(cols, axis=1, inplace=True)
    return df
train_new = combine_features(train, cols=[col for col in train.columns if col.startswith('pared')], name='wall')
print('Wall type count by target.');
sns.factorplot("wall", col="Target", col_wrap=4, data=train_new, kind="count");
train_new = combine_features(train_new, cols=[col for col in train_new.columns if col.startswith('piso')], name='floor')
print('Floor type count by target.');
sns.factorplot("floor", col="Target", col_wrap=4, data=train_new, kind="count");
train_new = combine_features(train_new, cols=[col for col in train_new.columns if col.startswith('techo')], name='roof')
print('Roof type count by target.');
sns.factorplot("roof", col="Target", col_wrap=4, data=train_new, kind="count");
train_new = combine_features(train_new, cols=[col for col in train_new.columns if col.startswith('abasta')], name='water')
print('Water provision type count by target.');
sns.factorplot("water", col="Target", col_wrap=4, data=train_new, kind="count");
train_new = combine_features(train_new, cols=['public', 'planpri', 'noelec', 'coopele'], name='electricity')
print('Electricity source type count by target.');
sns.factorplot("electricity", col="Target", col_wrap=4, data=train_new, kind="count");
train_new = combine_features(train_new, cols=[col for col in train_new.columns if col.startswith('sanitario')], name='toilet')
print('Toilet connection type count by target.');
sns.factorplot("toilet", col="Target", col_wrap=4, data=train_new, kind="count");
train_new = combine_features(train_new, cols=[col for col in train_new.columns if col.startswith('energcocinar')], name='cooking')
print('Cooking sourse energy type count by target.');
sns.factorplot("cooking", col="Target", col_wrap=4, data=train_new, kind="count");
train_new = combine_features(train_new, cols=[col for col in train_new.columns if col.startswith('elimbasu')], name='rubbish')
print('Rubbish disposal type count by target.');
sns.factorplot("rubbish", col="Target", col_wrap=4, data=train_new, kind="count");
train_new = combine_features(train_new, cols=[col for col in train_new.columns if col.startswith('epared')], name='wall_quality')
print('Wall quality type count by target.');
sns.factorplot("wall_quality", col="Target", col_wrap=4, data=train_new, kind="count");
train_new = combine_features(train_new, cols=[col for col in train_new.columns if col.startswith('etecho')], name='roof_quality')
print('Roof quality type count by target.');
sns.factorplot("roof_quality", col="Target", col_wrap=4, data=train_new, kind="count");
train_new = combine_features(train_new, cols=[col for col in train_new.columns if col.startswith('eviv')], name='floor_quality')
print('Floor quality type count by target.');
sns.factorplot("floor_quality", col="Target", col_wrap=4, data=train_new, kind="count");
train_new = combine_features(train_new, cols=[col for col in train_new.columns if col.startswith('estadocivil')], name='family')
print('Family status count by target.');
sns.factorplot("family", col="Target", col_wrap=4, data=train_new, kind="count");
train_new = combine_features(train_new, cols=[col for col in train_new.columns if col.startswith('parentesco')], name='family_status')
train_new['family_status'].value_counts().plot(kind='bar');
plt.title('Family status count.');
train_new = combine_features(train_new, cols=[col for col in train_new.columns if col.startswith('instlevel')], name='education')
print('Education level count by target.');
sns.factorplot("education", col="Target", col_wrap=4, data=train_new, kind="count");
train_new = combine_features(train_new, cols=[col for col in train_new.columns if col.startswith('tipovivi')], name='home_own')
print('Home ownership type count by target.');
sns.factorplot("home_own", col="Target", col_wrap=4, data=train_new, kind="count");
train_new = combine_features(train_new, cols=[col for col in train_new.columns if col.startswith('lugar')], name='region')
print('Region count by target.');
sns.factorplot("region", col="Target", col_wrap=4, data=train_new, kind="count");
train['edjefe'].unique(),train['SQBedjefe'].unique()
print(train.loc[train.edjefe == 'yes', 'SQBedjefe'].unique())
print(train.loc[train.edjefe == 'no', 'SQBedjefe'].unique())
train['edjefe'] = train['edjefe'].apply(lambda x: 0 if x == 'no' else (1 if x == 'yes' else int(x)))
train_new['edjefe'] = train_new['edjefe'].apply(lambda x: 0 if x == 'no' else (1 if x == 'yes' else int(x)))

# fixing edjefa as well
train['edjefa'] = train['edjefa'].apply(lambda x: 0 if x == 'no' else (1 if x == 'yes' else int(x)))
train_new['edjefa'] = train_new['edjefa'].apply(lambda x: 0 if x == 'no' else (1 if x == 'yes' else int(x)))
train['dependency'].unique(),train['SQBdependency'].unique()
print(train.loc[train.dependency == 'yes', 'SQBdependency'].unique())
print(train.loc[train.dependency == 'no', 'SQBdependency'].unique())
train['dependency'] = train['dependency'].apply(lambda x: 0 if x == 'no' else (1 if x == 'yes' else float(x)))
train_new['dependency'] = train_new['dependency'].apply(lambda x: 0 if x == 'no' else (1 if x == 'yes' else float(x)))
for col in ['v2a1', 'v18q1', 'rez_esc', 'meaneduc', 'SQBmeaned']:
    train[col] = train[col].fillna(0)
    train_new[col] = train_new[col].fillna(0)
def create_new_features(data, new=False):
    data['v2a1'] = np.log1p(data['v2a1'])
    data['rent_per_room'] = data['v2a1'] / data['rooms']
    data['males_to_females'] = data['r4h3'] / data['r4m3']
    data['persons_per_room'] = data['tamviv'] / data['rooms']
    
    data['bedrooms_to_rooms'] = data['bedrooms']/data['rooms']
    data['r4t3_to_tamhog'] = data['r4t3']/data['tamhog']
    data['r4t3_to_rooms'] = data['r4t3']/data['rooms']
    data['v2a1_to_r4t3'] = data['v2a1']/data['r4t3']
    data['v2a1_to_r4t3'] = data['v2a1']/(data['r4t3'] - data['r4t1'])
    data['hhsize_to_rooms'] = data['hhsize']/data['rooms']
    data['rent_to_hhsize'] = data['v2a1']/data['hhsize']
    
    for col in ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin',
       'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq', 'v2a1', 'meaneduc']:
        data['idhogar_mean_' + col] = data.groupby('idhogar')[col].transform('mean')
        data['idhogar_std_' + col] = data.groupby('idhogar')[col].transform('std')
        data['idhogar_sum_' + col] = data.groupby('idhogar')[col].transform('sum')
    
    if new:
        for col in ['wall', 'floor', 'roof', 'water', 'electricity', 'toilet', 'cooking', 'rubbish', 'wall_quality', 'roof_quality', 'floor_quality', 'family', 'family_status',
                    'education', 'home_own', 'region']:
            data[col + '_rent_mean'] = data.groupby(col)['v2a1'].transform('mean')
    return data
train = create_new_features(train)
train_new = create_new_features(train_new, new=True)
test_new = combine_features(test, cols=[col for col in test.columns if col.startswith('pared')], name='wall')
test_new = combine_features(test_new, cols=[col for col in test_new.columns if col.startswith('piso')], name='floor')
test_new = combine_features(test_new, cols=[col for col in test_new.columns if col.startswith('techo')], name='roof')
test_new = combine_features(test_new, cols=[col for col in test_new.columns if col.startswith('abasta')], name='water')
test_new = combine_features(test_new, cols=['public', 'planpri', 'noelec', 'coopele'], name='electricity')
test_new = combine_features(test_new, cols=[col for col in test_new.columns if col.startswith('sanitario')], name='toilet')
test_new = combine_features(test_new, cols=[col for col in test_new.columns if col.startswith('energcocinar')], name='cooking')
test_new = combine_features(test_new, cols=[col for col in test_new.columns if col.startswith('elimbasu')], name='rubbish')
test_new = combine_features(test_new, cols=[col for col in test_new.columns if col.startswith('epared')], name='wall_quality')
test_new = combine_features(test_new, cols=[col for col in test_new.columns if col.startswith('etecho')], name='roof_quality')
test_new = combine_features(test_new, cols=[col for col in test_new.columns if col.startswith('eviv')], name='floor_quality')
test_new = combine_features(test_new, cols=[col for col in test_new.columns if col.startswith('estadocivil')], name='family')
test_new = combine_features(test_new, cols=[col for col in test_new.columns if col.startswith('parentesco')], name='family_status')
test_new = combine_features(test_new, cols=[col for col in test_new.columns if col.startswith('instlevel')], name='education')
test_new = combine_features(test_new, cols=[col for col in test_new.columns if col.startswith('tipovivi')], name='home_own')
test_new = combine_features(test_new, cols=[col for col in test_new.columns if col.startswith('lugar')], name='region')
test['edjefe'] = test['edjefe'].apply(lambda x: 0 if x == 'no' else (1 if x == 'yes' else int(x)))
test_new['edjefe'] = test_new['edjefe'].apply(lambda x: 0 if x == 'no' else (1 if x == 'yes' else int(x)))
test['dependency'] = test['dependency'].apply(lambda x: 0 if x == 'no' else (1 if x == 'yes' else float(x)))
test_new['dependency'] = test_new['dependency'].apply(lambda x: 0 if x == 'no' else (1 if x == 'yes' else float(x)))
# fixing edjefa as well
test['edjefa'] = test['edjefa'].apply(lambda x: 0 if x == 'no' else (1 if x == 'yes' else int(x)))
test_new['edjefa'] = test_new['edjefa'].apply(lambda x: 0 if x == 'no' else (1 if x == 'yes' else int(x)))
for col in ['v2a1', 'v18q1', 'rez_esc', 'meaneduc', 'SQBmeaned']:
    test[col] = test[col].fillna(0)
    test_new[col] = test_new[col].fillna(0)
test = create_new_features(test)
test_new = create_new_features(test_new, new=True)
le.fit(list(train_new['idhogar'].values) + list(test_new['idhogar'].values))
train['idhogar'] = le.transform(train['idhogar'])
train_new['idhogar'] = le.transform(train_new['idhogar'])
test['idhogar'] = le.transform(test['idhogar'])
test_new['idhogar'] = le.transform(test_new['idhogar'])
X = train.drop(['Id', 'Target'], axis=1)
y = train['Target']
X_new = train_new.drop(['Id', 'Target'], axis=1)
y_new = train_new['Target']
X_test = test.drop(['Id'], axis=1)
X_test_new = test_new.drop(['Id'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y - 1, test_size=0.10, random_state=42, stratify=y)
params = {
    
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'max_depth': 5,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.5,
    'bagging_freq': 5,
    'num_threads': 6,
    'lambda_l2': 1.0,
    'num_class': 4,}
model = lgb.train(params, lgb.Dataset(X_train, y_train), 5000, [lgb.Dataset(X_train, y_train), lgb.Dataset(X_valid, y_valid)], verbose_eval=500, early_stopping_rounds=50)
lgb.plot_importance(model, max_num_features=30, figsize=(24, 18));
X_train, X_valid, y_train, y_valid = train_test_split(X_new, y_new - 1, test_size=0.10, random_state=42, stratify=y_new)
params = {
    'boosting': 'gbdt',
    'application': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 4,
    'learning_rate': 0.05,
    'num_leaves': 7,
    'max_depth': 3,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': 2,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'num_threads': 6}
model1 = lgb.train(params, lgb.Dataset(X_train, y_train), 5000, [lgb.Dataset(X_train, y_train), lgb.Dataset(X_valid, y_valid)], verbose_eval=500, early_stopping_rounds=50)
lgb.plot_importance(model1, max_num_features=30, figsize=(24, 18));
%%time
## predicting on folds
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'max_depth': 5,
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.5,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 6,
    'lambda_l2': 1.0,
    'min_gain_to_split': 0,
    'num_class': 4,}
prediction = np.zeros((X_test.shape[0], 4))
for i, (train_i, test_i) in enumerate(kf.split(X, y)):
    print(f'Fold {i}.')
    X_train = X.values[train_i]
    y_train = y.values[train_i] - 1
    X_valid = X.values[test_i]
    y_valid = y.values[test_i] - 1
    model = lgb.train(params, lgb.Dataset(X_train, y_train), 5000, [lgb.Dataset(X_train, y_train), lgb.Dataset(X_valid, y_valid)], verbose_eval=500, early_stopping_rounds=50)
    pred = model.predict(X_test)
    prediction += pred

%%time
## predicting on folds
params = {
    'boosting': 'gbdt',
    'application': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 4,
    'learning_rate': 0.01,
    'num_leaves': 9,
    'max_depth': 128,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'scale_pos_weight': 2,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'num_threads': 6}
prediction1 = np.zeros((X_test.shape[0], 4))
for i, (train_i, test_i) in enumerate(kf.split(X_new, y_new)):
    print(f'Fold {i}.')
    X_train = X_new.values[train_i]
    y_train = y_new.values[train_i] - 1
    X_valid = X_new.values[test_i]
    y_valid = y_new.values[test_i] - 1
    model = lgb.train(params, lgb.Dataset(X_train, y_train), 5000, [lgb.Dataset(X_train, y_train), lgb.Dataset(X_valid, y_valid)], verbose_eval=500, early_stopping_rounds=50)
    pred = model.predict(X_test_new)
    prediction1 += pred

full_prediction = np.argmax(prediction + prediction1, axis=1)
submission['Target'] = full_prediction + 1
submission.to_csv('blend.csv', index=False)
