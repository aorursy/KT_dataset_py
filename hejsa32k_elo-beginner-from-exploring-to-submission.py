
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 400)
%matplotlib inline

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', parse_dates=["first_active_month"])
test = pd.read_csv('../input/test.csv', parse_dates=["first_active_month"])

train.head(5)

train_test = pd.concat([train, test], ignore_index=False, keys=['train', 'test'], sort=True)
print('Rows in train: {}'.format(len(train_test[['card_id']])))
print('Unique card_ids in train: {}'.format(len(train_test.card_id.value_counts())))

# no issues regarding duplicate card_id's
print('train missing values:')
print(train_test.loc['train'].isnull().sum())

print('test missing values:')
print(train_test.loc['test'].isnull().sum())

print('missing value:')
print(train_test[train_test['first_active_month'].isnull()])
# calculate mean of datetime series

mean_date = (train_test.first_active_month - train_test.first_active_month.min()).mean() + train_test.first_active_month.min()

# fill with mean
train_test['first_active_month'] = train_test['first_active_month'].fillna(mean_date)
# data pull seems to be 2018-04-30 as this is max in new_merchant_transactions

train_test['Days_since_first_active'] = (pd.to_datetime('2018-04-30') - train_test['first_active_month']).dt.days

train_test['Months_since_first_active'] = train_test['Days_since_first_active'] // 30 #floor division

#clean dataset

keep_columns = ['card_id', 'first_active_month', 'feature_1', 'feature_2', 'feature_3', 'target', 
                'Months_since_first_active']

train_test = train_test[keep_columns]
import seaborn as sns; sns.set() #set plot theme
import matplotlib.pyplot as plt

print(train_test.loc['train'].describe())
print(train_test.loc['test'].describe())
#feature_1
feature_1_train = train_test.loc['train'].feature_1.value_counts().sort_index()
feature_1_train = feature_1_train / len(train_test.loc['train'])

feature_1_test = train_test.loc['test'].feature_1.value_counts().sort_index()
feature_1_test = feature_1_test / len(train_test.loc['test'])

#feature_2
feature_2_train = train_test.loc['train'].feature_2.value_counts().sort_index()
feature_2_train = feature_2_train / len(train_test.loc['train'])

feature_2_test = train_test.loc['test'].feature_2.value_counts().sort_index()
feature_2_test = feature_2_test / len(train_test.loc['test'])

#feature_3
feature_3_train = train_test.loc['train'].feature_3.value_counts().sort_index()
feature_3_train = feature_3_train / len(train_test.loc['train'])

feature_3_test = train_test.loc['test'].feature_3.value_counts().sort_index()
feature_3_test = feature_3_test / len(train_test.loc['test'])

#Months_since_first_active
Months_since_first_active_train = train_test.loc['train'].Months_since_first_active.value_counts().sort_index()
Months_since_first_active_train = Months_since_first_active_train / len(train_test.loc['train'])

Months_since_first_active_test = train_test.loc['test'].Months_since_first_active.value_counts().sort_index()
Months_since_first_active_test = Months_since_first_active_test / len(train_test.loc['test'])
fig, ax = plt.subplots(2, 2, figsize=(14,14))

# =============================================================================
# #plotting independt variables checking for consistency vs train and test data
# =============================================================================

#plotting feature_1
ax[0,0].bar(feature_1_train.index, feature_1_train, label='train', alpha=0.3)
ax[0,0].bar(feature_1_test.index, feature_1_test, label='test', alpha=0.3)
ax[0,0].legend()
ax[0,0].set_title('feature_1')

#plotting feature_2
ax[0,1].bar(feature_2_train.index, feature_2_train, label='train', alpha=0.3)
ax[0,1].bar(feature_2_test.index, feature_2_test, label='test', alpha=0.3)
ax[0,1].legend()
ax[0,1].set_title('feature_2')

#plotting feature_3
ax[1,0].bar(feature_3_train.index, feature_3_train, label='train', alpha=0.3)
ax[1,0].bar(feature_3_test.index, feature_3_test, label='test', alpha=0.3)
ax[1,0].legend()
ax[1,0].set_title('feature_3')

#plotting Months_since_first_active
ax[1,1].bar(Months_since_first_active_train.index, Months_since_first_active_train, label='train', alpha=0.3)
ax[1,1].bar(Months_since_first_active_test.index, Months_since_first_active_test, label='test', alpha=0.3)
ax[1,1].legend()
ax[1,1].set_title('Months_since_first_active');
f, ax = plt.subplots(figsize=(14,7))
sns.distplot(train_test.loc['train'].target)
plt.title('target distribution');
#plotting feature_1 - feature_2 vs target
fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(14,7))

fig = sns.boxplot(x='feature_1', y='target', data=train_test.loc['train'],ax=ax1)
ax1.set_title('target distribution [feature_1]')

sns.boxplot(x='feature_2', y='target', data=train_test.loc['train'],ax=ax2)
ax2.set_title('target distribution [feature_2]')

#plotting feature_1 - feature_2 vs target
fig = plt.subplots(nrows=1, ncols=1, figsize=(7,7))
fig = sns.boxplot(x='feature_3', y='target', data=train_test.loc['train'])
plt.title('target distribution [feature_3]')

#plotting jointplot, taking every 60th observation to lower load time
fig = plt.subplots(nrows=1, ncols=1, figsize=(14,7))
fig = sns.swarmplot(x='Months_since_first_active', y='target', data=train_test.loc['train'::60])
plt.title('Months_since_first_active vs. target')
plt.xticks(rotation='90');
hist_transactions = pd.read_csv('../input/historical_transactions.csv', parse_dates=['purchase_date'])

#function found in https://www.kaggle.com/fabiendaniel/elo-world
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

hist_transactions = reduce_mem_usage(hist_transactions)
                                
print('missing values %:') 
print(hist_transactions.isnull().sum() / len(hist_transactions))
##### authorized_flag
hist_transactions.authorized_flag.head()
hist_transactions['authorized_flag'] = hist_transactions['authorized_flag'].map({'Y' : 1, 'N' : 0})

fig = plt.subplots(nrows=1, ncols=1, figsize=(7,7))
sns.barplot(x = hist_transactions['authorized_flag'].value_counts().index,
            y = hist_transactions['authorized_flag'].value_counts() / len(hist_transactions),
            order = hist_transactions['authorized_flag'].value_counts().index)
plt.ylabel('freq %')
plt.title('authorized_flag');
##### city_id
hist_transactions.city_id.head(10)
print(hist_transactions.city_id.value_counts().head()) #-1 probalbly missing values, we treet them as a seperate group for now

show = 20
fig = plt.subplots(nrows=1, ncols=1, figsize=(14,7))
sns.barplot(x = hist_transactions['city_id'].value_counts().index[:show],
            y = hist_transactions['city_id'].value_counts()[:show] / len(hist_transactions['city_id']),
            order = hist_transactions['city_id'].value_counts().index[:show])
plt.ylabel('freq %')
plt.title('Most {}th frequent city_ids'.format(show))

#cumulative barplot

city_id_freq = hist_transactions['city_id'].value_counts() / len(hist_transactions['city_id'])

city_id_cum = pd.DataFrame(city_id_freq)
city_id_cum.columns = ['city_id_freq']
 
city_id_cum = city_id_cum.reset_index()

cum_var = np.empty((0,1),float)

for i in city_id_cum.index:
    if i == 0:
        cum_var = np.append(cum_var,(city_id_cum.loc[i,'city_id_freq']))
    else: 
        cum_var = np.append(cum_var, (cum_var[i-1] + city_id_cum.loc[i,'city_id_freq']))

city_id_cum['city_id_cum'] = cum_var    

fig = plt.subplots(nrows=1, ncols=1, figsize=(14,7))
sns.barplot(x = city_id_cum['index'][:show],
            y = city_id_cum['city_id_cum'][:show],
            order = city_id_cum['index'][:show])
plt.ylabel('freq %')
plt.title('Most {}th frequent city_ids (cumulative)'.format(show)); #we see that a relatively small number of cities make up a large part of the transactions
##### state_id
hist_transactions.state_id.head(10)


fig = plt.subplots(nrows=1, ncols=1, figsize=(14,7))
sns.barplot(x = hist_transactions['state_id'].value_counts().index,
            y = hist_transactions['state_id'].value_counts() / len(hist_transactions['state_id']),
            order = hist_transactions['state_id'].value_counts().index)
plt.ylabel('freq %')
plt.title('state_id')

#cumulative barplot

state_id_freq = hist_transactions['state_id'].value_counts() / len(hist_transactions['state_id'])

state_id_cum = pd.DataFrame(state_id_freq)
state_id_cum.columns = ['state_id_freq']
 
state_id_cum = state_id_cum.reset_index()

cum_var = np.empty((0,1),float)

for i in state_id_cum.index:
    if i == 0:
        cum_var = np.append(cum_var,(state_id_cum.loc[i,'state_id_freq']))
    else: 
        cum_var = np.append(cum_var, (cum_var[i-1] + state_id_cum.loc[i,'state_id_freq']))

state_id_cum['state_id_cum'] = cum_var    

fig = plt.subplots(nrows=1, ncols=1, figsize=(14,7))
sns.barplot(x = state_id_cum['index'],
            y = state_id_cum['state_id_cum'],
            order = state_id_cum['index'])
plt.ylabel('freq %')
plt.title('state_id cummulative dist'); #we see that a relatively small number of states make up a large part of the transactions
##### category_1
hist_transactions.category_1.head()
hist_transactions.category_1.unique()

hist_transactions['category_1'] = hist_transactions['category_1'].map({'Y' : 1, 'N' : 0})

fig = plt.subplots(nrows=1, ncols=1, figsize=(7,7))
sns.barplot(x = hist_transactions['category_1'].value_counts().index,
            y = hist_transactions['category_1'].value_counts() / len(hist_transactions['category_1']),
            order = hist_transactions['category_1'].value_counts().index)
plt.ylabel('freq %')
plt.title('category_1')

##### category_2
hist_transactions.category_2.head()
hist_transactions.category_2.value_counts(dropna=False)

hist_transactions['category_2'] = hist_transactions.category_2.fillna(6.0)

fig = plt.subplots(nrows=1, ncols=1, figsize=(7,7))
sns.barplot(x = hist_transactions['category_2'].value_counts().index,
            y = hist_transactions['category_2'].value_counts() / len(hist_transactions['category_2']),
            order = hist_transactions['category_2'].value_counts().index)
plt.ylabel('freq %')
plt.title('category_2')


##### category_3
hist_transactions.category_3.head()
hist_transactions.category_3.value_counts(dropna=False)

#we see relatively small number of NAs. We handle them as a seperate group.
hist_transactions['category_3'] = hist_transactions.category_3.fillna('NA')

hist_transactions.category_3.value_counts(dropna=False)

fig = plt.subplots(nrows=1, ncols=1, figsize=(7,7))
sns.barplot(x = hist_transactions['category_3'].value_counts().index,
            y = hist_transactions['category_3'].value_counts() / len(hist_transactions['category_3']),
            order = hist_transactions['category_3'].value_counts().index)
plt.ylabel('% freq')
plt.title('category_3');

##### installments
hist_transactions.installments.unique()
print('Values:')
print(hist_transactions.installments.value_counts()) 

hist_transactions['installments'] = hist_transactions.installments.replace({-1 : 0, 999 : 0})
hist_transactions.installments.value_counts()

fig = plt.subplots(nrows=1, ncols=1, figsize=(7,7))
sns.barplot(x = hist_transactions['installments'].value_counts().index,
            y = hist_transactions['installments'].value_counts() / len(hist_transactions['installments']),
            order = hist_transactions['installments'].value_counts().index)
plt.xlabel('# of installments')
plt.ylabel('% freq')
plt.title('installments');
##### merchant_category_id
hist_transactions.merchant_category_id.value_counts().head(15)

fig = plt.subplots(nrows=1, ncols=1, figsize=(14,7))
sns.barplot(x = hist_transactions['merchant_category_id'].value_counts().index[:30],
            y = hist_transactions['merchant_category_id'].value_counts()[:30] / len(hist_transactions['merchant_category_id']),
            order = hist_transactions['merchant_category_id'].value_counts().index[:30])
plt.ylabel('freq %')
plt.xlabel('merchant_category_id')
plt.title('merchant_category_id')

#cumulative barplot

merchant_category_id_freq = hist_transactions['merchant_category_id'].value_counts() / len(hist_transactions['merchant_category_id'])

merchant_category_id_cum = pd.DataFrame(merchant_category_id_freq)
merchant_category_id_cum.columns = ['merchant_category_id_freq']
 
merchant_category_id_cum = merchant_category_id_cum.reset_index()

cum_var = np.empty((0,1),float)

for i in merchant_category_id_cum.index:
    if i == 0:
        cum_var = np.append(cum_var,(merchant_category_id_cum.loc[i,'merchant_category_id_freq']))
    else: 
        cum_var = np.append(cum_var, (cum_var[i-1] + merchant_category_id_cum.loc[i,'merchant_category_id_freq']))

merchant_category_id_cum['merchant_category_id_cum'] = cum_var    

fig = plt.subplots(nrows=1, ncols=1, figsize=(14,7))
sns.barplot(x = merchant_category_id_cum['index'][:30],
            y = merchant_category_id_cum['merchant_category_id_cum'][:30],
            order = merchant_category_id_cum['index'][:30])
plt.xlabel('merchant_category_id')
plt.ylabel('freq %')
plt.title('merchant_category_id cummulative dist'); #we see that a relatively small number of merchant_category_id's make up a large part of the transactions
##### month_lag / purchase_date
hist_transactions['month_lag'].head()
hist_transactions['month_lag'].value_counts()

#calculate so it is consistent with the formula used on the training set

hist_transactions['Days_since_trans'] = (pd.to_datetime('2018-03-01') - hist_transactions['purchase_date']).dt.days

hist_transactions['MonthsSince_trans'] = hist_transactions['Days_since_trans'] // 30 #floor division

fig = plt.subplots(nrows=1, ncols=1, figsize=(7,7))
sns.barplot(x = hist_transactions['MonthsSince_trans'].value_counts().index,
            y = hist_transactions['MonthsSince_trans'].value_counts() / len(hist_transactions['MonthsSince_trans']),)
plt.ylabel('% freq')
plt.xlabel('# months since transaction')
plt.title('MonthsSince_trans');

show = 20
fig = plt.subplots(nrows=1, ncols=1, figsize=(14,7))
sns.barplot(x = hist_transactions['purchase_date'].dt.date.value_counts().index[:show],
            y = hist_transactions['purchase_date'].dt.date.value_counts()[:show] / len(hist_transactions['purchase_date']),
            order = hist_transactions['purchase_date'].dt.date.value_counts().index[:show])
plt.xticks(rotation='45')
plt.ylabel('% freq')
plt.title('Most {}th frequent purchase_dates'.format(show));
##### purchase_amount

print(hist_transactions['purchase_amount'].describe())
##### subsector_id

hist_transactions.subsector_id.value_counts().head(15)

fig = plt.subplots(nrows=1, ncols=1, figsize=(14,7))
sns.barplot(x = hist_transactions['subsector_id'].value_counts().index[:30],
            y = hist_transactions['subsector_id'].value_counts()[:30] / len(hist_transactions['subsector_id']),
            order = hist_transactions['subsector_id'].value_counts().index[:30])
plt.xlabel('subsector_id')
plt.ylabel('freq %')
plt.title('subsector_id')

#cumulative barplot

subsector_id_freq = hist_transactions['subsector_id'].value_counts() / len(hist_transactions['subsector_id'])

subsector_id_cum = pd.DataFrame(subsector_id_freq)
subsector_id_cum.columns = ['subsector_id_freq']
 
subsector_id_cum = subsector_id_cum.reset_index()

cum_var = np.empty((0,1),float)

for i in subsector_id_cum.index:
    if i == 0:
        cum_var = np.append(cum_var,(subsector_id_cum.loc[i,'subsector_id_freq']))
    else: 
        cum_var = np.append(cum_var, (cum_var[i-1] + subsector_id_cum.loc[i,'subsector_id_freq']))

subsector_id_cum['subsector_id_cum'] = cum_var    

fig = plt.subplots(nrows=1, ncols=1, figsize=(14,7))
sns.barplot(x = subsector_id_cum['index'][:30],
            y = subsector_id_cum['subsector_id_cum'][:30],
            order = subsector_id_cum['index'][:30])
plt.xlabel('subsector_id')
plt.ylabel('freq %')
plt.title('subsector_id cummulative dist');

#### merchant_id

hist_transactions.merchant_id.value_counts().head(15)
print('Number of unique merchant_id~s: {}'.format(len(hist_transactions.merchant_id.unique())))

show = 20
fig = plt.subplots(nrows=1, ncols=1, figsize=(14,7))
sns.barplot(x = hist_transactions['merchant_id'].value_counts().index[:show],
            y = hist_transactions['merchant_id'].value_counts()[:show] / len(hist_transactions['merchant_id']),
            order = hist_transactions['merchant_id'].value_counts().index[:show])
plt.xticks(rotation='60')
plt.ylabel('freq %')
plt.xlabel('merchant_id')
plt.title('Most {}th frequent merchant_id~s'.format(show))

#cumulative barplot

merchant_id_freq = hist_transactions['merchant_id'].value_counts() / len(hist_transactions['merchant_id'])

merchant_id_cum = pd.DataFrame(merchant_id_freq)
merchant_id_cum.columns = ['merchant_id_freq']
 
merchant_id_cum = merchant_id_cum.reset_index()

cum_var = np.empty((0,1),float)

for i in merchant_id_cum.index:
    if i == 0:
        cum_var = np.append(cum_var,(merchant_id_cum.loc[i,'merchant_id_freq']))
    else: 
        cum_var = np.append(cum_var, (cum_var[i-1] + merchant_id_cum.loc[i,'merchant_id_freq']))

merchant_id_cum['merchant_id_cum'] = cum_var    

fig = plt.subplots(nrows=1, ncols=1, figsize=(14,7))
sns.barplot(x = merchant_id_cum['index'][:show],
            y = merchant_id_cum['merchant_id_cum'][:show],
            order = merchant_id_cum['index'][:show])
plt.xticks(rotation='60')
plt.ylabel('freq %')
plt.xlabel('merchant_id')
plt.title('merchant_id cummulative dist'); 
agg_func = {
        'MonthsSince_trans' : ['min', 'max', 'mean', 'std'],
        'purchase_date' : ['count'],
        'authorized_flag': ['min', 'max', 'sum', 'mean'],
        'category_3': ['nunique'],
        'installments': ['min', 'max', 'mean', 'sum', 'std'],
        'category_1' : ['min', 'max', 'mean'],
        'merchant_category_id' : ['nunique'],
        'subsector_id' : ['nunique'],
        'merchant_id' : ['nunique'],
        'purchase_amount' : ['min', 'max', 'sum', 'mean', 'std'],
        'city_id' : ['nunique'],
        'state_id' : ['nunique'],
        'category_2' : ['nunique', 'min', 'max', 'mean']
        }

hist_trans_agg = hist_transactions.groupby(['card_id']).agg(agg_func)

hist_trans_agg.columns = ['hist_' + '_'.join(col).strip() 
                           for col in hist_trans_agg.columns.values]

hist_trans_agg.reset_index(inplace=True)

del hist_transactions
train = train_test.loc['train'].set_index('card_id').drop('first_active_month', axis=1)
test = train_test.loc['test'].set_index('card_id').drop('first_active_month', axis=1)

train = train.merge(hist_trans_agg, left_on='card_id', right_on='card_id',
                              how='left').set_index('card_id')

test = test.merge(hist_trans_agg, left_on='card_id', right_on='card_id',
                              how='left').set_index('card_id')
# =============================================================================
# #transforming categorical features
# =============================================================================
features = list(train.columns)
categorical_feats = [col for col in features if 'feature_' in col]
for col in categorical_feats:
    print(col, 'have', train[col].value_counts().shape[0], 'categories.')

from sklearn.preprocessing import LabelEncoder
for col in categorical_feats:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))

df_all = pd.concat([train, test])
df_all = pd.get_dummies(df_all, columns=categorical_feats)

len_train = train.shape[0]

train = df_all[:len_train]
test = df_all[len_train:]
#inspired by https://www.kaggle.com/garethjns/microsoft-lightgbm-with-parameter-tuning-0-823

import lightgbm as lgb

#set params

lgb_params = {'max_depth' : 6,
          'objective': 'regression',
          'num_leaves': 55,
          'max_bin' : 60,
          'learning_rate': 0.05,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'subsample_for_bin': 200,
          'reg_alpha': 1,
          'reg_lambda': 1,    
          'min_child_weight': 1,
          'min_child_samples': 12,
          'min_split_gain': 0.5,
          'scale_pos_weight': 1,
          'metric' : 'rmse'}

gridParams = {
    'learning_rate': [0.05],
    'n_estimators': [40],
    'num_leaves': [8,16,32,64],
    'objective' : ['regression'],
    'random_state' : [501], # Updated from 'seed'
    'colsample_bytree' : [0.6, 0.8],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.5],
    'reg_lambda' : [1,1.2,1.5],
    }


mdl = lgb.LGBMRegressor(boosting_type= 'gbdt',
          n_jobs = 3, # Updated from 'nthread'
          silent = True,
          **lgb_params)

# =============================================================================
# #grid search
# =============================================================================

#dropping 10 features with low importance from first run
to_drop = ['hist_authorized_flag_max', 'hist_category_1_min', 'hist_authorized_flag_min', 
            'feature_3_1', 'feature_1_3', 'hist_category_1_max', 'feature_3_0', 'feature_1_2', 
            'hist_category_2_max', 'feature_1_0']

target = train['target']

train = train.drop(to_drop + ['target'], axis=1)
test = test.drop(to_drop + ['target'], axis=1)

from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

       
mean_squared_error_ = make_scorer(mean_squared_error, greater_is_better=False)

#setting up GridSearchCV
grid = GridSearchCV(mdl, gridParams,
                    verbose=4,
                    cv=4,
                    n_jobs=2,  #paralel computing
                    scoring=mean_squared_error_) 

grid.fit(train, target)
    
# Print the best parameters found

print('mean square error for best params {}'.format(np.abs(grid.best_score_)))
print('root mean square error for best params {}'.format(np.sqrt(np.abs(grid.best_score_))))

lgb_params['colsample_bytree'] = grid.best_params_['colsample_bytree']
lgb_params['learning_rate'] = grid.best_params_['learning_rate']
# params['max_bin'] = grid.best_params_['max_bin']
lgb_params['num_leaves'] = grid.best_params_['num_leaves']
lgb_params['reg_alpha'] = grid.best_params_['reg_alpha']
lgb_params['reg_lambda'] = grid.best_params_['reg_lambda']
lgb_params['subsample'] = grid.best_params_['subsample']


print('Fitting with params: ')
print(lgb_params)
# =============================================================================
# Model Build / Best Params
# =============================================================================

# inspiration from https://www.kaggle.com/chocozzz/simple-data-exploration-with-python-lb-3-764

from sklearn.model_selection import KFold


FOLDs = KFold(n_splits=5, shuffle=True, random_state=1987)

oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

features_lgb = list(train.columns)
feature_importance_df_lgb = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train)):
    trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])

    print("LGB " + str(fold_) + "-" * 50)
    num_round = 10000
    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 60)
    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df_lgb = pd.DataFrame()
    fold_importance_df_lgb["feature"] = features_lgb
    fold_importance_df_lgb["importance"] = clf.feature_importance()
    fold_importance_df_lgb["fold"] = fold_ + 1
    feature_importance_df_lgb = pd.concat([feature_importance_df_lgb, fold_importance_df_lgb], axis=0)
    predictions_lgb += clf.predict(test, num_iteration=clf.best_iteration) / FOLDs.n_splits

print(np.sqrt(mean_squared_error(oof_lgb, target)))

# =============================================================================
# inspecting var importance
# =============================================================================


cols = (feature_importance_df_lgb[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df_lgb.loc[feature_importance_df_lgb.feature.isin(cols)]

plt.figure(figsize=(14,14))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
# =============================================================================
# best_features dataframe
# =============================================================================

#best_features = best_features.drop('fold', axis=1).groupby("feature").mean().sort_values(by="importance", ascending=True)

#print(best_features.index[:10])

# =============================================================================
# submission
# =============================================================================

sub_df = pd.read_csv('../input/sample_submission.csv')
sub_df["target"] = predictions_lgb 
#sub_df.to_csv("submission_lgb.csv", index=False)

print(sub_df.head(10))