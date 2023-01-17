# Import libraries



import pandas as pd

import numpy as np



# Check the folder structure

import os

import datetime

print(os.listdir('../input'))



#Visualisations

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')



import warnings

warnings.filterwarnings("ignore")



#Model

from sklearn.model_selection import KFold, StratifiedKFold

import lightgbm as lgb



#Metrics

from sklearn.metrics import roc_auc_score, roc_curve
#Let's load the train and test datasets

df_train = pd.read_csv('../input/train.csv', parse_dates=['Date'])

df_test = pd.read_csv('../input/test-3.csv', index_col='index', parse_dates=['Date'])
df_train.head()
df_test.head()
df_train.info()
df_test.info()
df_test = df_test.drop('Referee', axis=1)
df_train.describe()
df_test.describe()
df_train.shape, df_test.shape
df_train['FTR'].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))

sns.barplot(x=df_train['FTR'].value_counts().index, y = df_train['FTR'].value_counts().values, color='orange')

ax.set_title('Distribution of Full Time Result', fontsize=14)

ax.set_xlabel('Full Time Result', fontsize=12)

ax.set_ylabel('Count', fontsize=12)

plt.show()
df_train[df_train['AF'].isnull()]
drop_index = df_train[(df_train['HomeTeam'].isnull()) & (df_train['AwayTeam'].isnull())].index

df_train.drop(drop_index, inplace=True)
#Define a function which returns a null dataframe

def check_null(df):

    df_null = df.isna().sum().reset_index()

    df_null.columns = ['Column', 'Null_Count']

    df_null = df_null[df_null['Null_Count'] > 0]

    df_null = df_null.sort_values(by='Null_Count', ascending=False).reset_index(drop=True)

    return df_null
df_null_train = check_null(df_train)

df_null_train
#Let's fill null values grouped by the hometeam

for _, item in df_null_train.iterrows():

    column = item['Column']

    df_train[column] = df_train.groupby(['HomeTeam'])[column].transform(lambda x: x.fillna(x.mode()[0]))
check_null(df_train)
check_null(df_test)
def check_unique(df):

    df_unique = df.nunique().reset_index()

    df_unique.columns = ['Column', 'Unique_Count']

    df_unique = df_unique[df_unique['Unique_Count'] < 2]

    df_unique = df_unique.sort_values(by='Unique_Count', ascending=False).reset_index(drop=True)

    return df_unique
check_unique(df_train)
check_unique(df_test)
def plot_feature_distributions(df, features, palette):

    i = 0

    plt.figure()

    fig, ax = plt.subplots(len(features),1,figsize=(14,35))

    plt.subplots_adjust(bottom=0.001)



    for feature in features:

        i += 1

        plt.subplot(len(features),1,i)

        sns.barplot(df[feature].value_counts().index, df[feature].value_counts().values, palette=palette)

        plt.title('Distribution of {0}'.format(feature), fontsize=14)

        plt.xlabel(feature, fontsize=12)

        plt.ylabel('Count', fontsize=12)

    plt.show()
def plot_feature_distribution_hue_target(df, features, palette):

    i = 0

    plt.figure()

    fig, ax = plt.subplots(len(features),1,figsize=(14,35))

    plt.subplots_adjust(bottom=0.001)



    for feature in features:

        i += 1

        plt.subplot(len(features),1,i)

        sns.countplot(x=feature, hue='FTR', data=df, palette=palette)

        plt.title('Distribution of {0} by Full Time Result'.format(feature), fontsize=14)

        plt.xlabel(feature, fontsize=12)

        plt.ylabel('Count', fontsize=12)

    plt.show()
features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HTAG']

plot_feature_distributions(df_train, features, 'RdBu')
features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HTAG']

plot_feature_distributions(df_test, features, 'PuOr')
#Let's bin Away Team Corners values greater than 11.0 to 11.0

df_train['AC'] = df_train['AC'].transform(lambda x: 11.0 if x > 11.0 else x)

df_test['AC'] = df_test['AC'].transform(lambda x: 11.0 if x > 11.0 else x)



#Let's bin values greater than 30.0 to 30.0 and test values greater than 26.0 to 26.0

df_train['AF'] = df_train['AF'].transform(lambda x: 30.0 if x > 30.0 else x)

df_test['AF'] = df_test['AF'].transform(lambda x: 26.0 if x > 26.0 else x)



#Let's bin Away Team shots values greater than 27.0 to 27.0

df_train['AS'] = df_train['AS'].transform(lambda x: 27.0 if x > 27.0 else x)

df_test['AS'] = df_test['AS'].transform(lambda x: 27.0 if x > 27.0 else x)



#Let's bin Away Team shots on Target values greater than 12.0 to 12.0

df_train['AST'] = df_train['AST'].transform(lambda x: 12.0 if x > 12.0 else x)

df_test['AST'] = df_test['AST'].transform(lambda x: 12.0 if x > 12.0 else x)



#Let's bin Away Team Yellow Card values greater than 6.0 to 6.0

df_train['AY'] = df_train['AY'].transform(lambda x: 6.0 if x > 6.0 else x)

df_test['AY'] = df_test['AY'].transform(lambda x: 6.0 if x > 6.0 else x)
features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HTAG']

plot_feature_distribution_hue_target(df_train, features, 'coolwarm')
features = ['HC', 'HF', 'HR', 'HS', 'HST', 'HTHG', 'HY']

plot_feature_distributions(df_train, features, 'RdYlBu')
features = ['HC', 'HF', 'HR', 'HS', 'HST', 'HTHG', 'HY']

plot_feature_distributions(df_test, features, 'YlGnBu')
#Let's bin Home Team Corners values greater than 14.0 to 14.0

df_train['HC'] = df_train['HC'].transform(lambda x: 14.0 if x > 11.0 else x)

df_test['HC'] = df_test['HC'].transform(lambda x: 11.0 if x > 11.0 else x)



#Let's bin Home Team values greater than 27.0 to 27.0 and test values greater than 24.0 to 24.0

df_train['HF'] = df_train['HF'].transform(lambda x: 27.0 if x > 27.0 else x)

df_test['HF'] = df_test['HF'].transform(lambda x: 24.0 if x > 24.0 else x)



#Let's bin Home Team shots values greater than 30.0 to 30.0

df_train['HS'] = df_train['HS'].transform(lambda x: 30.0 if x > 30.0 else x)

df_test['HS'] = df_test['HS'].transform(lambda x: 30.0 if x > 30.0 else x)



#Let's bin Home Team shots on Target values greater than 14.0 to 14.0 and 12.0 to 12.0 for test set

df_train['HST'] = df_train['HST'].transform(lambda x: 14.0 if x > 14.0 else x)

df_test['HST'] = df_test['HST'].transform(lambda x: 12.0 if x > 12.0 else x)



#Let's bin Away Team Yellow Card values greater than 6.0 to 6.0

df_train['HY'] = df_train['HY'].transform(lambda x: 6.0 if x > 6.0 else x)

df_test['HY'] = df_test['HY'].transform(lambda x: 6.0 if x > 6.0 else x)
features = ['HC', 'HF', 'HR', 'HS', 'HST', 'HTHG', 'HY']

plot_feature_distribution_hue_target(df_train, features, 'coolwarm')
fig, ax = plt.subplots(figsize=(10, 35))

sns.barplot(x=df_train['HomeTeam'].value_counts().values, y=df_train['HomeTeam'].value_counts().index, color='Orange')

ax.set_title('Matches played by teams home between 2009 and 2017', fontsize=14)

ax.set_xlabel('Matches Played', fontsize=12)

ax.set_ylabel('Teams', fontsize=12)

plt.show()
fig, ax = plt.subplots(figsize=(10, 35))

sns.barplot(x=df_train['AwayTeam'].value_counts().values, y=df_train['AwayTeam'].value_counts().index, color='Green')

ax.set_title('Matches played by teams away between 2009 and 2017')

ax.set_xlabel('Matches Played', fontsize=12)

ax.set_ylabel('Teams', fontsize=12)

plt.show()
df_train['Year'] = df_train['Date'].dt.year

df_train['Month'] = df_train['Date'].dt.month

df_train['FTR'] = df_train['FTR'].map({'H':0, 'D':1, 'A':2})

df_test['Year'] = df_test['Date'].dt.year

df_test['Month'] = df_test['Date'].dt.month
df_HTHG = df_train.groupby(['HomeTeam'])['HomeTeam', 'HTHG'].sum().reset_index()

df_HTHG = df_HTHG.sort_values(by='HTHG', ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 35))

sns.barplot(x=df_HTHG['HTHG'], y=df_HTHG['HomeTeam'], color='yellow')

ax.set_title('Half Time goals by home teams', fontsize=14)

ax.set_xlabel('Goals', fontsize=12)

ax.set_ylabel('Teams', fontsize=12)

plt.show()
df_HTAG = df_train.groupby(['AwayTeam'])['AwayTeam', 'HTAG'].sum().reset_index()

df_HTAG = df_HTAG.sort_values(by='HTAG', ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 35))

sns.barplot(x=df_HTAG['HTAG'], y=df_HTAG['AwayTeam'], color='blue')

ax.set_title('Half Time goals by away teams', fontsize=14)

ax.set_xlabel('Goals', fontsize=12)

ax.set_ylabel('Teams', fontsize=12)

plt.show()
df_HTHG_season = df_train.groupby(['Year'])['HTHG'].sum().reset_index()

fig, ax = plt.subplots(figsize=(8, 8))

sns.barplot(x=df_HTHG_season['Year'], y=df_HTHG_season['HTHG'], palette='RdBu')

ax.set_title('Half Time goals by Home Team by season', fontsize=14)

ax.set_xlabel('Year', fontsize=12)

ax.set_ylabel('Goals', fontsize=12)

plt.show()
df_HTAG_season = df_train.groupby(['Year'])['HTAG'].sum().reset_index()

fig, ax = plt.subplots(figsize=(8, 8))

sns.barplot(x=df_HTAG_season['Year'], y=df_HTAG_season['HTAG'], palette='RdBu')

ax.set_title('Half Time goals by Away Team by season', fontsize=14)

ax.set_xlabel('Year', fontsize=12)

ax.set_ylabel('Goals', fontsize=12)

plt.show()
conditions = [df_train['FTR']==2,df_train['FTR']==0,df_train['FTR']==1]

select = [df_train['AwayTeam'],df_train['HomeTeam'],'Draw']

df_train['FTW']=np.select(conditions, select)
df_Winner = df_train['FTW'].value_counts().reset_index()

df_Winner.columns = ['Team', 'Win_Counts']



#Dropping Winner Feature as we will not be able to produce the same in test set and for modelling

df_train.drop('FTW', axis=1, inplace=True)





#Drop Draws from the dataframe

df_Winner.drop(df_Winner.head(1).index, axis=0, inplace=True)

df_Winner = df_Winner.head(20)

fig, ax = plt.subplots(figsize=(12, 8))

sns.barplot(x=df_Winner['Team'], y=df_Winner['Win_Counts'], palette='GnBu_r')

ax.set_title('Teams with maximum full-time wins', fontsize=14)

ax.set_xlabel('Teams', fontsize=12)

ax.set_ylabel('Wins', fontsize=12)

ax.set_xticklabels(df_Winner['Team'], rotation=45)

plt.show()
#Lambda Function

def select_winner(x):

    if x > 0:

        return 0

    elif x < 0:

        return 2

    else:

        return 1



def transform_HTR(df):

    df['HTW'] = df['HTHG'] - df['HTAG']

    df['HTW'] = df['HTW'].transform(lambda x: select_winner(x))

    conditions = [df['HTW']==2,df['HTW']==0,df['HTW']==1]

    select = [df['AwayTeam'],df['HomeTeam'],'Draw']

    df['HTW']=np.select(conditions, select)

    return df['HTW']

    

df_train['HTW'] = transform_HTR(df_train)
df_Winner = df_train['HTW'].value_counts().reset_index()

df_Winner.columns = ['Team', 'Win_Counts']



#Dropping Winner Feature as we will not be able to produce the same in test set and for modelling

df_train.drop('HTW', axis=1, inplace=True)



#Drop Draws from the dataframe

df_Winner.drop(df_Winner.head(1).index, axis=0, inplace=True)

df_Winner = df_Winner.head(20)

fig, ax = plt.subplots(figsize=(12, 8))

sns.barplot(x=df_Winner['Team'], y=df_Winner['Win_Counts'], palette='GnBu_r')

ax.set_title('Teams with maximum half-time wins', fontsize=14)

ax.set_xlabel('Teams', fontsize=12)

ax.set_ylabel('Wins', fontsize=12)

ax.set_xticklabels(df_Winner['Team'], rotation=45)

plt.show()
plt.figure(figsize=(16,8))

features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HC','HF', 'HR', 'HS', 'HST', 'HTAG', 'HTHG', 'HY']

plt.title("Distribution of mean values per row in the train and test set")

sns.distplot(df_train[features].mean(axis=1),color="green", kde=True,bins=120, label='train')

sns.distplot(df_test[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,8))

features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HC','HF', 'HR', 'HS', 'HST', 'HTAG', 'HTHG', 'HY']

plt.title("Distribution of std values per row in the train and test set")

sns.distplot(df_train[features].std(axis=1),color="red", kde=True,bins=120, label='train')

sns.distplot(df_test[features].std(axis=1),color="blue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,8))

features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HC','HF', 'HR', 'HS', 'HST', 'HTAG', 'HTHG', 'HY']

plt.title("Distribution of min values per row in the train and test set")

sns.distplot(df_train[features].min(axis=1),color="green", kde=True,bins=120, label='train')

sns.distplot(df_test[features].min(axis=1),color="blue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,8))

features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HC','HF', 'HR', 'HS', 'HST', 'HTAG', 'HTHG', 'HY']

plt.title("Distribution of max values per row in the train and test set")

sns.distplot(df_train[features].max(axis=1),color="gold", kde=True,bins=120, label='train')

sns.distplot(df_test[features].max(axis=1),color="darkblue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,8))

features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HC','HF', 'HR', 'HS', 'HST', 'HTAG', 'HTHG', 'HY']

plt.title("Distribution of skew values per row in the train and test set")

sns.distplot(df_train[features].skew(axis=1),color="gold", kde=True,bins=120, label='train')

sns.distplot(df_test[features].skew(axis=1),color="darkblue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,8))

features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HC','HF', 'HR', 'HS', 'HST', 'HTAG', 'HTHG', 'HY']

plt.title("Distribution of kurtosis values per row in the train and test set")

sns.distplot(df_train[features].kurtosis(axis=1),color="red", kde=True,bins=120, label='train')

sns.distplot(df_test[features].kurtosis(axis=1),color="orange", kde=True,bins=120, label='test')

plt.legend()

plt.show()
df_train_corr = df_train.corr()
fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(df_train_corr, cmap='RdYlBu_r', annot=True)

ax.set_title('Correlation of training set features', fontsize=14)

plt.show()
df_test_corr = df_test.corr()
fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(df_train_corr, cmap='RdYlBu_r', annot=True)

ax.set_title('Correlation of test set features', fontsize=14)

plt.show()
%%time

idx = features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HC','HF', 'HR', 'HS', 'HST', 'HTAG', 'HTHG', 'HY']

for df in [df_test, df_train]:

    df['sum'] = df[idx].sum(axis=1)  

    df['max'] = df[idx].max(axis=1)

    df['mean'] = df[idx].mean(axis=1)

    df['std'] = df[idx].std(axis=1)

    df['skew'] = df[idx].skew(axis=1)

    df['kurt'] = df[idx].kurtosis(axis=1)

    df['med'] = df[idx].median(axis=1)
%%time

for df in [df_test, df_train]:

    df['weekofyear'] = df['Date'].dt.weekofyear

    df['dayofweek'] = df['Date'].dt.dayofweek

    df['weekend'] = (df['Date'].dt.dayofweek >= 5).astype('int')

    df['quarter'] = df['Date'].dt.quarter

    df['is_month_start'] = df['Date'].dt.is_month_start

    df['month_diff'] = ((datetime.datetime.today() - df['Date']).dt.days)//30
def aggregate_away_metrics(df, prefix):

    agg_func = {

        'AC': ['sum', 'mean', 'max', 'std', 'count'],

        'AF': ['sum', 'mean', 'max', 'std', 'count'],

        'AR': ['sum', 'mean', 'max', 'std', 'count'],

        'AS': ['sum', 'mean', 'max', 'std', 'count'],

        'AST': ['sum', 'mean', 'max', 'std', 'count'],

        'AY': ['sum', 'mean', 'max', 'std', 'count'],

        'HTAG': ['sum', 'mean', 'max', 'std', 'count']

    }

    

    agg_transactions = df.groupby(['HomeTeam']).agg(agg_func)

    agg_transactions.columns = [prefix + '_'.join(col).strip() 

                           for col in agg_transactions.columns.values]

    agg_transactions.reset_index(inplace=True)

    return agg_transactions
agg_away = aggregate_away_metrics(df_train, 'away_')
df_train = pd.merge(df_train, agg_away, on='HomeTeam', how='left')

df_train.shape
agg_away = aggregate_away_metrics(df_test, 'away_')
df_test = pd.merge(df_test, agg_away, on='HomeTeam', how='left')

df_test.shape
def aggregate_home_metrics(df, prefix):

    agg_func = {

        'HC': ['sum', 'mean', 'max', 'std', 'count'],

        'HF': ['sum', 'mean', 'max', 'std', 'count'],

        'HR': ['sum', 'mean', 'max', 'std', 'count'],

        'HS': ['sum', 'mean', 'max', 'std', 'count'],

        'HST': ['sum', 'mean', 'max', 'std', 'count'],

        'HY': ['sum', 'mean', 'max', 'std', 'count'],

        'HTHG': ['sum', 'mean', 'max', 'std', 'count']

    }

    

    agg_transactions = df.groupby(['AwayTeam']).agg(agg_func)

    agg_transactions.columns = [prefix + '_'.join(col).strip() 

                           for col in agg_transactions.columns.values]

    agg_transactions.reset_index(inplace=True)

    return agg_transactions
agg_home = aggregate_home_metrics(df_train, 'home_')
df_train = pd.merge(df_train, agg_home, on='AwayTeam', how='left')

df_train.shape
agg_home = aggregate_home_metrics(df_test, 'home_')
df_test = pd.merge(df_test, agg_home, on='AwayTeam', how='left')

df_test.shape
df_train = df_train.join(pd.get_dummies(df_train['league']))

df_train.drop('league', axis=1, inplace=True)

df_train.head()
df_test = df_test.join(pd.get_dummies(df_test['league']))

df_test.drop('league', axis=1, inplace=True)

df_test.head()
df_train.drop(['Date', 'HomeTeam', 'AwayTeam'], axis=1, inplace=True)
df_test.drop(['Date', 'HomeTeam', 'AwayTeam'], axis=1, inplace=True)
df_train.shape, df_test.shape
Y_train = df_train['FTR']

X_train = df_train.drop(['FTR'], axis=1)

X_test = df_test
param = {

    'max_bin': 119,

    'min_data_in_leaf': 11,

    'learning_rate': 0.001,

    'min_sum_hessian_in_leaf': 0.00245,

    'bagging_fraction': 0.7, 

    'bagging_freq': 5, 

    'lambda_l1': 4.972,

    'lambda_l2': 2.276,

    'min_gain_to_split': 0.65,

    'max_depth': 14,

    'save_binary': True,

    'seed': 1337,

    'feature_fraction_seed': 1337,

    'bagging_seed': 1337,

    'drop_seed': 1337,

    'data_random_seed': 1337,

    'verbose': 1,

    'is_unbalance': True,

    'boost': 'gbdt',

    'feature_fraction' : 0.8,  # colsample_bytree

    'metric':'multi_logloss',

    'num_leaves': 30,

    'objective' : 'multiclass',

    'num_class' : 3,

    'verbosity': 1

}
folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)

oof = np.zeros((len(X_train), 3))

predictions = np.zeros((len(X_test), 3))

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, Y_train.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(X_train.iloc[trn_idx][X_train.columns], label=Y_train.iloc[trn_idx])

    val_data = lgb.Dataset(X_train.iloc[val_idx][X_train.columns], label=Y_train.iloc[val_idx])



    num_round = 20000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=2000, early_stopping_rounds = 500)

    oof[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    print('Fold Validation Set Accuracy: {0}'.format(np.mean(Y_train[val_idx] == np.argmax(oof[val_idx],axis=1))))

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = X_train.columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(X_test[features], num_iteration=clf.best_iteration) / folds.n_splits

cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:150].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,28))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('Features importance (averaged/folds)')

plt.tight_layout()

plt.savefig('FI.png')
output = pd.DataFrame({ 'index' : df_test.index, 'FTR': np.argmax(predictions, axis=1) })

output.tail()