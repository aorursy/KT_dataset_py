# https://www.kaggle.com/gpreda/santander-eda-and-prediction/data

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import gc, os, logging, datetime, warnings

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import roc_auc_score



from sklearn.model_selection import StratifiedKFold





# Any results you write to the current directory are saved as output.
%%time

PATH ='/kaggle/input/santander-customer-transaction-prediction/'

train_df = pd.read_csv(PATH+"train.csv") #(200000, 202)

test_df = pd.read_csv(PATH+"test.csv") #(200000, 201)
train_df.head()
train_df.shape
def missing(data):

    total = data.isnull().sum()

    length=data.shape[0]

    percentage=total*100/data.count()

    miss = pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])

    types= []

    for col in data.columns:

        dtypes=str(data[col].dtype)

        types.append(dtypes)

    miss['Types'] = types

    return miss.T

    

missing(train_df)
test_df.describe()
# corrmat = train_df.T[:10].corr()

# fig, axes = plt.subplots(figsize=(20, 15))

# fig, axes = sns.heatmap(corrmat, vmin=0, vmax=1)

# fig.show()



def plot_features(df1, df2, features):

    i = 1

    fig, ax = plt.subplots(4, 4, figsize=(14,14))

    for f in features:

        plt.subplot(4,4,i)

        plt.scatter(df1[f], df2[f], marker='+', label=f)

        plt.xlabel(f, fontsize=9)

        i+=1

    plt.show()



feats = ['var_0','var_1','var_2','var_3','var_4','var_5','var_6','var_7','var_8','var_9','var_10','var_11','var_12','var_13','var_14','var_15']

plot_features(train_df, test_df, feats)
sns.countplot(train_df['target'], palette='Set3')

plt.show()
train_df['target'].value_counts()
ones = train_df['target'].value_counts()[1]

print("There are {}% target with value 1".format(100*ones/train_df['target'].count()))
def plot_dist(df1, df2, features):

    sns.set_style("whitegrid")

    fig, axes = plt.subplots(5,5,figsize=(20,24))

    t0 = train_df.loc[train_df['target']==0]

    t1 = train_df.loc[train_df['target']==1]

    i=0

    for f in features:

        i+=1

        plt.subplot(5,5,i)

        sns.distplot(t0[f], hist=False, label="0")

        sns.distplot(t1[f], hist=False, label="1")

    plt.show()



features = ['var_%s'%i for i in range(0,25)]

plot_dist(train_df, test_df, features)
# Plot mean distribution of all variables over all training samples

# Per row

plt.subplots(1,2,figsize=(16,6))

features = train_df.columns.values[102:202]

plt.subplot(1,2,1)

sns.distplot(train_df[features].mean(axis=1), hist=True, bins=50, kde=True, rug=True, label='Train')

plt.subplot(1,2,2)

sns.distplot(test_df[features].mean(axis=1), bins=50, kde=True, rug=True, label='Test')

plt.show()

# Distribution of mean values per column in the train and test set

plt.subplots(1,2,figsize=(16,6))

features = train_df.columns.values[102:202]

plt.subplot(1,2,1)

sns.distplot(train_df[features].mean(axis=0), hist=True, bins=50, kde=True, rug=True, label='Train')

plt.legend()



plt.subplot(1,2,2)

sns.distplot(test_df[features].mean(axis=0), bins=50, kde=True, rug=True, label='Test')

plt.legend()



plt.figure(figsize=(16,6))

sns.distplot(train_df[features].mean(axis=0), color="magenta", label='Train', bins=120)

sns.distplot(test_df[features].mean(axis=0), color="darkblue", label='Test', bins=120)

plt.legend()

plt.show()

# Plot mean distribution of all variables over all training samples

# Per row

plt.subplots(1,2,figsize=(16,6))

features = train_df.columns.values[102:202]

plt.subplot(1,2,1)

sns.distplot(train_df[features].std(axis=1), hist=True, bins=50, kde=True, rug=True, label='Train')

plt.subplot(1,2,2)

sns.distplot(test_df[features].std(axis=1), bins=50, kde=True, rug=True, label='Test')

plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of std values per row in the train and test set")

sns.distplot(train_df[features].min(axis=1), color='black', bins=120, label="Train")

plt.legend();plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of std values per row in the train and test set")

sns.distplot(test_df[features].min(axis=1), color='red', bins=120, label="Test")

plt.legend();plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of skew per row in the train and test set")

sns.distplot(train_df[features].skew(axis=1), color='orange', kde=True, bins=120, label='Train')

plt.legend();plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of skew per row in the train and test set")

sns.distplot(test_df[features].skew(axis=1), color='red', kde=True, bins=120, label='Train')

plt.legend();plt.show()
corre = train_df.drop(['target', 'ID_code'], axis=1).corr().abs().unstack().sort_values(kind='quicksort').reset_index()

type(corre)

corre = corre[corre['level_0']!=corre['level_1']]

# Least correlated

corre.head(10)
# Most correlated

corre.tail(10)
train_df.columns
params = {

    'bagging_freq': 5,

    'bagging_fraction': 0.4,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.05,

    'learning_rate': 0.01,

    'max_depth': -1,  

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary', 

    'verbosity': 1

}
import lightgbm as lgb

train_df = pd.read_csv(PATH+"train.csv")
oof = np.zeros(len(train_df))



K = 10 # Number of splits

skf = StratifiedKFold(n_splits=K, shuffle=False, random_state=1226)



y = train_df['target']

X = train_df.drop(['target', 'ID_code'], axis=1)

features = list(X.columns)



data_split = skf.split(X.values, y.values)

feature_importance_df = pd.DataFrame()



predictions = np.zeros(len(test_df))



test = test_df.drop(['ID_code'], axis=1)



for k, (train, val) in enumerate(data_split):

    print("Fold {}".format(k))

    X_train, X_val = X.loc[train], X.loc[val]

    y_train, y_val = y.loc[train], y.loc[val]

    

    train_data = lgb.Dataset(X_train, y_train)

    val_data = lgb.Dataset(X_val, y_val)

    

    num_round = 1000000

    clf = lgb.train(params, train_data, num_round, valid_sets=[train_data, val_data], verbose_eval=1000, early_stopping_rounds=3000)

    oof[val] = clf.predict(X_val.astype('float32'), num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["Importance"] = clf.feature_importance()

    fold_importance_df["Fold"] = k+1

    

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(test, num_iteration=clf.best_iteration)/skf.n_splits

    lgb.save(clf, "model.txt")



print("CV Score: {%8.5f}".format(roc_auc_score(target, oof)))

# train_df

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