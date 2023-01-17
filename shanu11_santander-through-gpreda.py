import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sna
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")
test = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")
train.shape, test.shape
train.head()
train.describe()
test.describe()
sna.countplot(train['target'], palette='Set3')
def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sna.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sna.distplot(df1[feature], hist=False,label=label1)
        sna.distplot(df2[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();
t0 = train.loc[train["target"]==0]
t1 = train.loc[train["target"]==1]
features = train.columns.values[2:102]
plot_feature_distribution(t0, t1, 0,1, features)
plot_feature_distribution(train, test,"train","test", features)
plt.figure(figsize=(16,6))
features = train.columns.values[2:202]
plt.title("Distribution of mean values per row in the train and test set")
sna.distplot(train[features].mean(axis=1),color="green", kde=True,bins=120, label='train')
sna.distplot(test[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()
plt.figure(figsize=(16,6))
features = train.columns.values[2:202]
plt.title("Distribution of mean values per row in the train and test set")
sna.distplot(train[features].mean(axis=0),color="green", kde=True,bins=120, label='train')
sna.distplot(test[features].mean(axis=0),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()
plt.figure(figsize=(16,6))
features = train.columns.values[2:202]
plt.title("Distribution of mean values per row in the train and test set")
sna.distplot(train[features].std(axis=1),color="green", kde=True,bins=120, label='train')
sna.distplot(test[features].std(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()
plt.figure(figsize=(16,6))
features = train.columns.values[2:202]
plt.title("Distribution of mean values per row in the train and test set")
sna.distplot(train[features].std(axis=0),color="green", kde=True,bins=120, label='train')
sna.distplot(test[features].std(axis=0),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()
plt.figure(figsize = (16,6))
t0 = train.loc[train["target"]==0]
t1 = train.loc[train["target"]==1]
plt.title("Distribution of mean values per row in the train set with respect to target values")
sna.distplot(t0[features].mean(axis=1),color="red", kde=True,bins=120, label='target = 0')
sna.distplot(t1[features].mean(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
plt.figure(figsize = (16,6))
t0 = train.loc[train["target"]==0]
t1 = train.loc[train["target"]==1]
plt.title("Distribution of mean values per column in the train set with respect to target values")
sna.distplot(t0[features].mean(axis=0),color="red", kde=True,bins=120, label='target = 0')
sna.distplot(t1[features].mean(axis=0),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
plt.figure(figsize = (16,6))
t0 = train.loc[train["target"]==0]
t1 = train.loc[train["target"]==1]
plt.title("Distribution of std values per row in the train set with respect to target values")
sna.distplot(t0[features].std(axis=1),color="red", kde=True,bins=120, label='target = 0')
sna.distplot(t1[features].std(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
plt.figure(figsize = (16,6))
t0 = train.loc[train["target"]==0]
t1 = train.loc[train["target"]==1]
plt.title("Distribution of std values per column in the train set with respect to target values")
sna.distplot(t0[features].std(axis=0),color="red", kde=True,bins=120, label='target = 0')
sna.distplot(t1[features].std(axis=0),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
plt.figure(figsize=(16,6))
features = train.columns.values[2:202]
plt.title("Distribution of min values per row in the train and test set")
sna.distplot(train[features].min(axis=1),color="red", kde=True,bins=120, label='train')
sna.distplot(test[features].min(axis=1),color="orange", kde=True,bins=120, label='test')
plt.legend()
plt.show()
plt.figure(figsize=(16,6))
features = train.columns.values[2:202]
plt.title("Distribution of min values per column in the train and test set")
sna.distplot(train[features].min(axis=0),color="red", kde=True,bins=120, label='train')
sna.distplot(test[features].min(axis=0),color="orange", kde=True,bins=120, label='test')
plt.legend()
plt.show()
plt.figure(figsize=(16,6))
features = train.columns.values[2:202]
plt.title("Distribution of max values per row in the train and test set")
sna.distplot(train[features].max(axis=1),color="brown", kde=True,bins=120, label='train')
sna.distplot(test[features].max(axis=1),color="yellow", kde=True,bins=120, label='test')
plt.legend()
plt.show()
plt.figure(figsize=(16,6))
features = train.columns.values[2:202]
plt.title("Distribution of max values per columns in the train and test set")
sna.distplot(train[features].max(axis=0),color="brown", kde=True,bins=120, label='train')
sna.distplot(test[features].max(axis=0),color="yellow", kde=True,bins=120, label='test')
plt.legend()
plt.show()
t0 = train.loc[train['target'] == 0]
t1 = train.loc[train['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of min values per row in the train set")
sna.distplot(t0[features].min(axis=1),color="orange", kde=True,bins=120, label='target = 0')
sna.distplot(t1[features].min(axis=1),color="darkblue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
t0 = train.loc[train['target'] == 0]
t1 = train.loc[train['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of min values per column in the train set")
sna.distplot(t0[features].min(axis=0),color="orange", kde=True,bins=120, label='target = 0')
sna.distplot(t1[features].min(axis=0),color="darkblue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
t0 = train.loc[train['target'] == 0]
t1 = train.loc[train['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of max values per row in the train set")
sna.distplot(t0[features].max(axis=1),color="orange", kde=True,bins=120, label='target = 0')
sna.distplot(t1[features].max(axis=1),color="darkblue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
t0 = train.loc[train['target'] == 0]
t1 = train.loc[train['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of max values per column in the train set")
sna.distplot(t0[features].max(axis=0),color="orange", kde=True,bins=120, label='target = 0')
sna.distplot(t1[features].max(axis=0),color="darkblue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
plt.figure(figsize=(16,6))
plt.title("Distribution of skew per row in the train and test set")
sna.distplot(train[features].skew(axis=1),color="red", kde=True,bins=120, label='train')
sna.distplot(test[features].skew(axis=1),color="orange", kde=True,bins=120, label='test')
plt.legend()
plt.show()
plt.figure(figsize=(16,6))
plt.title("Distribution of skew per column in the train and test set")
sna.distplot(train[features].skew(axis=0),color="red", kde=True,bins=120, label='train')
sna.distplot(test[features].skew(axis=0),color="orange", kde=True,bins=120, label='test')
plt.legend()
plt.show()
plt.figure(figsize=(16,6))
plt.title("Distribution of kurtosis per row in the train and test set")
sna.distplot(train[features].kurtosis(axis=1),color="darkblue", kde=True,bins=120, label='train')
sna.distplot(test[features].kurtosis(axis=1),color="yellow", kde=True,bins=120, label='test')
plt.legend()
plt.show()
plt.figure(figsize=(16,6))
plt.title("Distribution of kurtosis per column in the train and test set")
sna.distplot(train[features].kurtosis(axis=0),color="darkblue", kde=True,bins=120, label='train')
sna.distplot(test[features].kurtosis(axis=0),color="yellow", kde=True,bins=120, label='test')
plt.legend()
plt.show()
correlations = train[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]
correlations.head(10)
features = train.columns.values[2:202]
unique_max_train = []
unique_max_test = []
for feature in features:
    values = train[feature].value_counts()
    unique_max_train.append([feature, values.max(), values.idxmax()])
    values = test[feature].value_counts()
    unique_max_test.append([feature, values.max(), values.idxmax()])
np.transpose((pd.DataFrame(unique_max_train, columns=['Feature', 'Max duplicates', 'Value'])).\
            sort_values(by = 'Max duplicates', ascending=False).head(15))
np.transpose((pd.DataFrame(unique_max_test, columns=['Feature', 'Max duplicates', 'Value'])).\
            sort_values(by = 'Max duplicates', ascending=False).head(15))
idx = features = train.columns.values[2:202]
for df in [test, train]:
    df['sum'] = df[idx].sum(axis=1)  
    df['min'] = df[idx].min(axis=1)
    df['max'] = df[idx].max(axis=1)
    df['mean'] = df[idx].mean(axis=1)
    df['std'] = df[idx].std(axis=1)
    df['skew'] = df[idx].skew(axis=1)
    df['kurt'] = df[idx].kurtosis(axis=1)
    df['med'] = df[idx].median(axis=1)
train.head()
features = [c for c in train.columns if c not in ['ID_code', 'target']]
target = train['target']
param = {
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
folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = predictions
sub_df.to_csv("submission.csv", index=False)