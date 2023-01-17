import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



import xgboost as xgb



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
color = sns.color_palette()
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sub = pd.read_csv('../input/sample_submission.csv')
print("Number of rows and columns in train set : ",train.shape)

print("Number of rows and columns in test set : ",test.shape)
train.head()
test.head()
train.info()
test.info()
train.columns
test.columns
train.describe()
test.describe()
train.target.value_counts()
sns.countplot(train['target'], palette='Set2')
train.isnull().sum()
test.isnull().sum()
feats = [f for f in train.columns if f not in ['id','target']]

for i in feats:

    print ('==' + str(i) + '==')

    print ('train:' + str(train[i].nunique()/train.shape[0]))

    print ('test:' + str(test[i].nunique()/test.shape[0]))
def plot_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(26,10,figsize=(18,22))



    for feature in features:

        i += 1

        plt.subplot(26,10,i)

        sns.distplot(df1[feature], hist=False,label=label1)

        sns.distplot(df2[feature], hist=False,label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)

        plt.tick_params(axis='y', which='major', labelsize=6)

    plt.show();

    

t0 = train[feats].loc[train['target'] == 0]

t1 = train[feats].loc[train['target'] == 1]

features = train[feats].columns.values

plot_feature_distribution(t0, t1, '0', '1', features)    
plt.figure(figsize=(16,6))

features = train[feats].columns.values

plt.title("Distribution of mean values per row in the train and test set")

sns.distplot(train[features].mean(axis=1),color="green", kde=True,bins=120, label='train')

sns.distplot(test[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train[feats].columns.values

plt.title("Distribution of mean values per column in the train and test set")

sns.distplot(train[features].mean(axis=0),color="yellow",kde=True,bins=50, label='train')

sns.distplot(test[features].mean(axis=0),color="red", kde=True,bins=50, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train[feats].columns.values

plt.title("Distribution of std values per row in the train and test set")

sns.distplot(train[features].std(axis=1),color="green", kde=True,bins=120, label='train')

sns.distplot(test[features].std(axis=1),color="blue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train[feats].columns.values

plt.title("Distribution of std values per row in the train and test set")

sns.distplot(train[features].std(axis=0),color="red", kde=True,bins=50, label='train')

sns.distplot(test[features].std(axis=0),color="yellow", kde=True,bins=50, label='test')

plt.legend()

plt.show()
correlations = train[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()

correlations = correlations[correlations['level_0'] != correlations['level_1']]

correlations.head()
correlations.tail()
feats_target = [f for f in train.columns if f not in ['id']]

correlations = train[feats_target].corr().abs().unstack().sort_values(kind="quicksort").reset_index()

correlations = correlations[correlations['level_0'] != correlations['level_1']]

corr = correlations[correlations['level_0']=='target']

corr.head(10)
random_state = 42

np.random.seed(random_state)
xgb_params = {

        'objective': 'binary:logistic',

        #'objective':'reg:linear',

        'tree_method': 'gpu_hist',

        'eta':0.1,

        'num_round':120000,

        'max_depth': 8,

        'silent':1,

        'subsample':0.5,

        'colsample_bytree': 0.5,

        'min_child_weight': 100,

        'eval_metric': 'auc',

        'verbose_eval': 1000,

    }
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

oof = train[['id', 'target']]

oof['predict'] = 0

predictions = test[['id']]

val_aucs = []

features = [col for col in train.columns if col not in ('id', 'target')]
for fold, (trn_idx, val_idx) in enumerate(skf.split(train,train['target'])):

    X_train, y_train = train.iloc[trn_idx][features],train.iloc[trn_idx]['target']

    X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx]['target']

    train_dataset = xgb.DMatrix(X_train, y_train)

    valid_dataset = xgb.DMatrix(X_valid, y_valid)

    watchlist = [(train_dataset, 'train'), (valid_dataset, 'valid')]

    xgb_clf = xgb.train(xgb_params,

                        train_dataset,

                        evals=watchlist,

                        num_boost_round=12000,

                        early_stopping_rounds=300,

                        verbose_eval=1000

                       )

    p_valid = xgb_clf.predict(valid_dataset, ntree_limit=xgb_clf.best_iteration)

    yp = xgb_clf.predict(xgb.DMatrix(test[features]), ntree_limit=xgb_clf.best_iteration)

    

    oof['predict'][val_idx] = p_valid

    val_score = roc_auc_score(y_valid, p_valid)

    val_aucs.append(val_score)

    predictions['fold{}'.format(fold+1)] = yp
all_auc = roc_auc_score(oof['target'], oof['predict'])

print('ROC mean: %.6f, std: %.6f.' % (np.mean(val_aucs), np.std(val_aucs)))

print('Ensemble ROC: %.6f' % (all_auc))
predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['id', 'target']]].values, axis=1)

sub["target"] = predictions['target']

sub.to_csv("submission.csv", index=False)

oof.to_csv('oof.csv', index=False)