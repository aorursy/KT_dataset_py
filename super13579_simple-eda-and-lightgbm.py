import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import seaborn as sns

import lightgbm as lgb

from sklearn.model_selection import GroupKFold,StratifiedKFold

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from sklearn import metrics



print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
print ("Total Train Set : %d" %len(train))

print ('Total Test Set : %d' %len(test))
train['scalar_coupling_constant'].describe()
sns.distplot(train['scalar_coupling_constant'])
typelist = list(train['type'].value_counts().index)

typelist
plt.figure(figsize=(26, 24))

for i, col in enumerate(typelist):

    plt.subplot(4,2, i + 1)

    sns.distplot(train[train['type']==col]['scalar_coupling_constant'],color ='orange')

    plt.title(col)
structures = pd.read_csv('../input/structures.csv')
structures.head()
#https://www.kaggle.com/inversion/atomic-distance-benchmark/output

def map_atom_info(df, atom_idx):

    df = pd.merge(df, structures, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df



train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)
train.head()
#https://www.kaggle.com/inversion/atomic-distance-benchmark/output



train['dist'] = ((train['x_1'] - train['x_0'])**2 +

             (train['y_1'] - train['y_0'])**2 +

             (train['z_1'] - train['z_0'])**2 ) ** 0.5



test['dist'] = ((test['x_1'] - test['x_0'])**2 +

             (test['y_1'] - test['y_0'])**2 +

             (test['z_1'] - test['z_0'])**2 ) ** 0.5
molecules = train.pop('molecule_name')

test = test.drop('molecule_name', axis=1)

id_train = train.pop('id')

id_test = test.pop('id')



y = train.pop('scalar_coupling_constant')



# Label Encoding

for f in ['type', 'atom_0', 'atom_1']:

    lbl = LabelEncoder()

    lbl.fit(list(train[f].values) + list(test[f].values))

    train[f] = lbl.transform(list(train[f].values))

    test[f] = lbl.transform(list(test[f].values))
train.head()
## Evaluate matric

## https://www.kaggle.com/abhishek/competition-metric

def metric(df, preds):

    df["prediction"] = preds

    maes = []

    for t in df.type.unique():

        y_true = df[df.type==t].scalar_coupling_constant.values

        y_pred = df[df.type==t].prediction.values

        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))

        maes.append(mae)

    return np.mean(maes)

#df for evaluate

eval_df = pd.DataFrame({"type":train["type"]})

eval_df["scalar_coupling_constant"] = y
n_splits = 5 # Number of K-fold Splits



splits = list(GroupKFold(n_splits=n_splits).split(train, y, groups=molecules))

splits[:3]
params = {"learning_rate" : 0.1,

          "depth": 9,

          'metric':'MAE',

          'min_samples_leaf': 3,

          "loss_function": "MAE"}
oof = np.zeros(len(train))

predictions = np.zeros(len(test))

feature_importance_df = pd.DataFrame()

features = [c for c in train.columns if c not in ['id']]



for i, (train_idx, valid_idx) in enumerate(splits):  

    print(f'Fold {i + 1}')

    x_train = np.array(train)

    y_train = np.array(y)

    trn_data = lgb.Dataset(x_train[train_idx.astype(int)], label=y_train[train_idx.astype(int)])

    val_data = lgb.Dataset(x_train[valid_idx.astype(int)], label=y_train[valid_idx.astype(int)])

    

    num_round = 10000

    clf = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 200)

    oof[valid_idx] = clf.predict(x_train[valid_idx], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = i + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #predictions[fake_data.index] += clf.predict(fake_data, num_iteration=clf.best_iteration) / n_splits

    predictions += clf.predict(test, num_iteration=clf.best_iteration) / n_splits



#print("CV score: {:<8.5f}".format(np.log(metrics.mean_absolute_error(train, oof))))
print("CV score: {:<8.5f}".format(metric(eval_df, oof)))
cols = (feature_importance_df[["feature", "importance"]]

        .groupby("feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]



plt.figure(figsize=(14,5))

sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('LightGBM Features (averaged over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')
sample_submission = pd.read_csv('../input/sample_submission.csv')



benchmark = sample_submission.copy()

benchmark['scalar_coupling_constant'] = predictions

benchmark.to_csv('LGBM_submission.csv',index=False)
benchmark.head()