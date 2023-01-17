from gc import collect

# Data manipulation and set-up

import numpy as np

import pandas as pd



# Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=1)



# Modelling (including set-up & evaluation)

from sklearn.model_selection import KFold, StratifiedKFold

import lightgbm as lgb

from sklearn.metrics import roc_auc_score



# Ignore warnings

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')
random_state = 2434

np.random.seed(random_state)

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

oof = train[['ID_code', 'target']]

oof['predict'] = 0

predictions = test[['ID_code']]

val_aucs = []

feature_importance = pd.DataFrame()
features = [col for col in train.columns if col not in ['target', 'ID_code']]

X_test = test[features]
from tqdm import tqdm

all_df = pd.concat([train.iloc[:, 2:], test.iloc[:, 1:]]) 

unique_maps = dict()

for col in tqdm(list(all_df)):

    map_ = all_df[col].value_counts()

    unique_maps[col] = map_
import scipy.stats as st

delta_dics = {}

delta2_dics = {}

for col in ['var_108']:

    kde = st.gaussian_kde(all_df[col])

    uvar = np.unique(np.hstack([all_df[col].unique(), all_df[col].unique() + .0001, all_df[col].unique() - .0001]))

    uvar.sort()

    gs = []

    for i in tqdm(range(-(-uvar.shape[0]//20))):

        gs.extend(list(kde(uvar[20*i:20*(i+1)])))

    dic_ = {uvar[i]: gs[i]-gs[i-1] for i in range(1, uvar.shape[0])}

    delta_dics[col] = dic_
def FE(X):

    unique_df = pd.DataFrame()

    for col in unique_maps.keys():

        unique_df["unique_" + col] = X[col].map(unique_maps[col])

    #X["rotated_0"] = (scl.transform(X.values) @ R)[:, 0]

    X["var_uniques"] = (unique_df <= 2).mean(axis=1)

    #X["var_duplicates"] = (unique_df).mean(axis=1)

    for col in unique_maps.keys():

        X[col + "_uniques"] = unique_df["unique_" + col]

    from datetime import datetime

    epoch_datetime = pd.datetime(1900, 1, 1)

    s = (X['var_68']*10000 - 7000 + epoch_datetime.toordinal()).astype(int).map(datetime.fromordinal)

    #X["dayofweek"] = s.dt.dayofweek

    X["dayofyear"] = s.dt.dayofyear

    for key in ['var_108']:

        X[key + "_delta"] = X[key].map(delta_dics[key])

    return  X
# Data augmentation

def augment(x,y,t=2):

    xs,xn = [],[]

    for i in tqdm(range(t)):

        mask = y>0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xs.append(x1)



    for i in range(t//6):

        mask = y==0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xn.append(x1)



    xs = np.vstack(xs)

    xn = np.vstack(xn)

    ys = np.ones(xs.shape[0])

    yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn])

    y = np.concatenate([y,ys,yn])

    return x,y
from catboost import CatBoostRegressor, Pool, CatBoostClassifier



features = [col for col in train.columns if col not in ['target', 'ID_code']]

X_test = test[features]

X_test = FE(X_test)

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train['target'])):

    X_train, y_train = train.iloc[trn_idx][features], train.iloc[trn_idx]['target']

    X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx]['target']

    

    N = 5

    p_valid,yp = 0,0

    X_valid = FE(X_valid)

    for i in range(N):

        X_t, y_t = augment(X_train.values, y_train.values, t=12)

        X_t = pd.DataFrame(X_t)

        X_t = X_t.add_prefix('var_')

        X_t = FE(X_t)

        dev_pool = Pool(X_t,

                        y_t)

        val_pool = Pool(X_valid,

                        y_valid)

        evals_result = {}

        model = CatBoostClassifier(eval_metric="AUC",

                                   custom_loss='Logloss',

                                   depth=3,

                                   subsample=.2,

                                   l2_leaf_reg=1,

                                   verbose=1000,

                                   early_stopping_rounds=500,

                                   bootstrap_type="Bernoulli",

                                   learning_rate=0.02,

                                   task_type="GPU",

                                   random_seed=2434 + fold*5 + i,

                                   od_type="Iter",                           

                                   iterations=60000,

                                   )

        model.fit(dev_pool, eval_set=val_pool)

        p_valid += model.predict_proba(X_valid)[:, 1]

        yp += model.predict_proba(X_test)[:, 1]

        del X_t, dev_pool

        collect()

    oof['predict'][val_idx] = p_valid/N

    val_score = roc_auc_score(y_valid, p_valid)

    val_aucs.append(val_score)

    

    predictions['fold{}'.format(fold+1)] = yp/N
# Submission

predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)

predictions.to_csv('cat_all_predictions.csv', index=None)

sub = pd.DataFrame({"ID_code":test["ID_code"].values})

sub["target"] = predictions['target']

sub.to_csv("cat_submission.csv", index=False)

oof.to_csv('cat_oof.csv', index=False)
resdf = pd.DataFrame([model.feature_importances_,

              model.feature_names_]).T.sort_values(0, ascending=False)
original_feat = [f"var_{i}" for i in range(200)]

search_cols = resdf[resdf[1].isin(original_feat)][1].tolist()