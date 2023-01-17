!pip install catboost==0.13.1
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=UserWarning)
import gc

import numpy as np

import pandas as pd



from time import time

from sklearn.metrics import roc_auc_score, classification_report

from catboost import Pool, CatBoost, CatBoostClassifier, MetricVisualizer
basepath = "../input/"

df_train = pd.read_csv(f"{basepath}train/train.csv", index_col=0, low_memory=False)

df_test = pd.read_csv(f"{basepath}test/test.csv", index_col=0, low_memory=False)



group_col = 'group_id'

target = 'target'



df_train.shape, df_test.shape
df_train = df_train.drop(df_train.index[[0,1,2,3]])



df_train = df_train.drop(['feature_13', 'feature_32'], axis=1)

df_test = df_test.drop(['feature_13', 'feature_32'], axis=1)



df_train = df_train.rename(columns={'feature_18': 'pos'})

df_test = df_test.rename(columns={'feature_18': 'pos'})
num_features = list(df_train.select_dtypes(include=[np.float64, np.int64]).columns)

num_features.remove(target)

num_features.remove(group_col)



mins = df_train[num_features].min()

maxs = df_train[num_features].max()

difs = np.abs(maxs - mins)

vals = mins - 5 * difs



for col in num_features:

    df_train[col] = df_train[col].fillna(vals[col])

    df_test[col] = df_test[col].fillna(vals[col])



df_train[num_features] = df_train[num_features].astype(np.float32)

df_test[num_features] = df_test[num_features].astype(np.float32)



f"{gc.collect()} objects collected"
def add_aggregates(df, col):

    # Aggregations

    df[f"{col}_min"] = df.groupby(group_col)[col].transform(np.min)

    df[f"{col}_avg"] = df.groupby(group_col)[col].transform(np.mean)

    df[f"{col}_q50"] = df.groupby(group_col)[col].transform(np.median)

    df[f"{col}_max"] = df.groupby(group_col)[col].transform(np.max)

    df[f"{col}_var"] = df.groupby(group_col)[col].transform(np.var)

    df[f"{col}_sum"] = df.groupby(group_col)[col].transform(np.sum)

    # Comparisons

    df[f"{col}_cmp_min"] = df[col] == df[f"{col}_min"]

    df[f"{col}_cmp_max"] = df[col] == df[f"{col}_max"]

    df[f"{col}_cmp_avg"] = df[col] <= df[f"{col}_avg"]

    df[f"{col}_cmp_q50"] = df[col] <= df[f"{col}_q50"]

    # Scaled values

    df[f"{col}_scaled_min"] = df[col] / df[f"{col}_min"]

    df[f"{col}_scaled_avg"] = df[col] / df[f"{col}_avg"]

    df[f"{col}_scaled_q50"] = df[col] / df[f"{col}_q50"]

    df[f"{col}_scaled_max"] = df[col] / df[f"{col}_max"]

    df[f"{col}_scaled_sum"] = df[col] / df[f"{col}_sum"]
t_start = time()

for i, feature in enumerate(num_features):

    add_aggregates(df_train, feature)

    add_aggregates(df_test, feature)

    print(f"{i+1}/{len(num_features)}: {feature} transformed")

t_elapsed = int(time() - t_start)

m, s = t_elapsed // 60, t_elapsed % 60

    

num_features = list(df_train.select_dtypes(include=[np.bool, np.float32, np.int64]).columns)

num_features.remove(target)

num_features.remove(group_col)

print(f"Total: {len(num_features)} numeric features")

print(f"Process time: {m}m {s}s")
cat_features = list(df_train.select_dtypes(include=[np.object]).columns)



df_train[cat_features] = df_train[cat_features].fillna('N/A')

df_test[cat_features] = df_test[cat_features].fillna('N/A')



all_data = pd.concat((df_train, df_test))

for col in cat_features:

    cats = all_data[col].unique()

    df_train[col] = df_train[col].astype('category', categories = cats)

    df_test[col] = df_test[col].astype('category', categories = cats)

    

for col in cat_features:

    df_train[col] = df_train[col].cat.codes + 1

    df_test[col] = df_test[col].cat.codes + 1

    

del all_data

f"{gc.collect()} objects collected"
features = num_features + cat_features

f"Total {len(features)} features: {len(num_features)} num., {len(cat_features)} cat."
def score_model(model, pool):

    ys = pool.get_label()

    ys_pred = model.predict(pool)

    results = {'auc': roc_auc_score(ys, ys_pred)}

    results.update(classification_report(ys, ys_pred, output_dict=True))

    return results



def print_results(r):

    if 'time' in r:

        print(f"Train time: {r['time'][0]:2d}m {r['time'][1]:2d}s")

    print(f"AUC score:  {r['auc']:1.5f}")

    print(f"Class Neg:")

    print(f"   precision: {r['0']['precision']:1.5f}")

    print(f"   recall:    {r['0']['recall']:1.5f}")

    print(f"Class Pos:")

    print(f"   precision: {r['1']['precision']:1.5f}")

    print(f"   recall:    {r['1']['recall']:1.5f}")
def make_pool(df, labeled=True):

    return Pool(

        data = df[features],

        label = df[target] if labeled else None,

        cat_features = cat_features,

        group_id = df[group_col],

    )



def space_split(df, val_size=0.1):

    df.sort_values([group_col, 'pos'])

    cutoff = df[group_col].iloc[int((1 - val_size) * len(df))]

    pt1, pt2 = df[df[group_col] < cutoff], df[df[group_col] >= cutoff]

    return make_pool(pt1), make_pool(pt2)
def full_train(model, train_data, test_data, name):

    train_pool = make_pool(train_data)

    # Fit model with `train_data` and measure time

    t_start = time()

    model.fit(train_pool)

    t_elapsed = int(time() - t_start)

    m, s = t_elapsed // 60, t_elapsed % 60

    print(f"Train time: {m}m {s}s")

    # Score model if target present in `test_data`

    if target in test_data.columns:

        test_pool = make_pool(test_data)

        return score_model(model, test_pool)

    # Else predict probabilities and write submission

    else:

        id_test = test_data.index

        xs_test = make_pool(test_data, labeled=False)

        ys_prob = model.predict_proba(xs_test)

        yP_prob = [y2 for y1,y2 in ys_prob]

        answer = pd.DataFrame({'Id': id_test, 'target': yP_prob})

        answer.to_csv(f'{name}.csv', sep=',', index=False)
def validate(model, data, val_size=0.1, rounds=100, fit=True):

    results = {}

    # Split `data` dataset to `learn` and `validate`

    learn, val = space_split(data, val_size)

    # Fit model

    if fit:

        t_start = time()

        model.fit(

            X=learn,

            eval_set=val,

            use_best_model=True,

            early_stopping_rounds=rounds,

            verbose=False,

            plot=True

        )

        t_elapsed = int(time() - t_start)

        results['time'] = t_elapsed // 60, t_elapsed % 60

    # Scores on `validate` with different metrics

    results.update(score_model(model, val))

    return results
common = {

    'loss_function': 'CrossEntropy',

    'eval_metric': 'F1',

    

    'od_type': 'IncToDec',

    'od_pval': 1e-5,

    

    'boosting_type': 'Ordered',

    'bootstrap_type': 'Bernoulli',

    'one_hot_max_size': 3,

    

    'random_seed': 51,

    'task_type': 'GPU',

}
params = {

    **common, 'train_dir': 'p08-2naf',

    

    'iterations': 1500,      # p08-2naf best iteration: 1384

    'learning_rate': 0.05,

    'depth': 8,

    

    'l2_leaf_reg': 12.0,

    'random_strength': 3.0,

    'subsample': 0.5,

}
# model = CatBoostClassifier(**params)

# result = validate(model, df_train, rounds=150)

# best_iteration = model.best_iteration_

# print_results(result)

# print(f"Best iteration: {best_iteration}")
model = CatBoostClassifier(**params)

full_train(model, df_train, df_test, 'm13-kernel-catboost')