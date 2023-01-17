# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plots

import seaborn as sns



from pathlib import Path

from tqdm import tqdm



random_state = 42



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/predict-volcanic-eruptions-ingv-oe/train.csv")

train.describe()
sequence = pd.read_csv("../input/predict-volcanic-eruptions-ingv-oe/train/1000015382.csv", dtype="Int16")

sequence.describe()
sequence.tail()
sequence.fillna(0).plot(subplots=True, figsize=(25, 10))

plt.tight_layout()

plt.show()
def agg_stats(df, idx):

    df = df.agg(['sum', 'min', "mean", "std", "median", "skew", "kurtosis"])

    df_flat = df.stack()

    df_flat.index = df_flat.index.map('{0[1]}_{0[0]}'.format)

    df_out = df_flat.to_frame().T

    df_out["segment_id"] = int(idx)

    return df_out
summary_stats = pd.DataFrame()

for csv in tqdm(Path("../input/predict-volcanic-eruptions-ingv-oe/train/").glob("**/*.csv"), total=4501):

    df = pd.read_csv(csv)

    summary_stats = summary_stats.append(agg_stats(df, csv.stem))
test_data = pd.DataFrame()

for csv in tqdm(Path("../input/predict-volcanic-eruptions-ingv-oe/test/").glob("**/*.csv"), total=4501):

    df = pd.read_csv(csv)

    test_data = test_data.append(agg_stats(df, csv.stem))
features = list(summary_stats.drop(["segment_id"], axis=1).columns)

target_name = ["time_to_eruption"]

summary_stats = summary_stats.merge(train, on="segment_id")

summary_stats.head()
summary_stats.describe()
import lightgbm as lgbm

from sklearn.model_selection import KFold

import gc





n_fold = 7

folds = KFold(n_splits=n_fold, shuffle=True, random_state=random_state)



data = summary_stats



params = {

    "n_estimators": 2000,

    "boosting_type": "gbdt",

    "metric": "mae",

    "num_leaves": 66,

    "learning_rate": 0.005,

    "feature_fraction": 0.9,

    "bagging_fraction": 0.8,

    "agging_freq": 3,

    "max_bins": 2048,

    "verbose": 0,

    "random_state": random_state,

    "nthread": -1,

    "device": "gpu",

}



sub_preds = np.zeros(test_data.shape[0])

feature_importance = pd.DataFrame(index=list(range(n_fold)), columns=features)



for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data)):

    trn_x, trn_y = data[features].iloc[trn_idx], data[target_name].iloc[trn_idx]

    val_x, val_y = data[features].iloc[val_idx], data[target_name].iloc[val_idx]

    

    model = lgbm.LGBMRegressor(**params)

    

    model.fit(trn_x, trn_y, 

            eval_set= [(trn_x, trn_y), (val_x, val_y)], 

            eval_metric="mae", verbose=0, early_stopping_rounds=150

           )



    feature_importance.iloc[n_fold, :] = model.feature_importances_

    

    sub_preds += model.predict(test_data[features], num_iteration=model.best_iteration_) / folds.n_splits

best = feature_importance.mean().sort_values(ascending=False)

best_idx = best[best > 5].index



plt.figure(figsize=(14,26))

sns.boxplot(data=feature_importance[best_idx], orient="h")

plt.title("Features Importance per Fold")

plt.tight_layout()
submission = pd.DataFrame()

submission['segment_id'] = test_data["segment_id"]

submission['time_to_eruption'] = sub_preds

submission.to_csv('submission.csv', header=True, index=False)