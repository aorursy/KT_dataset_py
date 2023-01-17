import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('darkgrid')

sns.set_palette('bone')



pd.options.display.float_format = '{:,.3f}'.format
df = pd.read_csv("../input/data.csv")

df.shape
df.describe(include=['number'])
df.describe(include=['object'])
df.head()
df.describe().T[["mean","min","max"]]
vc = df.shot_made_flag.value_counts(normalize=True).to_frame()

vc.plot.bar()

vc.T
df.game_id.nunique(), df.game_date.nunique()
df.season.unique()
df.team_id.unique(), df.team_name.unique()
df.game_event_id.nunique()
df.action_type.unique()
df.action_type.value_counts()[:10]
(df["minutes_remaining"] * 60 + df["seconds_remaining"]).hist()
_ = df[(df["minutes_remaining"] == 0) & (df["seconds_remaining"] <= 10)]

_.mean()["shot_made_flag"], _.count()["shot_made_flag"]
df["game_year"] = df["game_date"].str[0:4].astype(int)

df["game_month"] = df["game_date"].str[5:7].astype(int)

df['action_first_words'] = df["action_type"].str.split(' ').str[0]

df['action_last_words'] = df["action_type"].str.split(' ').str[-2]

df['season_start_year'] = df.season.str.split('-').str[0].astype(int)



df["remaining"] = df["minutes_remaining"] * 60 + df["seconds_remaining"]

df["hurry_shot"] = ((df["minutes_remaining"] == 0) & (df["seconds_remaining"] < 10)).astype(int)

df["home_game"] = df["matchup"].apply(lambda x: 1 if (x.find('@') < 0) else 0)



df['distance_bin'] = pd.cut(df.shot_distance, bins=10, labels=range(10)).astype('int')



import math as m

df["angle"] = df.apply(lambda row: 90 if row["loc_y"]==0 else m.degrees(m.atan(row["loc_x"]/abs(row["loc_y"]))),axis=1)

df["angle_bin"] = pd.cut(df.angle, 7, labels=range(7)).astype(int)
df.drop(["team_id", "team_name", "game_date", "game_event_id", "matchup"], axis=1, inplace=True)
nullcount = df.isnull().sum()

nullcount[nullcount > 0]
df.shot_made_flag.mean()
_ = pd.concat([df.game_id, df.period, df.shot_made_flag, df.game_id.shift(1), df.period.shift(1), df.shot_made_flag.shift(1)], axis=1)

_.columns = ["game_id", "period", "shot_made_flag", "pre_game_id", "pre_period", "pre_shot_made_flag"]

_.dropna()

_ = _[(_["game_id"] == _["pre_game_id"]) & (_["period"] == _["pre_period"])]

_.groupby(["pre_shot_made_flag"]).mean()["shot_made_flag"]
df_enc = df.copy()
from sklearn.preprocessing import LabelEncoder



for i, t in df_enc.dtypes.iteritems():

    if t == object:

        le = LabelEncoder()

        le.fit(df_enc[i].astype(str))

        df_enc[i] = le.transform(df_enc[i].astype(str))
plt.figure(figsize=(18,10))

sns.heatmap(df_enc.corr(), annot=True, linewidths=.6, fmt='.1f', vmax=1, vmin=-1, center=0, cmap='Blues')
train = df[~df.shot_made_flag.isnull()]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))



train.groupby(["shot_distance"]).mean()["shot_made_flag"].plot(ylim=(0,1), ax=axes[0])



_g = train.groupby(train["shot_distance"] // 5 * 5)

_ = pd.concat([_g.count()["shot_made_flag"], _g.mean()["shot_made_flag"]], axis=1)

_.columns = ["shot_count", "shot_mean"]

_[_.shot_count >= 10].plot.bar(y="shot_mean", ylim=(0,1), ax=axes[1])
def shot_mean(group_col):

    return train.groupby([group_col]).mean()["shot_made_flag"]



def sorted_shot_mean(group_col):

    return train.groupby([group_col]).mean()["shot_made_flag"].sort_values(ascending=False)
fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(18,4))

sorted_shot_mean("shot_zone_range").plot.bar(ylim=(0,1), ax=axes[0])

sorted_shot_mean("shot_zone_basic").plot.bar(ylim=(0,1), ax=axes[1])

sorted_shot_mean("shot_zone_area").plot.bar(ylim=(0,1), ax=axes[2])

sorted_shot_mean("combined_shot_type").plot.bar(ylim=(0,1), ax=axes[3])

sorted_shot_mean("action_first_words").plot.bar(ylim=(0,1), ax=axes[4])

sorted_shot_mean("action_last_words").plot.bar(ylim=(0,1), ax=axes[5])
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(18,4))

shot_mean("shot_type").plot.bar(ylim=(0,1), ax=axes[0])

shot_mean("home_game").plot.bar(ylim=(0,1), ax=axes[1])

shot_mean("hurry_shot").plot.bar(ylim=(0,1), ax=axes[2])

shot_mean("period").plot.bar(ylim=(0,1), ax=axes[3])

shot_mean("angle_bin").plot.bar(ylim=(0,1), ax=axes[4])
sorted_shot_mean("opponent").plot.bar(figsize=(10,4))
X_train = df_enc[~df_enc.shot_made_flag.isnull()]

X_game_id = X_train.pop('game_id')

Y_train = X_train['shot_made_flag']

X_train = X_train.drop(['shot_id','shot_made_flag'], axis=1)



X_test = df_enc[df_enc.shot_made_flag.isnull()].drop(['game_id','shot_id','shot_made_flag'], axis=1)
from sklearn.model_selection import GroupKFold

from sklearn.preprocessing import minmax_scale

import lightgbm as lgb



params={'learning_rate': 0.03,

        'objective':'binary',

        'metric':'binary_logloss',

        'num_leaves': 31,

        'verbose': 1,

        'random_state':42,

        'bagging_fraction': 1,

        'feature_fraction': 0.8

       }



folds = GroupKFold(n_splits=10)



oof_preds = np.zeros(X_train.shape[0])

sub_preds = np.zeros(X_test.shape[0])

for fold_, (trn_, val_) in enumerate(folds.split(X_train, Y_train, groups=X_game_id)):

    trn_x, trn_y = X_train.iloc[trn_], Y_train.iloc[trn_]

    val_x, val_y = X_train.iloc[val_], Y_train.iloc[val_]

    

    reg = lgb.LGBMRegressor(**params, n_estimators=3000)

    reg.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=100, verbose=500)

    

    oof_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)

    sub_preds += reg.predict(X_test, num_iteration=reg.best_iteration_) / folds.n_splits



pred = sub_preds
from sklearn import metrics



fpr, tpr, thresholds = metrics.roc_curve(Y_train, oof_preds)

auc = metrics.auc(fpr, tpr)



plt.plot(fpr, tpr, label='ROC curve (area = %.3f)'%auc)

plt.legend()

plt.title('ROC curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.grid(True)
# Plot feature importance

feature_importance = reg.feature_importances_

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

#sorted_idx = sorted_idx[len(feature_importance) - 10:]

pos = np.arange(sorted_idx.shape[0]) + .5



plt.figure(figsize=(12,8))

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X_train.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
submission = pd.DataFrame({

    "shot_id": df[df.shot_made_flag.isnull()]["shot_id"],

    "shot_made_flag": pred

})

submission.to_csv("submission.csv", index=False)
submission['shot_made_flag'].hist(bins=25)

np.mean(submission['shot_made_flag'])