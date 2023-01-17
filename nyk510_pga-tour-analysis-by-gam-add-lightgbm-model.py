# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# ref: https://www.kaggle.com/mesadowski/moneyball-golf-scikit-learn-and-tf-estimator-api



# read data

df = pd.read_csv('../input/pga-tour-20102018-data/PGA_Data_Historical.csv')



# unstack

df = df.set_index(['Player Name', 'Variable', 'Season'])['Value'].unstack('Variable').reset_index()



# drop non-numeric features

keep_columns = [

    'Player Name',

    'Season',

    'Total Money (Official and Unofficial) - (MONEY)', # 年間獲得賞金 (ドル)

    'Driving Distance - (AVG.)', # ドライバーの平均飛距離 (ヤード)

    'Driving Accuracy Percentage - (%)', # ドライバーのフェアウェイキープ率

    'Total Distance Efficiency - (AVERAGE DISTANCE (YARDS))', # ドライバーの、ボール スピードに対する飛距離 (高いほうが効率よく飛んでいる)

    'Average Distance to Hole After Tee Shot - (AVG)', # 平均の平地ショット飛距離

    'Ball Speed - (AVG.)', # 平均ボール速度

    'Scrambling from the Sand - (%)', # バンカーからのスクランブル率 (パーオン出来なかったホールで、パー以上であがること)

    'Scrambling from the Fringe - (%)', # グリーンのフリンジからのスクランブル率

    'Scrambling from the Rough - (%)', # ラフからのスクランブル率

    '3-Putt Avoidance - (%)', # 3 パットしたホールの割合

    'Birdie or Better Conversion Percentage - (%)' # バーディーより良い成績で挙がるホールの割合

]

df = df[keep_columns].dropna()



# rename the columns to something shorter

df.rename(columns = {'Total Money (Official and Unofficial) - (MONEY)':'Money'}, inplace = True)

df.rename(columns = {'3-Putt Avoidance - (%)':'ThreePuttRate'}, inplace = True)

df.rename(columns = {'Average Distance to Hole After Tee Shot - (AVG)':'NonDrivingDistance'}, inplace = True)

df.rename(columns = {'Total Distance Efficiency - (AVERAGE DISTANCE (YARDS))':'DistanceEfficiency'}, inplace=True)

df.rename(columns = {'Ball Speed - (AVG.)':'BallSpeed'}, inplace=True)

df.rename(columns = {'Driving Distance - (AVG.)':'DrivingDistance'}, inplace = True)

df.rename(columns = {'Driving Accuracy Percentage - (%)':'DrivingAccuracy'}, inplace=True)

df.rename(columns = {'Scrambling from the Sand - (%)':'ScramblingSand'}, inplace = True)

df.rename(columns = {'Scrambling from the Fringe - (%)':'ScramblingFringe'}, inplace=True)

df.rename(columns = {'Scrambling from the Rough - (%)':'ScramblingRough'}, inplace=True)

df.rename(columns = {'Birdie or Better Conversion Percentage - (%)':'BirdieConversion'}, inplace=True)



# remove $ and commas from Money

df['Money']= df['Money'].str.replace('$','')

df['Money']= df['Money'].str.replace(',','')



# make all variables into number

for col in  df.columns[2:]:

   df[col] = df[col].astype(float)
np.random.seed(0)

index = np.random.randint(df.shape[0], size=10)

df.iloc[index,:]
df.mean()
df.groupby("Season").mean()
df.loc[df.groupby("Season")["Money"].idxmax()]
corr = df[df.columns[2:]].corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)]= True



f, ax = plt.subplots(figsize=(11, 15))

heatmap = sns.heatmap(corr, 

                      square = True,

                      mask = mask,

                      linewidths = .5,

                      cmap = 'coolwarm',

                      cbar_kws = {'shrink': .4, 

                                'ticks' : [-1, -.5, 0, 0.5, 1]},

                      vmin = -1, 

                      vmax = 1,

                      annot = True,

                      annot_kws = {"size": 12})



ax.set_yticklabels(corr.columns, rotation = 0)

ax.set_xticklabels(corr.columns)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
# 説明変数と目的変数

X = df[df.columns[3:]]

y = df["Money"]



print('X', X.shape)

print('y', y.shape)
from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



# 訓練データとテストデータに分離

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2018)
!pip install pygam
from pygam import s, LinearGAM



gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9))

gam = gam.fit(X_train, y_train)
def plot_splines(gam):

    fig, axes = plt.subplots(2, 5, figsize=(18, 12))

    axes = np.array(axes).flatten()

    for i, (ax, title, p_value) in enumerate(zip(axes, X_train.columns, gam.statistics_['p_values'])):

        XX = gam.generate_X_grid(term=i)

        ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))

        ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')

        ax.axhline(0, c='#cccccc')

        ax.set_title("{0:} (p={1:.2})".format(title, p_value))

        ax.set_yticks([])

        

plot_splines(gam)
from pygam import l, LinearGAM



glm = LinearGAM(l(0) + l(1) + l(2) + l(3) + l(4) + l(5) + l(6) + l(7) + l(8) + l(9))

glm = glm.fit(X_train, y_train)
plot_splines(glm)
!pip install interpret
from interpret import show

from interpret.data import Marginal



marginal = Marginal().explain_data(X_test, y_test, name = 'test data')

show(marginal)
from interpret.glassbox import ExplainableBoostingRegressor, LinearRegression, RegressionTree



ebm = ExplainableBoostingRegressor(random_state=0, scoring="mean_squared_error")

ebm.fit(X_train, y_train)
ebm_global = ebm.explain_global(name='EBM')

show(ebm_global)
ebm_local = ebm.explain_local(X_test[:5], y_test[:5], name='EBM')

show(ebm_local)
from interpret import show

from interpret.perf import RegressionPerf



ebm_perf = RegressionPerf(ebm.predict).explain_perf(X_test, y_test, name='EBM')

show(ebm_perf)
from interpret.glassbox import LinearRegression, RegressionTree



lr = LinearRegression(random_state=0)

lr.fit(X_train, y_train)

rt = RegressionTree(random_state=0)

rt.fit(X_train, y_train)



model_names = [

    (glm, "GLM"),

    (gam, "GAM"),

    (ebm, "InterpretML (EBM)"),

    (lr, "InterpretML (LR)"),

    (rt, "InterpretML (RT)")

]

result = pd.DataFrame()



for (model, name) in model_names:

    y_pred = model.predict(X_test)

    series = pd.Series()

    series["model"] = name

    series["MSE"] = mean_squared_error(y_test, y_pred)

    result = result.append(series, ignore_index=True)

result
from lightgbm import LGBMRegressor

from sklearn.model_selection import KFold



def train_lgbm(params, log=False):

    fold = KFold(n_splits=8)

    oof = np.zeros_like(y_train)

    models = []

    

    if log:

        y = np.log1p(y_train.values)

    else:

        y = y_train.values



    for idx_train, idx_valid in fold.split(X_train, y):

        clf = LGBMRegressor(**params)

        clf.fit(X_train.values[idx_train], y[idx_train], 

                eval_set=(X_train.values[idx_valid], y[idx_valid]), 

               early_stopping_rounds=100,

               verbose=50)

        oof[idx_valid] = clf.predict(X_train.values[idx_valid])

        models.append(clf)

    if log:

        oof = np.expm1(oof)

    return models, oof
!pip install git+https://gitlab.com/nyker510/vivid
params = {

    'objective': 'poisson',

    'learning_rate': .05,

    'n_estimators': 1000,

    'reg_lambda': 10.,

    'colsample_bytree': .7,

    'importance_type': 'gain',

    'max_depth': 2 # 多重共線性が悪さしているようで depth が深いと精度が出ない

}



models, oof = train_lgbm(params, log=False)
from vivid.metrics import regression_metrics
regression_metrics(oof, y_train)
from vivid.visualize import visualize_feature_importance



fig, ax, importance_df = visualize_feature_importance(models, columns=X_train.columns)
importance_df.groupby('column').mean()
class LGBMEnsumble:

    def __init__(self, models, log=False):

        self.models = models

        self.log = log

        

    def predict(self, x):

        p = np.mean([m.predict(x) for m in self.models], axis=0)

        if self.log:

            p = np.expm1(p)

        return p
lgbm_ens = LGBMEnsumble(models)
import seaborn as sns

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(oof, y_train)


model_names = [

    (glm, "GLM"),

    (gam, "GAM"),

    (ebm, "InterpretML (EBM)"),

    (lr, "InterpretML (LR)"),

    (rt, "InterpretML (RT)"),

    (lgbm_ens, 'lgbm')

]

result = pd.DataFrame()



fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=True, sharey=True)

axes = [a for x in axes for a in x]



for (model, name), ax in zip(model_names, axes):

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    ax.scatter(y_pred, y_test, label=name)

    ax.set_title(f'{mse:.3e}')

    ax.legend()

    series = pd.Series()

    series["model"] = name

    series["MSE"] = mse

    result = result.append(series, ignore_index=True)



fig.tight_layout()



result
result.plot(kind='bar', x='model')
from vivid.out_of_fold.ensumble import RFRegressorFeatureOutOfFold

from vivid.out_of_fold.boosting import OptunaXGBRegressionOutOfFold, OptunaXGBRegressionOutOfFold
rf = RFRegressorFeatureOutOfFold(name='rf')
rf.fit(X_train, y_train.values)

p = rf.predict(X_test)

regression_metrics(y_test, p)
optuna_lgbm = OptunaXGBRegressionOutOfFold(name='optuna_lgbm', n_trials=200)
optuna_lgbm.fit(X_train, y_train.values)
p = optuna_lgbm.predict(X_test)



regression_metrics(y_test, p)
optuna_xgb = OptunaXGBRegressionOutOfFold(n_trials=200, name='optuna_xgb')
optuna_xgb.fit(X_train, y_train.values)
y = optuna_xgb.predict(X_test)

regression_metrics(y_test, y)