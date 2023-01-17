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
def plot_corr(corr, figsize=(11,15)):

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)]= True



    f, ax = plt.subplots(figsize=figsize)

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

    

corr = df[df.columns[2:]].corr()

plot_corr(corr)
def cond(corr):

    eig_values = np.linalg.eigvals(corr)

    return eig_values.max() / (eig_values.min() + 1e-10)



def validate_cond(cond):

    # https://www3.nd.edu/~rwilliam/stats2/l11.pdf

    if cond > 30:

        return 'danger'

    elif cond > 15:

        return 'warning'

    else:

        return 'good'

    

print('相関行列の条件数:', cond(corr))

print('信号:', validate_cond(cond(corr)))
df["Money"].describe()
fig, axes = plt.subplots(figsize=(10,4), ncols=2)

df["Money"].plot.hist(bins=100, ax=axes[0])

df["Money"].apply(np.log1p).plot.hist(bins=100, ax=axes[1])

axes[0].set_title('Money')

axes[1].set_title('log(Money)')

plt.show()
from sklearn.model_selection import train_test_split



X = df[df.columns[3:]]

y = df["Money"]



# 特徴量選択用のデータセット

rate_for_feature_selection = 0.1

X_fs, X, y_fs, y = train_test_split(X, y, train_size=rate_for_feature_selection, random_state=2018)
X_fs.shape, y_fs.shape
from lightgbm import LGBMRegressor

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFECV

from sklearn.metrics import mean_squared_error



rfecv = RFECV(estimator=LGBMRegressor(),

              cv=3,

              scoring='neg_mean_squared_error')

rfecv.fit(X_fs, np.log1p(y_fs))



print("Optimal number of features : %d" % rfecv.n_features_)

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
from mlxtend.feature_selection import SequentialFeatureSelector as SFS



sfs = SFS(LGBMRegressor(),

          k_features=4,

          forward=True,

          floating=False,

          scoring='neg_mean_squared_error',

          cv=3)



sfs.fit(X_fs, y_fs)

selected_features = X_fs.columns[list(sfs.k_feature_idx_)]



print(selected_features)
corr = df[selected_features].corr()

plot_corr(corr, figsize=(6,4))

print('相関行列の条件数:', cond(corr))

print('信号:', validate_cond(cond(corr)))
# > rate_for_feature_selection = 0.1

# > X_fs, X, y_fs, y = train_test_split(X, y, train_size=rate_for_feature_selection, random_state=2018)



X = X[selected_features]

y = y
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2018)



print('X_train', X_train.shape)

print('X_test', X_test.shape)
!pip install pygam
from pygam import s, LinearGAM



lineargam = LinearGAM(s(0) + s(1) + s(2) + s(3)).fit(X_train, y_train)

lineargam.summary()
def plot_splines(gam):

    fig, axes = plt.subplots(ncols=4, figsize=(14, 5))

    axes = np.array(axes).flatten()

    for i, (ax, title, p_value) in enumerate(zip(axes, X_train.columns, gam.statistics_['p_values'])):

        XX = gam.generate_X_grid(term=i)

        ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))

        ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')

        ax.axhline(0, c='#cccccc')

        ax.set_title("{0:} (p={1:.2})".format(title, p_value))

        ax.set_yticks([])

        

plot_splines(lineargam)
y.plot.hist(bins=100)
from pygam import GammaGAM



gammagam = GammaGAM(s(0) + s(1) + s(2) + s(3)).fit(X_train, y_train)

gammagam.summary()
print('GAM (gaussian)', lineargam.statistics_['AIC'])

print('GAM (gamma)   ', gammagam.statistics_['AIC'])
print('GAM (gaussian)', lineargam.statistics_['GCV'])

print('GAM (gamma)   ', gammagam.statistics_['GCV'])
def msle(y_pred, y_test):

    return np.sum((np.log1p(y_pred) - np.log1p(y_test))**2) / len(y_test)



def gam_msle(gamobj):

    y_pred = gamobj.predict(X_test)

    return msle(y_pred, y_test)



print('LinearGAM', gam_msle(lineargam))

print('GammaGAM',  gam_msle(gammagam))
from pygam import l, GammaGAM



gammaglm = GammaGAM(l(0) + l(1) + l(2) + l(3)).fit(X_train, y_train)

gammaglm.summary()
plot_splines(gammaglm)
print('GAM (dist=gaussian, link=identity)')

plot_splines(lineargam)

plt.show()



print('GAM (dist=gamma, link=log)')

plot_splines(gammagam)

plt.show()



print('GLM (dist=gamma, link=log)')

plot_splines(gammaglm)

plt.show()
print('GAM (gaussian)', lineargam.statistics_['AIC'])

print('GAM (gamma)   ', gammagam.statistics_['AIC'])

print('GLM (gamma)   ', gammaglm.statistics_['AIC'])
print('GAM (gaussian)', lineargam.statistics_['GCV'])

print('GAM (gamma)   ', gammagam.statistics_['GCV'])

print('GLM (gamma)   ', gammaglm.statistics_['GCV'])
print('GAM (gaussian)', gam_msle(lineargam))

print('GAM (gamma)   ', gam_msle(gammagam))

print('GLM (gamma)   ', gam_msle(gammaglm))
!pip install interpret
from interpret import show

from interpret.data import Marginal



# ログ変換を使う

y_train_log, y_test_log = np.log1p(y_train), np.log1p(y_test)



marginal = Marginal().explain_data(X_test, y_test_log, name='test data')

show(marginal)
from interpret.glassbox import ExplainableBoostingRegressor, LinearRegression, RegressionTree



ebm = ExplainableBoostingRegressor(random_state=0, scoring="mean_squared_error")

ebm.fit(X_train, y_train_log)
ebm_global = ebm.explain_global(name='EBM')

show(ebm_global)
ebm_local = ebm.explain_local(X_test[:5], y_test_log[:5], name='EBM')

show(ebm_local)
from interpret import show

from interpret.perf import RegressionPerf



ebm_perf = RegressionPerf(ebm.predict).explain_perf(X_test, y_test_log, name='EBM')

show(ebm_perf)
from sklearn.metrics import mean_squared_error

from interpret.glassbox import LinearRegression, RegressionTree

from lightgbm import LGBMRegressor



# Train some models

# -----------------------

lr = LinearRegression(random_state=0)

lr.fit(X_train, y_train_log)



rt = RegressionTree(random_state=0)

rt.fit(X_train, y_train_log)



lgb = LGBMRegressor()

lgb.fit(X_train, y_train_log)





# Evaluate the models

# -----------------------



# model, name, is_log_pred_model

models = [

    (lineargam, "GAM (Normal)",             False),

    (gammagam,  "GAM (Gamma)",              False),

    (gammaglm,  "GLM (Gamma)",              False),

    (ebm,       "InterpretML (EBM = GA2M)", True),

    (lr,        "InterpretML (LR)",         True),

    (rt,        "InterpretML (RT)",         True),

    (lgb,       "LightGBM",                 True)

]



result = pd.DataFrame()

for (model, name, is_log_pred_model) in models:

    if is_log_pred_model:

        y_pred_log = model.predict(X_test)

        error = mean_squared_error(y_test_log, y_pred_log)

    else:

        y_pred = model.predict(X_test)

        error = msle(y_test, y_pred)

        

    series = pd.Series()

    series["model"] = name

    series["MSLE"] = error

    result = result.append(series, ignore_index=True)
result.sort_values('MSLE')