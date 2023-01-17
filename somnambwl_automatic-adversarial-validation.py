import collections

import gc

import pickle

import sys

import time



import numpy as np

import pandas as pd



import sklearn

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.utils import resample



import lightgbm as lgb

from lightgbm import LGBMRegressor, LGBMClassifier

import lightgbm



%matplotlib notebook

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from bokeh.models import HoverTool

from bokeh.io import output_notebook

from bokeh.plotting import figure, show



from tqdm.auto import tqdm
# Seaborn advanced settings



sns.set(style='ticks',          # 'ticks', 'darkgrid'

        palette='colorblind',   # 'colorblind', 'pastel', 'muted', 'bright'

        #palette=sns.color_palette('Accent'),   # 'Set1', 'Set2', 'Dark2', 'Accent'

        rc = {

           'figure.autolayout': True,

           'figure.figsize': (14, 8),

           'legend.frameon': True,

           'patch.linewidth': 2.0,

           'lines.markersize': 6,

           'lines.linewidth': 2.0,

           'font.size': 20,

           'legend.fontsize': 20,

           'axes.labelsize': 16,

           'axes.titlesize': 22,

           'axes.grid': True,

           'grid.color': '0.9',

           'grid.linestyle': '-',

           'grid.linewidth': 1.0,

           'xtick.labelsize': 20,

           'ytick.labelsize': 20,

           'xtick.major.size': 8,

           'ytick.major.size': 8,

           'xtick.major.pad': 10.0,

           'ytick.major.pad': 10.0,

           }

       )



plt.rcParams['image.cmap'] = 'viridis'
# Set general parameters here

N_samples = int(1e5)   # Double of this amount is used (train and test)

drop_max_columns = -1   # -1 is for: repeat until all but one columns are dropped

allowed_time = 7.5 * 60 * 60



N_folds = 2   # Minimum is 2, but you can fork and set different CV 



n_estimators = 50                                                                                                                                                       

verbose = 100                                                                                                                                                               

early_stopping_rounds = 50 
# Set LightGBM hyperparameters here

params = {'num_leaves': 1000,

          'min_child_weight': 0.03,

          'feature_fraction': 1,

          'subsample': 0.5,

          'min_data_in_leaf': 100,

          'objective': 'binary',

          'max_depth': 500,

          'learning_rate': 0.007,

          "boosting_type": "gbdt",

          "bagging_seed": 312,

          "metric": 'auc',

#           "verbosity": -1,

          'reg_alpha': 0.4,

          'reg_lambda': 0.6,

          'random_state': 312,

          'use_missing': True,

         }
# Load your train and test data in this cell

with open("../input/ieee-the-most-basic-preprocessing/Train-dtypes.pkl", "rb") as f:

    train_dtypes = pickle.load(f)

with open("../input/ieee-the-most-basic-preprocessing/Test-dtypes.pkl", "rb") as f:

    test_dtypes = pickle.load(f)

    

train = pd.read_csv("../input/ieee-the-most-basic-preprocessing/Train.csv", dtype=train_dtypes)

test = pd.read_csv("../input/ieee-the-most-basic-preprocessing/Test.csv", dtype=test_dtypes)
features = test.columns

train = train[features]
thing_to_learn = "target"



train['target'] = 0

test['target'] = 1



train["target"] = train["target"].astype("int16")

test["target"] = test["target"].astype("int16")

train_dtypes["target"] = train["target"].dtype
dset = pd.concat([train.sample(N_samples), test.sample(N_samples)], ignore_index=True).reindex()

for column in tqdm(dset.columns):

    dset[column] = dset[column].astype(train_dtypes[column])

print(dset.shape)

del train, test

gc.collect()
drop_columns = list()

dropped_values = list()

CVs = list()

if drop_max_columns == -1:

    drop_max_columns = dset.shape[1] - 1

if drop_max_columns > dset.shape[1]:

    drop_max_columns = dset.shape[1] - 1
dropped_already = -1

start_time = time.time()

current_time = time.time()

while dropped_already < drop_max_columns and (current_time - start_time) < allowed_time:

    CV = 0

    feature_importance = pd.DataFrame() 

    kf = KFold(n_splits=N_folds, random_state=312, shuffle=True)

    fold_n = 0

    for train_index, valid_index in kf.split(dset):

        X_train, X_valid = dset.drop([thing_to_learn]+drop_columns, axis=1).loc[train_index], dset.drop([thing_to_learn]+drop_columns, axis=1).loc[valid_index]

        y_train, y_valid = dset.loc[train_index, thing_to_learn], dset.loc[valid_index, thing_to_learn]

        dtrain = lgb.Dataset(X_train, label=y_train)

        dvalid = lgb.Dataset(X_valid, label=y_valid)

        model = lgb.train(params, dtrain, n_estimators, valid_sets=[dtrain, dvalid], verbose_eval=verbose, early_stopping_rounds=500)

        fold_importance = pd.DataFrame()

        fold_importance["feature"] = dset.drop([thing_to_learn]+drop_columns, axis=1).columns                                                                                                                                     

        fold_importance["importance"] = model.feature_importance()                                                                                                              

        fold_importance["fold"] = fold_n + 1                  

        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)                                                                                        

        CV += model.best_score["valid_1"]["auc"] 

        fold_n += 1

    CV /= N_folds

    fi = pd.DataFrame({"Feature":feature_importance.pivot(index="fold", columns="feature", values="importance").columns.values,                                   

                      "Value":feature_importance.pivot(index="fold", columns="feature", values="importance").mean().values})

    row = fi.iloc[fi['Value'].idxmax()]

    feature = row["Feature"]

    value = row["Value"]

    drop_columns.append(feature)

    dropped_values.append(value)

    CVs.append(CV)

    dropped_already += 1

    current_time = time.time()
print(list(drop_columns))
drop_columns = drop_columns[:-1]

dropped_values = dropped_values[:-1]

drop_columns.insert(0, "Nothing")

dropped_values.insert(0, 0)

deltas = list(np.array(CVs)[:-1] - np.array(CVs)[1:])

deltas.insert(0, 0)

df = pd.DataFrame({"Dropped_feature":drop_columns, "CV_change":deltas, "N_dropped":np.arange(len(CVs)), "CV":CVs})

df.to_csv("Results.csv")
output_notebook()



fig = figure(plot_height=600,

             plot_width=700,

             x_axis_label='Number of dropped features',

             x_range=(0, drop_max_columns),

             y_axis_label='AUC',

             title='AUC vs number of dropped features',

             toolbar_location='below',

             tools='save, box_zoom, pan, wheel_zoom, reset')



tooltips = [

            ('Dropped feature','@Dropped_feature'),

            ('CV change', '@CV_change'),

            ("Number of dropped features", "@N_dropped"),

            ("Current CV", "@CV")

           ]



scatter = fig.circle(x='N_dropped', y='CV', source=df,

        size=15, alpha=1,

        hover_fill_color='cyan', hover_alpha=0.3)



fig.add_tools(HoverTool(tooltips=tooltips, renderers=[scatter]))



show(fig)
pd.set_option('display.max_rows', 100)

df.head(100)
plt.figure()

plt.plot(np.arange(len(CVs)), CVs, "o")

plt.ylabel("AUC")

plt.xlabel("Number of dropped features")

plt.savefig("AUCvsDroppedFeatures.png")

plt.show()