INPUT_DIR = '../input/'
OUTPUT_DIR = './'
!ls -laFh {INPUT_DIR}
!pip install --upgrade sparklines
!pip install --upgrade treeinterpreter
!pip install --upgrade nmlu
%load_ext autoreload
%autoreload 2
%matplotlib inline

import re

import pprint
pp = pprint.PrettyPrinter(indent=2).pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # do sns.set() if you want its default styles globally applied

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

# NMLU (Nano Machine Learning Utils) - my lib of a few simple helpers to keep boring code DRY
from nmlu.qinspect.nb import df_peek, df_display_all
from nmlu.qinspect.common import df_types_and_stats
import nmlu.etl as etl
import nmlu.eda as eda
import nmlu.model_analysis as ma
import nmlu.model_utils as mu

eda.set_plot_sane_defaults()
df_raw = pd.read_csv(f'{INPUT_DIR}train.csv', low_memory=False)
df_peek(df_raw, 'df_raw')  # summary with stats and datatypes
# categories
df_raw.Survived = df_raw.Survived.astype('category').cat.as_ordered()
df_raw.Pclass = df_raw.Pclass.astype('category').cat.as_ordered()
etl.train_cats(df_raw)
df_peek(df_raw)
# quick look at things
eda.plot_pairs_corr(df_raw.drop(columns='PassengerId'));
eda.plot_heatmap(df_raw);
# make fully numericalize version of data
nzDf_raw = etl.numericalized_df(df_raw)
pp(nzDf_raw.dtypes)
nzDf_raw[:5]
# take a deeper look at things
eda.plot_pairs_dists(nzDf_raw, 'Survived');
eda.plot_heatmap(nzDf_raw);
for n, c in df_raw.items():
    if n in {'PassengerId', 'Survived', 'Cabin', 'Name', 'Ticket'}:
        continue
    if not pd.api.types.is_categorical(c):
        continue
    print(f"--- {n}")
    eda.show_cat_feature_vs_y(df_raw, n, 'Survived')
eda.plot_dendrogram(nzDf_raw.drop('Survived', axis=1));
def get_score(model, x_trn, y_trn, x_val, y_val):
    score_trn = model.score(x_trn, y_trn)
    score_val = model.score(x_val, y_val)
    r = dict({
        "Score training": score_trn,
        "Score validation": score_val
    })
    if hasattr(model, 'oob_score_'):
        r["OOB score"] = model.oob_score_
    return r
# simplest non-useless model we can think of,
# to get some feature importances
x, y, nas = etl.proc_df(
    nzDf_raw, 'Survived', skip_flds=['PassengerId'])

x_trn, x_val, y_trn, y_val = train_test_split(
    x, y, test_size=int(0.25 * len(x)))

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,  # ~ log2(features) * 2
    oob_score=True,
)
# repeat model training and feature importance to make sure
# we don't have a model so random it's useless
for i in range(4):
    model.fit(x_trn, y_trn)
    pp(get_score(model, x_trn, y_trn, x_val, y_val))
    ma.rf_show_plot_fi(model, x_val)
# also test predictions of this simple model
df_test = pd.read_csv(f'{INPUT_DIR}test.csv')

etl.apply_cats(df_test, df_raw)

# df_peek(df_test)

x_test, y_test, nas_test = etl.proc_df(df_test, na_dict=nas.copy())

model.fit(x, y)

print("OOB score:", model.oob_score_)

preds_final = model.predict(x_test.drop(columns='PassengerId'))
result = pd.DataFrame({'PassengerId': x_test.PassengerId,
                       'Survived': preds_final})
display(result.head())

result.to_csv(f'{OUTPUT_DIR}results_eda.csv', index=False)
# do FI with 1-hot-encoded data too
# also, turning binary cats like Sex to dummies only confuses
# FI (and likely doesn't help the model either), hence
# pass no_binary_dummies=True
oheX, oheY, oheNas = etl.proc_df(
    df_raw, 'Survived',
    max_n_cat=10,
    no_binary_dummies=True,
    skip_flds=['PassengerId'])

oheX_trn, oheX_val, oheY_trn, oheY_val = train_test_split(
    oheX, oheY, test_size=int(0.25 * len(x)))

df_peek(oheX)
oheModel = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,  # ~ log2(features) * 2
    oob_score=True,
)
# repeat model training and feature importance to make sure
# we don't have a model so random it's useless
for i in range(4):
    oheModel.fit(oheX_trn, oheY_trn)
    pp(get_score(oheModel, oheX_trn, oheY_trn, oheX_val, oheY_val))
    ma.rf_show_plot_fi(oheModel, oheX_val)
# also test predictions of OHE simple model
oheDf_test = pd.read_csv(f'{INPUT_DIR}test.csv')

etl.apply_cats(oheDf_test, df_raw)

# df_peek(oheDf_test)

oheX_test, oheY_test, oheNas_test = etl.proc_df(
    oheDf_test, na_dict=oheNas.copy())

oheModel.fit(oheX, oheY)

print("OOB score:", oheModel.oob_score_)

ohePreds_final = model.predict(oheX_test.drop(columns='PassengerId'))
oheResult = pd.DataFrame({'PassengerId': oheX_test.PassengerId,
                          'Survived': ohePreds_final})
display(oheResult.head())

oheResult.to_csv(f'{OUTPUT_DIR}oheResults_eda.csv', index=False)
# just check how well CV correlates with kaggle scores
# (model: 0.66028, oheModel: 0.65550 before, ? now)
def do_cv(model, x, y):
    scores = cross_val_score(model, x, y, cv=10)
    r = "Accuracy: %0.2f (+/- %0.2f)" % (
        scores.mean(), scores.std() * 2)
    return r

def do_shuffle_cv(model, x, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    scores = cross_val_score(model, x, y, cv=cv)
    r = "Accuracy (shuffled): %0.2f (+/- %0.2f)" % (
        scores.mean(), scores.std() * 2)
    return r
do_cv(model, x, y)
do_cv(oheModel, oheX, oheY)