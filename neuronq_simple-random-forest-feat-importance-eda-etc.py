## I/O dirs, other global params
INPUT_DIR = '../input/'
OUTPUT_DIR = './'
!ls -lah {INPUT_DIR}
# DEPS
!pip install --upgrade nmlu  # personal library of a few little helpers (has nice readable explained code)
!pip install --upgrade sparklines
!pip install --upgrade treeinterpreter
## Settings
%load_ext autoreload
%autoreload 2
%matplotlib inline
## Imports
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # do sns.set() if you want its default styles globally applied

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

# NMLU (Nano Machine Learning Utils) - my lib of a few simple helpers to keep boring code DRY
from nmlu.qinspect.nb import df_peek, df_display_all
from nmlu.qinspect.common import df_types_and_stats
import nmlu.etl as etl
import nmlu.eda as eda
import nmlu.model_analysis as ma
import nmlu.model_utils as mu
df_raw = pd.read_csv(f'{INPUT_DIR}train.csv', low_memory=False)
df_peek(df_raw, 'df_raw')  # summary with stats and datatypes
# take a look through a larger random sample
df_display_all(etl.get_sample(df_raw, 100))
# auto-categorize stringy columns, dependent variable
# and everyth else that makes sense to be a category
etl.train_cats(df_raw)
df_raw.Survived = df_raw.Survived.astype('category').cat.as_ordered()
df_raw.Pclass = df_raw.Pclass.astype('category').cat.as_ordered()
df_raw.dtypes
# fully numericalized (but not one-hot-encoded) data for some data analysis and stuff
df = etl.numericalized_df(df_raw)
print(df.dtypes)
df[:5]
# numericalized and one-hot-encoded data for use in prediction
x, y, nas = etl.proc_df(df_raw, 'Survived', max_n_cat=5)
print(x.shape, y.shape, nas)
display(etl.get_sample(x, 10))
x.dtypes
eda.set_plot_sane_defaults()
eda.plot_pairs_corr(df_raw);
eda.plot_pairs_dists(df, 'Survived');
eda.plot_heatmap(df);
# figure out what can be bar-plotted vs. response variable
df_raw.drop('Survived', axis=1).agg(['dtype', 'nunique', 'unique']).T
for c in ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']:
    print(f"--- Feature {c}:")
    eda.show_cat_feature_vs_y(df_raw, c, 'Survived')
eda.plot_dendrogram(df.drop('Survived', axis=1));
TRN_F = 0.8
trn_sz = int(len(df) * TRN_F)

sx, sy = shuffle(x, y)

x_trn, x_val = etl.arr_split(sx, trn_sz)
y_trn, y_val = etl.arr_split(sy, trn_sz)

print(f"shapes: x_trn~{x_trn.shape} y_trn~{y_trn.shape} x_val~{x_val.shape} y_val~{y_val.shape}")
x_trn[:5]
def rmse(x, y):
    return np.sqrt(((x - y)**2).mean())


def print_score(model):
    rmse_trn = rmse(model.predict(x_trn), y_trn)
    rmse_val = rmse(model.predict(x_val), y_val)
    score_trn = model.score(x_trn, y_trn)
    score_val = model.score(x_val, y_val)
    print(f"RMSE training: {rmse_trn:.6f}, validation: {rmse_val:.6f}")
    print(f"Score training: {score_trn:.6f}, validation: {score_val:.6f}")
    if hasattr(model, 'oob_score_'):
        print(f"OOB score: {model.oob_score_:.6f}")
m0 = RandomForestClassifier(n_estimators=20, min_samples_leaf=3, max_features=0.75, n_jobs=-1, oob_score=True)
m0.fit(x_trn, y_trn)
print_score(m0)
ma.rf_show_plot_fi(m0, x_val)
ma.rf_predict_with_explanations(m0, x_val[:3])
m1 = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features=0.75, n_jobs=-1, oob_score=True)
m1.fit(x_trn, y_trn)
print_score(m1)
df_peek(df_raw, 'df_raw')
df_display_all(etl.get_sample(df_raw, 100))
df_raw2 = df_raw.copy()
df_raw2[['Name_family', 'Name_title', 'Name_first']] = df_raw2.Name.str.split(r'[,.]\s+', n=2, expand=True)
df_raw2.drop('Name', axis=1, inplace=True)

df_raw2['Ticket_pre'] = df_raw2.Ticket.str.extract(r'([^s]+)\s+')
df_raw2['Ticket_n'] = df_raw2.Ticket.str.extract(r'(\d+)$')

df_raw2['Cabin_pre'] = df_raw2.Cabin.str.extract(r'([^\d\s]+)')
df_raw2['Cabin_n'] = df_raw2.Cabin.str.extract(r'(\d+)$')
df_raw2['Cabin_len'] = df_raw2.Cabin.str.split(r'\s+').apply(
    lambda x: 0 if not hasattr(x, '__len__') else len(x)
)
df_peek(df_raw2)
df_display_all(etl.get_sample(df_raw2, 100))
df_raw2.dtypes
def process_data(df_raw_input, y_field='Survived', na_dict=None):
    raw = df_raw_input.copy()

    # feature processing/engineering
    raw[
        ['Name_family', 'Name_title', 'Name_first']
    ] = raw.Name.str.split(r'[,.]\s+', n=2, expand=True)
    raw.drop('Name', axis=1, inplace=True)

    raw['Ticket_pre'] = raw.Ticket.str.extract(r'([^s]+)\s+')
    raw['Ticket_n'] = raw.Ticket.str.extract(r'(\d+)$')

    raw['Cabin_pre'] = raw.Cabin.str.extract(r'([^\d\s]+)')
    raw['Cabin_n'] = raw.Cabin.str.extract(r'(\d+)$')
    raw['Cabin_len'] = raw.Cabin.str.split(r'\s+').apply(
        lambda x: 0 if not hasattr(x, '__len__') else len(x)
    )
    
    # auto-categorize stringy columns, dependent variable
    # and everyth else that makes sense to be a category
    etl.train_cats(raw)
    if y_field:
        raw[y_field] = raw[y_field].astype('category').cat.as_ordered()
    raw.Pclass = raw.Pclass.astype('category').cat.as_ordered()
    print("raw.dtypes:")
    print(raw.dtypes)
    
    # fully numericalized (but not one-hot-encoded) data for some data analysis and stuff
    nzd = etl.numericalized_df(raw)
    print("\nnzd.dtypes:")
    print(nzd.dtypes)
    print("\nnzd[:5]:")
    display(nzd[:5])
    
    # numericalized and one-hot-encoded data for use in prediction
    x, y, nas = etl.proc_df(raw, y_field, max_n_cat=5, na_dict=na_dict)
    print("x.shape, y.shape, nas:", x.shape, y.shape if y_field else None, nas)
    print("sample of 10 from x:")
    df_display_all(etl.get_sample(x, 50).T)
    print("x.dtypes:")
    print(x.dtypes)
    
    return raw, nzd, x, y, nas
df_raw2, df2, x2, y2, nas2 = process_data(df_raw)
sx2, sy2 = shuffle(x2, y2)

x_trn2, x_val2 = etl.arr_split(sx2, trn_sz)
y_trn2, y_val2 = etl.arr_split(sy2, trn_sz)

print(f"shapes: x_trn2~{x_trn2.shape} y_trn2~{y_trn2.shape} x_val2~{x_val2.shape} y_val2~{y_val2.shape}")
x_trn2[:5]
def print_score2(model):
    rmse_trn = rmse(model.predict(x_trn2), y_trn2)
    rmse_val = rmse(model.predict(x_val2), y_val2)
    score_trn = model.score(x_trn2, y_trn2)
    score_val = model.score(x_val2, y_val2)
    print(f"RMSE training: {rmse_trn:.6f}, validation: {rmse_val:.6f}")
    print(f"Score training: {score_trn:.6f}, validation: {score_val:.6f}")
    if hasattr(model, 'oob_score_'):
        print(f"OOB score: {model.oob_score_:.6f}")
m2 = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features=0.8, n_jobs=-1, oob_score=True)
m2.fit(x_trn2, y_trn2)
print_score2(m2)
mu.set_rf_samples(int(len(x_trn2) * 0.8))
m2b = RandomForestClassifier(n_estimators=1000, min_samples_leaf=3, max_features=0.8, n_jobs=-1, oob_score=True)
m2b.fit(x_trn2, y_trn2)
print_score2(m2b)
mu.reset_rf_samples()
def do_cv(model, x, y):
    scores = cross_val_score(model, x, y, cv=10)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
do_cv(m0, sx, sy)
do_cv(m1, sx, sy)
do_cv(m2, sx2, sy2)
def do_shuffle_cv(model, x, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    scores = cross_val_score(model, x, y, cv=cv)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
do_shuffle_cv(m0, sx, sy)
do_shuffle_cv(m1, sx, sy)
do_shuffle_cv(m2, sx2, sy2)
ma.rf_show_plot_fi(m2, x_val2)
eda.plot_dendrogram(df2.drop('Survived', axis=1));
flds_to_remove = ['Cabin_pre', 'Cabin', 'Cabin_len', 'Cabin_n', 'Ticket', 'Ticket_pre', 'Ticket_n',
                  ['Pclass_2.0', 'Pclass_3.0', 'Pclass_nan']]
def print_score_custom(model, x_trn, y_trn, x_val, y_val):
    rmse_trn = rmse(model.predict(x_trn), y_trn)
    rmse_val = rmse(model.predict(x_val), y_val)
    score_trn = model.score(x_trn, y_trn)
    score_val = model.score(x_val, y_val)
    print(f"RMSE training: {rmse_trn:.6f}, validation: {rmse_val:.6f}")
    print(f"Score training: {score_trn:.6f}, validation: {score_val:.6f}")
    if hasattr(model, 'oob_score_'):
        print(f"OOB score: {model.oob_score_:.6f}")
for fld in flds_to_remove:
    print(f"--- score without {fld}")
    x_trn2_keep = x_trn2.drop(fld, axis=1)
    mx = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features=0.8, n_jobs=-1, oob_score=True)
    mx.fit(x_trn2_keep, y_trn2)
    print_score_custom(mx, x_trn2_keep, y_trn2, x_val2.drop(fld, axis=1), y_val2)
x_trn3 = x_trn2.drop('Ticket_n', axis=1)
x_val3 = x_val2.drop('Ticket_n', axis=1)
sx3 = sx2.drop('Ticket_n', axis=1)
m3 = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features=0.8, n_jobs=-1, oob_score=True)
m3.fit(x_trn3, y_trn2)
print_score_custom(m3, x_trn3, y_trn2, x_val3, y_val2)
do_shuffle_cv(m3, sx3, sy2)
model = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features=0.8, n_jobs=-1, oob_score=True)
model.fit(x2, y2)
print("OOB score:", model.oob_score_)
df_test_orig = pd.read_csv(f'{INPUT_DIR}test.csv')
df_peek(df_test_orig)
df_test_raw, df_test, x_test, y_test, nas_test = process_data(df_test_orig, y_field=None, na_dict=nas2)
preds_final = model.predict(x_test)
result = pd.DataFrame({'PassengerId': df_test.PassengerId,
                       'Survived': preds_final})
result.head()
result.to_csv(f'{OUTPUT_DIR}results.csv', index=False)