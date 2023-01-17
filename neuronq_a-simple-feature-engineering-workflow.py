INPUT_DIR = '../input/'
OUTPUT_DIR = './'
!ls -lah {INPUT_DIR}
!pip install --upgrade sparklines
!pip install --upgrade nmlu
!pip install --upgrade treeinterpreter
%load_ext autoreload
%autoreload 2
%matplotlib inline

import re

import pprint
pp = pprint.PrettyPrinter(indent=2).pprint

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    cross_val_score, ShuffleSplit, train_test_split, GridSearchCV)
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

# NMLU (Nano Machine Learning Utils) - my lib of a few simple helpers
# to keep boring code DRY
from nmlu.qinspect.nb import df_peek, df_display_all
from nmlu.qinspect.common import df_types_and_stats
import nmlu.etl as etl
import nmlu.eda as eda
import nmlu.model_analysis as ma
import nmlu.model_utils as mu

eda.set_plot_sane_defaults()
sns.set()
mpl.rcParams['figure.facecolor'] = 'white'
df_raw = pd.read_csv(f'{INPUT_DIR}train.csv', low_memory=False)
orig_df_raw = df_raw.copy()
df_test_raw = pd.read_csv(f'{INPUT_DIR}test.csv', low_memory=False)
def make_train_test_data(input_df, test_f=0.2):
    df = input_df.copy()
    etl.train_cats(df)
    x, y, nas = etl.proc_df(
        df,
        y_fld='Survived',
        max_n_cat=10,
        no_binary_dummies=True,
    )
    x_trn, x_val, y_trn, y_val = train_test_split(
        x, y, test_size=int(test_f * len(x)),
    )
    return (
        df,
        x, y, nas,
        x_trn, y_trn, x_val, y_val
    )
def make_test_data(input_df_test_raw, df_train, nas):
    df_test_raw = input_df_test_raw.copy()
    

    etl.apply_cats(df_test_raw, df_train)
    
    print("-- types b4:", df_test_raw.dtypes)
    
    
    
    test_x, _, _ = etl.proc_df(
        df_test_raw,
        max_n_cat=10,
        no_binary_dummies=True,
        na_dict=nas.copy()
    )
    
    return df_test_raw, test_x
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

def do_cv(model, x, y):
    scores = cross_val_score(model, x, y, cv=10, n_jobs=-1)
    r = "Accuracy: %0.2f (+/- %0.2f)" % (
        scores.mean() * 100, scores.std() * 2 * 100)
    return r

def do_shuffle_cv(model, x, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    scores = cross_val_score(model, x, y, cv=cv, n_jobs=-1)
    r = "Accuracy (shuffled): %0.2f +/- %0.2f [%.2f - %.2f]" % (
        scores.mean() * 100,
        scores.std() * 2 * 100,
        np.min(scores) * 100, np.max(scores) * 100)
    return r

def try_model(model, x, y, x_trn, y_trn, x_val, y_val):
    model.fit(x_trn, y_trn)
    pp(get_score(model, x_trn, y_trn, x_val, y_val))
    pp(do_shuffle_cv(model, x, y))
(
    orig_df,
    orig_x, orig_y, orig_nas,
    orig_x_trn, orig_y_trn, orig_x_val, orig_y_val
) = make_train_test_data(orig_df_raw)
df_test, test_x = make_test_data(df_test_raw, orig_df, orig_nas)
m0 = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_depth=5,
    max_features=10,
    oob_score=True, n_jobs=-1)
try_model(m0, orig_x, orig_y,
          orig_x_trn, orig_y_trn, orig_x_val, orig_y_val)
def df_proc_fare(df):
    df = df.copy()
    count_for_ticket = df.groupby('Ticket').Fare.count()
    ticket_size = df.Ticket.map(count_for_ticket)
    df['Fare_adj'] = df.Fare / ticket_size
    df.drop(columns='Fare', inplace=True)
    return df
# df_proc_fare(df_raw)[['Fare', 'Fare_adj']][:10]
df, x, y, nas, x_trn, y_trn, x_val, y_val = make_train_test_data(
    df_proc_fare(df_raw))
df_raw.Fare.hist(bins=80, figsize=(16, 4))
plt.show()
df.Fare_adj.hist(bins=80, figsize=(16, 4))
try_model(m0, x, y, x_trn, y_trn, x_val, y_val)
def df_proc_age_by_pclass(df, age_by_pclass=None):
    df = df.copy()
    if age_by_pclass is None:
        age_by_pclass = df[
            ['Pclass', 'Age']].groupby('Pclass').Age.median()
    idxs = df.Age.isnull()
    df.loc[idxs, 'Age'] = df.loc[idxs, 'Pclass'].map(age_by_pclass)
    return df, age_by_pclass
# df_proc_age_by_pclass(df_raw)[0][['Age', 'Pclass']][:10]
df, x, y, nas, x_trn, y_trn, x_val, y_val = make_train_test_data(
    df_proc_age_by_pclass(df_raw)[0])
try_model(m0, x, y, x_trn, y_trn, x_val, y_val)
def df_proc_name(df):
    df = df.copy()
    df[
        ['Name_family', 'Name_title', 'Name_first']
    ] = df.Name.str.split(r'[,.]\s+', n=2, expand=True)
    df['Name_len'] = df.Name.str.len()
    df['Name_words'] = df.Name.apply(
        lambda s: len(s.split()))
    df.drop('Name', axis=1, inplace=True)
    return df
# df_proc_name(df_raw)[
#     ['Name', 'Name_title', 'Name_family', 'Name_first']][:10]
df, x, y, nas, x_trn, y_trn, x_val, y_val = make_train_test_data(
    df_proc_name(df_raw))
try_model(m0, x, y, x_trn, y_trn, x_val, y_val)
x.shape
m1 = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_depth=5,
    max_features=10,
    oob_score=True, n_jobs=-1)
try_model(m1, x, y, x_trn, y_trn, x_val, y_val)
for fld in [[], 'Name_first', 'Name_family', 'Name_len', 'Name_words']:
    print(f"--without {fld}")
    np.random.seed(42)
    df, x, y, nas, x_trn, y_trn, x_val, y_val = make_train_test_data(
        df_proc_name(df_raw).drop(columns=fld))
    try_model(m0, x, y, x_trn, y_trn, x_val, y_val)
df, x, y, nas, x_trn, y_trn, x_val, y_val = make_train_test_data(
    df_proc_name(df_raw).drop(columns=['Name_first', 'Name_len']))
try_model(m0, x, y, x_trn, y_trn, x_val, y_val)
def df_proc_name_family_survived(df, lname_survived=None):
    df = df.copy()
    if lname_survived is None:
        lname_survived = df[['Name_family', 'Survived']]\
            .groupby('Name_family').Survived.sum()
    df['Name_family_survived'] = df.Name_family.map(lname_survived)
    df['Name_family_survived'] = df.Name_family_survived.fillna(0)
    return df, lname_survived
df_proc_name_family_survived(df)[0][:10]
x.shape
m = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_depth=5,
    max_features=15,
    oob_score=True, n_jobs=-1)
df_name_family_survived, lname_survived = df_proc_name_family_survived(
    df_proc_name(df_raw).drop(columns=['Name_first', 'Name_len'])
)
df, x, y, nas, x_trn, y_trn, x_val, y_val = make_train_test_data(
    df_name_family_survived
)
try_model(m, x, y, x_trn, y_trn, x_val, y_val)
def df_proc_ticket(df):
    df = df.copy()
    
    df['Ticket_pre'] = df.Ticket.str.extract(r'([^s]+)\s+')
    df['Ticket_n'] = df.Ticket.str.extract(r'(\d+)$')
    df['Ticket_len'] = df.Ticket.apply(
        lambda s: len(s.split()))
    
    return df

df_proc_ticket(df_raw)[[
    'Ticket', 'Ticket_pre', 'Ticket_n', 'Ticket_len'
]][:10]
df, x, y, nas, x_trn, y_trn, x_val, y_val =\
make_train_test_data(df_proc_ticket(df_raw))
try_model(m, x, y, x_trn, y_trn, x_val, y_val)
def df_proc_ticket_survived(
    df, ticket_survived=None,
):
    df = df.copy()
    
    if ticket_survived is None:
        ticket_survived = df[
            ['Ticket', 'Survived']
        ].groupby('Ticket').Survived.sum()
    df['Ticket_survived'] = df.Ticket.map(ticket_survived)
    df['Ticket_survived'] = df.Ticket_survived.fillna(0)

#     if ticket_pre_survived is None:
#         ticket_pre_survived = df[
#             ['Ticket_pre', 'Survived']
#         ].groupby('Ticket_pre').Survived.sum()
#     df['Ticket_pre_survived'] =\
#         df.Ticket_pre.map(ticket_pre_survived)
#     df['Ticket_pre_survived'] =\
#         df.Ticket_pre_survived.fillna(0)
    
    return df, ticket_survived

df_proc_ticket_survived(
    df_proc_ticket(df_raw)
)[0]
df, x, y, nas, x_trn, y_trn, x_val, y_val =\
make_train_test_data(
    df_proc_ticket_survived(
        df_proc_ticket(df_raw)
    )[0]
)
try_model(m, x, y, x_trn, y_trn, x_val, y_val)
def df_proc_cabin(df):
    df = df.copy()
    
    df['Cabin_pre'] = df.Cabin.str.extract(r'([^\d\s]+)')
    df['Cabin_n'] = df.Cabin.str.extract(r'(\d+)$')
    df['Cabin_len'] = df.Cabin.str.split(r'\s+').apply(
        lambda x: (
            0
            if not hasattr(x, '__len__') else
            len(x)
        )
    )
    
    return df

df_proc_cabin(df_raw)[[
    'Cabin', 'Cabin_pre', 'Cabin_n', 'Cabin_len'
]][:10]
df, x, y, nas, x_trn, y_trn, x_val, y_val =\
make_train_test_data(df_proc_cabin(df_raw))
try_model(m, x, y, x_trn, y_trn, x_val, y_val)
def df_proc_cabin_survived(
    df, cabin_survived=None
):
    df = df.copy()
    
    if cabin_survived is None:
        cabin_survived = df[
            ['Cabin', 'Survived']
        ].groupby('Cabin').Survived.sum()
    df['Cabin_survived'] = df.Cabin.map(cabin_survived)
    df['Cabin_survived'] = df.Cabin_survived.fillna(0)
    
    return df, cabin_survived

df_proc_cabin_survived(
    df_proc_cabin(df_raw)
)[0][:10]
df, x, y, nas, x_trn, y_trn, x_val, y_val =\
make_train_test_data(
    df_proc_cabin_survived(
        df_proc_cabin(df_raw)
    )[0]
)
try_model(m, x, y, x_trn, y_trn, x_val, y_val)
def df_proc_all(
    df,
    age_by_pclass=None,
    lname_survived=None,
    ticket_survived=None,
#     ticket_pre_survived=None,
#     cabin_survived=None,
    drop=[],
):
    df = df.copy()
    df = df_proc_fare(df)
    df, age_by_pclass = df_proc_age_by_pclass(
        df, age_by_pclass)
    df = df_proc_name(df)
    df, lname_survived = df_proc_name_family_survived(
        df, lname_survived)
    df = df_proc_ticket(df)
#     df, ticket_survived =\
#         df_proc_ticket_survived(
#             df, ticket_survived)
    df = df_proc_cabin(df)
#     df, cabin_survived = df_proc_cabin_survived(
#         df, cabin_survived)
    df.drop(columns=drop, inplace=True)
    return (
        df,
        age_by_pclass,
        lname_survived,
#         ticket_survived,
#         ticket_pre_survived,
#         cabin_survived,
        drop,
    )

df_display_all(
    df_proc_all(orig_df_raw)[0][:10].T.sort_index())
(
    df_proc,
    age_by_pclass,
    lname_survived,
#     ticket_survived,
#     ticket_pre_survived,
#     cabin_survived,
    drop,
) = df_proc_all(
    orig_df_raw,
    drop=[
        'Name_first', 'Name_len',
        'Ticket',
        'Cabin',
    ],
)

df, x, y, nas, x_trn, y_trn, x_val, y_val =\
    make_train_test_data(df_proc)
x.shape
df
# mu.reset_rf_samples()
m = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_depth=4,
    max_features=18,
    oob_score=True,
    n_jobs=-1)
try_model(m, x, y, x_trn, y_trn, x_val, y_val)
params = dict(
    n_estimators=100,
    criterion='entropy',
    n_jobs=-1
)
plt.figure(figsize=(24, 30))
max_depths = [2, 4, 5, 8]
max_feature_ns = [5, 10, 12, 16, 18]
i = 1
for ir, max_features in enumerate(max_feature_ns):
    for ic, max_depth in enumerate(max_depths):
        model = RandomForestClassifier(
            **params, max_features=max_features, max_depth=max_depth)
        plt.subplot(len(max_feature_ns), len(max_depths), i)
        i += 1
        ma.plot_train_vs_test(model, x, y, step=100, n_runs=5, ylim=(0.8, 1))
        plt.title(f"max_features={max_features}, max_depth={max_depth}")
plt.tight_layout()
df_test_raw = pd.read_csv(f'{INPUT_DIR}test.csv',
                          low_memory=False)
df_test = df_proc_all(
    df_test_raw,
    age_by_pclass=age_by_pclass,
    lname_survived=lname_survived,
#     ticket_survived,
#     ticket_pre_survived,
#     cabin_survived,
    drop=drop,
)[0]

# df_peek(df_test)

df_test, test_x = make_test_data(
    df_test,
    df,
    nas
)


m
m.fit(x, y)
print("OOB score:", m.oob_score_)
preds_final = m.predict(test_x)
print("Predict % survived:", np.mean(preds_final))
result = pd.DataFrame({
    'PassengerId': test_x.PassengerId,
    'Survived': preds_final,
})
display(result)
result.to_csv(
    f'{OUTPUT_DIR}results_proc_all_tweaked.csv',
    index=False)