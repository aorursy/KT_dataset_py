%matplotlib inline

import pandas as pd

import numpy as np

import gc, os, sys, re, time

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from collections import Counter, defaultdict

import matplotlib.pyplot as plt

from IPython.display import display, HTML
DTYPE = {

    'TransactionID': 'int32',

    'isFraud': 'int8',

    'TransactionDT': 'int32',

    'TransactionAmt': 'float32',

    'ProductCD': 'category',

    'card1': 'int16',

    'card2': 'float32',

    'card3': 'float32',

    'card4': 'category',

    'card5': 'float32',

    'card6': 'category',

    'addr1': 'float32',

    'addr2': 'float32',

    'dist1': 'float32',

    'dist2': 'float32',

    'P_emaildomain': 'category',

    'R_emaildomain': 'category',

}



IDX = 'TransactionID'

TGT = 'isFraud'



CCOLS = [f'C{i}' for i in range(1, 15)]

DCOLS = [f'D{i}' for i in range(1, 16)]

MCOLS = [f'M{i}' for i in range(1, 10)]

VCOLS = [f'V{i}' for i in range(1, 340)]



DTYPE.update((c, 'float32') for c in CCOLS)

DTYPE.update((c, 'float32') for c in DCOLS)

DTYPE.update((c, 'float32') for c in VCOLS)

DTYPE.update((c, 'category') for c in MCOLS)



IN_DIR = '../input'



NR = None



tran = pd.read_csv(f'{IN_DIR}/train_transaction.csv', index_col=IDX, nrows=NR, dtype=DTYPE)

tran.shape
# utility: encode binary 0/1 columns as bits in a single integer

def encode_bits(binary_df):

    ncols = binary_df.shape[1]

    assert ncols < 64

    return binary_df @ (1 << np.arange(ncols))
to_count = tran.columns[2:].tolist()



for c in to_count:

    s = tran[c]

    if hasattr(s, 'cat'):

        s = s.cat.codes

    vc = s.value_counts(dropna=False)

    tran[f'{c}_count'] = s.map(vc).astype(np.int32)
tran['TimeInDay'] = tran.TransactionDT % 86400

tran['Cents'] = tran.TransactionAmt % 1

tran['C_bin'] = encode_bits(tran[CCOLS]>0)

tran['D_bin'] = encode_bits(tran[DCOLS].isnull())

tran['M_bin'] = encode_bits(tran[MCOLS].isnull())

tran['addr_bin'] = encode_bits(tran[['addr1','addr2','dist1','dist2']].isnull())

tran['email_bin'] = encode_bits(tran[['R_emaildomain','P_emaildomain']].isnull())
split = tran.TransactionDT.quantile(0.75)

istrain = tran.TransactionDT < split

train_df = tran.loc[istrain]

valid_df = tran.loc[~istrain]

print(train_df.shape, valid_df.shape)
params = {

    'num_leaves': 64,

    'objective': 'binary',

    'min_data_in_leaf': 12,

    'learning_rate': 0.01,

    'feature_fraction': 0.6,

    'bagging_fraction': 0.9,

    'bagging_freq': 1,

    'max_cat_to_onehot': 128,

    'metric': 'auc',

    'num_threads': 8,

    'seed': 42,

}
y_tr = train_df[TGT].values

y_va = valid_df[TGT].values



use = [c for c in train_df.columns if c != TGT]

train = train_df[use]

valid = valid_df[use]



dtrain = lgb.Dataset(train, y_tr, params=params)

dvalid = lgb.Dataset(valid, y_va, params=params, reference=dtrain)



clf = lgb.train(params,

                dtrain,

                num_boost_round=3000,

                valid_sets=(dvalid,),

                early_stopping_rounds=100,

                verbose_eval=100)
roc_auc_score(y_va, clf.predict(valid))
_ = clf.save_model('ieee_fraud_lgb_model.txt')
# uncomment to see model structure

# clf.dump_model(num_iteration=2)['tree_info']
# NOTE: lightgbm.Booster has a new get_split_value_histogram API which counts split points used.

# This code pre-dates that, and sums gain instead of counting appearances.

def get_split_point_stats(clf):

    split_points = defaultdict(Counter)

    names = clf.feature_name()



    def visit_node(d):

        if 'tree_info' in d:

            for tree in d['tree_info']: # a list of trees

                visit_node(tree)

        for k in ['tree_structure', 'left_child', 'right_child' ]:

            if k in d:

                visit_node(d[k])

        if 'split_feature' in d:

            split_points[names[d['split_feature']]] [d['threshold']] += d['split_gain']



    visit_node(clf.dump_model())

    return split_points
split_points = get_split_point_stats(clf)
split_points['C1'].most_common(5)
with pd.ExcelWriter('ieee_fraud_split_points.xlsx') as writer:

    for feat in use:

        counter = split_points[feat]

        df = pd.Series(counter, name=feat).sort_index().to_frame('GainSum')

        df.to_excel(writer, feat, index_label=feat)



    for sheet in writer.sheets.values():

        sheet.set_column(0, 0, 30)
MAX_SHOW = 50
ADJS = 'abundant:common:ubiquitous:omnipresent:rampant:rife:permeant:widespread:legendary:popular:fashionable:frequent:usual:useful:predominant:recurrent:repetitive:repetitious:marked:prevalent:prevalent:prevalent'.split(':')

COLORS = [

    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',

    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'

]

GAINS = pd.Series(index=clf.feature_name(), data=clf.feature_importance('gain'))

COUNTS = pd.Series(index=clf.feature_name(), data=clf.feature_importance())

np.random.seed(42)



def plot_it(col):

    counts = split_points[col]

    ser = pd.Series(dict(counts)).sort_values(ascending=False)

    if hasattr(tran[col], 'cat'):

        # remap categories from int -> cat value

        try:

            ser.index = tran[col].cat.categories[ser.index.astype(int)]

        except:

            # e.g. TypeError: Cannot cast Index to dtype <class 'int'>

            # a categorical with many categories and '1||4||7' etc type splits

            # leave it as it is

            pass

    adj = np.random.choice(ADJS)

    display(

        HTML(

            f'<h1 id="plot_{col}">{col}</h1>'

            f'<p>Used {COUNTS[col]} times, total gain is {GAINS[col]}.'

            f'<p>{len(ser)} split point values used. '

            f'Most {adj} is {ser.index[0]} with gain of {ser.values[0]}.'

        )

    )

    ser = ser.head(MAX_SHOW).sort_index()

    ax = ser.plot.bar(title=f'{col} — Split points by gain',

                      rot=90, fontsize=12, figsize=(15,5),

                      width=0.7, color=COLORS)

    plt.show()
for col in use:

    counts = split_points[col]

    if len(counts) >= 4:

        plot_it(col)
def make_importances(clf, divisions):

    max_n = clf.num_trees()

    thres = np.arange(1, divisions+1) / divisions



    idx = pd.Index(clf.feature_name(), name='Feature')

    importances = pd.DataFrame(index=idx)



    for t in thres:

        n = int(max_n * t)

        c = f'count_{t*100:.0f}'

        dat = clf.feature_importance(iteration=n)

        importances[c] = pd.Series(dat, index=idx).astype(np.int32)



    for t in thres:

        n = int(max_n * t)

        c = f'gain_{t*100:.0f}'

        dat = clf.feature_importance('gain', iteration=n)

        importances[c] = pd.Series(dat, index=idx).astype(np.float32)



    return importances
importances = make_importances(clf, 5)
importances.sort_values('gain_100', ascending=False).head(10)
importances.to_csv('ieee_fraud_lgb_importances.csv')
toplot = importances.sort_values('gain_100').tail(80)

toplot['gain_100'].plot.barh(figsize=(12,20), legend=True, color='red', title='Feature Gain at 40% and 100%')

toplot['gain_40'].plot.barh(figsize=(12,20), legend=True, color='royalblue')
gaincols = importances.columns[importances.columns.str.startswith('gain')].tolist()

toplot = importances.sort_values('gain_100', ascending=False).head(10).copy()

toplot['gain_0'] = 0

toplot[['gain_0'] + gaincols].T.plot(figsize=(14,6), title='Cumulative Gain')
gaincols = importances.columns[importances.columns.str.startswith('gain')].tolist()

toplot = importances.sort_values('count_20', ascending=False).head(10).copy()

toplot['gain_0'] = 0

toplot[['gain_0'] + gaincols].T.plot(figsize=(14,6), title='Cumulative Gain')