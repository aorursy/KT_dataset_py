%matplotlib inline

import pandas as pd

import numpy as np

from pandas.api.types import union_categoricals

import gc, os, sys, re, time

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

from collections import Counter, defaultdict

import matplotlib.pyplot as plt

from IPython.display import display, Image, HTML
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





DTYPE_ID = {

    'TransactionID': 'int32',

    'DeviceType': 'category',

    'DeviceInfo': 'category',

}



ID_COLS = [f'id_{i:02d}' for i in range(1, 39)]

ID_CATS = [

    'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30',

    'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'

]



DTYPE_ID.update(((c, 'float32') for c in ID_COLS))

DTYPE_ID.update(((c, 'category') for c in ID_CATS))



IN_DIR = '../input'



NR = None

NTRAIN = 590540



def read_both(t):

    df = pd.read_csv(f'{IN_DIR}/{t}_transaction.csv',

                     index_col=IDX,

                     nrows=NR,

                     dtype=DTYPE)

    df = df.join(

        pd.read_csv(f'{IN_DIR}/{t}_identity.csv',

                    index_col=IDX,

                    nrows=NR,

                    dtype=DTYPE_ID))

    print(t, df.shape)

    return df



def read_dataset():

    train = read_both('train')

    test = read_both('test')

    

    train.pop('isFraud')

    

    train['isTest'] = 0

    test['isTest'] = 1

    

    ntrain = train.shape[0]

    for c in train.columns:

        s = train[c]

        if hasattr(s, 'cat'):

            u = union_categoricals([train[c], test[c]], sort_categories=True)

            train[c] = u[:ntrain]

            test[c] = u[ntrain:]

    

    uni = train.append(test)

    return uni
uni = read_dataset()

uni.shape
uni['TimeInDay'] = uni.TransactionDT % 86400

uni['Cents'] = uni.TransactionAmt % 1
params = {

    'num_leaves': 64,

    'objective': 'binary',

    'min_data_in_leaf': 10,

    'learning_rate': 0.1,

    'feature_fraction': 0.5,

    'bagging_fraction': 0.9,

    'bagging_freq': 1,

    'max_cat_to_onehot': 128,

    'metric': 'auc',

    'num_threads': 8,

    'seed': 42,

    'subsample_for_bin': uni.shape[0]

}
class LightGbmSnoop:

    def __init__(self):

        self.train_logs = []

        self.valid_logs = []

    def _callback(self, env):

        self.model = env.model

        self.train_logs.append( [b.eval_train()[0][2] for b in self.model.boosters] )

        self.valid_logs.append( [b.eval_valid()[0][2] for b in self.model.boosters] )

    def train_log(self):

        return pd.DataFrame(self.train_logs).add_prefix('train_')

    def valid_log(self):

        return pd.DataFrame(self.valid_logs).add_prefix('valid_')

    def logs(self):

        return pd.concat((self.train_log(), self.valid_log()), 1)

    def get_oof(self, n):

        oof = np.zeros(n, dtype=float)

        for i, b in enumerate(self.model.boosters):

            vs = b.valid_sets[0]  # validation data

            idx = vs.used_indices

            # Note: this uses all trees, not the early stopping peak count.

            # You can use b.rollback_one_iter() to drop trees :)

            p = b._Booster__inner_predict(1) # 0 = train; 1 = valid

            oof[idx] = p

        return oof



TGT = 'isTest'

FEATS = uni.columns.tolist()

FEATS.remove(TGT)  

FEATS.remove('TransactionDT') # makes train/test trivially separable, remove it

print(len(FEATS), 'features')



folds = list(KFold(n_splits=4, shuffle=True, random_state=42).split(uni[FEATS]))

ds = lgb.Dataset(uni[FEATS], uni[TGT], params=params)

s = LightGbmSnoop()

res = lgb.cv(params,

             ds,

             folds=folds,

             num_boost_round=3000,

             early_stopping_rounds=100,

             verbose_eval=100,

             callbacks=[s._callback])
OOF = s.get_oof(uni.shape[0])

np.save('ieee_fraud_adversarial_lgb_oof', OOF)

roc_auc_score(uni[TGT], OOF)
pd.Series(OOF).plot.hist(bins=100, title='Histogram of predictions of p(isTest)')
pd.Series(OOF).plot(figsize=(14,6), title='OOF Prediction of p(isTest)')
pd.Series(OOF).rolling(500).mean().plot(figsize=(14,6), title='Smoothed OOF prediction of p(isTest)')
pd.Series(OOF[:NTRAIN]).plot.hist(bins=100, title='Histogram of predictions of p(isTest) - train set rows')
pd.Series(OOF[NTRAIN:]).plot.hist(bins=100, title='Histogram of predictions of p(isTest) - test set rows')
for i, b in enumerate(s.model.boosters):

    b.save_model(f'ieee_fraud_adversarial_lgb_model_{i}.txt')
s.logs().to_csv(f'ieee_fraud_adversarial_lgb_auc_logs.csv', index_label='Round')
logs = pd.DataFrame({'train':s.train_log().mean(1), 'valid':s.valid_log().mean(1)})

logs.train.plot(legend=True, title='Adversarial AUC Logs')

logs.valid.plot(legend=True)
def make_importances(clf, importance_type):

    return pd.Series(data=clf.feature_importance(importance_type), index=clf.feature_name())



IMPORTANCES = pd.concat((make_importances(b, 'gain')

                         for b in s.model.boosters), 1).sum(1).to_frame('Gain')

IMPORTANCES['Count'] = pd.concat((make_importances(b, 'split') for b in s.model.boosters), 1).sum(1)

IMPORTANCES.sort_values('Gain', ascending=False).head()
IMPORTANCES.to_csv('ieee_fraud_adversarial_lgb_importances.csv')
COLORS = [

    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',

    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'

]

toplot = IMPORTANCES.sort_values('Gain').tail(80)

toplot['Gain'].plot.barh(figsize=(12,20), color=COLORS, title='Adversarial Feature Gain')
# uncomment to see model structure

# clf.dump_model(num_iteration=2)['tree_info']
# NOTE: lightgbm.Booster has a new get_split_value_histogram API which counts split points used.

# This code pre-dates that, and sums gain instead of counting appearances.

# Here it is adapted from the original to use a collection of models, and sum the overall data.

def get_split_point_stats_multi(clfs):

    split_points = defaultdict(Counter)



    def visit_node(d):

        if 'tree_info' in d:

            for tree in d['tree_info']: # a list of trees

                visit_node(tree)

        for k in ['tree_structure', 'left_child', 'right_child' ]:

            if k in d:

                visit_node(d[k])

        if 'split_feature' in d:

            split_points[names[d['split_feature']]] [d['threshold']] += d['split_gain']



    for clf in clfs:

        names = clf.feature_name()

        visit_node(clf.dump_model())

    return split_points
split_points = get_split_point_stats_multi(s.model.boosters)
split_points['card1'].most_common(5)
with pd.ExcelWriter('ieee_fraud_adversarial_split_points.xlsx') as writer:

    for feat in FEATS:

        counter = split_points[feat]

        df = pd.Series(counter, name=feat).sort_index().to_frame('GainSum')

        df.to_excel(writer, feat, index_label=feat)



    for sheet in writer.sheets.values():

        sheet.set_column(0, 0, 30)
MAX_SHOW = 50
ADJS = 'abundant:common:ubiquitous:omnipresent:rampant:rife:permeant:widespread:legendary:popular:fashionable:frequent:usual:useful:predominant:recurrent:repetitive:repetitious:marked:prevalent:prevalent:prevalent'.split(':')



np.random.seed(42)



def plot_it(col):

    counts = split_points[col]

    ser = pd.Series(dict(counts)).sort_values(ascending=False)

    total_gain = IMPORTANCES.loc[col, 'Gain']

    total_splits = IMPORTANCES.loc[col, 'Count']

    if hasattr(uni[col], 'cat'):

        # remap categories from int -> cat value

        try:

            ser.index = uni[col].cat.categories[ser.index.astype(int)]

        except:

            # e.g. TypeError: Cannot cast Index to dtype <class 'int'>

            # a categorical with many categories and '1||4||7' etc type splits

            # leave it as it is

            pass

    adj = np.random.choice(ADJS)

    display(

        HTML(

            f'<h1 id="plot_{col}">{col}</h1>'

            f'<p>Used {total_splits} times, total gain is {total_gain}.'

            f'<p>{len(ser)} split point values used. '

            f'Most {adj} is {ser.index[0]} with gain of {ser.values[0]}.'

        )

    )

    ser = ser.head(MAX_SHOW).sort_index()

    ax = ser.plot.bar(title=f'{col} — Adversarial split points by gain',

                      rot=90, fontsize=12, figsize=(15,5),

                      width=0.7, color=COLORS)

    plt.show()
for col in FEATS:

    counts = split_points[col]

    if len(counts) >= 2:

        plot_it(col)