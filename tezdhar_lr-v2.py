import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import gmean

from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold, StratifiedShuffleSplit

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline, make_union

from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, MinMaxScaler, StandardScaler
ROOT = '/kaggle/input/cat-in-the-dat/'



class FieldNames:

    idx = 'id'

    bin_0 = 'bin_0'

    bin_1 = 'bin_1'

    bin_2 = 'bin_2'

    bin_3 = 'bin_3'

    bin_4 = 'bin_4'

    nom_0 = 'nom_0'

    nom_1 = 'nom_1'

    nom_2 = 'nom_2'

    nom_3 = 'nom_3'

    nom_4 = 'nom_4'

    nom_5 = 'nom_5'

    nom_6 = 'nom_6'

    nom_7 = 'nom_7'

    nom_8 = 'nom_8'

    nom_9 = 'nom_9'

    ord_0 = 'ord_0'

    ord_1 = 'ord_1'

    ord_2 = 'ord_2'

    ord_3 = 'ord_3'

    ord_4 = 'ord_4'

    ord_5 = 'ord_5'

    ord_5a = 'ord_5a'

    ord_5b = 'ord_5b'

    day = 'day'

    month = 'month'

    target = 'target'





class FileNames:

    train = 'train.csv'

    test = 'test.csv'

    train_v1 = 'train_v1.ftr'

    test_v1 = 'test_v1.ftr'

    sub = 'submission.csv'





class ColumnGroup:

    binary = [FieldNames.bin_0, FieldNames.bin_1, FieldNames.bin_2,

              FieldNames.bin_3, FieldNames.bin_4]

    ordinal = [FieldNames.ord_0, FieldNames.ord_1, FieldNames.ord_2,

               FieldNames.ord_3, FieldNames.ord_4, FieldNames.ord_5,

               ]

    nominal = [FieldNames.nom_0, FieldNames.nom_1, FieldNames.nom_2,

               FieldNames.nom_3, FieldNames.nom_4, FieldNames.nom_5,

               FieldNames.nom_6, FieldNames.nom_7, FieldNames.nom_8,

               FieldNames.nom_9]
"""All the transformers go here."""



from abc import abstractmethod

from collections import Counter, defaultdict

import itertools



import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from scipy.spatial.distance import cosine

from tqdm import tqdm





def _convert_to_2d_array(X):

    X = np.array(X)

    if X.ndim == 1:

        return X.reshape(-1, 1)

    return X





class BaseTransformer(BaseEstimator, TransformerMixin):

    """Base interface for transformer."""



    def fit(self, X, y=None):

        """Learn something from data."""

        return self



    @abstractmethod

    def _transform(self, X):

        pass



    def transform(self, X, y=None):

        """Transform data."""

        return self._transform(X)





class ArrayTransformer(BaseTransformer):

    """Transformer to be used for returnng 2d arrays."""



    def transform(self, X):

        """Transform data and return 2d array."""

        Xt = self._transform(X)

        return _convert_to_2d_array(Xt)





class SelectCols(BaseTransformer):

    """Select column of a dataframe."""



    def __init__(self, cols):

        """Initialie columns to be selected."""

        self.cols = cols



    def _transform(self, X, y=None):

        return X[self.cols]





class LabelEncoderWithThresh(ArrayTransformer):

    """Transform categoricals to inetgers having count more than threshold."""



    def __init__(self, threshold=0):

        self.threshold = threshold

        self.label2int = []



    def fit(self, X, y=None):

        self.label2int = []

        X = np.asarray(X)

        m = X.shape[1]

        if not isinstance(self.threshold, list):

            thresholds = [self.threshold]*m

        else:

            thresholds = self.threshold



        for j in range(m):

            cnts = Counter(X[:, j])

            self.label2int.append({lbl: i+1 for i, (lbl, cnt) in enumerate(cnts.items()) if cnt >

                                   thresholds[j]})

        return self



    def _transform(self, X):

        X = np.asarray(X)

        Xt = np.zeros_like(X)

        for i in range(len(self.label2int)):

            Xt[:, i] = [self.label2int[i].get(val, 0) for val in X[:, i]]

        return Xt

def map_ordinals(df):

    ord_1_map = {'Novice': 0, 'Contributor': 1, 'Expert': 2, 'Master': 3, 'Grandmaster': 4.0}

    ord_2_map = {'Freezing': 0, 'Cold': 1, 'Warm': 2, 'Hot': 3, 'Boiling Hot': 4, 'Lava Hot': 5.3}

    ord_3_map = {char: i for i, char in enumerate(sorted(df.ord_3.unique()))}

    ord_4_map = {char: i for i, char in enumerate(sorted(df.ord_4.unique()))}

    ord_5_map = {char: i for i, char in enumerate(sorted(df.ord_5.unique()))}

    bin_4_map = {'N': 0, 'Y': 1}

    bin_3_map = {'F': 0, 'T': 1}



    df[FieldNames.ord_1] = df[FieldNames.ord_1].map(ord_1_map)

    df[FieldNames.ord_2] = df[FieldNames.ord_2].map(ord_2_map)

    df[FieldNames.ord_3] = df[FieldNames.ord_3].map(ord_3_map)

    df[FieldNames.ord_4] = df[FieldNames.ord_4].map(ord_4_map)

    df[FieldNames.ord_5a] = df[FieldNames.ord_5].str[0]

    df[FieldNames.ord_5b] = df[FieldNames.ord_5].str[1]

    ord_5a_map = {char: i for i, char in enumerate(sorted(df.ord_5a.unique()))}

    ord_5b_map = {char: i for i, char in enumerate(sorted(df.ord_5b.unique()))}

    df[FieldNames.ord_5a] = df[FieldNames.ord_5a].map(ord_5a_map)

    df[FieldNames.ord_5b] = df[FieldNames.ord_5b].map(ord_5b_map)



    df[FieldNames.ord_5] = df[FieldNames.ord_5].map(ord_5_map)

    df[FieldNames.bin_3] = df[FieldNames.bin_3].map(bin_3_map)

    df[FieldNames.bin_4] = df[FieldNames.bin_4].map(bin_4_map)

    return df
tr = pd.read_csv(ROOT + FileNames.train)

te = pd.read_csv(ROOT + FileNames.test)



# tr.loc[tr.ord_3 == 'm', 'ord_3'] = 'zzz'

# tr.loc[tr.ord_3 == 'l', 'ord_3'] = 'm'

# tr.loc[tr.ord_3 == 'zzz', 'ord_3'] = 'l'



tr = map_ordinals(tr)

te = map_ordinals(te)



#tr = tr.loc[~((tr.day == 6) & (tr.month==6))]

print('Mapping binary and ordinal columns done.')

y = tr.target.values.astype(int)

# cvlist1 = list(StratifiedKFold(5, random_state=12345786, shuffle=True).split(tr, y))

# cvlist = list(StratifiedKFold(20, random_state=123457869).split(tr, y))

cvlist = list(StratifiedShuffleSplit(n_splits=5, test_size=0.05, random_state=123457869).split(tr, y))
# y[14932] = 0

# tr.ord_5.value_counts()
qtr_map = {1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3, 10:4, 11:4, 12:4}

tr['qtr'] = tr['month'].map(qtr_map)

te['qtr'] = te['month'].map(qtr_map)
day_map = {4:0, 5:1, 3:1, 6:2, 2:2, 7:3, 1:3}

tr['day2'] = tr['day'].map(day_map)

te['day2'] = te['day'].map(day_map)
mm_map = {1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:3, 8:3, 9:4, 10:4, 11:5, 12:5}

tr['month2'] = (tr['month'] >= 6).astype(int)

te['month2'] = (te['month'] >= 6).astype(int)
bin_cols = ColumnGroup.binary

ord_cols = ColumnGroup.ordinal

nom_cols = ColumnGroup.nominal

# nom_cols = [col for col in nom_cols if col not in ['nom_3']]

cyc_cols = ['day', 'month']
pd.set_option('display.max_columns', 200)

tr.loc[(tr.ord_4 == 11) & (tr.ord_5 == 11)]
#y[113760] = 1
plt.plot(tr.groupby('ord_5b')['target'].mean())
from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb
feat_pipe2 = make_union(

    make_pipeline(SelectCols(nom_cols+['month', 'day2']),

                  LabelEncoderWithThresh(3),

                  OneHotEncoder(),

                  ),

    make_pipeline(SelectCols(bin_cols)),

    make_pipeline(SelectCols(ord_cols + ['qtr', 'ord_5a']), 

                  MinMaxScaler((0, 1))),

)



feat_pipe1 = make_union(

    make_pipeline(SelectCols(nom_cols+['month', 'day']),

                  LabelEncoderWithThresh(1),

                  OneHotEncoder(),

                  ),

    make_pipeline(SelectCols(bin_cols)),

    make_pipeline(SelectCols(ord_cols + ['ord_5a']), 

                  MinMaxScaler((0, 1))),

)





model = LogisticRegression(C=0.11, solver='lbfgs', random_state=12345786, max_iter=5000, tol=1e-8, fit_intercept=True)

#model = lgb.LGBMClassifier(n_estimators=10000, num_leaves=2, subsample=0.7, colsample_bytree=0.1, metric='None',

#                           categorical_feature=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], max_bin=512,

#                          lambda_l1=1.0, lambda_l2=10, cat_l2=10.0, cat_smooth=1, min_child_samples=2, min_data_per_group=2)

# pipe1 = make_pipeline(feat_pipe2, model)

# pipe1.named_steps
def cross_val_pred(pipe, model, tr, y, te, cvlist):

    scores = []

    val_preds = []

    y_trues = []

    test_preds = []

    pipe.fit(pd.concat([tr, te]))

    for trix, vlix in cvlist:

        xtr, ytr = tr.iloc[trix], y[trix]

        xvl, yvl = tr.iloc[vlix], y[vlix]

        pipe.fit(xtr, ytr)

        xtr = pipe.transform(xtr)

        xvl = pipe.transform(xvl)

        print(xtr.shape)

        model.fit(xtr, ytr)

        pvl = model.predict_proba(xvl)[:, 1]

        # pte = model.predict_proba(xte)[:, 1]

        val_preds.extend(pvl)

        y_trues.extend(yvl)

        # test_preds.append(pte)

        score = roc_auc_score(yvl, pvl)

        print("Score ", score)

        scores.append(score)

        # break

    print("Avg score ", np.mean(scores), np.std(scores))

    print("Overall score ", roc_auc_score(y_trues, val_preds))

    # test_preds = gmean(test_preds, 0)

    xtr = pipe.transform(tr)

    xte = pipe.transform(te)

    test_preds = model.fit(xtr, y).predict_proba(xte)[:, 1]

    return val_preds, y_trues, test_preds
#41493, 14932
val_preds1, y_trues, test_preds1 = cross_val_pred(feat_pipe2, model, tr, y, te, cvlist)

# val_preds2, y_trues, test_preds2 = cross_val_pred(feat_pipe1, model, tr, y, te, cvlist)
# roc_auc_score(y_trues, np.mean([val_preds1, val_preds2], 0))
sns.distplot(val_preds1)

sns.distplot(test_preds1)

plt.show()
sub = te[['id']]

sub['target'] = test_preds1

sub.to_csv('submission.csv', index=False)
sub.head()